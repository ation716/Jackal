# -*- coding: utf-8 -*-
# @Time    : 2026/4/2 11:50
# @Author  : gaolei
# @FileName: big_fund.py
"""
big_fund.py — Real-time large-order detector for A-share stocks.

Features:
  - Fetches intraday tick data via tushare realtime_tick (sina source).
  - Restricts ticks to the 09:30-15:00 trading session.
  - Price-tiered large-order thresholds (lots):
      price < 6         -> >= 5000
      6  <= price < 12  -> >= 3000
      12 <= price < 24  -> >= 1500
      24 <= price < 48  -> >= 800
      price >= 48       -> >= 500
  - Appends each session's large-order details to results/big_fund/{code}.csv.
  - Fields written: time, current change%, price, volume (lots), order type.
  - Calculates and appends the weighted average cost of buy-side large orders.
"""

import os
import tushare as ts
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'big_fund')
TUSHARE_TOKEN = "1bf9b910cdda6f0cd856f55b97c1c1419860237f7be8156aacac3259"

# Trading session window (Asia/Shanghai)
SESSION_START = '09:30:00'
SESSION_END = '15:00:00'

# Price-tiered large-order thresholds. Each tuple: (upper_price_exclusive, min_lots, label).
# A price falls into the first tier whose upper bound it is strictly less than;
# the final tier (float('inf')) catches everything else.
PRICE_TIERS = [
    (6,             5000, 'Tier-1(<6)'),
    (12,            3000, 'Tier-2(6-12)'),
    (24,            1500, 'Tier-3(12-24)'),
    (48,             800, 'Tier-4(24-48)'),
    (float('inf'),   500, 'Tier-5(>=48)'),
]


def _tier_for(price: float):
    """Return the (min_lots, label) tier for *price*."""
    for upper, min_lots, label in PRICE_TIERS:
        if price < upper:
            return min_lots, label
    # Unreachable — last tier has upper == inf — but keep a safe fallback.
    return PRICE_TIERS[-1][1], PRICE_TIERS[-1][2]


def _big_threshold(price: float) -> int:
    """Dynamic large-order threshold (lots) keyed by price."""
    return _tier_for(price)[0]


def get_big_orders(ts_code: str, page_count: int = 60):
    """
    Fetch large-order ticks for *ts_code* and append them to a per-stock CSV.

    Parameters
    ----------
    ts_code    : stock code in tushare format, e.g. '600000.SH'
    page_count : number of pages to pull from realtime_tick (default 60)
    """
    ts.set_token(TUSHARE_TOKEN)

    df = ts.realtime_tick(ts_code=ts_code, src='sina', page_count=page_count)

    if df is None or len(df) == 0:
        print(f"{ts_code}: no tick data returned")
        return

    # Normalise column names to upper-case
    df.columns = [c.upper() for c in df.columns]

    for col in ('PRICE', 'VOLUME', 'CHANGE'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['PRICE', 'VOLUME'], inplace=True)

    # Restrict to the 09:30 - 15:00 trading session
    time_col_all = 'TIME' if 'TIME' in df.columns else df.columns[0]
    time_str = df[time_col_all].astype(str).str.strip()
    df = df[(time_str >= SESSION_START) & (time_str <= SESSION_END)].copy()
    if df.empty:
        print(f"{ts_code}: no ticks within {SESSION_START}-{SESSION_END}")
        return

    # Print every tick detail first
    print(f"\n===== {ts_code} all tick details ({len(df)} rows) =====")
    for _, row in df.iterrows():
        t = row.get(time_col_all, '')
        price = row.get('PRICE', '')
        vol = row.get('VOLUME', '')
        typ = row.get('TYPE', '')
        print(f"{t}  price={price}  volume={vol}  type={typ}")

    # Apply price-dependent large-order threshold per tick
    df['_threshold'] = df['PRICE'].apply(_big_threshold)
    big = df[df['VOLUME'] >= df['_threshold']].copy()
    if big.empty:
        print(f"\n{ts_code}: no large orders found under dynamic thresholds")
        return

    # Order type: tier label (by price band) + direction
    big['order_type'] = big.apply(
        lambda row: f"{_tier_for(row['PRICE'])[1]}-{row.get('TYPE', 'Unknown')}",
        axis=1,
    )

    # Current change %: CHANGE / prev_close * 100
    if 'CHANGE' in big.columns:
        prev_close = big['PRICE'] - big['CHANGE']
        big['change_pct'] = (
            big['CHANGE'] / prev_close.replace(0, float('nan')) * 100
        ).round(2).astype(str) + '%'
    else:
        big['change_pct'] = 'N/A'

    time_col = 'TIME' if 'TIME' in big.columns else big.columns[0]

    out = big[[time_col, 'change_pct', 'PRICE', 'VOLUME', 'order_type']].copy()
    out.columns = ['time', 'change_pct', 'price', 'volume_lots', 'order_type']

    # Print large orders at the end
    print(f"\n===== {ts_code} large orders (price-tiered thresholds) =====")
    for _, row in out.iterrows():
        print(
            f"{row['time']}  change={row['change_pct']}  price={row['price']}  "
            f"volume={row['volume_lots']}  type={row['order_type']}"
        )

    # Append to CSV (write header only on first write)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stock_code = ts_code.split('.')[0]
    filepath = os.path.join(RESULTS_DIR, f"{stock_code}.csv")
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    out.to_csv(filepath, mode='a', header=write_header, index=False, encoding='utf-8-sig')

    # Weighted average cost for buy-side orders (buy + neutral)
    buy_mask = big['TYPE'].isin(['买盘', '中性']) if 'TYPE' in big.columns else pd.Series(True, index=big.index)
    buy_orders = big[buy_mask]
    if not buy_orders.empty:
        total_vol = buy_orders['VOLUME'].sum()
        avg_cost = (buy_orders['PRICE'] * buy_orders['VOLUME']).sum() / total_vol
        with open(filepath, 'a', encoding='utf-8-sig') as f:
            f.write(f"buy_avg_cost,{avg_cost:.3f}\n")
        print(f"{ts_code}: {len(buy_orders)} buy-side large orders  avg cost {avg_cost:.3f}")
    else:
        print(f"{ts_code}: no buy-side large orders")

    print(f"{ts_code}: {len(out)} large orders total, appended to {filepath}")


if __name__ == '__main__':
    get_big_orders('002185.SZ')
