# -*- coding: utf-8 -*-
# @Time    : 2026/4/2 11:50
# @Author  : gaolei
# @FileName: big_fund.py
"""
big_fund.py — Real-time large-order detector for A-share stocks.

Features:
  - Fetches intraday tick data via tushare realtime_tick (sina source).
  - Classifies orders as "Super-Large" (>= 5000 lots) or "Large" (>= 2000 lots).
  - Appends each session's large-order details to results/big_fund/{code}.csv.
  - Fields written: time, current change%, price, volume (lots), order type.
  - Calculates and appends the weighted average cost of buy-side large orders.
"""

import os
import tushare as ts
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'big_fund')
TUSHARE_TOKEN = "1bf9b910cdda6f0cd856f55b97c1c1419860237f7be8156aacac3259"

VERY_BIG_ORDER = 5000   # Super-large order threshold (lots)
BIG_ORDER = 2000        # Large order threshold (lots)


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

    # Keep only large orders (>= 2000 lots)
    big = df[df['VOLUME'] >= BIG_ORDER].copy()
    if big.empty:
        print(f"{ts_code}: no large orders found (>= {BIG_ORDER} lots)")
        return

    # Order type: size category + direction
    def _classify(row):
        size = 'Super-Large' if row['VOLUME'] >= VERY_BIG_ORDER else 'Large'
        return f"{size}-{row.get('TYPE', 'Unknown')}"

    big['order_type'] = big.apply(_classify, axis=1)

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
    get_big_orders('600396.SH')
