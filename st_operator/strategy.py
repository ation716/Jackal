# -*- coding: utf-8 -*-
# @Time    : 2026/4/3
# @Author  : gaolei
# @FileName: strategy.py
"""
Jackal system — async main loop.

Continuously:
  - Reads user input (non-blocking) and dispatches commands:
      'a'       → feature A: plot today's market breadth (rising / falling / sentiment)
      'b<code>' → feature B: buy suggestion for stock code, e.g. 'b600001'
      'b'       → feature B: batch score all stocks in config/aim.py
      'c'       → feature C: show sector overview — change % and limit-up count per sector
      'q'       → quit
  - Runs RedMonitor in background: collects market breadth data every 2 min
    before 10:00, every 5 min after 10:00; writes to CSV and in-memory cache.
  - Monitors a watch function in a background task; pops up a dialog
    if an anomaly is detected.

"""


import asyncio
import io
import logging
import os
import sys
import threading

from alert_tem import SimpleDialog
from red import RedMonitor
from score import StockScorer
from guass_smoother import AdaptiveForwardGaussianSmoother

# Global monitor shared across features
_monitor = RedMonitor()

# ───────────────────────── Logger setup ──────────────────────────────────

_LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)


def _get_logger() -> logging.Logger:
    """Return a logger whose file handler points to today's log file."""
    import datetime
    today = datetime.date.today().strftime('%Y-%m-%d')
    log_path = os.path.join(_LOG_DIR, f'strategy-{today}.log')

    logger = logging.getLogger(f'strategy.{today}')
    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _log_feature(tag: str, text: str):
    """Write each non-empty line as: YYYY-MM-DD HH:MM:SS [TAG] line"""
    import datetime
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger = _get_logger()
    for line in text.splitlines():
        if line.strip():
            logger.info('%s [%s] %s', ts, tag, line)


# ───────────────────────── Feature A ─────────────────────────────────────

def _plot_market_breadth(df, auto_close_sec: int = 20):
    """Render the chart and block for up to *auto_close_sec* seconds, then close."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import time

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle("Market Breadth — Today", fontsize=13, fontweight='bold')

    ax1.plot(df['ts'], df['rising'],  color='tomato',         lw=1.5, label='Rising')
    ax1.plot(df['ts'], df['falling'], color='mediumseagreen', lw=1.5, label='Falling')
    ax1.set_ylabel('Stock count', color='black')
    ax1.tick_params(axis='y')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    fig.autofmt_xdate(rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df['ts'], df['sentiment'], color='darkorange', lw=1.5,
             linestyle='--', label='Sentiment %')
    ax2.set_ylabel('Sentiment (%)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show(block=False)

    # Poll until the window is closed manually or the timeout expires
    deadline = time.time() + auto_close_sec
    while time.time() < deadline:
        if not plt.fignum_exists(fig.number):
            break          # user closed manually
        plt.pause(0.5)

    try:
        plt.close(fig)
    except Exception:
        pass


async def feature_a():
    """
    Plot today's market breadth in a background thread (non-blocking).
    Left  y-axis : rising count (red)  +  falling count (green)
    Right y-axis : sentiment / activity % (orange)
    X-axis       : time
    Auto-closes after 20 s; user can close manually before that.
    """
    df = _monitor.get_plot_data()
    if df is None or df.empty:
        print("[A] No data available yet for today.")
        return

    n = len(df)
    threading.Thread(
        target=_plot_market_breadth,
        args=(df,),
        kwargs={'auto_close_sec': 20},
        daemon=True,
    ).start()
    print(f"[A] Chart opened — {n} data points  (auto-closes in 20s)")


# ───────────────────────── Feature B helpers ─────────────────────────────

def _print_score_report(result: dict):
    """Print full score report with per-dimension breakdown."""
    SEP  = '═' * 60
    SEP2 = '─' * 60
    code  = result['code']
    total = result['total']
    price = result.get('current_price')
    sector = result.get('sector') or 'N/A'

    print(f"\n{SEP}")
    print(f"  Stock {code}   Price: {price:.3f}   Sector: {sector}")
    print(f"  Total Score: {total:.1f} / 100")
    print(SEP2)

    dim_labels = {
        'breakout':          ('Breakout Strength',    12),
        'recent_change':     ('Recent Change',         12),
        'relative_strength': ('Relative Strength',    10),
        'fund_flow':         ('Fund Flow',             15),
        'chip':              ('Chip Distribution',      7),
        'technical':         ('Technical Indicators',  10),
        'hot_rank':          ('Hot Rank',               7),
        'seal_quality':      ('Seal Quality',           8),
        'sector_strength':   ('Sector Strength',        8),
        'consec_limit':      ('Consecutive Limit',     11),
    }
    for key, (label, max_score) in dim_labels.items():
        s = result['scores'].get(key, {})
        score  = s.get('score', 0)
        detail = s.get('detail', '')
        bar_len = int(score / max_score * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {label:<22} {score:>5.1f}/{max_score}  {bar}  {detail}")

    print(SEP)


def _print_sector_info(fetcher, code: str):
    """Print sector name and recent sector index performance."""
    SEP2 = '─' * 60
    sector_name = fetcher.get_sector_name(code)
    print(f"\n{SEP2}")
    print(f"  Sector Info")
    print(SEP2)

    if not sector_name:
        print("  Sector: N/A")
        return

    print(f"  Sector: {sector_name}")

    sec_df = fetcher.get_sector_hist(sector_name, days=10)
    if sec_df.empty:
        print("  Sector history: unavailable")
        return

    sc = '收盘' if '收盘' in sec_df.columns else sec_df.columns[-1]
    if sc not in sec_df.columns or len(sec_df) < 2:
        return

    vals = sec_df[sc].values
    chg1d = (vals[-1] - vals[-2]) / vals[-2] * 100 if vals[-2] > 0 else 0.0
    chg5d = (vals[-1] - vals[max(0, len(vals)-6)]) / vals[max(0, len(vals)-6)] * 100 \
            if len(vals) >= 6 and vals[max(0, len(vals)-6)] > 0 else 0.0
    print(f"  Sector 1-day change: {chg1d:+.2f}%   5-day change: {chg5d:+.2f}%")
    print(f"  Latest sector close: {vals[-1]:.2f}")
    print(SEP2)


def _print_gaussian_prediction(fetcher, code: str):
    """Fetch 150-day history and predict next 2 trading days via Gaussian smoother."""
    SEP2 = '─' * 60
    print(f"\n{SEP2}")
    print(f"  Gaussian Forecast (based on last 150 trading days)")
    print(SEP2)

    hist = fetcher.get_hist(code, days=150)
    if hist.empty or '收盘' not in hist.columns or len(hist) < 10:
        print("  Insufficient history data for prediction.")
        print(SEP2)
        return

    close_prices = hist['收盘'].tolist()
    current = close_prices[-1]

    smoother = AdaptiveForwardGaussianSmoother(
        min_sigma=0.3, max_sigma=2.0, base_window_size=7, sensitivity=3
    )
    _, smooth_B, _, _ = smoother.smooth_array(close_prices)

    pred_d1 = smoother.predict_next()
    smoother.history.append(pred_d1)
    pred_d2 = smoother.predict_next()

    chg_d1 = (pred_d1 - current) / current * 100 if current > 0 else 0.0
    chg_d2 = (pred_d2 - current) / current * 100 if current > 0 else 0.0

    print(f"  Current close : {current:.3f}")
    print(f"  D+1 forecast  : {pred_d1:.3f}  ({chg_d1:+.2f}% vs today)")
    print(f"  D+2 forecast  : {pred_d2:.3f}  ({chg_d2:+.2f}% vs today)")
    print(f"  History used  : {len(close_prices)} days")
    print(SEP2)


def _run_feature_b(code: str):
    """Synchronous execution of feature B (runs in executor thread)."""
    scorer = StockScorer()

    print(f"\n[B] Analysing {code}...")

    # 1. Score
    result = scorer.score_stock(code)
    _print_score_report(result)

    # 2. Sector info
    _print_sector_info(scorer.fetcher, code)

    # 3. Gaussian prediction
    _print_gaussian_prediction(scorer.fetcher, code)

    # Recommendation hint
    total = result['total']
    if total >= 75:
        hint = "Strong buy signal"
    elif total >= 60:
        hint = "Moderate buy signal"
    elif total >= 45:
        hint = "Neutral / watch"
    else:
        hint = "Weak / avoid"
    print(f"\n  Recommendation: {hint}  (score={total:.1f})\n")


# ───────────────────────── Feature B (async entry) ───────────────────────

def _run_feature_b_batch(aim_codes: dict):
    """Score all stocks in aim.py and print a ranked summary table."""
    scorer = StockScorer()
    SEP  = '═' * 52
    SEP2 = '─' * 52

    print(f"\n[B] Scoring {len(aim_codes)} stocks from aim.py ...\n")
    rows = []
    for code, meta in aim_codes.items():
        try:
            result = scorer.score_stock(code)
            rows.append({
                'code':  code,
                'name':  meta.get('name', ''),
                'total': result['total'],
                'price': result.get('current_price') or 0.0,
            })
            print(f"  {code} {meta.get('name',''):8s}  {result['total']:.1f}")
        except Exception as e:
            print(f"  {code} error: {e}")

    rows.sort(key=lambda r: r['total'], reverse=True)
    print(f"\n{SEP}")
    print(f"  {'代码':<8} {'名称':<10} {'总分':>6} {'现价':>8}")
    print(SEP2)
    for r in rows:
        print(f"  {r['code']:<8} {r['name']:<10} {r['total']:>6.1f} {r['price']:>8.3f}")
    print(SEP)


async def feature_b(arg: str):
    """
    Buy suggestion for a stock, or batch score all stocks in aim.py.
    arg: 6-digit stock code, e.g. '600001'; empty → score all from aim.py
    """
    loop = asyncio.get_event_loop()

    if not arg.strip():
        import importlib.util, os
        aim_path = os.path.join(os.path.dirname(__file__), 'config', 'aim.py')
        spec = importlib.util.spec_from_file_location('aim', aim_path)
        aim_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aim_mod)
        await loop.run_in_executor(None, _run_feature_b_batch, aim_mod.codes)
        return

    code = arg.strip().zfill(6)
    if not code.isdigit() or len(code) != 6:
        print(f"[B] Invalid stock code: {arg!r}  (expected 6 digits, e.g. b600001)")
        return

    await loop.run_in_executor(None, _run_feature_b, code)


# ───────────────────────── Feature C ─────────────────────────────────────

def _run_feature_c():
    """Synchronous execution of feature C (runs in executor thread)."""
    import akshare as ak
    import datetime

    today = datetime.date.today().strftime('%Y%m%d')
    print("\n[C] 获取板块数据...")

    try:
        sector_df = ak.stock_board_industry_spot_em()
    except Exception as e:
        print(f"[C] 获取板块行情失败: {e}")
        return

    zt_counts: dict = {}
    try:
        zt_df = ak.stock_zt_pool_em(date=today)
        if '所属行业' in zt_df.columns:
            zt_counts = zt_df['所属行业'].value_counts().to_dict()
    except Exception as e:
        print(f"[C] 获取涨停池失败: {e}")

    if '涨跌幅' not in sector_df.columns or '板块名称' not in sector_df.columns:
        print("[C] 数据格式异常")
        return

    sector_df = sector_df.copy()
    sector_df['涨停数'] = sector_df['板块名称'].map(lambda n: zt_counts.get(n, 0))
    sector_df = sector_df.sort_values(['涨停数', '涨跌幅'], ascending=[False, False])

    SEP = '─' * 36
    print(f"\n{'板块':<14} {'涨幅':>6} {'涨停':>4}")
    print(SEP)
    for _, row in sector_df.iterrows():
        name = str(row['板块名称'])
        chg  = float(row['涨跌幅'])
        zt   = int(row['涨停数'])
        print(f"{name:<14} {chg:>6.2f} {zt:>4}")
    print(SEP)


async def feature_c():
    """展示板块综合情况：板块涨幅、涨停个股数。"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_feature_c)


# ───────────────────────── Anomaly monitor ───────────────────────────────

def get_monitor_value() -> float | None:
    """Returns the current value to watch, or None. Replace with real source."""
    return None


def is_anomaly(value: float) -> bool:
    """Returns True when *value* should trigger an alert. Replace with real logic."""
    return False


async def anomaly_loop(interval: float = 5.0):
    """Poll get_monitor_value() every *interval* seconds; alert on anomaly."""
    print(f"[Anomaly] started, polling every {interval}s")
    while True:
        try:
            value = get_monitor_value()
            if value is not None and is_anomaly(value):
                msg = f"Anomaly detected: value={value}"
                print(f"[Anomaly] {msg}")
                SimpleDialog.auto_close(msg, title="Jackal Alert", seconds=10)
        except Exception as e:
            print(f"[Anomaly] error: {e}")
        await asyncio.sleep(interval)


# ───────────────────────── Input reader ──────────────────────────────────

async def read_input() -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sys.stdin.readline)


# ───────────────────────── Command dispatcher ────────────────────────────

async def dispatch(raw: str):
    cmd = raw.strip()
    if not cmd:
        return

    # Determine feature tag before running
    if cmd == 'a':
        tag = 'A'
    elif cmd.startswith('b'):
        tag = 'B'
    elif cmd in ('c', 'r'):
        tag = 'C'
    else:
        tag = None

    if tag:
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            if tag == 'A':
                await feature_a()
            elif tag == 'B':
                await feature_b(cmd[1:])
            elif tag == 'C':
                await feature_c()
        finally:
            sys.stdout = real_stdout
            captured = buf.getvalue()
            print(captured, end='')
            _log_feature(tag, captured)
    elif cmd == 'q':
        print("Quitting...")
        raise SystemExit(0)
    else:
        print(f"[Input] unknown command: {cmd!r}  (a / b<code> / c / q)")


# ───────────────────────── Main loop ─────────────────────────────────────

async def main():
    print("=" * 60)
    print("  Jackal system started")
    print("  Commands:  a | b<code> | c | q (quit)")
    print("  Example:   b600001  → buy suggestion for 600001")
    print("=" * 60)

    tasks = [
        asyncio.create_task(_monitor.run()),
        asyncio.create_task(anomaly_loop(interval=5.0)),
    ]

    try:
        while True:
            print(">> ", end='', flush=True)
            raw = await read_input()
            await dispatch(raw)
    except (SystemExit, KeyboardInterrupt):
        print("\nShutting down.")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
