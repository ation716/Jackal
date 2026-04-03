# Kelly-formula-based investment strategy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from heston import *
from guass_smoother import AdaptiveForwardGaussianSmoother
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# Chinese font support for matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ───────────────────────── Kelly Strategy Core ──────────────────────────

def kalley1(p, b):
    """Classic Kelly criterion:  p - (1-p)/b"""
    return p - (1 - p) / b


def kalley2(p, r):
    """Zero-sum Kelly criterion:  (2p-1)/r"""
    if r == 0:
        r = 0.0001
    return (2 * p - 1) / r


def kellly_strategy_continues(p_matrix, b_matrix):
    """Kelly strategy (classic Kelly)."""
    p_matrix = np.array(p_matrix, dtype=float)
    result = np.zeros_like(p_matrix)
    for i in range(len(p_matrix)):
        if b_matrix[i] < 0:
            result[i] = 0
        else:
            tem = kalley1(p_matrix[i], b_matrix[i] * 10)
            if tem < 0:
                result[i] = 0
            elif tem < 1:
                result[i] = tem
            else:
                result[i] = 1
    return result


def kellly_strategy_continues2(p_matrix, b_matrix):
    """Kelly strategy (zero-sum)."""
    p_matrix = np.array(p_matrix, dtype=float)
    result = np.zeros_like(p_matrix)
    for i in range(len(p_matrix)):
        if b_matrix[i] < 0:
            result[i] = 0
        tem = kalley2(p_matrix[i], b_matrix[i] * 10)
        if tem < 0:
            result[i] = 0
        elif tem < 1:
            result[i] = tem
        else:
            result[i] = 1
    return result


def continuse_gain(kelly_weight, path, s0):
    """Compute Kelly strategy equity curve."""
    result = np.zeros(len(path) + 1)
    for i in range(len(result)):
        if i < 2:
            result[i] = s0
        else:
            result[i] = (result[i - 1] * kelly_weight[i - 2] * (1 + path[i - 1])
                         + result[i - 1] * (1 - kelly_weight[i - 2]))
    return result


# ───────────────────────── Main Analysis Function ──────────────────────────

def analyze_stock(stock_code: str, start_date: str, end_date: str):
    """
    Analyse a stock over the given date range:
      1. Plot Gaussian-smoothed forecast curve (incl. D+1/D+2 prediction + accuracy)
      2. Plot Kelly strategy equity curve

    Parameters
    ----------
    stock_code : str   stock code, e.g. "600410"
    start_date : str   start date YYYYMMDD, e.g. "20250101"
    end_date   : str   end date   YYYYMMDD, e.g. "20260312"
    """
    # ── 1. Fetch historical data ──
    df = ak.stock_zh_a_hist(
        symbol=stock_code, period="daily",
        start_date=start_date, end_date=end_date, adjust="qfq"
    )
    if df.empty or len(df) < 10:
        print(f"[Error] {stock_code} has insufficient data ({len(df)} rows) for {start_date}~{end_date}")
        return None

    close_prices = df['收盘'].tolist()
    dates        = [str(d)[:10] for d in df['日期'].tolist()]
    N  = len(close_prices)
    S0 = close_prices[0]
    S_array = np.array(close_prices, dtype=float)

    # ── 2. Adaptive Gaussian smoothing & historical forecast ──
    smoother = AdaptiveForwardGaussianSmoother(
        min_sigma=0.3, max_sigma=2.0, base_window_size=7, sensitivity=3
    )
    predict_B, smooth_B, sigmas, windows = smoother.smooth_array(close_prices)
    # predict_B[i] ≈ prediction for close_prices[i+1]  (valid for i = 0..N-2)

    # ── 3. Forecast next 2 trading days ──
    # smoother.history now contains all N historical prices
    pred_day1 = smoother.predict_next()          # D+1 forecast
    smoother.history.append(pred_day1)
    pred_day2 = smoother.predict_next()          # D+2 forecast (based on history + pred_day1)

    # ── 4. Attempt to fetch actual next-2-day prices for validation ──
    end_dt   = datetime.strptime(end_date, "%Y%m%d")
    next_str = (end_dt + timedelta(days=1)).strftime("%Y%m%d")
    far_str  = (end_dt + timedelta(days=14)).strftime("%Y%m%d")  # 14-day buffer to cover 2 trading days

    future_prices = []
    future_dates  = []
    try:
        df_fut = ak.stock_zh_a_hist(
            symbol=stock_code, period="daily",
            start_date=next_str, end_date=far_str, adjust="qfq"
        )
        if not df_fut.empty:
            future_prices = df_fut['收盘'].tolist()[:2]
            future_dates  = [str(d)[:10] for d in df_fut['日期'].tolist()][:2]
    except Exception:
        pass

    # ── 5. Historical forecast accuracy ──
    # predict_B[i] forecasts S_array[i+1]; N-1 pairs in total
    hist_preds  = predict_B[:-1]
    hist_actual = S_array[1:]
    hist_acc    = np.clip(1 - np.abs(hist_preds - hist_actual) * 5 / hist_actual, 0, 1)
    hist_acc_sm = np.clip(test_adaptive_smoothing(hist_acc, s=1), 0, 1)

    # Future 2-day accuracy (only when actual data is available)
    future_acc = []
    for k, pred in enumerate([pred_day1, pred_day2]):
        if k < len(future_prices):
            acc = max(0.0, min(1.0, 1 - abs(pred - future_prices[k]) * 5 / future_prices[k]))
            future_acc.append(acc)

    # ── 6. Kelly strategy ──
    gain_loss_p   = np.clip((predict_B[1:] - predict_B[:-1]) / predict_B[:-1], -0.1, 0.1)
    kelly_weights = kellly_strategy_continues2(hist_acc_sm, gain_loss_p)
    gain_loss_r   = (S_array[1:] - S_array[:-1]) / S_array[:-1]
    assets        = continuse_gain(kelly_weights, gain_loss_r, S0)

    # ── 7. Plot ──
    _plot_analysis(
        stock_code, start_date, end_date,
        S_array, dates,
        predict_B, smooth_B,
        pred_day1, pred_day2,
        future_prices, future_dates, future_acc,
        hist_acc, hist_acc_sm,
        kelly_weights, assets, S0
    )

    # ── 8. Console summary ──
    mean_acc_20  = float(np.mean(hist_acc[-20:]))
    final_val    = assets[-1]
    total_return = (final_val / S0 - 1) * 100
    bh_return    = (S_array[-1] / S0 - 1) * 100

    print(f"\n{'='*45}")
    print(f"  Stock {stock_code}  {start_date} ~ {end_date}  {N} trading days")
    print(f"{'='*45}")
    print(f"  Avg forecast accuracy (last 20 days): {mean_acc_20*100:.1f}%")
    print(f"\n  Next 2-day forecast:")
    for k, (pred, label) in enumerate([(pred_day1, 'D+1'), (pred_day2, 'D+2')]):
        date_str = future_dates[k] if k < len(future_dates) else label
        line = f"    {date_str}: predicted={pred:.3f}"
        if k < len(future_prices):
            line += f"  actual={future_prices[k]:.3f}  accuracy={future_acc[k]*100:.1f}%"
        print(line)
    print(f"\n  Kelly strategy:")
    print(f"    initial={S0:.3f}  final={final_val:.3f}  total return={total_return:+.1f}%")
    print(f"    Buy-and-hold return: {bh_return:+.1f}%")
    print(f"{'='*45}\n")

    return {
        'close_prices':   close_prices,
        'dates':          dates,
        'predict_B':      predict_B,
        'smooth_B':       smooth_B,
        'pred_day1':      pred_day1,
        'pred_day2':      pred_day2,
        'future_prices':  future_prices,
        'future_acc':     future_acc,
        'hist_acc':       hist_acc,
        'kelly_weights':  kelly_weights,
        'assets':         assets,
    }


# ───────────────────────── Plot Helper ──────────────────────────

def _plot_analysis(
    stock_code, start_date, end_date,
    S_array, dates,
    predict_B, smooth_B,
    pred_day1, pred_day2,
    future_prices, future_dates, future_acc,
    hist_acc, hist_acc_sm,
    kelly_weights, assets, S0
):
    N = len(S_array)

    fig, axes = plt.subplots(3, 1, figsize=(15, 12),
                             gridspec_kw={'height_ratios': [3, 1, 2]})
    fig.suptitle(
        f'{stock_code}   {start_date[:4]}-{start_date[4:6]}-{start_date[6:]} '
        f'~ {end_date[:4]}-{end_date[4:6]}-{end_date[6:]}',
        fontsize=14, fontweight='bold'
    )
    ax_price, ax_acc, ax_kelly = axes

    # ── X-axis ticks (dates) ──
    tick_step = max(1, N // 10)
    tick_pos  = list(range(0, N, tick_step))
    if N - 1 not in tick_pos:
        tick_pos.append(N - 1)
    tick_pos += [N, N + 1]

    tick_labels_price = [dates[i] for i in tick_pos[:-2]]
    tick_labels_price += [
        future_dates[0] if future_dates else 'D+1',
        future_dates[1] if len(future_dates) > 1 else 'D+2',
    ]

    # ────────── Panel 1: price + Gaussian forecast ──────────
    ax_price.plot(range(N), S_array, lw=1.5, color='steelblue',
                  label='Actual price', zorder=3)

    # Historical forecast: predict_B[i] maps to close_prices[i+1] → plotted at i+1
    ax_price.plot(range(1, N), predict_B[:-1], lw=1.2, color='darkorange',
                  linestyle='--', alpha=0.85, label='Gaussian forecast (hist)', zorder=2)

    # Gaussian smoothed curve (actual smoothed values, not shifted)
    ax_price.plot(range(N), smooth_B, lw=1, color='mediumpurple',
                  linestyle=':', alpha=0.6, label='Gaussian smooth', zorder=2)

    # Next 2-day forecast line
    fut_x = [N - 1, N, N + 1]
    fut_y = [S_array[-1], pred_day1, pred_day2]
    ax_price.plot(fut_x, fut_y, lw=2.5, color='tomato', linestyle='--',
                  marker='o', markersize=9, label='Forecast (D+1/D+2)', zorder=5)

    # Plot validation data if available
    if future_prices:
        ax_price.plot(range(N, N + len(future_prices)), future_prices,
                      lw=1.5, color='limegreen', marker='s', markersize=8,
                      label='Actual (validation)', zorder=6)

    # Prediction value & accuracy annotations
    price_range = S_array.max() - S_array.min()
    offset = price_range * 0.06
    for k, (px, py, pred) in enumerate(
        [(N, pred_day1, pred_day1), (N + 1, pred_day2, pred_day2)]
    ):
        lines = [f'D+{k+1}: {pred:.2f}']
        if k < len(future_acc):
            lines.append(f'Accuracy: {future_acc[k]*100:.1f}%')
            color = 'green' if future_acc[k] > 0.7 else ('orange' if future_acc[k] > 0.5 else 'red')
        else:
            color = 'tomato'
        ax_price.annotate(
            '\n'.join(lines),
            xy=(px, py),
            xytext=(px - max(N // 8, 6), py + offset * (1.2 + k * 0.8)),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            fontsize=9, color=color,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9)
        )

    # Recent historical accuracy label
    mean_acc_20 = float(np.mean(hist_acc[-20:]))
    ax_price.text(
        0.01, 0.97,
        f'Avg forecast accuracy (last 20 days): {mean_acc_20*100:.1f}%',
        transform=ax_price.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9)
    )

    ax_price.set_xticks(tick_pos)
    ax_price.set_xticklabels(tick_labels_price, rotation=45, ha='right', fontsize=7)
    ax_price.set_title('Price & Gaussian Smooth Forecast')
    ax_price.set_ylabel('Price')
    ax_price.legend(loc='upper left', fontsize=9)
    ax_price.grid(True, alpha=0.3)

    # ────────── Panel 2: historical forecast accuracy ──────────
    ax_acc.fill_between(range(1, N), hist_acc, alpha=0.3, color='orange', label='Daily accuracy')
    ax_acc.plot(range(1, N), hist_acc_sm, lw=1.2, color='darkorange', label='Smoothed accuracy')
    ax_acc.axhline(0.7, color='green', lw=0.8, linestyle='--', alpha=0.6, label='70% baseline')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xticks(tick_pos[:-2])
    ax_acc.set_xticklabels(tick_labels_price[:-2], rotation=45, ha='right', fontsize=7)
    ax_acc.set_title('Historical Forecast Accuracy')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower left', fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    # ────────── Panel 3: Kelly strategy equity ──────────
    t_asset = np.arange(N)
    ax_kelly.plot(t_asset, assets, lw=1.8, color='mediumseagreen',
                  label='Kelly strategy equity', zorder=3)
    ax_kelly.plot(t_asset, S_array, lw=1, color='steelblue',
                  linestyle=':', alpha=0.7, label='Buy-and-hold (price)', zorder=2)
    ax_kelly.axhline(S0, color='silver', linestyle='--', lw=0.8, alpha=0.7)

    final_val    = assets[-1]
    total_return = (final_val / S0 - 1) * 100
    bh_return    = (S_array[-1] / S0 - 1) * 100

    # Final value annotation
    xt = max(0, N - N // 5)
    yt = final_val * (0.88 if total_return > 0 else 1.08)
    ax_kelly.annotate(
        f'Final: {final_val:.2f}\nKelly: {total_return:+.1f}%\nB&H: {bh_return:+.1f}%',
        xy=(N - 1, final_val),
        xytext=(xt, yt),
        arrowprops=dict(arrowstyle='->', color='mediumseagreen', lw=1.5),
        fontsize=9, color='mediumseagreen',
        bbox=dict(boxstyle='round,pad=0.3', fc='honeydew', alpha=0.9)
    )

    ax_kelly.set_xticks(tick_pos[:-2])
    ax_kelly.set_xticklabels(tick_labels_price[:-2], rotation=45, ha='right', fontsize=7)
    ax_kelly.set_title(f'Kelly Strategy Equity Curve  Total: {total_return:+.1f}%  (B&H: {bh_return:+.1f}%)')
    ax_kelly.set_ylabel('Asset value')
    ax_kelly.set_xlabel('Date')
    ax_kelly.legend(loc='upper left', fontsize=9)
    ax_kelly.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ───────────────────────── Entry point ──────────────────────────

if __name__ == "__main__":
    analyze_stock("600410", "20250101", "20260312")
