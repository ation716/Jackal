# 基于凯利公式的投资策略

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from heston import *
from guass_smoother import AdaptiveForwardGaussianSmoother
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# 中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ───────────────────────── 凯利策略核心函数 ──────────────────────────

def kalley1(p, b):
    """经典凯利  p - (1-p)/b"""
    return p - (1 - p) / b


def kalley2(p, r):
    """零和博弈的凯利 (2p-1)/r"""
    if r == 0:
        r = 0.0001
    return (2 * p - 1) / r


def kellly_strategy_continues(p_matrix, b_matrix):
    """凯莉策略（经典凯利）"""
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
    """凯莉策略（零和博弈）"""
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
    """计算 Kelly 策略资产曲线"""
    result = np.zeros(len(path) + 1)
    for i in range(len(result)):
        if i < 2:
            result[i] = s0
        else:
            result[i] = (result[i - 1] * kelly_weight[i - 2] * (1 + path[i - 1])
                         + result[i - 1] * (1 - kelly_weight[i - 2]))
    return result


# ───────────────────────── 主分析函数 ──────────────────────────

def analyze_stock(stock_code: str, start_date: str, end_date: str):
    """
    根据股票代码、起始时间、终止时间进行分析：
      1. 绘制高斯平滑后的预测曲线（含未来2天预测 + 准确率）
      2. 绘制凯利策略收益曲线

    Parameters
    ----------
    stock_code : str   股票代码，如 "600410"
    start_date : str   起始日期 YYYYMMDD，如 "20250101"
    end_date   : str   终止日期 YYYYMMDD，如 "20260312"
    """
    # ── 1. 拉取历史数据 ──
    df = ak.stock_zh_a_hist(
        symbol=stock_code, period="daily",
        start_date=start_date, end_date=end_date, adjust="qfq"
    )
    if df.empty or len(df) < 10:
        print(f"[错误] 股票 {stock_code} 在 {start_date}~{end_date} 数据不足（{len(df)} 条）")
        return None

    close_prices = df['收盘'].tolist()
    dates        = [str(d)[:10] for d in df['日期'].tolist()]
    N  = len(close_prices)
    S0 = close_prices[0]
    S_array = np.array(close_prices, dtype=float)

    # ── 2. 高斯自适应平滑 & 历史预测 ──
    smoother = AdaptiveForwardGaussianSmoother(
        min_sigma=0.3, max_sigma=2.0, base_window_size=7, sensitivity=3
    )
    predict_B, smooth_B, sigmas, windows = smoother.smooth_array(close_prices)
    # predict_B[i]  ≈ 对 close_prices[i+1] 的预测  (i=0..N-2 有效)

    # ── 3. 预测未来2天 ──
    # smoother.history 此时包含全部 N 个历史价格
    pred_day1 = smoother.predict_next()          # D+1 预测
    smoother.history.append(pred_day1)
    pred_day2 = smoother.predict_next()          # D+2 预测（基于 history + pred_day1）

    # ── 4. 尝试获取未来2个交易日的实际价格（用于验证准确率） ──
    end_dt   = datetime.strptime(end_date, "%Y%m%d")
    next_str = (end_dt + timedelta(days=1)).strftime("%Y%m%d")
    far_str  = (end_dt + timedelta(days=14)).strftime("%Y%m%d")  # 多留14天保证覆盖2个交易日

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

    # ── 5. 历史预测准确率 ──
    # gauss_d[i] 预测 S_array[i+1]，共 N-1 对
    hist_preds  = predict_B[:-1]          # 对 S_array[1..N-1] 的预测
    hist_actual = S_array[1:]
    hist_acc    = np.clip(1 - np.abs(hist_preds - hist_actual) * 5 / hist_actual, 0, 1)
    hist_acc_sm = np.clip(test_adaptive_smoothing(hist_acc, s=1), 0, 1)

    # 未来2天准确率（仅在有实际数据时计算）
    future_acc = []
    for k, pred in enumerate([pred_day1, pred_day2]):
        if k < len(future_prices):
            acc = max(0.0, min(1.0, 1 - abs(pred - future_prices[k]) * 5 / future_prices[k]))
            future_acc.append(acc)

    # ── 6. 凯利策略 ──
    gain_loss_p   = np.clip((predict_B[1:] - predict_B[:-1]) / predict_B[:-1], -0.1, 0.1)
    kelly_weights = kellly_strategy_continues2(hist_acc_sm, gain_loss_p)
    gain_loss_r   = (S_array[1:] - S_array[:-1]) / S_array[:-1]
    assets        = continuse_gain(kelly_weights, gain_loss_r, S0)

    # ── 7. 绘图 ──
    _plot_analysis(
        stock_code, start_date, end_date,
        S_array, dates,
        predict_B, smooth_B,
        pred_day1, pred_day2,
        future_prices, future_dates, future_acc,
        hist_acc, hist_acc_sm,
        kelly_weights, assets, S0
    )

    # ── 8. 控制台摘要 ──
    mean_acc_20  = float(np.mean(hist_acc[-20:]))
    final_val    = assets[-1]
    total_return = (final_val / S0 - 1) * 100
    bh_return    = (S_array[-1] / S0 - 1) * 100

    print(f"\n{'='*45}")
    print(f"  股票 {stock_code}  {start_date} ~ {end_date}  共 {N} 交易日")
    print(f"{'='*45}")
    print(f"  近20日历史预测准确率均值: {mean_acc_20*100:.1f}%")
    print(f"\n  未来2天预测:")
    for k, (pred, label) in enumerate([(pred_day1, 'D+1'), (pred_day2, 'D+2')]):
        date_str = future_dates[k] if k < len(future_dates) else label
        line = f"    {date_str}: 预测={pred:.3f}"
        if k < len(future_prices):
            line += f"  实际={future_prices[k]:.3f}  准确率={future_acc[k]*100:.1f}%"
        print(line)
    print(f"\n  凯利策略:")
    print(f"    初始={S0:.3f}  终值={final_val:.3f}  总收益={total_return:+.1f}%")
    print(f"    买入持有收益: {bh_return:+.1f}%")
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


# ───────────────────────── 绘图辅助函数 ──────────────────────────

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

    # ── X 轴刻度（日期） ──
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

    # ────────── 子图1：价格 + 高斯预测 ──────────
    ax_price.plot(range(N), S_array, lw=1.5, color='steelblue',
                  label='实际价格', zorder=3)

    # 历史预测曲线：predict_B[i] 对应 close_prices[i+1] → 绘制于 i+1 位置
    ax_price.plot(range(1, N), predict_B[:-1], lw=1.2, color='darkorange',
                  linestyle='--', alpha=0.85, label='高斯预测(历史)', zorder=2)

    # 高斯平滑曲线（实际平滑值，不偏移）
    ax_price.plot(range(N), smooth_B, lw=1, color='mediumpurple',
                  linestyle=':', alpha=0.6, label='高斯平滑', zorder=2)

    # 未来2天预测线
    fut_x = [N - 1, N, N + 1]
    fut_y = [S_array[-1], pred_day1, pred_day2]
    ax_price.plot(fut_x, fut_y, lw=2.5, color='tomato', linestyle='--',
                  marker='o', markersize=9, label='未来2天预测', zorder=5)

    # 如果有验证数据，画出来
    if future_prices:
        ax_price.plot(range(N, N + len(future_prices)), future_prices,
                      lw=1.5, color='limegreen', marker='s', markersize=8,
                      label='实际(验证)', zorder=6)

    # 预测值 & 准确率标注
    price_range = S_array.max() - S_array.min()
    offset = price_range * 0.06
    for k, (px, py, pred) in enumerate(
        [(N, pred_day1, pred_day1), (N + 1, pred_day2, pred_day2)]
    ):
        lines = [f'D+{k+1}: {pred:.2f}']
        if k < len(future_acc):
            lines.append(f'准确率: {future_acc[k]*100:.1f}%')
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

    # 近期历史准确率文字提示
    mean_acc_20 = float(np.mean(hist_acc[-20:]))
    ax_price.text(
        0.01, 0.97,
        f'近20日历史预测准确率: {mean_acc_20*100:.1f}%',
        transform=ax_price.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9)
    )

    ax_price.set_xticks(tick_pos)
    ax_price.set_xticklabels(tick_labels_price, rotation=45, ha='right', fontsize=7)
    ax_price.set_title('价格走势 & 高斯平滑预测')
    ax_price.set_ylabel('价格')
    ax_price.legend(loc='upper left', fontsize=9)
    ax_price.grid(True, alpha=0.3)

    # ────────── 子图2：历史预测准确率 ──────────
    ax_acc.fill_between(range(1, N), hist_acc, alpha=0.3, color='orange', label='单日准确率')
    ax_acc.plot(range(1, N), hist_acc_sm, lw=1.2, color='darkorange', label='平滑准确率')
    ax_acc.axhline(0.7, color='green', lw=0.8, linestyle='--', alpha=0.6, label='70%基准')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xticks(tick_pos[:-2])
    ax_acc.set_xticklabels(tick_labels_price[:-2], rotation=45, ha='right', fontsize=7)
    ax_acc.set_title('历史预测准确率')
    ax_acc.set_ylabel('准确率')
    ax_acc.legend(loc='lower left', fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    # ────────── 子图3：凯利策略收益 ──────────
    t_asset = np.arange(N)
    ax_kelly.plot(t_asset, assets, lw=1.8, color='mediumseagreen',
                  label='凯利策略资产', zorder=3)
    ax_kelly.plot(t_asset, S_array, lw=1, color='steelblue',
                  linestyle=':', alpha=0.7, label='买入持有(股价)', zorder=2)
    ax_kelly.axhline(S0, color='silver', linestyle='--', lw=0.8, alpha=0.7)

    final_val    = assets[-1]
    total_return = (final_val / S0 - 1) * 100
    bh_return    = (S_array[-1] / S0 - 1) * 100

    # 最终值标注
    xt = max(0, N - N // 5)
    yt = final_val * (0.88 if total_return > 0 else 1.08)
    ax_kelly.annotate(
        f'终值: {final_val:.2f}\n凯利: {total_return:+.1f}%\n持有: {bh_return:+.1f}%',
        xy=(N - 1, final_val),
        xytext=(xt, yt),
        arrowprops=dict(arrowstyle='->', color='mediumseagreen', lw=1.5),
        fontsize=9, color='mediumseagreen',
        bbox=dict(boxstyle='round,pad=0.3', fc='honeydew', alpha=0.9)
    )

    ax_kelly.set_xticks(tick_pos[:-2])
    ax_kelly.set_xticklabels(tick_labels_price[:-2], rotation=45, ha='right', fontsize=7)
    ax_kelly.set_title(f'凯利策略收益曲线  总收益: {total_return:+.1f}%  (买入持有: {bh_return:+.1f}%)')
    ax_kelly.set_ylabel('资产价值')
    ax_kelly.set_xlabel('日期')
    ax_kelly.legend(loc='upper left', fontsize=9)
    ax_kelly.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ───────────────────────── 入口 ──────────────────────────

if __name__ == "__main__":
    analyze_stock("600410", "20250101", "20260312")
