# -*- coding: utf-8 -*-
# @Time    : 2026/4/17 13:05
# @Author  : gaolei
# @FileName: pingjia.py
# @Software: PyCharm
"""
在这里生成评价 predict_tes 的代码

个股数据 skl_predict/predict_tes_data/sz.002580.csv
板块数据 skl_predict/predict_tes_data/881281.csv
指数数据 skl_predict/predict_tes_data/sh.000001.csv

这些数据都是从 20250101 开始到 20260401
我想通过 skl_predict/predict_tes.py 里的代码，持续预测，记录数据

最后用 matplotlib 画出两幅图，

图1 有该个股股价K线走势图，
还有 预测 k 线图，K线使用 25% 的概率涨幅值和 75% 的概率涨幅值，如果 75% 的概率涨幅值 大于 25% 的概率涨幅值，则为红色，反之则为绿色
为区分，两个 k 色差明显一点

图2 同样，该个股股价K线走势图，叠加 40% 的概率涨幅值和 60% 的概率涨幅值 的K线走势图

保存图片

"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from predict_tes import fit_distribution, predict_distribution

BASE    = Path(__file__).parent / 'predict_tes_data'
OUT_DIR = Path(__file__).parent
WINDOW  = 60   # 滚动窗口：60 个交易日

# ── 1. 加载数据 ───────────────────────────────────────────────────────────────
stock_df  = pd.read_csv(BASE / 'sz.002580.csv',  parse_dates=['date']).sort_values('date').set_index('date')
index_df  = pd.read_csv(BASE / 'sh.000001.csv',  parse_dates=['date']).sort_values('date').set_index('date')
sector_df = pd.read_csv(BASE / '881281.csv', parse_dates=['日期']).sort_values('日期').set_index('日期')

index_ret  = index_df['close'].pct_change()
sector_ret = sector_df['收盘价'].pct_change()

# 取三者共同交易日
common = (
    stock_df.index
    .intersection(index_ret.dropna().index)
    .intersection(sector_ret.dropna().index)
    .sort_values()
)

price_s  = stock_df.loc[common, 'close']
index_s  = index_ret.loc[common]
sector_s = sector_ret.loc[common]

# ── 2. 滚动预测 ───────────────────────────────────────────────────────────────
def get_quantile(dist_df: pd.DataFrame, q: float) -> float:
    idx = (dist_df['cdf'] - q).abs().idxmin()
    return float(dist_df.loc[idx, 'return'])

records = []
dates   = common.tolist()
n       = len(dates)

print(f"开始滚动预测，窗口={WINDOW}，共 {n - WINDOW - 1} 个预测点...")
for i in range(WINDOW, n - 1):
    p_s  = price_s.iloc[i - WINDOW: i + 1]
    i_s  = index_s.iloc[i - WINDOW: i + 1]
    se_s = sector_s.iloc[i - WINDOW: i + 1]

    try:
        model = fit_distribution(p_s, se_s, i_s)
        dist  = predict_distribution(model)
    except Exception:
        continue

    records.append({
        'date':       dates[i + 1],
        'prev_close': float(price_s.iloc[i]),
        'p25': get_quantile(dist, 0.25),
        'p40': get_quantile(dist, 0.40),
        'p60': get_quantile(dist, 0.60),
        'p75': get_quantile(dist, 0.75),
    })

    if (i - WINDOW + 1) % 50 == 0:
        print(f"  {i - WINDOW + 1}/{n - WINDOW - 1}")

pred_df = pd.DataFrame(records).set_index('date')
print(f"预测完成，共 {len(pred_df)} 条记录")

# ── 3. 合并实际数据，计算预测价格 ─────────────────────────────────────────────
plot_df = stock_df.loc[pred_df.index].copy()
plot_df = plot_df.join(pred_df[['prev_close', 'p25', 'p40', 'p60', 'p75']])

plot_df['pred25'] = plot_df['prev_close'] * (1 + plot_df['p25'])
plot_df['pred40'] = plot_df['prev_close'] * (1 + plot_df['p40'])
plot_df['pred60'] = plot_df['prev_close'] * (1 + plot_df['p60'])
plot_df['pred75'] = plot_df['prev_close'] * (1 + plot_df['p75'])

# ── 4. K线绘制工具 ────────────────────────────────────────────────────────────
def draw_candles(ax, xs, opens, highs, lows, closes,
                 up_color, down_color, width=0.35, alpha=1.0, lw=0.8):
    """在 ax 上绘制蜡烛图。"""
    for x, o, h, l, c in zip(xs, opens, highs, lows, closes):
        color = up_color if c >= o else down_color
        # 实体
        body_y = min(o, c)
        body_h = max(abs(c - o), (h - l) * 0.01)   # 十字星给最小高度
        rect = mpatches.Rectangle(
            (x - width / 2, body_y), width, body_h,
            facecolor=color, edgecolor=color, alpha=alpha, linewidth=0
        )
        ax.add_patch(rect)
        # 上下影线
        ax.plot([x, x], [l, h], color=color, linewidth=lw, alpha=alpha)


xs = np.arange(len(plot_df))

# ── 5. 通用绘图函数：将数据切成 4 段，每段一个子图，上下排列 ──────────────────
def plot_4panel(plot_df, pred_open_col, pred_close_col,
                title, legend_up_label, legend_down_label,
                out_path):
    n_total = len(plot_df)
    seg_size = n_total // 4
    # 每段起止索引（最后一段吸收余数）
    segs = [(i * seg_size, (i + 1) * seg_size if i < 3 else n_total)
            for i in range(4)]

    fig, axes = plt.subplots(4, 1, figsize=(18, 20))
    fig.suptitle(title, fontsize=14, y=0.995)

    legend_handles = [
        mpatches.Patch(color='#e8001c', label='actual up'),
        mpatches.Patch(color='#00a800', label='actual down'),
        mpatches.Patch(color='#ff8c00', label=legend_up_label),
        mpatches.Patch(color='#1e90ff', label=legend_down_label),
    ]

    for panel_idx, (s, e) in enumerate(segs):
        ax   = axes[panel_idx]
        sub  = plot_df.iloc[s:e]
        xs_s = np.arange(len(sub))
        d_s  = sub.index

        # 实际K线
        draw_candles(ax, xs_s,
                     sub['open'].values, sub['high'].values,
                     sub['low'].values,  sub['close'].values,
                     up_color='#e8001c', down_color='#00a800',
                     width=0.35, alpha=0.9)

        # 预测K线
        p_o = sub[pred_open_col].values
        p_c = sub[pred_close_col].values
        draw_candles(ax, xs_s,
                     p_o, np.maximum(p_o, p_c), np.minimum(p_o, p_c), p_c,
                     up_color='#ff8c00', down_color='#1e90ff',
                     width=0.35, alpha=0.55, lw=0.5)

        # 坐标轴
        step = max(1, len(xs_s) // 10)
        ax.set_xlim(-1, len(xs_s))
        ax.set_xticks(xs_s[::step])
        ax.set_xticklabels(
            [d.strftime('%Y-%m-%d') for d in d_s[::step]],
            rotation=30, ha='right', fontsize=8
        )
        ax.set_ylabel('Price (CNY)', fontsize=8)
        ax.grid(True, alpha=0.2)

        # 只在第一个子图显示图例
        if panel_idx == 0:
            ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")


# ── 6. 图1：[P25, P75] ────────────────────────────────────────────────────────
plot_4panel(
    plot_df,
    pred_open_col='pred25', pred_close_col='pred75',
    title='sz.002580  Actual K-line  vs  Predicted Range [P25, P75]',
    legend_up_label='predict [P25,P75] up',
    legend_down_label='predict [P25,P75] down',
    out_path=OUT_DIR / 'pingjia_p25_p75.png',
)

# ── 7. 图2：[P40, P60] ────────────────────────────────────────────────────────
plot_4panel(
    plot_df,
    pred_open_col='pred40', pred_close_col='pred60',
    title='sz.002580  Actual K-line  vs  Predicted Range [P40, P60]',
    legend_up_label='predict [P40,P60] up',
    legend_down_label='predict [P40,P60] down',
    out_path=OUT_DIR / 'pingjia_p40_p60.png',
)

plt.show()
