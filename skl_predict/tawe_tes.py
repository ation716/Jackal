import taew
import mplfinance as mpf
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from basic_s import ChipDistributionAnalyzer as analyzer

path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.join(path, 'data', 'opportunity', 'opportunity.json')

al = analyzer()
today = datetime.now().date()
date_before = today - timedelta(days=120)
end = today.strftime('%Y%m%d')
start = date_before.strftime('%Y%m%d')

with open(path, 'r', encoding='utf-8') as ff:
    data = json.load(ff)

df = None
for industry, codes in data.items():
    for code in codes.get('codes', []):
        df = al.get_daily_bs(code, start, end)
        if not df.empty:
            break
    if df is not None and not df.empty:
        break

if df is None or df.empty:
    raise ValueError("未获取到有效数据")

# 准备 mplfinance 所需格式：DatetimeIndex + 大写列名
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

# ========== 计算波浪 ==========
prices = df['Close'].tolist()
waves = taew.Alternative_ElliottWave_label_upward(prices)

positions = [w['z'] for w in waves]
values = [w['x'] for w in waves]

# 上升波：偶数索引为波谷，奇数索引为波峰
peaks_idx, peaks_val = [], []
troughs_idx, troughs_val = [], []
for i, (pos, val) in enumerate(zip(positions, values)):
    if i % 2 == 0:
        troughs_idx.append(pos)
        troughs_val.append(val)
    else:
        peaks_idx.append(pos)
        peaks_val.append(val)

# ========== 构建散点序列 ==========
peak_series = pd.Series(index=df.index, dtype=float)
trough_series = pd.Series(index=df.index, dtype=float)
for idx, val in zip(peaks_idx, peaks_val):
    if idx < len(df):
        peak_series.iloc[idx] = val
for idx, val in zip(troughs_idx, troughs_val):
    if idx < len(df):
        trough_series.iloc[idx] = val
print('波峰', peak_series.dropna())
print('波谷', trough_series.dropna())

# ========== 绘制 K 线 + 成交量 + 波浪标记 ==========
# 波峰标记贴在 high 上方，波谷标记贴在 low 下方
offset = df['Close'].mean() * 0.01  # 约1%偏移，让标记紧贴K线

peak_plot = pd.Series(index=df.index, dtype=float)
trough_plot = pd.Series(index=df.index, dtype=float)
for idx, val in zip(peaks_idx, peaks_val):
    if idx < len(df):
        peak_plot.iloc[idx] = df['High'].iloc[idx] + offset
for idx, val in zip(troughs_idx, troughs_val):
    if idx < len(df):
        trough_plot.iloc[idx] = df['Low'].iloc[idx] - offset

ap = []
if not peak_plot.dropna().empty:
    ap.append(mpf.make_addplot(peak_plot, type='scatter', markersize=80, marker='^', color='red'))
if not trough_plot.dropna().empty:
    ap.append(mpf.make_addplot(trough_plot, type='scatter', markersize=80, marker='v', color='green'))

mpf.plot(df, type='candle', volume=True, addplot=ap, style='charles', figsize=(12, 6))