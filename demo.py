import akshare as ak

# 获取实时行情
stock_zh_a_spot = ak.stock_zh_a_spot()
print(stock_zh_a_spot[stock_zh_a_spot['代码'] == '000001'])

# 获取分时数据
stock_zh_a_minute = ak.stock_zh_a_minute(symbol='sh000001', period='1')
print(stock_zh_a_minute.tail())

# 计算MACD（需自行计算）
import talib
close_prices = stock_zh_a_minute['close'].astype(float)
macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
print("MACD:", macd[-1], "Signal:", signal[-1])

# 1. 获取 A 股实时行情（包含量价数据）
df_spot = ak.stock_zh_a_spot()
df_stock = df_spot[df_spot['代码'] == '000001']  # 筛选平安银行
print(df_stock[['代码', '名称', '最新价', '成交量', '涨跌幅']])

# 2. 获取分时数据（每分钟）
df_minute = ak.stock_zh_a_minute(symbol='sh000001', period='1', adjust='qfq')
print(df_minute[['时间', '开盘', '收盘', '成交量']].tail())