# -*- coding: utf-8 -*-
# @Time    : 2025/9/7 16:56
# @Author  : gaolei
# @FileName: new_t.py
# @Software: PyCharm
import akshare as ak
import pandas as pd

# 假设这是您的自选股列表，股票代码需要符合规范的格式（例如：sh600000, sz000001）
your_watchlist = ['sh600036', 'sz000333', 'sh601888']

# 1. 获取板块资金流向（示例：行业板块）
# 注意：AkShare中不同函数获取的资金流向数据口径和格式可能不同，请以最新文档为准
sector_df = ak.stock_sector_fund_flow_rank(indicator="今日")
print("=== 板块资金流入排名（今日）===")
# 通常包含板块名称、涨跌幅、主力净流入等字段
print(sector_df[['板块名称', '涨跌幅', '主力净流入']].head(10))

# 2. 获取自选股的实时行情信息
print("\n=== 自选股实时行情 ===")
for stock_code in your_watchlist:
    # 注意：stock_zh_a_spot_em 或其他实时行情函数可能一次获取全市场或需要特定代码格式
    # 更常见的做法是先获取全市场股票实时行情，再过滤出你的自选股
    pass

# 更高效的方式：先获取全市场实时概况，再筛选
# 注意：这个函数可能会返回大量数据，请谨慎调用
all_spot_df = ak.stock_zh_a_spot_em()
watchlist_spot_df = all_spot_df[all_spot_df['代码'].isin([code[2:] for code in your_watchlist])] # 假设your_watchlist是带市场前缀的，而返回的'代码'字段不含前缀
print(watchlist_spot_df[['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额']])

# 3. 获取个股实时资金流向（示例）
# AkShare可能有专门的函数，例如 stock_individual_fund_flow
# 注意：不同接口的资金流向数据更新频率和计算口径可能不同
print("\n=== 自选股资金流向（示例） ===")
for stock_code in your_watchlist:
    # 这里需要根据AkShare最新文档查找正确的函数和参数
    # individual_flow_df = ak.stock_individual_fund_flow(stock=stock_code, market=None)
    # print(individual_flow_df)
    pass

# 由于实时资金流向接口在AkShare中可能不稳定或需要特定参数，您可能需要探索其他数据源。