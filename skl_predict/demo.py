# -*- coding: utf-8 -*-
# @Time    : 2026/4/15 16:21
# @Author  : gaolei
# @FileName: demo.py
# @Software: PyCharm
import os
from datetime import date, timedelta
from basic_s import ChipDistributionAnalyzer as Analyzer
import baostock as bs
import pandas as pd


# if __name__ == '__main__':
#     fec=Analyzer()
#     aim_path = os.path.join(os.path.dirname(__file__), '..', 'data')
#     print(aim_path)
#     # codes = ['sz.000678', 'sh.600396']
#     codes = ['sz.002580']
#     # codes = ['sh.000001']
#     start_d = date.today()
#     end = start_d.strftime('%Y%m%d')  # 例如：20260415
#
#     # 计算90天前的日期
#     days_ago = start_d - timedelta(days=90)
#     start = days_ago.strftime('%Y%m%d')  # 例如：20260115
#     for code in codes:
#         file_path =os.path.join(os.path.dirname(__file__), 'predict_tes_data',f'{code}.csv')
#         # 登录系统
#         lg = bs.login()
#
#         # 获取平安银行从2023年1月1日到2023年12月31日的日K线数据
#         rs = bs.query_history_k_data_plus(code,
#                                           "date,open,high,low,close,volume",
#                                           start_date='2025-01-01', end_date='2026-04-16',
#                                           frequency="d", adjustflag="3")
#
#         # 将结果转为DataFrame
#         df_list = []
#         while (rs.error_code == '0') & rs.next():
#             df_list.append(rs.get_row_data())
#         df = pd.DataFrame(df_list, columns=rs.fields)
#         print(df.head())
#
#         # 登出系统
#         bs.logout()
#
#         # df=fec.get_daily_ak(code,start,end)
#         df.to_csv(f'{file_path}', index=False, encoding='utf-8-sig')
#
import akshare as ak
import os
# 1. 获取概念板块指数数据
# df = ak.stock_board_concept_index_ths(
#     symbol="固态电池",       # 板块名称
#     start_date="20250101",   # 开始日期
#     end_date="20260416"      # 结束日期
# )
# file_path =os.path.join(os.path.dirname(__file__), 'predict_tes_data',f'{881281}.csv')
# df.to_csv(f'{file_path}', index=False, encoding='utf-8-sig')
# print("机器人概念板块指数历史数据：")
# print(stock_board_concept_index_ths_df.head())
# print(stock_board_concept_index_ths_df.columns)

# 2. 获取概念板块的成份股数据（如果需要）
# 注意：根据搜索结果，这个接口可能因数据源限制，已不再维护[reference:1]
# stock_board_concept_cons_ths_df = ak.stock_board_concept_cons_ths(symbol="机器人概念")
# print("机器人概念成份股：")
# print(stock_board_concept_cons_ths_df.head())

# ── 收益率分布可视化 ──────────────────────────────────────────────────────────
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
#
# result_path = os.path.join(os.path.dirname(__file__), 'result.csv')
# df = pd.read_csv(result_path)
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# fig.suptitle('Return Distribution', fontsize=14, fontweight='bold')
#
# # PDF
# ax1.plot(df['return'], df['pdf'], color='steelblue', linewidth=1.5)
# ax1.fill_between(df['return'], df['pdf'], alpha=0.25, color='steelblue')
# ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='return=0')
# ax1.set_ylabel('PDF')
# ax1.legend(fontsize=9)
# ax1.grid(True, alpha=0.3)
#
# # 标注 PDF 峰值
# peak_idx = df['pdf'].idxmax()
# ax1.annotate(
#     f"peak: {df.loc[peak_idx, 'return']:.4f}",
#     xy=(df.loc[peak_idx, 'return'], df.loc[peak_idx, 'pdf']),
#     xytext=(df.loc[peak_idx, 'return'] + 0.01, df.loc[peak_idx, 'pdf'] * 0.9),
#     arrowprops=dict(arrowstyle='->', color='gray'),
#     fontsize=8,
# )
#
# # CDF
# ax2.plot(df['return'], df['cdf'], color='darkorange', linewidth=1.5)
# ax2.axvline(0, color='red', linestyle='--', linewidth=1, label='return=0')
# ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='CDF=0.5')
# ax2.set_ylabel('CDF')
# ax2.set_xlabel('Return')
# ax2.legend(fontsize=9)
# ax2.grid(True, alpha=0.3)
# ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
#
# # 标注 CDF=0.5 对应的收益率（中位数）
# median_idx = (df['cdf'] - 0.5).abs().idxmin()
# median_ret = df.loc[median_idx, 'return']
# ax2.annotate(
#     f"median: {median_ret:.4f}",
#     xy=(median_ret, 0.5),
#     xytext=(median_ret + 0.01, 0.4),
#     arrowprops=dict(arrowstyle='->', color='gray'),
#     fontsize=8,
# )
#
# plt.tight_layout()
# plt.show()

import akshare as ak
import pandas as pd


def get_single_stock_announcement(stock_code="603757", date="20260611"):
    """
    查询单只股票的公告
    """
    try:
        # 确保股票代码是6位格式
        stock_code = stock_code.zfill(6)

        # 获取指定日期的全部公告
        df_all = ak.stock_notice_report(date=date)

        # 筛选出目标股票的公告
        df_stock = df_all[df_all['代码'] == stock_code]

        return df_stock
    except Exception as e:
        print(f"获取失败: {e}")
        return pd.DataFrame()


# 示例：查询平安银行（000001）在2024年6月13日的公告
result = get_single_stock_announcement("603757", "20260610")
print(result)