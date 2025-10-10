# 一个新的获取股票数据的库 tushare, 目前部分接口有积分限制, 这里只用来获取筹码分布信息
"""
tushare 官方文档 https://tushare.pro/document/2?doc_id=27
常用接口:

1. 日线行情
df = pro.daily(ts_code='000001.SZ,600000.SH', start_date='20180701', end_date='20180718')
返回字段

ts_code	str	股票代码
trade_date	str	交易日期
open	float	开盘价
high	float	最高价
low	float	最低价
close	float	收盘价
pre_close	float	昨收价【除权价，前复权】 （当公司进行分红、送股、配股等行为时，会在除权除息日对股价进行调整，这个调整后的价格就是除权价。前复权是以当前价格为基准，将历史价格按除权规则进行调整，使得历史价格与当前价格具有可比性。）
change	float	涨跌额
pct_chg	float	涨跌幅 【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收 】
vol	float	成交量 （手）
amount	float	成交额 （千元）

"""

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ChipDistributionAnalyzer:
    def __init__(self, token=None):
        """
        初始化Tushare Pro
        注册地址：https://tushare.pro/register
        """
        if token is None:
            token = "1bf9b910cdda6f0cd856f55b97c1c1419860237f7be8156aacac3259"
        self.pro = ts.pro_api(token)

    def get_daily(self,ts_code,start_date,end_date):
        """获取日线行情
        详情参考 https://tushare.pro/document/2?doc_id=27
        """
        try:
            # 获取日线行情
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None


    def get_stock_chip_distribution(self, ts_code, start_date, end_date):
        """
        获取个股筹码分布数据
        :return:
        ts_code	str	Y	股票代码
        trade_date	str	Y	交易日期
        his_low	float	Y	历史最低价
        his_high	float	Y	历史最高价
        cost_5pct	float	Y	5分位成本
        cost_15pct	float	Y	15分位成本
        cost_50pct	float	Y	50分位成本
        cost_85pct	float	Y	85分位成本
        cost_95pct	float	Y	95分位成本
        weight_avg	float	Y	加权平均成本
        winner_rate	float	Y	胜率
        """
        try:
            # 获取筹码分布数据
            df = self.pro.cyq_perf(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def get_stock_chip_distribute_detail(self, ts_code, start_date, end_date):
        """
        获取详细筹码分布
        :return:
        ts_code	str	Y	股票代码
        trade_date	str	Y	交易日期
        price	float	Y	成本价格
        percent	float	Y	价格占比（%）
        """
        try:
            # 获取详细筹码分布数据
            df = self.pro.cyq_chips(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def get_report_rc(self, ts_code, report_date=None,start_date=None, end_date=None):
        """券商预测数据
        详情参考 https://tushare.pro/document/2?doc_id=292
        """
        try:
            # 获取详细筹码分布数据
            if report_date is None:
                df = self.pro.report_rc(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = self.pro.report_rc(ts_code=ts_code, report_date='20220429')
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None


# 使用示例
def demo_tushare():
    # 需要先注册获取token：https://tushare.pro/register

    analyzer = ChipDistributionAnalyzer()

    # 获取贵州茅台的数据
    df = analyzer.get_stock_chip_distribution(
        ts_code="002115.SZ",
        start_date="20250919",
        end_date="20250929"
    )


    return df

if __name__ == '__main__':
    demo_tushare()