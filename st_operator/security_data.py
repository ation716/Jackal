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
import time

import tushare as ts
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class SecurityFileds(Enum):
    STOCK_CODE = "ts_code"
    TRADE_DATE = "trade_date"
    HISTORICAL_LOW = "his_low"
    HISTORICAL_HIGH = "his_high"
    COST_5TH_PERCENTILE = "cost_5pct"
    COST_15TH_PERCENTILE = "cost_15pct"
    COST_50TH_PERCENTILE = "cost_50pct"
    COST_85TH_PERCENTILE = "cost_85pct"
    COST_95TH_PERCENTILE = "cost_95pct"
    WEIGHTED_AVERAGE_COST = "weight_avg"
    WIN_RATE = "winner_rate"
    TURNOVER_RATE ='换手率'
    OPEN='open'
    CLOSE='close'
    HIGH='high'
    LOW='low'
    VOLUME='vol'  # 单位手
    AMOUNT='amount'  # 单位千元
    CHANGE='change'
    PCT_CHG='pct_chg'




class ChipDistributionAnalyzer:
    def __init__(self, token=None):
        """
        初始化Tushare Pro
        注册地址：https://tushare.pro/register
        """
        if token is None:
            token = "1bf9b910cdda6f0cd856f55b97c1c1419860237f7be8156aacac3259"
        ts.set_token(token)
        self.pro = ts.pro_api(token)

    def normal_ts_code(self,ts_code):
        """标准化股票代码"""
        if ts_code.startswith('60'):
            return ts_code+'.SH'
        elif ts_code.startswith('00'):
            return ts_code+'.SZ'
        else:
            return ts_code

    def get_daily_tu(self,ts_code,start_date,end_date):
        """获取日线行情
        详情参考 https://tushare.pro/document/2?doc_id=27
        """
        try:
            # 获取日线行情
            ts_code = self.normal_ts_code(ts_code)
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def get_all_pec(self,ts_code,start_date,end_date):
        """
        获取所有筹码分布
        :return:
        """

        try:
            # 获取日线行情
            df = self.pro.cyq_chips(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None



    def get_stock_chip_distribution(self, ts_code, start_date, end_date):
        """
        获取个股筹码分布数据
        :return:
        ts_code	str	Y
        trade_date	str	Y
        his_low	float	Y
        his_high	float	Y
        cost_5pct	float	Y
        cost_15pct	float	Y
        cost_50pct	float	Y
        cost_85pct	float	Y
        cost_95pct	float	Y
        weight_avg	float	Y
        winner_rate	float	Y
        """
        try:
            # 获取筹码分布数据
            ts_code = self.normal_ts_code(ts_code)
            df = self.pro.cyq_perf(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def get_stock_chip_akshare(self,ts_code,qfq=""):
        """akshare 的筹码分布"""
        return ak.stock_cyq_em(symbol=ts_code, adjust=qfq)

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

    def get_realtime_tick(self,ts_code):
        """获取实时行情
        """
        try:
            # 获取实时行情数据
            df = ts.realtime_quote(ts_code=ts_code)
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def get_daily_ak(self,symbol,start_date,end_date):
        # 获取股票历史数据
        if symbol.endswith(".SH"):
            symbol = symbol.replace(".SH","")
        if symbol.endswith(".SZ"):
            symbol = symbol.replace(".SZ","")
        df = ak.stock_zh_a_hist(symbol=symbol,
                                period='daily',
                                start_date=start_date,
                                end_date=end_date,
                                adjust="")        # 不复权
        return df

    def get_daily_limit_up(self,date):
        """"""
        column_mapping = {
            '序号': 'serial_number',
            '代码': 'symbol',
            '名称': 'name',
            '涨跌幅': 'change_percent',
            '最新价': 'latest_price',
            '成交额': 'turnover',
            '流通市值': 'circulating_market_cap',
            '总市值': 'total_market_cap',
            '换手率': 'turnover_rate',
            '封板资金': 'sealing_capital',
            '首次封板时间': 'first_sealing_time',
            '最后封板时间': 'last_sealing_time',
            '炸板次数': 'board_broken_times',
            '涨停统计': 'limit_up_stats',
            '连板数': 'continuous_boards',
            '所属行业': 'industry'
        }
        df = ak.stock_zt_pool_em(date="20251015")
        df = df.copy()
        # 只转换存在的列名
        existing_columns = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        df.rename(columns=existing_columns, inplace=True)
        return df

    def get_main_business_th(self,symbol):
        """
        :return
        主营业务，
        产品类型，
        产品名称，
        经营范围，
        """
        if symbol.endswith(".SH"):
            symbol = symbol.replace(".SH","")
        if symbol.endswith(".SZ"):
            symbol = symbol.replace(".SZ","")
        return ak.stock_zyjs_ths(symbol=symbol)

    def get_main_business_dc(self,symbol):
        """"""
        if symbol.endswith(".SH"):
            symbol = symbol.replace(".SH","")
        if symbol.endswith(".SZ"):
            symbol = symbol.replace(".SZ","")
        return ak.stock_zygc_em(symbol=symbol)

    def get_emotion(self):
        return ak.stock_market_activity_legu()

    def get_limit_up(self,date):
        return ak.stock_zt_pool_em(date=date)

    def get_strong(self,date):
        return ak.stock_zt_pool_strong_em(date=date)

    def get_price_crush(self,date):
        return ak.stock_zt_pool_zbgc_em(date=date)

    def get_limit_down(self,date):
        return ak.stock_zt_pool_dtgc_em(date=date)

    def get_daily_info(self,symbol):
        return ak.stock_info_global_cls(symbol=symbol)

    def get_daily_jgcyd(self,symbol):
        return ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)



# 使用示例
def demo_tushare():
    # 需要先注册获取token：https://tushare.pro/register

    analyzer = ChipDistributionAnalyzer()

    # 获取贵州茅台的数据
    df1 = analyzer.get_stock_chip_distribution(
        ts_code="601162.SH",
        start_date="20240924",
        end_date="20251010"
    )
    df2=analyzer.get_daily(
        ts_code="601162.SH",
        start_date="20240924",
        end_date="20251010")

    combined_df = pd.concat([df1, df2], axis=1)
    return combined_df

if __name__ == '__main__':
    # demo_tushare()
    ts_code = "603696"
    # start_date = "20240924"
    # end_date = "20251010"
    # analyzer = ChipDistributionAnalyzer()
    # df1=analyzer.get_daily_ak(ts_code,start_date,end_date)
    # df2=analyzer.get_stock_chip_distribution(ts_code,start_date,end_date)
    # df3=analyzer.get_daily_tu(ts_code,start_date,end_date)
    # df2=df2.iloc[::-1].reset_index(drop=True)
    # df3=df3.iloc[::-1].reset_index(drop=True)
    # combined_df = pd.concat([df2, df3.iloc[:,2:11], df1.iloc[:,11:12]], axis=1)
    date='20260112'
    start_date='20251127'
    end_date='20251223'
    analyzer = ChipDistributionAnalyzer()
    name='dmgx'
    jgcyd=analyzer.get_daily_jgcyd(ts_code)
    # '002632'
    # data=analyzer.get_stock_chip_distribute_detail(ts_code='002632.SZ',start_date=start_date,end_date=end_date)
    # data=analyzer.get_stock_chip_akshare(ts_code='002632')
    # data.to_csv(f'../st_operator/{name}.csv',
    #                    index=False,  # 不保存索引
    #                    encoding='utf_8_sig',  # 支持中文
    #                    sep=',')  # 分隔符
    limit_up=analyzer.get_limit_up(date=date)
    # strong=analyzer.get_strong(date=date)
    crush=analyzer.get_price_crush(date=date)
    # limit_down=analyzer.get_limit_down(date=date)
    # info=analyzer.get_daily_info('全部')
    # info=analyzer.get_all_pec(ts_code,start_date,end_date)
    # info.to_csv('tem2',mode="a", index=False, header=False)
    time.sleep(8)
