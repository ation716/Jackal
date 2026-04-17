"""
使用 akshare， tushare，baostock 获取证券数据
tushare code 需要后面接 .SH 或 .SZ
"""
import json
import os
import time

import tushare as ts
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import baostock as bs


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
    TURNOVER_RATE = '换手率'
    OPEN = 'open'
    CLOSE = 'close'
    HIGH = 'high'
    LOW = 'low'
    VOLUME = 'vol'   # 单位手
    AMOUNT = 'amount'  # 单位千元
    CHANGE = 'change'
    PCT_CHG = 'pct_chg'


class ChipDistributionAnalyzer:
    def __init__(self, token=None):
        """
        初始化 Tushare Pro API。

        参数:
             token (str): tushare token，默认使用内置 token。

        注册地址：https://tushare.pro/register
        """
        if token is None:
            token = "1bf9b910cdda6f0cd856f55b97c1c1419860237f7be8156aacac3259"
        ts.set_token(token)
        self.pro = ts.pro_api(token)
        self._bs_logged_in = False

    def __del__(self):
        """对象销毁时登出 baostock"""
        if self._bs_logged_in:
            bs.logout()

    def _bs_login(self):
        """内部方法：登录 baostock（仅登录一次）"""
        if not self._bs_logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise Exception(f"baostock 登录失败: {lg.error_msg}")
            self._bs_logged_in = True

    def normal_ts_code(self, ts_code):
        """
        将 6 位纯数字股票代码标准化为 tushare 格式（带交易所后缀）。

        参数:
            ts_code (str): 6 位股票代码，如 '600000' 或 '000001'。

        返回示例:
            '600000' -> '600000.SH'
            '000001' -> '000001.SZ'
        """
        if ts_code.startswith('60'):
            return ts_code + '.SH'
        elif ts_code.startswith('00'):
            return ts_code + '.SZ'
        else:
            return ts_code

    def get_daily_tu(self, ts_code, start_date, end_date):
        """
        通过 tushare 获取个股日线行情数据。

        参数:
            ts_code (str): 股票代码，6 位或带后缀，如 '600000' 或 '600000.SH'。
            start_date (str): 开始日期，格式 'YYYYMMDD'，如 '20240101'。
            end_date (str): 结束日期，格式 'YYYYMMDD'，如 '20240131'。

        返回示例:
            DataFrame，列：ts_code, trade_date, open, high, low, close, pre_close,
                          change, pct_chg, vol, amount
        """
        ts_code = ts_code if len(ts_code) > 6 else self.normal_ts_code(ts_code)
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def get_all_pec(self, ts_code, start_date, end_date):
        """
        获取个股详细筹码分布（每个价位的持仓占比）。

        参数:
            ts_code (str): 带后缀的股票代码，如 '000001.SZ'。
            start_date (str): 开始日期，格式 'YYYYMMDD'。
            end_date (str): 结束日期，格式 'YYYYMMDD'。

        返回示例:
            DataFrame，列：ts_code, trade_date, price, percent
        """
        ts_code = ts_code if len(ts_code) > 6 else self.normal_ts_code(ts_code)
        df = self.pro.cyq_chips(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def get_stock_chip_distribution(self, ts_code, start_date, end_date):
        """
        获取个股筹码分布关键指标（成本分位、获利盘等）。

        参数:
            ts_code (str): 股票代码，6 位或带后缀。
            start_date (str): 开始日期，格式 'YYYYMMDD'。
            end_date (str): 结束日期，格式 'YYYYMMDD'。

        返回示例:
            DataFrame，列：ts_code, trade_date, his_low, his_high,
                          cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct,
                          weight_avg, winner_rate
        """
        ts_code = ts_code if len(ts_code) > 6 else self.normal_ts_code(ts_code)
        df = self.pro.cyq_perf(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def get_stock_chip_akshare(self, ts_code, qfq=""):
        """
        通过 akshare 获取个股筹码分布数据。

        参数:
            ts_code (str): 6 位股票代码，如 '000001'。
            qfq (str): 复权方式，'' 不复权，'qfq' 前复权，'hfq' 后复权。

        返回示例:
            DataFrame，列：日期, 获利比例, 平均成本, 90成本-低, 90成本-高,
                          90集中度, 70成本-低, 70成本-高, 70集中度
        """
        return ak.stock_cyq_em(symbol=ts_code, adjust=qfq)

    def get_stock_chip_distribute_detail(self, ts_code, start_date, end_date):
        """
        获取详细筹码分布（每个价位的持仓占比，与 get_all_pec 相同接口）。

        参数:
            ts_code (str): 带后缀的股票代码，如 '000001.SZ'。
            start_date (str): 开始日期，格式 'YYYYMMDD'。
            end_date (str): 结束日期，格式 'YYYYMMDD'。

        返回示例:
            DataFrame，列：ts_code, trade_date, price, percent
        """
        ts_code = ts_code if len(ts_code) > 6 else self.normal_ts_code(ts_code)
        df = self.pro.cyq_chips(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def get_stock_basic(self, ts_code):
        """
        获取股票基本信息（代码和所属行业）。

        参数:
            code (str): 带后缀的股票代码，如 '000001.SZ'。

        返回示例:
            DataFrame，列：ts_code, industry
            示例行：('000001.SZ', '银行')
        """
        ts_code = ts_code if len(ts_code) > 6 else self.normal_ts_code(ts_code)
        df = self.pro.stock_basic(ts_code=ts_code, fields='ts_code,industry')
        return df

    def get_report_rc(self, ts_code, report_date=None, start_date=None, end_date=None):
        """
        获取券商研报预测数据。

        参数:
            ts_code (str): 带后缀的股票代码，如 '000001.SZ'。
            report_date (str): 指定报告日期，格式 'YYYYMMDD'，与 start/end_date 二选一。
            start_date (str): 开始日期，格式 'YYYYMMDD'。
            end_date (str): 结束日期，格式 'YYYYMMDD'。

        返回示例:
            DataFrame，含研报日期、机构名称、评级、目标价等字段。
            详情参考 https://tushare.pro/document/2?doc_id=292
        """
        if report_date is None:
            df = self.pro.report_rc(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = self.pro.report_rc(ts_code=ts_code, report_date=report_date)
        return df

    def get_realtime_tick(self, ts_code):
        """
        获取个股实时行情快照。

        参数:
            ts_code (str): 带后缀的股票代码，如 '000001.SZ'。

        返回示例:
            DataFrame，列：TS_CODE, TRADE_DATE, PRE_CLOSE, OPEN, HIGH, LOW,
                          PRICE, CHANGE, PCT_CHANGE, VOL, AMOUNT
        """
        df = ts.realtime_quote(ts_code=ts_code)
        return df

    def get_daily_ak(self, symbol, start_date, end_date):
        """
        通过 akshare 获取个股日线行情（不复权）。

        参数:
            symbol (str): 股票代码，支持 6 位纯数字或带后缀格式。
            start_date (str): 开始日期，格式 'YYYYMMDD'，如 '20240101'。
            end_date (str): 结束日期，格式 'YYYYMMDD'，如 '20240131'。

        返回示例:
            DataFrame，列：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅,
                          涨跌幅, 涨跌额, 换手率
        """
        if symbol.endswith(".SH"):
            symbol = symbol.replace(".SH", "")
        if symbol.endswith(".SZ"):
            symbol = symbol.replace(".SZ", "")
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period='daily',
            start_date=start_date,
            end_date=end_date,
            adjust=""
        )
        return df

    def get_daily_limit_up(self, date):
        """
        获取指定日期的涨停股票池数据。

        参数:
            date (str): 交易日期，格式 'YYYYMMDD'，如 '20240101'。

        返回示例:
            DataFrame，列（英文化后）：symbol, name, change_percent, latest_price,
                          turnover, circulating_market_cap, total_market_cap,
                          turnover_rate, sealing_capital, first_sealing_time,
                          last_sealing_time, board_broken_times, limit_up_stats,
                          continuous_boards, industry
        """
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
        df = ak.stock_zt_pool_em(date=date)
        df = df.copy()
        existing_columns = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        df.rename(columns=existing_columns, inplace=True)
        return df

    def get_main_business_th(self, symbol):
        """
        通过同花顺接口获取个股主营业务信息。

        参数:
            symbol (str): 股票代码，支持 6 位纯数字或带后缀格式。

        返回示例:
            DataFrame，列：主营业务, 产品类型, 产品名称, 经营范围
        """
        if symbol.endswith(".SH"):
            symbol = symbol.replace(".SH", "")
        if symbol.endswith(".SZ"):
            symbol = symbol.replace(".SZ", "")
        return ak.stock_zyjs_ths(symbol=symbol)

    def get_main_business_dc(self, symbol):
        """
        通过东财接口获取个股主营构成数据。

        参数:
            symbol (str): 股票代码，支持 6 位纯数字或带后缀格式。

        返回示例:
            DataFrame，列：报告期, 分类方向, 主营构成, 主营收入, 收入比例,
                          主营成本, 成本比例, 主营利润, 利润比例, 毛利率
        """
        if symbol.endswith(".SH"):
            symbol = symbol.replace(".SH", "")
        if symbol.endswith(".SZ"):
            symbol = symbol.replace(".SZ", "")
        return ak.stock_zygc_em(symbol=symbol)

    def get_emotion(self):
        """
        获取市场情绪数据（乐股接口）。

        参数: 无

        返回示例:
            DataFrame，含涨停数、跌停数、上涨家数、下跌家数等市场情绪指标。
        """
        return ak.stock_market_activity_legu()

    def get_limit_up(self, date):
        """
        获取指定日期涨停股票池（原始列名，中文）。

        参数:
            date (str): 交易日期，格式 'YYYYMMDD'，如 '20240101'。

        返回示例:
            DataFrame，列：序号, 代码, 名称, 涨跌幅, 最新价, 封板资金, 首次封板时间,
                          最后封板时间, 炸板次数, 涨停统计, 连板数, 所属行业
        """
        return ak.stock_zt_pool_em(date=date)

    def get_strong(self, date):
        """
        获取指定日期强势股票池（连续涨停或高位强势股）。

        参数:
            date (str): 交易日期，格式 'YYYYMMDD'，如 '20240101'。

        返回示例:
            DataFrame，列：序号, 代码, 名称, 涨跌幅, 最新价, 涨停统计, 入选理由, 所属行业
        """
        return ak.stock_zt_pool_strong_em(date=date)

    def get_price_crush(self, date):
        """
        获取指定日期炸板股票池（曾涨停但未封住的股票）。

        参数:
            date (str): 交易日期，格式 'YYYYMMDD'，如 '20240101'。

        返回示例:
            DataFrame，列：序号, 代码, 名称, 涨跌幅, 最新价, 成交额, 炸板次数, 所属行业
        """
        return ak.stock_zt_pool_zbgc_em(date=date)

    def get_limit_down(self, date):
        """
        获取指定日期跌停股票池。

        参数:
            date (str): 交易日期，格式 'YYYYMMDD'，如 '20240101'。

        返回示例:
            DataFrame，列：序号, 代码, 名称, 涨跌幅, 最新价, 成交额, 换手率, 所属行业
        """
        return ak.stock_zt_pool_dtgc_em(date=date)

    def get_daily_info(self, symbol):
        """
        获取全球财经快讯（CLS 财联社）。

        参数:
            symbol (str): 查询关键词或股票代码。

        返回示例:
            DataFrame，含快讯标题、时间、内容等字段。
        """
        return ak.stock_info_global_cls(symbol=symbol)

    def get_daily_jgcyd(self, symbol):
        """
        获取个股机构参与度数据（东财接口）。

        参数:
            symbol (str): 6 位股票代码，如 '000001'。

        返回示例:
            DataFrame，含机构参与度、买入金额、卖出金额等字段。
        """
        return ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)

    def get_fund(self, symbol):
        """
        获取个股资金流向数据（akshare 接口）。

        参数:
            symbol (str): 6 位股票代码，如 '600000'（沪）或 '000001'（深）。

        返回示例:
            DataFrame，列：日期, 主力净流入, 小单净流入, 中单净流入, 大单净流入, 超大单净流入
        """
        market = 'sh' if symbol.startswith('6') else 'sz'
        return ak.stock_individual_fund_flow(symbol, market=market)

    def get_daily_bs(self, symbol, start_date, end_date, adjustflag='3', frequency='d'):
        """
        通过 baostock 获取个股日线数据（支持复权）。

        参数:
            symbol (str): 股票代码，支持6位数字（如 '600000'）或带后缀格式（如 '600000.SH'）。
            start_date (str): 开始日期，格式 'YYYYMMDD'，如 '20240101'。
            end_date (str): 结束日期，格式 'YYYYMMDD'，如 '20240131'。
            adjustflag (str): 复权类型，'1'后复权，'2'前复权，'3'不复权，默认'3'。
            frequency (str): 频率，'d'日线，'w'周线，'m'月线，'5'5分钟线等，默认'd'。

        返回:
            pd.DataFrame: 包含字段 date, open, high, low, close, volume, amount, adjustflag 等。
        """
        # 登录
        self._bs_login()

        # 标准化股票代码为 baostock 格式：sh.600000 或 sz.000001
        code = symbol.upper()
        if code.endswith('.SH'):
            code = code.replace('.SH', '')
            bs_code = f"sh.{code}"
        elif code.endswith('.SZ'):
            code = code.replace('.SZ', '')
            bs_code = f"sz.{code}"
        else:
            # 6位数字，根据首位判断市场
            if code.startswith('6'):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"

        # 日期格式转换：20240101 -> 2024-01-01
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        # 请求数据
        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields="date,open,high,low,close,volume,amount,adjustflag",
            start_date=start,
            end_date=end,
            frequency=frequency,
            adjustflag=adjustflag
        )

        # 将结果转为 DataFrame
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return pd.DataFrame()  # 无数据时返回空 DataFrame

        df = pd.DataFrame(data_list, columns=rs.fields)
        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df


if __name__ == '__main__':
    analyzer = ChipDistributionAnalyzer()
    symbol_sh = '600000'
    symbol_sz = '000001'
    ts_code_sh = '600000.SH'
    ts_code_sz = '000001.SZ'
    start = '20240101'
    end = '20240131'
    date = '20240131'

    print("=== normal_ts_code ===")
    print(analyzer.normal_ts_code('600000'))   # 600000.SH
    print(analyzer.normal_ts_code('000001'))   # 000001.SZ

    print("\n=== get_daily_tu ===")
    df = analyzer.get_daily_tu(symbol_sh, start, end)
    print(df.head())

    print("\n=== get_all_pec ===")
    df = analyzer.get_all_pec(ts_code_sz, start, end)
    print(df.head())

    print("\n=== get_stock_chip_distribution ===")
    df = analyzer.get_stock_chip_distribution(symbol_sz, start, end)
    print(df.head())

    print("\n=== get_stock_chip_akshare ===")
    df = analyzer.get_stock_chip_akshare(symbol_sz)
    print(df.head())

    print("\n=== get_stock_chip_distribute_detail ===")
    df = analyzer.get_stock_chip_distribute_detail(ts_code_sz, start, end)
    print(df.head())

    print("\n=== get_stock_basic ===")
    df = analyzer.get_stock_basic(ts_code_sz)
    print(df)

    print("\n=== get_report_rc ===")
    df = analyzer.get_report_rc(ts_code_sz, start_date=start, end_date=end)
    print(df.head())

    print("\n=== get_realtime_tick ===")
    df = analyzer.get_realtime_tick(ts_code_sz)
    print(df)

    print("\n=== get_daily_ak ===")
    df = analyzer.get_daily_ak(symbol_sh, start, end)
    print(df.head())

    print("\n=== get_daily_limit_up ===")
    df = analyzer.get_daily_limit_up(date)
    print(df.head())

    print("\n=== get_main_business_th ===")
    df = analyzer.get_main_business_th(symbol_sz)
    print(df.head())

    print("\n=== get_main_business_dc ===")
    df = analyzer.get_main_business_dc(symbol_sz)
    print(df.head())

    print("\n=== get_emotion ===")
    df = analyzer.get_emotion()
    print(df)

    print("\n=== get_limit_up ===")
    df = analyzer.get_limit_up(date)
    print(df.head())

    print("\n=== get_strong ===")
    df = analyzer.get_strong(date)
    print(df.head())

    print("\n=== get_price_crush ===")
    df = analyzer.get_price_crush(date)
    print(df.head())

    print("\n=== get_limit_down ===")
    df = analyzer.get_limit_down(date)
    print(df.head())

    print("\n=== get_daily_jgcyd ===")
    df = analyzer.get_daily_jgcyd(symbol_sz)
    print(df.head())

    print("\n=== get_fund ===")
    df = analyzer.get_fund(symbol_sh)
    print(df.head())
