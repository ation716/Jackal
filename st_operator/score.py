# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : gaolei
# @FileName: score.py
"""
股票评分系统 - 对一篮子 A 股主板股票进行实时综合评分

评分维度（满分 100 分）:
  1. historical_position  历史位置（150日高点位置与时间）  20 分
  2. recent_change        近期累计涨幅（3/7/15日）        20 分
  3. relative_strength    相对强度乖离（vs 上证/板块）     20 分
  4. fund_flow            大单资金流向（15/5/1日）         20 分
  5. chip                 筹码分布集中度                   10 分
  6. technical            技术指标（RSI、量比）             5 分
  7. hot_rank             热榜排名                         5 分
"""

import sys
import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

import akshare as ak
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from security_data import ChipDistributionAnalyzer


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

def to_ts_code(code):
    code = str(code).zfill(6)
    return code + '.SH' if code.startswith('6') else code + '.SZ'


def get_market(code):
    code = str(code).zfill(6)
    return 'sh' if code.startswith('6') else 'sz'


def safe_call(func, default=None, sleep_sec=0.3, **kwargs):
    try:
        result = func(**kwargs)
        time.sleep(sleep_sec)
        return result
    except Exception as e:
        print(f"  [接口异常] {func.__name__}: {e}")
        time.sleep(sleep_sec)
        return default


def calc_rsi(close_series, period=14):
    """计算 RSI，返回最新值"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return round(100 - 100 / (1 + rs), 2)


def calc_volume_ratio(vol_series, days=5):
    """量比 = 当日量 / 前 N 日均量"""
    if len(vol_series) < days + 1:
        return 1.0
    avg = vol_series.iloc[-(days + 1):-1].mean()
    if avg == 0:
        return 1.0
    return round(vol_series.iloc[-1] / avg, 2)


def pct_change_n(close_arr, n):
    """最后一根收盘 vs 往前 n 根收盘的涨跌幅（%）"""
    if len(close_arr) <= n:
        return 0.0
    base = close_arr[-(n + 1)]
    if base == 0:
        return 0.0
    return round((close_arr[-1] - base) / base * 100, 2)


# ─────────────────────────────────────────────────────────────────
# 数据获取层（含会话级缓存）
# ─────────────────────────────────────────────────────────────────

class DataFetcher:
    def __init__(self):
        self.analyzer = ChipDistributionAnalyzer()
        self._sh_cache = None       # 上证指数历史
        self._hot_cache = None      # 热榜
        self._realtime_map = {}     # {6位代码: 实时价格}

    # ── 实时行情（tushare realtime_quote 批量拉取）──────────────
    @staticmethod
    def _gen_ts_symbols(code_list):
        """将 6 位代码列表转成 tushare 接受的 ts_code 字符串"""
        parts = [to_ts_code(c) for c in code_list]
        return ','.join(parts)

    def prefetch_realtime(self, stock_list):
        """批量获取实时行情并缓存到 _realtime_map {code6: price}"""
        symbols = self._gen_ts_symbols(stock_list)
        df = self.analyzer.get_realtime_tick(ts_code=symbols)
        self._realtime_map = {}
        if df is None or df.empty:
            print("  [实时行情] 获取失败，价格将回退到历史最新收盘")
            return
        for i in range(len(df)):
            try:
                ts_code = str(df.iloc[i, 0])   # e.g. '600203.SH'
                price   = float(df.iloc[i, 6]) # PRICE 列
                self._realtime_map[ts_code[:6]] = price
            except Exception:
                pass

    def get_realtime_price(self, code):
        """从缓存的实时行情里查价格"""
        return self._realtime_map.get(str(code).zfill(6))

    # ── 个股历史日线 ─────────────────────────────────────────────
    def get_hist(self, code, days=165):
        """akshare 不复权日线，返回升序 DataFrame（约 days 行）"""
        end = datetime.date.today().strftime('%Y%m%d')
        start_dt = datetime.date.today() - datetime.timedelta(days=int(days * 1.8))
        start = start_dt.strftime('%Y%m%d')
        df = safe_call(
            ak.stock_zh_a_hist,
            default=None,
            sleep_sec=0.5,
            symbol=str(code).zfill(6),
            period='daily',
            start_date=start,
            end_date=end,
            adjust=''
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values('日期').tail(days).reset_index(drop=True)
        return df

    # ── 上证指数历史 ─────────────────────────────────────────────
    def get_sh_index(self, days=165):
        """上证指数日线（带缓存）"""
        if self._sh_cache is not None:
            return self._sh_cache
        df = safe_call(ak.stock_zh_index_daily, default=None, sleep_sec=0.5, symbol='sh000001')
        if df is None or df.empty:
            return pd.DataFrame()
        # 列: date, open, close, high, low, volume
        df = df.sort_values('date').tail(days).reset_index(drop=True)
        self._sh_cache = df
        return df

    # ── 个股资金流向 ─────────────────────────────────────────────
    def get_fund_flow(self, code):
        """东财个股资金流向（近 15 日）"""
        market = get_market(code)
        df = safe_call(
            ak.stock_individual_fund_flow,
            default=None,
            sleep_sec=0.8,
            stock=str(code).zfill(6),
            market=market
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.tail(15).reset_index(drop=True)

    # ── 筹码分布 ─────────────────────────────────────────────────
    def get_chip(self, code):
        """tushare cyq_perf，近 30 日"""
        ts_code = to_ts_code(code)
        end = datetime.date.today().strftime('%Y%m%d')
        start = (datetime.date.today() - datetime.timedelta(days=45)).strftime('%Y%m%d')
        try:
            df = self.analyzer.pro.cyq_perf(ts_code=ts_code, start_date=start, end_date=end)
            time.sleep(0.5)
            return df if (df is not None and not df.empty) else pd.DataFrame()
        except Exception as e:
            print(f"  [筹码] {code}: {e}")
            return pd.DataFrame()

    # ── 热榜 ─────────────────────────────────────────────────────
    def get_hot_rank(self):
        """东财热榜（带缓存）"""
        if self._hot_cache is not None:
            return self._hot_cache
        df = safe_call(ak.stock_hot_rank_em, default=None, sleep_sec=0.5)
        self._hot_cache = df
        return df

    # ── 板块信息 ─────────────────────────────────────────────────
    def get_sector_name(self, code):
        """获取个股所属行业板块名称"""
        df = safe_call(
            ak.stock_individual_info_em,
            default=None,
            sleep_sec=0.5,
            symbol=str(code).zfill(6)
        )
        if df is None or df.empty:
            return None
        row = df[df['item'] == '行业']
        if row.empty:
            return None
        return str(row.iloc[0]['value'])

    def get_sector_hist(self, sector_name, days=20):
        """行业板块历史行情"""
        end = datetime.date.today().strftime('%Y%m%d')
        start = (datetime.date.today() - datetime.timedelta(days=int(days * 2))).strftime('%Y%m%d')
        df = safe_call(
            ak.stock_board_industry_hist_em,
            default=None,
            sleep_sec=0.5,
            symbol=sector_name,
            start_date=start,
            end_date=end,
            period='日k',
            adjust='不复权'
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.tail(days).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# 评分模块
# ─────────────────────────────────────────────────────────────────

class StockScorer:
    """综合评分器，满分 100 分"""

    WEIGHTS = {
        'historical_position': 20,
        'recent_change':       20,
        'relative_strength':   20,
        'fund_flow':           20,
        'chip':                10,
        'technical':            5,
        'hot_rank':             5,
    }

    def __init__(self):
        self.fetcher = DataFetcher()

    # ── 1. 历史位置评分 ───────────────────────────────────────────
    def score_historical_position(self, hist_df, current_price):
        """
        满分 20 分
        - 过去 150 个交易日最高价距今天数（越近越好）
        - 当前价在 150 日高点的位置（越高越好）
        """
        w = self.WEIGHTS['historical_position']
        if hist_df.empty or current_price is None:
            return {'score': w * 0.5, 'detail': '数据不足取中性'}

        hist_150 = hist_df.tail(150)
        idx_max = hist_150['最高'].idxmax()
        max_price = float(hist_150.loc[idx_max, '最高'])
        max_date_str = str(hist_150.loc[idx_max, '日期'])

        try:
            max_date = pd.to_datetime(max_date_str).date()
        except Exception:
            max_date = datetime.date.today()

        days_ago = (datetime.date.today() - max_date).days
        price_ratio = current_price / max_price if max_price > 0 else 1.0

        # 时间得分：最高点越近越高
        if days_ago <= 5:
            ts = 1.00
        elif days_ago <= 15:
            ts = 0.85
        elif days_ago <= 30:
            ts = 0.68
        elif days_ago <= 60:
            ts = 0.45
        elif days_ago <= 100:
            ts = 0.28
        else:
            ts = 0.12

        # 价格位置得分
        if price_ratio >= 0.99:
            ps = 1.00
        elif price_ratio >= 0.95:
            ps = 0.80
        elif price_ratio >= 0.90:
            ps = 0.60
        elif price_ratio >= 0.85:
            ps = 0.40
        elif price_ratio >= 0.80:
            ps = 0.25
        else:
            ps = 0.10

        score = round(w * (ts * 0.65 + ps * 0.35), 2)
        detail = (f"150日最高={max_price:.2f}({max_date_str},{days_ago}天前), "
                  f"当前={current_price:.2f}({price_ratio*100:.1f}%位置)")
        return {
            'score': score,
            'detail': detail,
            'max_price': max_price,
            'max_date': max_date_str,
            'days_ago': days_ago,
            'price_ratio': round(price_ratio * 100, 1)
        }

    # ── 2. 近期累计涨幅评分 ────────────────────────────────────────
    def score_recent_change(self, hist_df):
        """
        满分 20 分
        3/7/15 日累计涨幅综合打分；适中上涨最优，过热或下跌扣分
        """
        w = self.WEIGHTS['recent_change']
        if hist_df.empty:
            return {'score': w * 0.5, 'detail': '数据不足'}

        closes = hist_df['收盘'].values
        chg3  = pct_change_n(closes, 3)
        chg7  = pct_change_n(closes, 7)
        chg15 = pct_change_n(closes, 15)

        def _seg(v, segs):
            """分段线性插值得分，segs: [(x0,s0),(x1,s1),...]，超出末端取末端分值"""
            for i in range(len(segs) - 1):
                x0, s0 = segs[i]
                x1, s1 = segs[i + 1]
                if x0 <= v <= x1:
                    t = (v - x0) / (x1 - x0) if x1 != x0 else 0
                    return s0 + t * (s1 - s0)
            return segs[-1][1] if v > segs[-1][0] else segs[0][1]

        # 3 日: -5~0 保守, 0~5% 最佳, 5~12% 稍热, >15% 过热
        s3_segs = [(-20, 0.0), (-5, 0.2), (0, 0.5), (4, 1.0), (8, 0.9), (12, 0.65), (20, 0.3)]
        # 7 日
        s7_segs = [(-20, 0.0), (-8, 0.2), (0, 0.45), (7, 1.0), (12, 0.85), (20, 0.55), (35, 0.2)]
        # 15 日
        s15_segs = [(-30, 0.0), (-10, 0.15), (0, 0.4), (10, 0.9), (18, 1.0), (30, 0.75), (50, 0.3)]

        combined = _seg(chg3, s3_segs) * 0.4 + _seg(chg7, s7_segs) * 0.35 + _seg(chg15, s15_segs) * 0.25
        score = round(w * combined, 2)
        detail = f"3日={chg3:+.2f}%, 7日={chg7:+.2f}%, 15日={chg15:+.2f}%"
        return {'score': score, 'detail': detail, 'chg3': chg3, 'chg7': chg7, 'chg15': chg15}

    # ── 3. 相对强度 / 乖离率评分 ──────────────────────────────────
    def score_relative_strength(self, hist_df, sh_df, sector_name):
        """
        满分 20 分
        - 近 10 日最大涨幅/跌幅当日个股 vs 上证乖离
        - 近 15 日个股累计 vs 板块累计乖离
        """
        w = self.WEIGHTS['relative_strength']
        if hist_df.empty:
            return {'score': w * 0.5, 'detail': '数据不足'}

        hist_10 = hist_df.tail(10).reset_index(drop=True)
        closes_10 = hist_10['收盘'].values
        if len(closes_10) < 2:
            return {'score': w * 0.5, 'detail': '数据过少'}

        # 每日涨幅
        if '涨跌幅' in hist_10.columns:
            stock_pct = hist_10['涨跌幅'].values.astype(float)
        else:
            stock_pct = np.diff(closes_10) / closes_10[:-1] * 100
            stock_pct = np.concatenate([[0.0], stock_pct])

        max_gain_idx = int(np.argmax(stock_pct))
        max_loss_idx = int(np.argmin(stock_pct))
        max_gain = float(stock_pct[max_gain_idx])
        max_loss = float(stock_pct[max_loss_idx])
        max_gain_date = str(hist_10.iloc[max_gain_idx]['日期'])
        max_loss_date = str(hist_10.iloc[max_loss_idx]['日期'])

        # 构建上证涨幅日期映射
        sh_pct_map = {}
        if not sh_df.empty and 'close' in sh_df.columns:
            sh_df = sh_df.copy()
            sh_df['pct'] = sh_df['close'].pct_change() * 100
            for _, row in sh_df.iterrows():
                key = str(row['date'])[:10].replace('-', '')
                sh_pct_map[key] = float(row['pct'])

        def _sh_pct(date_str):
            key = str(date_str)[:10].replace('-', '')
            return sh_pct_map.get(key, 0.0)

        gain_dev = max_gain - _sh_pct(max_gain_date)   # 个股超额（越高越好）
        loss_dev = max_loss - _sh_pct(max_loss_date)   # 跌幅当日个股 vs 上证（loss_dev>0 说明个股抗跌）

        # 近 15 日累计乖离
        hist_15 = hist_df.tail(16)
        c15 = hist_15['收盘'].values
        stock_15 = (c15[-1] - c15[0]) / c15[0] * 100 if c15[0] > 0 and len(c15) >= 2 else 0.0

        sector_deviation = 0.0
        if sector_name:
            sec_df = self.fetcher.get_sector_hist(sector_name, days=20)
            if not sec_df.empty:
                sc = '收盘' if '收盘' in sec_df.columns else sec_df.columns[-1]
                if sc in sec_df.columns and len(sec_df) >= 16:
                    sv = sec_df[sc].values
                    sector_15 = (sv[-1] - sv[-16]) / sv[-16] * 100 if sv[-16] > 0 else 0.0
                    sector_deviation = stock_15 - sector_15

        # 超额得分
        gain_dev_score = min(max(gain_dev / 6.0, 0.0), 1.0)
        # 抗跌得分（loss_dev > 0 说明个股比指数抗跌）
        loss_dev_score = min(max((loss_dev + 4.0) / 8.0, 0.0), 1.0)
        # 板块乖离得分
        sector_score = min(max((sector_deviation + 8.0) / 20.0, 0.0), 1.0)

        combined = gain_dev_score * 0.35 + loss_dev_score * 0.35 + sector_score * 0.30
        score = round(w * combined, 2)
        detail = (f"10日最大涨={max_gain:+.2f}%({max_gain_date},超额{gain_dev:+.2f}%), "
                  f"最大跌={max_loss:+.2f}%({max_loss_date},超额{loss_dev:+.2f}%), "
                  f"15日板块乖离={sector_deviation:+.2f}%")
        return {
            'score': score, 'detail': detail,
            'max_gain': max_gain, 'max_gain_date': max_gain_date,
            'max_loss': max_loss, 'max_loss_date': max_loss_date,
            'gain_deviation': round(gain_dev, 2),
            'loss_deviation': round(loss_dev, 2),
            'sector_deviation': round(sector_deviation, 2)
        }

    # ── 4. 大单资金流向评分 ────────────────────────────────────────
    def score_fund_flow(self, code):
        """
        满分 20 分
        近 15 日、近 5 日、昨日主力净流入
        """
        w = self.WEIGHTS['fund_flow']
        df = self.fetcher.get_fund_flow(code)
        if df is None or df.empty:
            return {'score': w * 0.5, 'detail': '资金流向数据不足'}

        # 自动识别主力净流入列
        candidate_cols = ['主力净流入', '主力净额', '超大单净流入', '大单净流入', '主力净流入-净额']
        inflow_col = next((c for c in candidate_cols if c in df.columns), None)
        if inflow_col is None:
            return {'score': w * 0.5, 'detail': f'未找到净流入列，可用: {df.columns.tolist()}'}

        inflow = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0)
        flow_15 = float(inflow.sum())
        flow_5  = float(inflow.tail(5).sum())
        flow_1  = float(inflow.iloc[-1]) if len(inflow) > 0 else 0.0

        def _flow_score(val, scale):
            """线性分段，scale 为参考基准（正负均有）"""
            ratio = val / scale if scale != 0 else 0
            if ratio >= 1.0:   return 1.00
            if ratio >= 0.5:   return 0.82
            if ratio >= 0.1:   return 0.65
            if ratio >= 0.0:   return 0.50
            if ratio >= -0.1:  return 0.35
            if ratio >= -0.5:  return 0.18
            return 0.05

        s15 = _flow_score(flow_15, 3e7)   # 15 日基准 3000 万
        s5  = _flow_score(flow_5,  1e7)   # 5  日基准 1000 万
        s1  = _flow_score(flow_1,  5e6)   # 昨日基准  500 万

        combined = s15 * 0.30 + s5 * 0.40 + s1 * 0.30
        score = round(w * combined, 2)
        detail = (f"15日主力净流入={flow_15/1e4:.0f}万, "
                  f"5日={flow_5/1e4:.0f}万, 昨日={flow_1/1e4:.0f}万")
        return {'score': score, 'detail': detail,
                'flow_15d': flow_15, 'flow_5d': flow_5, 'flow_1d': flow_1}

    # ── 5. 筹码分布评分 ────────────────────────────────────────────
    def score_chip(self, code):
        """
        满分 10 分
        - 集中度 = (cost_95pct - cost_5pct) / weight_avg，越小越集中
        - 获利盘 winner_rate：60~80% 最健康
        """
        w = self.WEIGHTS['chip']
        df = self.fetcher.get_chip(code)
        if df is None or df.empty:
            return {'score': w * 0.5, 'detail': '筹码数据不足'}

        df = df.sort_values('trade_date')
        latest = df.iloc[-1]
        try:
            cost_5   = float(latest.get('cost_5pct', 0) or 0)
            cost_95  = float(latest.get('cost_95pct', 0) or 0)
            w_avg    = float(latest.get('weight_avg', 0) or 0)
            winner   = float(latest.get('winner_rate', 50) or 50)

            spread_pct = (cost_95 - cost_5) / w_avg * 100 if w_avg > 0 else 100.0

            # 集中度评分（区间越小越好）
            if spread_pct <= 8:    cs = 1.00
            elif spread_pct <= 15: cs = 0.82
            elif spread_pct <= 25: cs = 0.62
            elif spread_pct <= 40: cs = 0.42
            elif spread_pct <= 60: cs = 0.25
            else:                  cs = 0.10

            # 获利盘评分：60~80% 最佳
            if 60 <= winner <= 80:        ws = 1.00
            elif 45 <= winner < 60:       ws = 0.75
            elif 80 < winner <= 90:       ws = 0.70
            elif 30 <= winner < 45:       ws = 0.45
            elif 90 < winner <= 96:       ws = 0.45
            else:                         ws = 0.20

            score = round(w * (cs * 0.60 + ws * 0.40), 2)
            detail = (f"筹码区间={spread_pct:.1f}%(5pct~95pct={cost_5:.2f}~{cost_95:.2f}), "
                      f"加权均价={w_avg:.2f}, 获利盘={winner:.1f}%")
            return {'score': score, 'detail': detail,
                    'spread_pct': round(spread_pct, 1),
                    'winner_rate': winner,
                    'weight_avg': w_avg}
        except Exception as e:
            return {'score': w * 0.5, 'detail': f'筹码计算异常: {e}'}

    # ── 6. 技术指标评分 ────────────────────────────────────────────
    def score_technical(self, hist_df):
        """
        满分 5 分
        - RSI(14): 40~65 健康区间
        - 量比: 1~3 活跃
        """
        w = self.WEIGHTS['technical']
        if hist_df.empty or '收盘' not in hist_df.columns:
            return {'score': w * 0.5, 'detail': '数据不足'}

        # RSI
        try:
            rsi = calc_rsi(hist_df['收盘'], period=14)
        except Exception:
            rsi = 50.0

        if 40 <= rsi <= 65:    rsi_s = 1.00
        elif 30 <= rsi < 40:   rsi_s = 0.72
        elif 65 < rsi <= 75:   rsi_s = 0.72
        elif 20 <= rsi < 30:   rsi_s = 0.40
        elif 75 < rsi <= 85:   rsi_s = 0.40
        else:                  rsi_s = 0.15

        # 量比
        vr = 1.0
        if '成交量' in hist_df.columns:
            try:
                vr = calc_volume_ratio(hist_df['成交量'], days=5)
            except Exception:
                vr = 1.0

        if 1.0 <= vr <= 3.0:   vr_s = 1.00
        elif 0.7 <= vr < 1.0:  vr_s = 0.72
        elif 3.0 < vr <= 5.0:  vr_s = 0.72
        elif 0.5 <= vr < 0.7:  vr_s = 0.45
        elif 5.0 < vr <= 8.0:  vr_s = 0.45
        else:                  vr_s = 0.20

        score = round(w * (rsi_s * 0.60 + vr_s * 0.40), 2)
        detail = f"RSI(14)={rsi:.1f}, 量比={vr:.2f}"
        return {'score': score, 'detail': detail, 'rsi': rsi, 'volume_ratio': vr}

    # ── 7. 热榜评分 ────────────────────────────────────────────────
    def score_hot_rank(self, code):
        """
        满分 5 分
        东财热榜排名越靠前得分越高
        """
        w = self.WEIGHTS['hot_rank']
        df = self.fetcher.get_hot_rank()
        if df is None or df.empty:
            return {'score': w * 0.4, 'detail': '热榜数据不足', 'rank': None}

        code = str(code).zfill(6)
        code_col = '代码' if '代码' in df.columns else None
        if code_col is None:
            return {'score': w * 0.4, 'detail': '热榜列名不匹配', 'rank': None}

        row = df[df[code_col] == code]
        if row.empty:
            return {'score': 0.0, 'detail': '未上热榜', 'rank': None}

        rank_col = '排名' if '排名' in df.columns else None
        rank = int(row.iloc[0][rank_col]) if rank_col else int(row.index[0]) + 1

        if rank <= 10:    hs = 1.00
        elif rank <= 30:  hs = 0.80
        elif rank <= 50:  hs = 0.60
        elif rank <= 100: hs = 0.40
        else:             hs = 0.20

        return {'score': round(w * hs, 2), 'detail': f'热榜第{rank}名', 'rank': rank}

    # ── 单只股票综合评分入口 ───────────────────────────────────────
    def score_stock(self, code):
        """
        对单只股票进行完整评分
        :param code: 6 位股票代码（字符串或整数）
        :return: dict，包含 total、scores、各维度 detail
        """
        code = str(code).zfill(6)

        # 数据获取
        hist_df      = self.fetcher.get_hist(code, days=165)
        current_price = self.fetcher.get_realtime_price(code)
        if current_price is None and not hist_df.empty:
            current_price = float(hist_df['收盘'].iloc[-1])

        sh_df       = self.fetcher.get_sh_index(days=165)
        sector_name = self.fetcher.get_sector_name(code)

        # 各维度评分
        scores = {
            'historical_position': self.score_historical_position(hist_df, current_price),
            'recent_change':       self.score_recent_change(hist_df),
            'relative_strength':   self.score_relative_strength(hist_df, sh_df, sector_name),
            'fund_flow':           self.score_fund_flow(code),
            'chip':                self.score_chip(code),
            'technical':           self.score_technical(hist_df),
            'hot_rank':            self.score_hot_rank(code),
        }

        total = round(sum(v['score'] for v in scores.values()), 2)
        return {
            'code': code,
            'current_price': current_price,
            'sector': sector_name,
            'total': total,
            'scores': scores,
        }

    # ── 批量评分入口 ───────────────────────────────────────────────
    def score_list(self, stock_list, interval=1.0):
        """
        对股票列表批量评分，返回按总分降序的 DataFrame
        :param stock_list:  股票代码列表（纯 6 位数字字符串）
        :param interval:    每只股票评分后的间隔秒数
        """
        # 预热会话级缓存
        print(">> 预加载实时行情 / 上证指数 / 热榜...")
        self.fetcher.prefetch_realtime(stock_list)
        self.fetcher.get_sh_index()
        self.fetcher.get_hot_rank()
        print(f">> 开始对 {len(stock_list)} 只股票评分\n")

        rows = []
        for i, code in enumerate(stock_list):
            print(f"  [{i+1:>2}/{len(stock_list)}] {code}", end='  ', flush=True)
            try:
                r = self.score_stock(code)
                s = r['scores']
                row = {
                    '代码':          r['code'],
                    '当前价':        r.get('current_price', '-'),
                    '板块':          r.get('sector', '-'),
                    '总分':          r['total'],
                    '历史位置':      s['historical_position']['score'],
                    '近期涨幅':      s['recent_change']['score'],
                    '相对强度':      s['relative_strength']['score'],
                    '资金流向':      s['fund_flow']['score'],
                    '筹码分布':      s['chip']['score'],
                    '技术指标':      s['technical']['score'],
                    '热榜':          s['hot_rank']['score'],
                    'RSI':           s['technical'].get('rsi', '-'),
                    '量比':          s['technical'].get('volume_ratio', '-'),
                    '3日涨%':        s['recent_change'].get('chg3', '-'),
                    '7日涨%':        s['recent_change'].get('chg7', '-'),
                    '15日涨%':       s['recent_change'].get('chg15', '-'),
                    '150日高距今天': s['historical_position'].get('days_ago', '-'),
                    '当前/高点%':    s['historical_position'].get('price_ratio', '-'),
                    '获利盘%':       s['chip'].get('winner_rate', '-'),
                    '15日主力流入万':  round(s['fund_flow'].get('flow_15d', 0) / 1e4, 0)
                                     if isinstance(s['fund_flow'].get('flow_15d'), float) else '-',
                }
                rows.append(row)
                print(f"总分={r['total']:>5.1f}  "
                      f"历史={s['historical_position']['score']:>4.1f}  "
                      f"涨幅={s['recent_change']['score']:>4.1f}  "
                      f"强度={s['relative_strength']['score']:>4.1f}  "
                      f"资金={s['fund_flow']['score']:>4.1f}")
            except Exception as e:
                print(f"评分失败: {e}")
                rows.append({'代码': str(code).zfill(6), '总分': -1.0})
            time.sleep(interval)

        df = pd.DataFrame(rows)
        if not df.empty and '总分' in df.columns:
            df = df.sort_values('总分', ascending=False).reset_index(drop=True)
        return df


# ─────────────────────────────────────────────────────────────────
# 实时循环监控
# ─────────────────────────────────────────────────────────────────

def run_monitor(stock_list, refresh_minutes=10):
    """
    实时监控模式：每隔 refresh_minutes 分钟重新评分并打印排名
    Ctrl-C 退出
    """
    scorer = StockScorer()
    round_num = 0
    try:
        while True:
            round_num += 1
            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'='*72}")
            print(f"  第 {round_num} 轮  |  {now_str}")
            print(f"{'='*72}")

            # 清除缓存保证数据新鲜
            scorer.fetcher._sh_cache    = None
            scorer.fetcher._hot_cache   = None
            scorer.fetcher._realtime_map = {}

            result_df = scorer.score_list(stock_list, interval=0.8)

            display_cols = [
                '代码', '当前价', '板块', '总分',
                '历史位置', '近期涨幅', '相对强度', '资金流向', '筹码分布',
                '技术指标', '热榜', 'RSI', '量比',
                '3日涨%', '7日涨%', '15日涨%',
            ]
            show = [c for c in display_cols if c in result_df.columns]

            print(f"\n{'─'*72}")
            print("  综合排名（高分优先）")
            print(f"{'─'*72}")
            print(result_df[show].to_string(index=True))
            print(f"\n  下次更新：{refresh_minutes} 分钟后  (Ctrl-C 退出)")
            time.sleep(refresh_minutes * 60)
    except KeyboardInterrupt:
        print("\n  已退出监控")


# ─────────────────────────────────────────────────────────────────
# 示例入口
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    stock_list = [
        '000537','601789','002298','002445','605268','002167','600410','600821','002261'
    ]

    # ── 单次评分 ──
    scorer = StockScorer()
    df = scorer.score_list(stock_list)
    print("\n" + "=" * 72)
    print("  最终评分结果")
    print("=" * 72)
    print(df.to_string(index=True))

    # ── 实时轮询（取消注释启用，默认 10 分钟刷新一次）──
    run_monitor(stock_list, refresh_minutes=10)
