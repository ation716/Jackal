# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : gaolei
# @FileName: score.py
"""
股票评分系统 - 短线龙头导向（持有 1~5 天）

1.评分维度（满分 100 分）:
  a. breakout             突破强度（60日高点位置与时间）        12 分
  b. recent_change        近期累计涨幅（1/3/5日）              12 分
  c. relative_strength    相对强度乖离（5日 vs 上证/板块）     10 分
  d. fund_flow            大单资金流向（5/3/1日，重近期）      15 分
  e. chip                 筹码分布（akshare，重控盘集中度）    7 分
  f. technical            技术指标（RSI强势区间、放量突破）     10 分
  g. hot_rank             热榜排名                           7 分

调整分值，另外增加以下 3 个，
  h. 封板质量    8 分 参考涨停时间（30%）封单量（20%）开板次数（30%）最低价（20%）
  i. 板块强度    8 分 50%（≥5只涨停：满分；3-4只：80%；1-2只：50%；0只：0分）30% 板块龙头强度，20 %板块涨幅
  j. 连扳梯度    11 分 当前连扳梯队，如果大于 2 板，考虑竞争晋级标的强度；连扳大于等于 5 板考虑更高位的压制；大于等于7板考虑监管压力

2.采用 Sigmoid 函数 将分段评分改成连续评分
  i. 当指标落在最优区间，可以给满分，最优区间定义：如果指标落在此区间内，在 T+1 规则下，有 95% 的把握人为第二天可以盈利
  ii. 可扩展其他评分函数
"""

import sys
import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import traceback
import akshare as ak
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from security_data import ChipDistributionAnalyzer
from crawl import Crawler


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
        # print(traceback.format_exc())
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
    _instance = None
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.analyzer = ChipDistributionAnalyzer()
        self.crawler = Crawler()
        self._sh_cache = None       # 上证指数历史
        self._hot_cache = None      # 热榜 (df, timestamp)
        self._hot_cache_ttl = 300    # 热榜缓存有效期（秒）
        self._realtime_map = {}     # {6位代码: 实时价格}（tushare）
        self._spot_cache  = None    # akshare 全市场实时行情，兜底用
        self._chip_manual_cache = {}  # {code6: df}，手动输入筹码数据缓存
        print("  [DataFetcher] 初始化完成")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

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
        new_map = {}
        if df is None or df.empty:
            print("  [实时行情] tushare 获取失败，将依赖 akshare spot 兜底")
        else:
            code_col  = 'TS_CODE' if 'TS_CODE' in df.columns else df.columns[0]
            price_col = 'PRICE'   if 'PRICE'   in df.columns else df.columns[6]
            for i in range(len(df)):
                try:
                    ts_code = str(df.iloc[i][code_col])
                    raw     = df.iloc[i][price_col]
                    price   = float(raw)
                    if np.isnan(price) or price <= 0:
                        continue
                    new_map[ts_code[:6]] = price
                except Exception as e:
                    print(f"  [实时行情] {df.iloc[i][code_col][:6]} 解析失败: {e}")
        self._realtime_map = new_map
        print(f"  [实时行情] tushare 缓存 {len(new_map)}/{len(stock_list)} 只")


    def get_realtime_price(self, code):
        """从缓存的实时行情里查价格"""
        return self._realtime_map.get(str(code).zfill(6))

    def prefetch_spot(self, stock_list):
        """akshare 全市场实时行情，仅保留 stock_list 的价格作为兜底缓存"""
        df = safe_call(ak.stock_zh_a_spot_em, default=None, sleep_sec=0.5)
        if df is None or df.empty:
            print("  [spot行情] akshare 获取失败")
            return
        code_col  = '代码'   if '代码'   in df.columns else None
        price_col = '最新价' if '最新价' in df.columns else None
        if code_col is None or price_col is None:
            print("  [spot行情] 列名不符预期，跳过")
            return
        need = set(str(c).zfill(6) for c in stock_list)
        self._spot_cache = (
            df[df[code_col].isin(need)][[code_col, price_col]]
            .set_index(code_col)[price_col]
            .to_dict()
        )
        print(f"  [spot行情] akshare 兜底缓存 {len(self._spot_cache)}/{len(stock_list)} 只")

    def get_spot_price(self, code):
        """从 akshare spot 缓存里取价格（兜底）"""
        if self._spot_cache is None:
            return None
        raw = self._spot_cache.get(str(code).zfill(6))
        if raw is None:
            return None
        try:
            price = float(raw)
            return price if not np.isnan(price) and price > 0 else None
        except Exception:
            return None

    # ── 个股历史日线 ─────────────────────────────────────────────
    def get_hist(self, code, days=80):
        """akshare 不复权日线，返回升序 DataFrame（约 days 行）"""
        end = datetime.date.today().strftime('%Y%m%d')
        start_dt = datetime.date.today() - datetime.timedelta(days=int(days * 1.8))
        start = start_dt.strftime('%Y%m%d')
        df = None

        # 先新浪
        df = safe_call(
            self.crawler.get_stock_history_simple,
            symbol='sh' + str(code) if str(code).startswith('6') else 'sz' + str(code),
            datalen=days,
        )
        df = df.rename(columns={'date': '日期', 'high': '最高', 'low': '最低', 'open': '开盘', 'close': '收盘',
                                'amount': '成交额'})
        # 再腾讯
        if df is None or df.empty:
            print("东财和新浪都异常")
            df = safe_call(
                ak.stock_zh_a_hist_tx,
                symbol='sh' + str(code) if str(code).startswith('6') else 'sz' + str(code),
                start_date=start,
                end_date=end,
                adjust=''
            )
            df = df.rename(columns={'date':'日期','high':'最高','low':'最低','open':'开盘','close':'收盘','amount':'成交额'})
        df = df.sort_values('日期').tail(days).reset_index(drop=True)
        if df is None or df.empty:
            return pd.DataFrame()

        return df

    # ── 上证指数历史 ─────────────────────────────────────────────
    def get_sh_index(self, days=80):
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
        """东财个股资金流向（近 10 日）"""
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
        return df.tail(10).reset_index(drop=True)

    # ── 筹码分布（akshare → 直接爬虫 → 手动输入，三级降级）────────
    def get_chip(self, code):
        """
        筹码分布三级降级：
          1. ak.stock_cyq_em（原版，依赖 py_mini_racer）
          2. Crawler.get_chip_em（纯 Python CYQ，无需 py_mini_racer）
          3. 手动输入（两种自动方式均失败时提示用户输入）
        """
        code6 = str(code).zfill(6)

        # ── 一、akshare ──────────────────────────────────────────
        df = safe_call(ak.stock_cyq_em, default=None, sleep_sec=0.5,
                       symbol=code6, adjust="")
        if df is not None and not df.empty:
            return df

        # ── 二、直接爬虫（绕过 py_mini_racer）───────────────────
        print(f"  [筹码] {code6}: akshare 失败，尝试直接爬虫...")
        try:
            df = self.crawler.get_chip_em(code6)
            if df is not None and not df.empty:
                print(f"  [筹码] {code6}: 直接爬虫成功")
                return df
        except Exception as e:
            print(f"  [筹码] {code6}: 直接爬虫异常: {e}")

        # ── 三、手动输入 ─────────────────────────────────────────
        return self._manual_chip_input(code6)

    def _manual_chip_input(self, code):
        """
        两种自动获取均失败时，提示用户手动输入筹码关键指标。
        直接回车跳过，返回空 DataFrame（评分取中性值）。
        已输入的结果会缓存，同一只股票本次运行不会重复询问。
        """
        code6 = str(code).zfill(6)
        if code6 in self._chip_manual_cache:
            return self._chip_manual_cache[code6]

        market_prefix = 'sh' if code6.startswith('6') else 'sz'
        em_url = f"https://quote.eastmoney.com/concept/{market_prefix}{code6}.html"
        print(f"\n  ┌── [{code6}] 筹码数据自动获取失败 ──────────────────────")
        print(f"  │  参考链接（日K→筹码分布）：{em_url}")
        print(  "  │  请手动输入（直接回车跳过，该股筹码取中性分）：")
        try:
            w = input("  │  获利盘比例（0~100，%）如 45.5 ：").strip()
            if not w:
                print("  └── 已跳过")
                self._chip_manual_cache[code6] = pd.DataFrame()
                return pd.DataFrame()

            c90  = input("  │  90集中度（0~1，越小越集中）如 0.12 ：").strip()
            c90l = input("  │  90成本-低（元）                    ：").strip()
            c90h = input("  │  90成本-高（元）                    ：").strip()
            avg  = input("  │  平均成本（元，可留空）             ：").strip()
            print("  └────────────────────────────────────────────────────")

            winner_val = float(w)
            # 统一转为 0~1 小数（与 akshare 格式一致）
            if winner_val > 1.0:
                winner_val /= 100.0

            df = pd.DataFrame([{
                '日期':    datetime.date.today(),
                '获利比例': winner_val,
                '平均成本': float(avg) if avg else 0.0,
                '90成本-低': float(c90l) if c90l else 0.0,
                '90成本-高': float(c90h) if c90h else 0.0,
                '90集中度':  float(c90) if c90 else 0.15,
                '70成本-低': 0.0, '70成本-高': 0.0, '70集中度': 0.0,
            }])
            self._chip_manual_cache[code6] = df
            return df

        except (EOFError, KeyboardInterrupt):
            print("  └── 已跳过")
            self._chip_manual_cache[code6] = pd.DataFrame()
            return pd.DataFrame()
        except Exception as e:
            print(f"  └── 输入解析失败: {e}，已跳过")
            self._chip_manual_cache[code6] = pd.DataFrame()
            return pd.DataFrame()

    # ── 热榜 ─────────────────────────────────────────────────────
    def get_hot_rank(self):
        """东财热榜（带 TTL 缓存，超过 _hot_cache_ttl 秒自动刷新）"""
        import time
        now = time.time()
        if self._hot_cache is not None:
            df, ts = self._hot_cache
            if now - ts < self._hot_cache_ttl:
                return df
        df = safe_call(self.crawler.get_ths_hot_rank, default=None, sleep_sec=0.5)
        self._hot_cache = (df, now)
        return df

    # ── 板块信息 ─────────────────────────────────────────────────
    def get_sector_name(self, code):
        """获取个股所属行业板块名称"""
        df = safe_call(
            ak.stock_individual_basic_info_xq,
            default=None,
            sleep_sec=0.5,
            symbol='SH'+str(code) if str(code).startswith('6') else 'SZ'+str(code),
        )
        row = df[df['item'] == 'affiliate_industry']
        if row.empty:
            return None
        return str(row.iloc[0]['value'].get('ind_name'))


    def get_sector_hist(self, sector_name, days=10):
        """行业板块历史行情"""
        end = datetime.date.today().strftime('%Y%m%d')
        start = (datetime.date.today() - datetime.timedelta(days=int(days * 2))).strftime('%Y%m%d')
        df = safe_call(
            ak.stock_board_industry_index_ths,
            symbol=sector_name,
            start_date=start,
            end_date=end
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.tail(days).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# CSV 输出
# ─────────────────────────────────────────────────────────────────

def save_scores_csv(df: pd.DataFrame, output_dir: str = None) -> str:
    """将评分结果保存到 results/scores/score_{date}.csv，返回文件路径"""
    if df is None or df.empty:
        return ''
    if output_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, 'results', 'scores')
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.date.today().strftime('%Y-%m-%d')
    fpath = os.path.join(output_dir, f'score_{date_str}.csv')
    fexist=os.path.exists(fpath)
    df.to_csv(fpath, mode='a', header=not fexist, index=False, encoding='utf-8-sig')
    print(f"  [CSV] 结果已写入 → {fpath}")
    return fpath


# ─────────────────────────────────────────────────────────────────
# 评分模块
# ─────────────────────────────────────────────────────────────────

class StockScorer:
    """
    短线龙头股综合评分器，满分 100 分
    目标持仓周期：1~5 个交易日
    """

    WEIGHTS = {
        'breakout':          20,  # 突破强度（60日高点位置/时间）
        'recent_change':     20,  # 近期累计涨幅（1/3/5日）
        'relative_strength': 20,  # 相对强度乖离（5日 vs 上证/板块）
        'fund_flow':         20,  # 大单资金流向（5/3/1日，重近期）
        'chip':              10,  # 筹码分布（akshare，重控盘）
        'technical':          5,  # 技术指标（RSI强势区间、量比）
        'hot_rank':           5,  # 热榜排名
    }

    def __init__(self):
        self.fetcher = DataFetcher()

    # ── 1. 突破强度评分 ───────────────────────────────────────────
    def score_breakout(self, hist_df, current_price):
        """
        满分 20 分（面向短线突破形态）
        - 近 60 个交易日最高价距今天数（越近越好，越近说明强势刚突破）
        - 当前价在 60 日高点的位置（越高越好）
        """
        w = self.WEIGHTS['breakout']
        if hist_df.empty or current_price is None:
            return {'score': w * 0.5, 'detail': '数据不足取中性'}

        hist_60 = hist_df.tail(60)
        idx_max = hist_60['最高'].idxmax()
        max_price = float(hist_60.loc[idx_max, '最高'])
        max_date_str = str(hist_60.loc[idx_max, '日期'])

        try:
            max_date = pd.to_datetime(max_date_str).date()
        except Exception:
            max_date = datetime.date.today()

        days_ago = (datetime.date.today() - max_date).days
        price_ratio = current_price / max_price if max_price > 0 else 1.0

        # 时间得分：高点距今越近越优（突破/攻击形态）
        if days_ago <= 3:
            ts = 1.00
        elif days_ago <= 7:
            ts = 0.90
        elif days_ago <= 15:
            ts = 0.75
        elif days_ago <= 30:
            ts = 0.52
        elif days_ago <= 45:
            ts = 0.30
        else:
            ts = 0.12

        # 价格位置得分：越接近或超过高点越好
        if price_ratio >= 0.99:
            ps = 1.00
        elif price_ratio >= 0.97:
            ps = 0.88
        elif price_ratio >= 0.94:
            ps = 0.72
        elif price_ratio >= 0.90:
            ps = 0.52
        elif price_ratio >= 0.85:
            ps = 0.30
        else:
            ps = 0.12

        score = round(w * (ts * 0.60 + ps * 0.40), 2)
        detail = (f"60日最高={max_price:.2f}({max_date_str},{days_ago}天前), "
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
        满分 20 分（短线版：1/3/5 日涨幅）
        - 1日：3~9% 最优（今日强势启动）
        - 3日：5~20% 最优（有节奏地上涨）
        - 5日：8~30% 最优（已确立强势但未超涨）
        - 过热惩罚：5日>45% 明显扣分
        """
        w = self.WEIGHTS['recent_change']
        if hist_df.empty:
            return {'score': w * 0.5, 'detail': '数据不足'}

        closes = hist_df['收盘'].values
        chg1  = pct_change_n(closes, 1)
        chg3  = pct_change_n(closes, 3)
        chg5  = pct_change_n(closes, 5)

        def _seg(v, segs):
            """分段线性插值得分"""
            for i in range(len(segs) - 1):
                x0, s0 = segs[i]
                x1, s1 = segs[i + 1]
                if x0 <= v <= x1:
                    t = (v - x0) / (x1 - x0) if x1 != x0 else 0
                    return s0 + t * (s1 - s0)
            return segs[-1][1] if v > segs[-1][0] else segs[0][1]

        # 1日：-5~0 保守，3~9% 最佳，>12% 涨停/超热稍扣（已超买）
        s1_segs = [(-10, 0.0), (-3, 0.2), (0, 0.45), (3, 0.85), (7, 1.0), (10, 0.92), (13, 0.72), (20, 0.40)]
        # 3日：5~20% 最佳
        s3_segs = [(-15, 0.0), (-5, 0.2), (0, 0.40), (5, 0.80), (12, 1.0), (20, 0.90), (30, 0.60), (45, 0.25)]
        # 5日：8~30% 最佳，>45% 过热扣分
        s5_segs = [(-20, 0.0), (-8, 0.15), (0, 0.35), (8, 0.75), (18, 1.0), (30, 0.85), (45, 0.50), (60, 0.20)]

        combined = _seg(chg1, s1_segs) * 0.40 + _seg(chg3, s3_segs) * 0.35 + _seg(chg5, s5_segs) * 0.25
        score = round(w * combined, 2)
        detail = f"1日={chg1:+.2f}%, 3日={chg3:+.2f}%, 5日={chg5:+.2f}%"
        return {'score': score, 'detail': detail, 'chg1': chg1, 'chg3': chg3, 'chg5': chg5}

    # ── 3. 相对强度 / 乖离率评分 ──────────────────────────────────
    def score_relative_strength(self, hist_df, sh_df, sector_name):
        """
        满分 20 分（短线版：缩短窗口至 5 日）
        - 近 5 日最大涨幅日个股 vs 上证超额
        - 近 5 日累计个股 vs 板块乖离
        """
        w = self.WEIGHTS['relative_strength']
        if hist_df.empty:
            return {'score': w * 0.5, 'detail': '数据不足'}

        hist_5 = hist_df.tail(5).reset_index(drop=True)
        closes_5 = hist_5['收盘'].values
        if len(closes_5) < 2:
            return {'score': w * 0.5, 'detail': '数据过少'}

        # 每日涨幅
        if '涨跌幅' in hist_5.columns:
            stock_pct = hist_5['涨跌幅'].values.astype(float)
        else:
            stock_pct = np.diff(closes_5) / closes_5[:-1] * 100
            stock_pct = np.concatenate([[0.0], stock_pct])

        max_gain_idx = int(np.argmax(stock_pct))
        max_loss_idx = int(np.argmin(stock_pct))
        max_gain = float(stock_pct[max_gain_idx])
        max_loss = float(stock_pct[max_loss_idx])
        max_gain_date = str(hist_5.iloc[max_gain_idx]['日期'])
        max_loss_date = str(hist_5.iloc[max_loss_idx]['日期'])

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

        gain_dev = max_gain - _sh_pct(max_gain_date)   # 最大涨幅日超额
        loss_dev = max_loss - _sh_pct(max_loss_date)   # 最大跌幅日抗跌情况

        # 近 5 日累计乖离
        hist_6 = hist_df.tail(6)
        c6 = hist_6['收盘'].values
        stock_5d = (c6[-1] - c6[0]) / c6[0] * 100 if c6[0] > 0 and len(c6) >= 2 else 0.0

        sector_deviation = 0.0
        if sector_name:
            sec_df = self.fetcher.get_sector_hist(sector_name, days=10)
            if not sec_df.empty:
                sc = '收盘' if '收盘' in sec_df.columns else sec_df.columns[-1]
                if sc in sec_df.columns and len(sec_df) >= 6:
                    sv = sec_df[sc].values
                    sector_5d = (sv[-1] - sv[-6]) / sv[-6] * 100 if sv[-6] > 0 else 0.0
                    sector_deviation = stock_5d - sector_5d

        # 超额得分（短线更激进：超额 5% 就满分）
        gain_dev_score = min(max(gain_dev / 5.0, 0.0), 1.0)
        # 抗跌得分
        loss_dev_score = min(max((loss_dev + 3.0) / 6.0, 0.0), 1.0)
        # 板块乖离得分
        sector_score = min(max((sector_deviation + 5.0) / 15.0, 0.0), 1.0)

        combined = gain_dev_score * 0.40 + loss_dev_score * 0.30 + sector_score * 0.30
        score = round(w * combined, 2)
        detail = (f"5日最大涨={max_gain:+.2f}%({max_gain_date},超额{gain_dev:+.2f}%), "
                  f"最大跌={max_loss:+.2f}%({max_loss_date},超额{loss_dev:+.2f}%), "
                  f"5日板块乖离={sector_deviation:+.2f}%")
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
        满分 20 分（短线版：重近期，窗口 5/3/1 日）
        - 权重：1日 40% + 3日 35% + 5日 25%
        """
        w = self.WEIGHTS['fund_flow']
        df = self.fetcher.get_fund_flow(code)
        if df is None or df.empty:
            return {'score': w * 0.5, 'detail': '资金流向数据不足'}

        candidate_cols = ['主力净流入', '主力净额', '超大单净流入', '大单净流入', '主力净流入-净额']
        inflow_col = next((c for c in candidate_cols if c in df.columns), None)
        if inflow_col is None:
            return {'score': w * 0.5, 'detail': f'未找到净流入列，可用: {df.columns.tolist()}'}

        inflow = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0)
        flow_5  = float(inflow.tail(5).sum())
        flow_3  = float(inflow.tail(3).sum())
        flow_1  = float(inflow.iloc[-1]) if len(inflow) > 0 else 0.0

        def _flow_score(val, scale):
            """线性分段，scale 为参考基准"""
            ratio = val / scale if scale != 0 else 0
            if ratio >= 1.5:   return 1.00
            if ratio >= 1.0:   return 0.92
            if ratio >= 0.5:   return 0.80
            if ratio >= 0.1:   return 0.62
            if ratio >= 0.0:   return 0.48
            if ratio >= -0.2:  return 0.30
            if ratio >= -0.5:  return 0.15
            return 0.05

        s5  = _flow_score(flow_5,  1.5e7)   # 5日基准 1500 万
        s3  = _flow_score(flow_3,  8e6)     # 3日基准  800 万
        s1  = _flow_score(flow_1,  5e6)     # 1日基准  500 万

        # 短线重视近期：1日 40% + 3日 35% + 5日 25%
        combined = s1 * 0.40 + s3 * 0.35 + s5 * 0.25
        score = round(w * combined, 2)
        detail = (f"5日主力净={flow_5/1e4:.0f}万, "
                  f"3日={flow_3/1e4:.0f}万, 1日={flow_1/1e4:.0f}万")
        return {'score': score, 'detail': detail,
                'flow_5d': flow_5, 'flow_3d': flow_3, 'flow_1d': flow_1}

    # ── 5. 筹码分布评分 ────────────────────────────────────────────
    def score_chip(self, code):
        """
        满分 10 分（akshare stock_cyq_em，短线控盘导向）
        - 90集中度：越小越集中，说明主力控盘力度强
        - 获利比例：30~70% 最佳（有充足上涨空间但已有启动）
        """
        w = self.WEIGHTS['chip']
        df = self.fetcher.get_chip(code)
        if df is None or df.empty:
            return {'score': w * 0.5, 'detail': '筹码数据不足'}

        # 验证必须列
        required = ['获利比例', '平均成本', '90集中度', '90成本-低', '90成本-高']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {'score': w * 0.5, 'detail': f'筹码列缺失: {missing}, 实际列: {df.columns.tolist()}'}

        # 取最近一行（最新交易日）
        latest = df.sort_values('日期').iloc[-1] if '日期' in df.columns else df.iloc[-1]

        try:
            winner      = float(latest['获利比例'])          # 0~1 小数
            avg_cost    = float(latest['平均成本'])
            conc_90     = float(latest['90集中度'])          # 相对宽度，越小越集中
            cost_90_low  = float(latest['90成本-低'])
            cost_90_high = float(latest['90成本-高'])

            # 获利比例转为百分比（akshare 返回 0~1 小数）
            winner_pct = winner * 100 if winner <= 1.0 else winner

            # 集中度评分（90集中度越小越好，0.06 以下属高度集中）
            if conc_90 <= 0.06:    cs = 1.00
            elif conc_90 <= 0.10:  cs = 0.85
            elif conc_90 <= 0.16:  cs = 0.68
            elif conc_90 <= 0.25:  cs = 0.48
            elif conc_90 <= 0.40:  cs = 0.28
            else:                  cs = 0.10

            # 获利盘评分（短线：30~70% 最佳，有上涨空间；>85% 套牢盘重）
            if 30 <= winner_pct <= 70:         ws = 1.00
            elif 20 <= winner_pct < 30:        ws = 0.78
            elif 70 < winner_pct <= 80:        ws = 0.75
            elif 10 <= winner_pct < 20:        ws = 0.50
            elif 80 < winner_pct <= 90:        ws = 0.45
            else:                              ws = 0.20

            score = round(w * (cs * 0.55 + ws * 0.45), 2)
            detail = (f"90集中度={conc_90:.3f}({cost_90_low:.2f}~{cost_90_high:.2f}), "
                      f"均成本={avg_cost:.2f}, 获利盘={winner_pct:.1f}%")
            return {'score': score, 'detail': detail,
                    'conc_90': round(conc_90, 4),
                    'winner_rate': round(winner_pct, 1),
                    'avg_cost': round(avg_cost, 2),
                    'cost_90_low': round(cost_90_low, 2),
                    'cost_90_high': round(cost_90_high, 2)}
        except Exception as e:
            return {'score': w * 0.5, 'detail': f'筹码计算异常: {e}'}

    # ── 6. 技术指标评分 ────────────────────────────────────────────
    def score_technical(self, hist_df):
        """
        满分 5 分（短线版）
        - RSI(14): 50~75 强势健康区间（短线不需要超低 RSI）
        - 量比: >= 2.0 有效放量突破加分
        """
        w = self.WEIGHTS['technical']
        if hist_df.empty or '收盘' not in hist_df.columns:
            return {'score': w * 0.5, 'detail': '数据不足'}

        # RSI
        try:
            rsi = calc_rsi(hist_df['收盘'], period=14)
        except Exception:
            rsi = 50.0

        # 短线：RSI 50~75 强势区间最优，过低（<40）不是龙头，超买（>80）需警惕
        if 50 <= rsi <= 75:    rsi_s = 1.00
        elif 40 <= rsi < 50:   rsi_s = 0.72
        elif 75 < rsi <= 82:   rsi_s = 0.68
        elif 30 <= rsi < 40:   rsi_s = 0.38
        elif 82 < rsi <= 90:   rsi_s = 0.38
        else:                  rsi_s = 0.15

        # 量比
        vr = 1.0
        if '成交量' in hist_df.columns:
            try:
                vr = calc_volume_ratio(hist_df['成交量'], days=5)
            except Exception:
                vr = 1.0

        # 短线：量比 >= 2.0 放量突破最优；过低（缩量）不理想
        if 2.0 <= vr <= 5.0:   vr_s = 1.00
        elif 1.5 <= vr < 2.0:  vr_s = 0.85
        elif 1.0 <= vr < 1.5:  vr_s = 0.65
        elif 5.0 < vr <= 8.0:  vr_s = 0.60
        elif 0.7 <= vr < 1.0:  vr_s = 0.40
        elif 8.0 < vr:         vr_s = 0.35
        else:                  vr_s = 0.20

        score = round(w * (rsi_s * 0.55 + vr_s * 0.45), 2)
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
        hist_df       = self.fetcher.get_hist(code, days=80)
        current_price = self.fetcher.get_realtime_price(code)
        if current_price is not None and np.isnan(current_price):
            current_price = None
        if current_price is None and not hist_df.empty:
            current_price = float(hist_df['收盘'].iloc[-1])
        if current_price is None:
            current_price = self.fetcher.get_spot_price(code)

        sh_df       = self.fetcher.get_sh_index(days=80)
        sector_name = self.fetcher.get_sector_name(code)

        # 各维度评分
        scores = {
            'breakout':          self.score_breakout(hist_df, current_price),
            'recent_change':     self.score_recent_change(hist_df),
            'relative_strength': self.score_relative_strength(hist_df, sh_df, sector_name),
            'fund_flow':         self.score_fund_flow(code),
            'chip':              self.score_chip(code),
            'technical':         self.score_technical(hist_df),
            'hot_rank':          self.score_hot_rank(code),
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
    def score_list(self, stock_list, interval=1.0, save=True):
        """
        对股票列表批量评分，返回按总分降序的 DataFrame
        :param stock_list:  股票代码列表（纯 6 位数字字符串）
        :param interval:    每只股票评分后的间隔秒数
        :param save:        是否自动写入 results/scores/ CSV
        """
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
                    '突破强度':      s['breakout']['score'],
                    '近期涨幅':      s['recent_change']['score'],
                    '相对强度':      s['relative_strength']['score'],
                    '资金流向':      s['fund_flow']['score'],
                    '筹码分布':      s['chip']['score'],
                    '技术指标':      s['technical']['score'],
                    '热榜':          s['hot_rank']['score'],
                    'RSI':           s['technical'].get('rsi', '-'),
                    '量比':          s['technical'].get('volume_ratio', '-'),
                    '1日涨%':        s['recent_change'].get('chg1', '-'),
                    '3日涨%':        s['recent_change'].get('chg3', '-'),
                    '5日涨%':        s['recent_change'].get('chg5', '-'),
                    '60日高距今天':  s['breakout'].get('days_ago', '-'),
                    '当前/高点%':    s['breakout'].get('price_ratio', '-'),
                    '获利盘%':       s['chip'].get('winner_rate', '-'),
                    '90集中度':      s['chip'].get('conc_90', '-'),
                    '5日主力流入万': round(s['fund_flow'].get('flow_5d', 0) / 1e4, 0)
                                     if isinstance(s['fund_flow'].get('flow_5d'), float) else '-',
                }
                rows.append(row)
                print(f"总分={r['total']:>5.1f}  "
                      f"突破={s['breakout']['score']:>4.1f}  "
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
        if save:
            save_scores_csv(df)
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

            scorer.fetcher._sh_cache    = None
            scorer.fetcher._realtime_map = {}

            result_df = scorer.score_list(stock_list, interval=0.8)

            display_cols = [
                '代码', '当前价', '板块', '总分',
                '突破强度', '近期涨幅', '相对强度', '资金流向', '筹码分布',
                '技术指标', '热榜', 'RSI', '量比',
                '1日涨%', '3日涨%', '5日涨%',
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
        # '000537','601789','002298',
        # '002445',
        # '605268', '002167',
        # '600410', '600821', '002261'
        # '605271','000020','600396'
        # xydz,shfa,hdln
        # '600410'
        # hstc
        # '603803'
        # rskd
        # '603687'
        # dsd
        # '002407'
        # dfd
        # '600645','603716','000815'
        # zyxh,slyl,mly
        # '603538','600666','600186','002685','603687','002218',
        # '002309',
        '000815'
    ]

    # ── 单次评分 ──
    scorer = StockScorer()
    df = scorer.score_list(stock_list)
    print("\n" + "=" * 72)
    print("  最终评分结果")
    print("=" * 72)
    print(df.to_string(index=True))

    # ── 实时轮询（取消注释启用，默认 10 分钟刷新一次）──
    # run_monitor(stock_list, refresh_minutes=10)
