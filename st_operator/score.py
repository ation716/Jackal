# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : gaolei
# @FileName: score.py
"""
Stock Scoring System - Short-term Leader Oriented (Hold 1~5 days)

1. Scoring Dimensions (Max 100 points):
  a. breakout             Breakout strength (60-day high position and time)        12 pts
  b. recent_change        Recent cumulative return (1/3/5 days)                    12 pts
  c. relative_strength    Relative strength deviation (5-day vs SH/Industry)      10 pts
  d. fund_flow            Large order money flow (5/3/1 days, recent weighted)    15 pts
  e. chip                 Chip distribution (akshare, focus on concentration)      7 pts
  f. technical            Technical indicators (RSI strong range, volume breakout)10 pts
  g. hot_rank             Hot list ranking                                         7 pts

Adjusted weights, added 3 more:
  h. seal_quality     8 pts  Factors: limit time (30%), seal amount (20%),
                              breakout count (30%), lowest price (20%)
  i. sector_strength  8 pts  50% (≥5 limit-up: full; 3-4: 80%; 1-2: 50%; 0: 0%)
                              30% sector leader strength, 20% sector return
  j. consec_limit     11 pts Current consecutive limit tier, if >2 boards consider
                              competition; >=5 boards consider higher suppression;
                              >=7 boards consider regulatory pressure

3. Using Sigmoid function to convert segmented scoring to continuous scoring
  i. Full score when indicator falls in optimal range, defined as: 95% confidence
     of profitable next day under T+1 rule
  ii. Extensible to other scoring functions
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
# Utility Functions
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
        print(f"  [API Exception] {func.__name__}: {e}")
        # print(traceback.format_exc())
        time.sleep(sleep_sec)
        return default


def calc_rsi(close_series, period=14):
    """Calculate RSI, return latest value"""
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
    """Volume ratio = current volume / avg volume of previous N days"""
    if len(vol_series) < days + 1:
        return 1.0
    avg = vol_series.iloc[-(days + 1):-1].mean()
    if avg == 0:
        return 1.0
    return round(vol_series.iloc[-1] / avg, 2)


def pct_change_n(close_arr, n):
    """Percentage change of last close vs close N days ago (%)"""
    if len(close_arr) <= n:
        return 0.0
    base = close_arr[-(n + 1)]
    if base == 0:
        return 0.0
    return round((close_arr[-1] - base) / base * 100, 2)


def range_score(x: float, opt_lo: float, opt_hi: float,
                left_k: float = 0.5, right_k: float = 0.5) -> float:
    """
    Continuous scoring (Sigmoid improved):
    - Indicator within optimal range [opt_lo, opt_hi] -> full score 1.0
    - Below opt_lo -> exp(-left_k * dist) exponential decay (k=0 means no penalty)
    - Above opt_hi -> exp(-right_k * dist) exponential decay (k=0 means no penalty)
    Continuous at boundaries (f(opt_lo)=f(opt_hi)=1.0).
    """
    if opt_lo <= x <= opt_hi:
        return 1.0
    if x < opt_lo:
        return float(np.exp(-left_k * (opt_lo - x))) if left_k > 0 else 1.0
    return float(np.exp(-right_k * (x - opt_hi))) if right_k > 0 else 1.0


def sigmoid_up(x: float, center: float, k: float = 1.0) -> float:
    """Monotonically increasing sigmoid score (higher is better), center ~0.5, approaches 0/1 at ends."""
    return float(1.0 / (1.0 + np.exp(-k * (x - center))))


# ─────────────────────────────────────────────────────────────────
# Data Fetching Layer (with session-level caching)
# ─────────────────────────────────────────────────────────────────

class DataFetcher:
    _instance = None
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.analyzer = ChipDistributionAnalyzer()
        self.crawler = Crawler()
        self._sh_cache = None       # Shanghai index history
        self._hot_cache = None      # Hot list (df, timestamp)
        self._hot_cache_ttl = 300    # Hot list cache TTL (seconds)
        self._realtime_map = {}     # {6-digit code: realtime price} (tushare)
        self._spot_cache  = None    # akshare full market spot, fallback
        self._chip_manual_cache = {}  # {code6: df}, manual chip data cache
        self._zt_cache = None         # Today's limit-up pool (df, timestamp)
        print("  [DataFetcher] Initialization complete")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ── Realtime quotes (tushare realtime_quote batch fetch) ──────────────
    @staticmethod
    def _gen_ts_symbols(code_list):
        """Convert 6-digit code list to tushare accepted ts_code string"""
        parts = [to_ts_code(c) for c in code_list]
        return ','.join(parts)

    def prefetch_realtime(self, stock_list):
        """Batch fetch realtime quotes and cache to _realtime_map {code6: price}"""
        symbols = self._gen_ts_symbols(stock_list)
        df = self.analyzer.get_realtime_tick(ts_code=symbols)
        new_map = {}
        if df is None or df.empty:
            print("  [Realtime] tushare fetch failed, will rely on akshare spot fallback")
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
                    print(f"  [Realtime] {df.iloc[i][code_col][:6]} parse failed: {e}")
        self._realtime_map = new_map
        print(f"  [Realtime] tushare cached {len(new_map)}/{len(stock_list)} stocks")


    def get_realtime_price(self, code):
        """Get price from cached realtime quotes"""
        return self._realtime_map.get(str(code).zfill(6))

    def prefetch_spot(self, stock_list):
        """akshare full market spot, keep only prices for stock_list as fallback cache"""
        df = safe_call(ak.stock_zh_a_spot_em, default=None, sleep_sec=0.5)
        if df is None or df.empty:
            print("  [Spot] akshare fetch failed")
            return
        code_col  = 'Code'   if 'Code'   in df.columns else '代码'
        price_col = 'Latest' if 'Latest' in df.columns else '最新价'
        if code_col is None or price_col is None:
            print("  [Spot] column names mismatch, skipping")
            return
        need = set(str(c).zfill(6) for c in stock_list)
        self._spot_cache = (
            df[df[code_col].isin(need)][[code_col, price_col]]
            .set_index(code_col)[price_col]
            .to_dict()
        )
        print(f"  [Spot] akshare fallback cached {len(self._spot_cache)}/{len(stock_list)} stocks")

    def get_spot_price(self, code):
        """Get price from akshare spot cache (fallback)"""
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

    # ── Individual stock historical daily ─────────────────────────────────────────────
    def get_hist(self, code, days=80):
        """akshare unadjusted daily, return ascending DataFrame (~days rows)"""
        end = datetime.date.today().strftime('%Y%m%d')
        start_dt = datetime.date.today() - datetime.timedelta(days=int(days * 1.8))
        start = start_dt.strftime('%Y%m%d')
        df = None

        # First try sina
        df = safe_call(
            self.crawler.get_stock_history_simple,
            symbol='sh' + str(code) if str(code).startswith('6') else 'sz' + str(code),
            datalen=days,
        )
        df = df.rename(columns={'date': 'Date', 'high': 'High', 'low': 'Low', 'open': 'Open', 'close': 'Close',
                                'amount': 'Volume'})
        # Then tencent
        if df is None or df.empty:
            print("Both eastmoney and sina failed")
            df = safe_call(
                ak.stock_zh_a_hist_tx,
                symbol='sh' + str(code) if str(code).startswith('6') else 'sz' + str(code),
                start_date=start,
                end_date=end,
                adjust=''
            )
            df = df.rename(columns={'date':'Date','high':'High','low':'Low','open':'Open','close':'Close','amount':'Volume'})
        df = df.sort_values('Date').tail(days).reset_index(drop=True)
        if df is None or df.empty:
            return pd.DataFrame()

        return df

    # ── Shanghai index history ─────────────────────────────────────────────
    def get_sh_index(self, days=80):
        """Shanghai index daily (with caching)"""
        if self._sh_cache is not None:
            return self._sh_cache
        df = safe_call(ak.stock_zh_index_daily, default=None, sleep_sec=0.5, symbol='sh000001')
        if df is None or df.empty:
            return pd.DataFrame()
        # Columns: date, open, close, high, low, volume
        df = df.sort_values('date').tail(days).reset_index(drop=True)
        self._sh_cache = df
        return df

    # ── Individual stock fund flow ─────────────────────────────────────────────
    def get_fund_flow(self, code, days=10):
        """Eastmoney individual stock fund flow (last *days* trading days)"""
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
        return df.tail(days).reset_index(drop=True)

    # ── Chip distribution (akshare → direct crawler → tushare → manual, 4-level fallback) ─
    def get_chip(self, code):
        """
        Chip distribution 4-level fallback:
          1. ak.stock_cyq_em (original, depends on py_mini_racer)
          2. Crawler.get_chip_em (pure Python CYQ, no py_mini_racer needed)
          3. ChipDistributionAnalyzer.get_stock_chip_distribution (tushare cyq_perf)
             Returns cols: ts_code/trade_date/his_low/his_high/cost_5pct/cost_15pct/
                    cost_50pct/cost_85pct/cost_95pct/weight_avg/winner_rate
             Normalized with _source='tushare' tag, scoring uses relaxed concentration rules
          4. Manual input (prompts user when all 3 automatic methods fail)
        """
        code6 = str(code).zfill(6)

        # ── 1. akshare ──────────────────────────────────────────
        df = safe_call(ak.stock_cyq_em, default=None, sleep_sec=0.5,
                       symbol=code6, adjust="")
        if df is not None and not df.empty:
            return df

        # ── 2. Direct crawler (bypass py_mini_racer) ───────────────────
        print(f"  [Chip] {code6}: akshare failed, trying direct crawler...")
        try:
            df = self.crawler.get_chip_em(code6)
            if df is not None and not df.empty:
                print(f"  [Chip] {code6}: direct crawler success")
                return df
        except Exception as e:
            print(f"  [Chip] {code6}: direct crawler exception: {e}")

        # ── 3. tushare cyq_perf ──────────────────────────────────
        print(f"  [Chip] {code6}: direct crawler failed, trying tushare cyq_perf...")
        try:
            ts_code = to_ts_code(code6)
            end_d   = datetime.date.today().strftime('%Y%m%d')
            start_d = (datetime.date.today() - datetime.timedelta(days=20)).strftime('%Y%m%d')
            ts_df = self.analyzer.get_stock_chip_distribution(ts_code, start_d, end_d)
            if ts_df is not None and not ts_df.empty:
                ts_df = ts_df.sort_values('trade_date').iloc[[-1]].reset_index(drop=True)
                r = ts_df.iloc[0]
                weight_avg  = float(r.get('weight_avg', 0) or 0)
                cost_50     = float(r.get('cost_50pct', 0) or 0)
                cost_5      = float(r.get('cost_5pct',  0) or 0)
                cost_95     = float(r.get('cost_95pct', 0) or 0)
                cost_15     = float(r.get('cost_15pct', 0) or 0)
                cost_85     = float(r.get('cost_85pct', 0) or 0)
                winner_raw  = float(r.get('winner_rate', 0) or 0)
                # weight_avg may be 0, use cost_50pct as fallback
                avg = weight_avg if weight_avg > 0 else cost_50
                # winner_rate in tushare is 0~100 (percentage), convert to 0~1
                winner_norm = winner_raw / 100.0 if winner_raw > 1.0 else winner_raw
                # Concentration proxy: 90% chip interval width / avg price (smaller = more concentrated)
                conc_90 = (cost_95 - cost_5)  / avg if avg > 0 else 0.20
                conc_70 = (cost_85 - cost_15) / avg if avg > 0 else 0.15
                df = pd.DataFrame([{
                    'Date':      str(r.get('trade_date', datetime.date.today())),
                    'Winner Ratio':  winner_norm,
                    'Avg Cost':  avg,
                    '90 Cost-Low': cost_5,
                    '90 Cost-High': cost_95,
                    '90 Concentration':  conc_90,
                    '70 Cost-Low': cost_15,
                    '70 Cost-High': cost_85,
                    '70 Concentration':  conc_70,
                    '_source':   'tushare',
                }])
                print(f"  [Chip] {code6}: tushare success"
                      f"(concentration proxy={conc_90:.3f}, winner ratio={winner_norm*100:.1f}%)")
                return df
        except Exception as e:
            print(f"  [Chip] {code6}: tushare exception: {e}")

        # ── 4. Manual input ─────────────────────────────────────────
        return self._manual_chip_input(code6)

    def _manual_chip_input(self, code):
        """
        When both automatic methods fail, prompt user to manually input key chip metrics.
        Press Enter to skip, returns empty DataFrame (scoring takes neutral value).
        Input results are cached, won't ask again for same stock in this run.
        """
        code6 = str(code).zfill(6)
        if code6 in self._chip_manual_cache:
            return self._chip_manual_cache[code6]

        market_prefix = 'sh' if code6.startswith('6') else 'sz'
        em_url = f"https://quote.eastmoney.com/concept/{market_prefix}{code6}.html"
        print(f"\n  ┌── [{code6}] Chip data auto-fetch failed ──────────────────────")
        print(f"  │  Reference link (Daily K → Chip distribution): {em_url}")
        print(  "  │  Please input manually (Enter to skip, chip takes neutral score):")
        try:
            w = input("  │  Winner ratio (0~100, %) e.g., 45.5 :").strip()
            if not w:
                print("  └── Skipped")
                self._chip_manual_cache[code6] = pd.DataFrame()
                return pd.DataFrame()

            c90  = input("  │  90 Concentration (0~1, smaller = more concentrated) e.g., 0.12 :").strip()
            c90l = input("  │  90 Cost-Low (yuan)                    :").strip()
            c90h = input("  │  90 Cost-High (yuan)                   :").strip()
            avg  = input("  │  Avg Cost (yuan, optional)             :").strip()
            print("  └────────────────────────────────────────────────────")

            winner_val = float(w)
            # Convert to 0~1 decimal (consistent with akshare format)
            if winner_val > 1.0:
                winner_val /= 100.0

            df = pd.DataFrame([{
                'Date':    datetime.date.today(),
                'Winner Ratio': winner_val,
                'Avg Cost': float(avg) if avg else 0.0,
                '90 Cost-Low': float(c90l) if c90l else 0.0,
                '90 Cost-High': float(c90h) if c90h else 0.0,
                '90 Concentration':  float(c90) if c90 else 0.15,
                '70 Cost-Low': 0.0, '70 Cost-High': 0.0, '70 Concentration': 0.0,
            }])
            self._chip_manual_cache[code6] = df
            return df

        except (EOFError, KeyboardInterrupt):
            print("  └── Skipped")
            self._chip_manual_cache[code6] = pd.DataFrame()
            return pd.DataFrame()
        except Exception as e:
            print(f"  └── Input parse failed: {e}, skipped")
            self._chip_manual_cache[code6] = pd.DataFrame()
            return pd.DataFrame()

    # ── Hot list ─────────────────────────────────────────────────────
    def get_hot_rank(self):
        """Eastmoney hot list (with TTL cache, auto-refresh after _hot_cache_ttl seconds)"""
        import time
        now = time.time()
        if self._hot_cache is not None:
            df, ts = self._hot_cache
            if now - ts < self._hot_cache_ttl:
                return df
        df = safe_call(self.crawler.get_ths_hot_rank, default=None, sleep_sec=0.5)
        self._hot_cache = (df, now)
        return df

    # ── Today's limit-up pool ─────────────────────────────────────────────
    def get_limit_up_pool(self):
        """Eastmoney today's limit-up pool (with TTL cache), includes consecutive limit count/seal amount/breakout count etc."""
        now = time.time()
        if self._zt_cache is not None:
            df, ts = self._zt_cache
            if now - ts < self._hot_cache_ttl:
                return df
        date_str = datetime.date.today().strftime('%Y%m%d')
        df = safe_call(ak.stock_zt_pool_em, default=None, sleep_sec=0.5, date=date_str)
        self._zt_cache = (df, now)
        return df

    # ── Sector information ─────────────────────────────────────────────────
    def get_sector_name(self, code):
        """Get industry sector name for individual stock"""
        symbol = str(code)+'.SH'  if str(code).startswith('6') else str(code) +'SZ'
        # df = ak.stock_individual_info_em(code)
        df = self.analyzer.get_stock_basic(code)

        row = df[df['item'] == '行业']
        if row.empty:
            return None
        return str(row.iloc[0]['value'].get('ind_name'))


    def get_sector_hist(self, sector_name, days=10):
        """Industry sector historical quotes"""
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
# CSV Output
# ─────────────────────────────────────────────────────────────────

def save_scores_csv(df: pd.DataFrame, output_dir: str = None) -> str:
    """Save scoring results to results/scores/score_{date}.csv, return file path"""
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
    print(f"  [CSV] Results written to → {fpath}")
    return fpath


# ─────────────────────────────────────────────────────────────────
# Scoring Modules
# ─────────────────────────────────────────────────────────────────

class StockScorer:
    """
    Short-term leader stock comprehensive scorer, max 100 points
    Target holding period: 1~5 trading days
    """

    WEIGHTS = {
        'breakout':          12,  # Breakout strength (60-day high position/time)
        'recent_change':     12,  # Recent cumulative return (1/3/5 days)
        'relative_strength': 10,  # Relative strength deviation (5-day vs SH/Sector)
        'fund_flow':         15,  # Large order flow (5/3/1 days, recent weighted)
        'chip':               7,  # Chip distribution (akshare, focus on control)
        'technical':         10,  # Technical indicators (RSI strong range, volume ratio)
        'hot_rank':           7,  # Hot list ranking
        'seal_quality':       8,  # Seal quality (limit time/seal amount/breakout/lowest)
        'sector_strength':    8,  # Sector strength (sector limit count/leader/return)
        'consec_limit':      11,  # Consecutive limit tier (count/competition/suppression/regulatory)
    }
    # WEIGHTS total = 100 points

    def __init__(self):
        self.fetcher = DataFetcher()

    # ── 1. Breakout strength score ───────────────────────────────────────────
    def score_breakout(self, hist_df, current_price):
        """
        Max 12 points (short-term breakout oriented)
        - Days since 60-day high (closer = better, indicates strong fresh breakout)
        - Current price position relative to 60-day high (higher = better)
        """
        w = self.WEIGHTS['breakout']
        if hist_df.empty or current_price is None:
            return {'score': w * 0.5, 'detail': 'Insufficient data, neutral'}

        hist_60 = hist_df.tail(60)
        idx_max = hist_60['High'].idxmax()
        max_price = float(hist_60.loc[idx_max, 'High'])
        max_date_str = str(hist_60.loc[idx_max, 'Date'])

        try:
            max_date = pd.to_datetime(max_date_str).date()
        except Exception:
            max_date = datetime.date.today()

        days_ago = (datetime.date.today() - max_date).days
        price_ratio = current_price / max_price if max_price > 0 else 1.0

        # Time score: closer to today is better; opt=[0,3] days, exponential decay per extra day
        ts = range_score(days_ago, 0, 3, left_k=0, right_k=0.04)

        # Price position score: closer to or above high is better; opt=[0.99,∞), exponential penalty below 0.99
        ps = range_score(price_ratio, 0.99, 9999, left_k=10, right_k=0)

        score = round(w * (ts * 0.60 + ps * 0.40), 2)
        detail = (f"60-day high={max_price:.2f}({max_date_str},{days_ago} days ago), "
                  f"current={current_price:.2f}({price_ratio*100:.1f}% position)")
        return {
            'score': score,
            'detail': detail,
            'max_price': max_price,
            'max_date': max_date_str,
            'days_ago': days_ago,
            'price_ratio': round(price_ratio * 100, 1)
        }

    # ── 2. Recent cumulative return score ────────────────────────────────────────
    def score_recent_change(self, hist_df):
        """
        Max 12 points (short-term version: 1/3/5 day returns)
        - 1-day: 3~9% optimal (strong start today)
        - 3-day: 5~20% optimal (rhythmic upward)
        - 5-day: 8~30% optimal (established strength but not overbought)
        - Overheat penalty: 5-day >45% significant deduction
        """
        w = self.WEIGHTS['recent_change']
        if hist_df.empty:
            return {'score': w * 0.5, 'detail': 'Insufficient data'}

        closes = hist_df['Close'].values
        chg1  = pct_change_n(closes, 1)
        chg3  = pct_change_n(closes, 3)
        chg5  = pct_change_n(closes, 5)

        # 1-day: opt=[3,9]% (strong start), penalty below 0, slight overbought penalty >12%
        s1 = range_score(chg1, 3.0, 9.0, left_k=0.15, right_k=0.10)
        # 3-day: opt=[5,20]% (rhythmic upward)
        s3 = range_score(chg3, 5.0, 20.0, left_k=0.08, right_k=0.04)
        # 5-day: opt=[8,30]% (established strength but not overbought), overheat penalty >45%
        s5 = range_score(chg5, 8.0, 30.0, left_k=0.06, right_k=0.02)
        combined = s1 * 0.40 + s3 * 0.35 + s5 * 0.25
        score = round(w * combined, 2)
        detail = f"1-day={chg1:+.2f}%, 3-day={chg3:+.2f}%, 5-day={chg5:+.2f}%"
        return {'score': score, 'detail': detail, 'chg1': chg1, 'chg3': chg3, 'chg5': chg5}

    # ── 3. Relative strength / deviation score ──────────────────────────────────
    def score_relative_strength(self, hist_df, sh_df, sector_name):
        """
        Max 10 points (short-term version: shortened window to 5 days)
        - Individual stock vs Shanghai excess on max gain day in last 5 days
        - Individual stock vs sector deviation cumulative over 5 days
        """
        w = self.WEIGHTS['relative_strength']
        if hist_df.empty:
            return {'score': w * 0.5, 'detail': 'Insufficient data'}

        hist_5 = hist_df.tail(5).reset_index(drop=True)
        closes_5 = hist_5['Close'].values
        if len(closes_5) < 2:
            return {'score': w * 0.5, 'detail': 'Too little data'}

        # Daily returns
        if 'Change' in hist_5.columns:
            stock_pct = hist_5['Change'].values.astype(float)
        else:
            stock_pct = np.diff(closes_5) / closes_5[:-1] * 100
            stock_pct = np.concatenate([[0.0], stock_pct])

        max_gain_idx = int(np.argmax(stock_pct))
        max_loss_idx = int(np.argmin(stock_pct))
        max_gain = float(stock_pct[max_gain_idx])
        max_loss = float(stock_pct[max_loss_idx])
        max_gain_date = str(hist_5.iloc[max_gain_idx]['Date'])
        max_loss_date = str(hist_5.iloc[max_loss_idx]['Date'])

        # Build Shanghai return date mapping
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

        gain_dev = max_gain - _sh_pct(max_gain_date)   # Excess on max gain day
        loss_dev = max_loss - _sh_pct(max_loss_date)   # Resilience on max loss day

        # Cumulative 5-day deviation
        hist_6 = hist_df.tail(6)
        c6 = hist_6['Close'].values
        stock_5d = (c6[-1] - c6[0]) / c6[0] * 100 if c6[0] > 0 and len(c6) >= 2 else 0.0

        sector_deviation = 0.0
        if sector_name:
            sec_df = self.fetcher.get_sector_hist(sector_name, days=10)
            if not sec_df.empty:
                sc = 'Close' if 'Close' in sec_df.columns else sec_df.columns[-1]
                if sc in sec_df.columns and len(sec_df) >= 6:
                    sv = sec_df[sc].values
                    sector_5d = (sv[-1] - sv[-6]) / sv[-6] * 100 if sv[-6] > 0 else 0.0
                    sector_deviation = stock_5d - sector_5d

        # Excess score: opt=[3,∞)%, fast penalty below 0 for no excess
        gain_dev_score = range_score(gain_dev, 3.0, 9999, left_k=0.18, right_k=0)
        # Resilience score: optimal when max loss day excess >=0 (resilient), penalty when loss greater than index
        loss_dev_score = range_score(loss_dev, 0.0, 9999, left_k=0.20, right_k=0)
        # Sector deviation score: opt=[2,∞)% excess
        sector_score = range_score(sector_deviation, 2.0, 9999, left_k=0.15, right_k=0)

        combined = gain_dev_score * 0.40 + loss_dev_score * 0.30 + sector_score * 0.30
        score = round(w * combined, 2)
        detail = (f"5-day max gain={max_gain:+.2f}%({max_gain_date},excess{gain_dev:+.2f}%), "
                  f"max loss={max_loss:+.2f}%({max_loss_date},excess{loss_dev:+.2f}%), "
                  f"5-day sector deviation={sector_deviation:+.2f}%")
        return {
            'score': score, 'detail': detail,
            'max_gain': max_gain, 'max_gain_date': max_gain_date,
            'max_loss': max_loss, 'max_loss_date': max_loss_date,
            'gain_deviation': round(gain_dev, 2),
            'loss_deviation': round(loss_dev, 2),
            'sector_deviation': round(sector_deviation, 2)
        }

    # ── 4. Large order fund flow score ────────────────────────────────────────
    def score_fund_flow(self, code):
        """
        资金流入评分，根据 5日流入，3日流入，1日流入情况，打分
        1，3，5 日的权重分别为 57% / 29% / 14%
        取 100 日数据，分别计算 3 日累计流入 60 分位，5日累计流入 60 分位，单日流入 60 分位，
        以这些分位为基准，按比值大小进行适当评分
        """
        w = self.WEIGHTS['fund_flow']
        df = self.fetcher.get_fund_flow(code, days=100)
        if df is None or df.empty:
            return {'score': w * 0.5, 'detail': 'Fund flow data insufficient'}

        candidate_cols = ['主力净流入-净额', '超大单净流入-净额', '大单净流入-净额']
        inflow_col = next((c for c in candidate_cols if c in df.columns), None)
        if inflow_col is None:
            return {'score': w * 0.5, 'detail': f'Net inflow column not found, available: {df.columns.tolist()}'}

        inflow = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0)

        # 当前值
        flow_5 = float(inflow.tail(5).sum())
        flow_3 = float(inflow.tail(3).sum())
        flow_1 = float(inflow.iloc[-1]) if len(inflow) > 0 else 0.0

        # 降级基准（当数据不足时使用）
        # 注意：这里的值会根据市值分档进一步调整
        FALLBACK_SCALES = {1: 5e6, 3: 8e6, 5: 1.5e7}

        def _get_market_cap_factor(self, code):
            """获取市值系数，用于降级时的基准调整"""
            try:
                # 获取流通市值（单位：元）
                market_cap = self.fetcher.get_float_market_cap(code)
                if market_cap is None or market_cap <= 0:
                    return 1.0
                # 以100亿为基准
                return max(0.3, min(3.0, market_cap / 1e10))
            except:
                return 1.0

        def _get_quantile_scale(self, series, period_days, target_quantile=0.6, min_history=20):
            """
            从股票自身历史分布中获取动态基准

            参数:
                series: 每日净流入序列
                period_days: 周期天数（1/3/5）
                target_quantile: 目标分位数，默认0.6（60%分位数）
                min_history: 最少需要的历史数据量

            返回:
                scale: 动态基准值
            """
            # 计算滚动累计（period_days=1时不需要滚动）
            if period_days == 1:
                rolling = series.copy()
            else:
                rolling = series.rolling(period_days).sum().dropna()

            # 检查数据量是否足够
            if len(rolling) < min_history:
                # 数据不足，使用降级基准
                fallback = FALLBACK_SCALES[period_days]
                # 如果有市值信息，按市值调整降级基准
                if hasattr(self, '_get_market_cap_factor'):
                    factor = self._get_market_cap_factor(self, code)  # 注意这里的code需要传入
                    return fallback * factor
                return fallback

            # 计算分位数
            scale = float(rolling.quantile(target_quantile))

            # 处理异常情况
            # 1. 如果基准为负数（说明该股票历史上持续净流出），使用中位数或降级基准
            if scale <= 0:
                # 尝试用正数部分的中位数
                positive_vals = rolling[rolling > 0]
                if len(positive_vals) >= min_history // 2:
                    scale = float(positive_vals.quantile(0.5))
                else:
                    # 全部为负，使用绝对值的中位数作为基准（但方向为负）
                    scale = float(rolling.quantile(0.5))
                    # 如果还是负数，使用降级基准
                    if scale <= 0:
                        fallback = FALLBACK_SCALES[period_days]
                        if hasattr(self, '_get_market_cap_factor'):
                            factor = self._get_market_cap_factor(self, code)
                            scale = fallback * factor
                        else:
                            scale = fallback

            # 2. 设置合理范围，避免极端值（100万 ~ 10亿）
            scale = max(scale, 1e6)
            scale = min(scale, 1e9)

            return scale

        def _flow_score( val, scale):
            """
            连续评分函数
            将 val/scale 的比值映射到 [0.5, 1.5] 区间

            当 val = scale（比值=1）时，得分为1.0（中性）
            当 val > scale 时，得分 > 1.0（加分），上限1.5
            当 val < scale 时，得分 < 1.0（减分），下限0.5

            使用指数衰减的非对称设计：
            - 左侧（流出）惩罚更重（right_k=0.5）
            - 右侧（流入）奖励更温和（left_k=1.5）
            """
            if scale == 0:
                return 1.0

            ratio = val / scale

            # 限制比值范围，避免极端值过度影响
            # 最大比值限制为3.0（超过3.0不再额外加分）
            # 最小比值限制为0.33（低于0.33不再额外减分）
            ratio = max(0.33, min(3.0, ratio))

            # 使用 range_score 函数（假设已定义）
            # 参数：ratio, 下限, 上限, 左侧斜率, 右侧斜率
            return range_score(ratio, 0.5, 1.5, left_k=1.5, right_k=0.5)

        # 计算各周期的动态基准
        scale_5 = _get_quantile_scale(self, inflow, 5, target_quantile=0.6)
        scale_3 = _get_quantile_scale(self, inflow, 3, target_quantile=0.6)
        scale_1 = _get_quantile_scale(self, inflow, 1, target_quantile=0.6)

        # 计算各周期的评分系数
        s5 = _flow_score(flow_5, scale_5)
        s3 = _flow_score(flow_3, scale_3)
        s1 = _flow_score(flow_1, scale_1)

        # 权重：1日57% + 3日29% + 5日14%（基于指数衰减 λ=0.35）
        combined = s1 * 0.57 + s3 * 0.29 + s5 * 0.14
        score = round(w * combined, 2)

        # 详细输出（包含基准信息，便于调试）
        detail = (f"当前(5d/3d/1d): {flow_5 / 1e4:.0f}万/{flow_3 / 1e4:.0f}万/{flow_1 / 1e4:.0f}万 | "
                  f"基准(5d/3d/1d): {scale_5 / 1e4:.0f}万/{scale_3 / 1e4:.0f}万/{scale_1 / 1e4:.0f}万 | "
                  f"系数(5d/3d/1d): {s5:.2f}/{s3:.2f}/{s1:.2f} | "
                  f"综合: {combined:.2f} → 得分: {score}")

        return {
            'score': score,
            'detail': detail,
            'flow_5d': flow_5,
            'flow_3d': flow_3,
            'flow_1d': flow_1,
            'scale_5d': scale_5,
            'scale_3d': scale_3,
            'scale_1d': scale_1,
            'score_5d': s5,
            'score_3d': s3,
            'score_1d': s1,
            'combined': combined
        }

    # ── 5. Chip distribution score ────────────────────────────────────────────
    def score_chip(self, code):
        """
        Max 7 points (short-term control oriented)
        Data source priority: akshare/crawler > tushare cyq_perf > manual input

        akshare/crawler mode (has precise 90 concentration):
          - Concentration 55%: opt=[0,0.10], fast decay beyond
          - Winner ratio 45%: opt=[30,70]%

        tushare mode (90 concentration is 95th-5th interval proxy, slightly less precise):
          - Concentration 35%: opt=[0,0.15], using looser threshold and slower decay
          - Winner ratio 65%: higher weight to compensate for concentration precision
          - detail will have [tushare] tag
        """
        w = self.WEIGHTS['chip']
        df = self.fetcher.get_chip(code)
        if df is None or df.empty:
            return {'score': w * 0.5, 'detail': 'Chip data insufficient'}

        # Verify required columns
        required = ['Winner Ratio', 'Avg Cost', '90 Concentration', '90 Cost-Low', '90 Cost-High']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {'score': w * 0.5, 'detail': f'Missing chip columns: {missing}, actual: {df.columns.tolist()}'}

        # Take most recent row (latest trading day)
        latest = df.sort_values('Date').iloc[-1] if 'Date' in df.columns else df.iloc[-1]
        source = latest['_source'] if '_source' in df.columns else 'akshare'

        try:
            winner       = float(latest['Winner Ratio'])
            avg_cost     = float(latest['Avg Cost'])
            conc_90      = float(latest['90 Concentration'])
            cost_90_low  = float(latest['90 Cost-Low'])
            cost_90_high = float(latest['90 Cost-High'])

            # Winner ratio normalized to percentage (akshare returns 0~1 decimal)
            winner_pct = winner * 100 if winner <= 1.0 else winner

            if source == 'tushare':
                # tushare: concentration is (95th-5th)/avg price proxy, less precise than akshare
                # Use looser threshold opt=[0,0.15], reduce concentration weight to 35%
                cs = range_score(conc_90, 0.0, 0.15, left_k=0, right_k=8)
                ws = range_score(winner_pct, 30.0, 70.0, left_k=0.06, right_k=0.06)
                score = round(w * (cs * 0.35 + ws * 0.65), 2)
                detail = (f"[tushare] concentration proxy={conc_90:.3f}"
                          f"({cost_90_low:.2f}~{cost_90_high:.2f}), "
                          f"avg cost={avg_cost:.2f}, winner ratio={winner_pct:.1f}%")
            else:
                # akshare/crawler: has precise 90 concentration
                cs = range_score(conc_90, 0.0, 0.10, left_k=0, right_k=15)
                ws = range_score(winner_pct, 30.0, 70.0, left_k=0.06, right_k=0.06)
                score = round(w * (cs * 0.55 + ws * 0.45), 2)
                detail = (f"90 concentration={conc_90:.3f}({cost_90_low:.2f}~{cost_90_high:.2f}), "
                          f"avg cost={avg_cost:.2f}, winner ratio={winner_pct:.1f}%")

            return {'score': score, 'detail': detail,
                    'conc_90': round(conc_90, 4),
                    'winner_rate': round(winner_pct, 1),
                    'avg_cost': round(avg_cost, 2),
                    'cost_90_low': round(cost_90_low, 2),
                    'cost_90_high': round(cost_90_high, 2),
                    'chip_source': source}
        except Exception as e:
            return {'score': w * 0.5, 'detail': f'Chip calculation exception: {e}'}

    # ── 6. Technical indicator score ────────────────────────────────────────────
    def score_technical(self, hist_df):
        """
        Max 10 points (short-term version)
        - RSI(14): 50~75 strong healthy range (short-term doesn't need very low RSI)
        - Volume ratio: >= 2.0 effective volume breakout bonus
        """
        w = self.WEIGHTS['technical']
        if hist_df.empty or 'Close' not in hist_df.columns:
            return {'score': w * 0.5, 'detail': 'Insufficient data'}

        # RSI
        try:
            rsi = calc_rsi(hist_df['Close'], period=14)
        except Exception:
            rsi = 50.0

        # Short-term: RSI opt=[50,75] strong healthy range, decay for too low/high
        rsi_s = range_score(rsi, 50.0, 75.0, left_k=0.10, right_k=0.08)

        # Volume ratio
        vr = 1.0
        if 'Volume' in hist_df.columns:
            try:
                vr = calc_volume_ratio(hist_df['Volume'], days=5)
            except Exception:
                vr = 1.0

        # Short-term: volume ratio opt=[2.0,5.0] volume breakout optimal; penalty for low volume or extreme high
        if 2.0 <= vr <= 5.0:   vr_s = 1.00
        elif 1.5 <= vr < 2.0:  vr_s = 0.85
        elif 1.0 <= vr < 1.5:  vr_s = 0.65
        elif 5.0 < vr <= 8.0:  vr_s = 0.60
        elif 0.7 <= vr < 1.0:  vr_s = 0.40
        elif 8.0 < vr:         vr_s = 0.35
        else:                  vr_s = 0.20

        score = round(w * (rsi_s * 0.55 + vr_s * 0.45), 2)
        detail = f"RSI(14)={rsi:.1f}, volume ratio={vr:.2f}"
        return {'score': score, 'detail': detail, 'rsi': rsi, 'volume_ratio': vr}

    # ── 7. Hot list score ────────────────────────────────────────────────
    def score_hot_rank(self, code):
        """
        Max 7 points
        Eastmoney hot list ranking, higher rank = higher score
        """
        w = self.WEIGHTS['hot_rank']
        df = self.fetcher.get_hot_rank()
        if df is None or df.empty:
            return {'score': w * 0.4, 'detail': 'Hot list data insufficient', 'rank': None}

        code = str(code).zfill(6)
        code_col = 'Code' if 'Code' in df.columns else 'code'
        if code_col is None:
            return {'score': w * 0.4, 'detail': 'Hot list column mismatch', 'rank': None}
        row = df[df[code_col] == code]
        if row.empty:
            return {'score': 0.0, 'detail': 'Not on hot list', 'rank': None}

        rank_col = 'Rank' if 'Rank' in df.columns else '排名'
        rank = int(row.iloc[0][rank_col]) if rank_col else int(row.index[0]) + 1

        # opt=[1,10], faster decay for lower rank
        hs = range_score(rank, 1, 10, left_k=0, right_k=0.05)

        return {'score': round(w * hs, 2), 'detail': f'Hot list rank {rank}', 'rank': rank}

    # ── 8. Seal quality score ────────────────────────────────────────────
    def score_seal_quality(self, code, hist_df):
        """
        Max 8 points (only valid for stocks that hit limit-up today, non-limit returns 0)
        - First seal time 30%: earlier is better, opt=[open,+30min]
        - Seal amount / float market cap 20%: larger amount = more solid
        - Breakout count 30%: 0 times full score, more times = more deduction
        - Today's low / limit price 20%: closer to 1 means less drop, stronger control
        """
        w = self.WEIGHTS['seal_quality']
        zt_df = self.fetcher.get_limit_up_pool()
        if zt_df is None or zt_df.empty:
            return {'score': 0.0, 'detail': 'Limit-up pool data unavailable'}

        code6 = str(code).zfill(6)
        code_col = next((c for c in ['Code', '代码'] if c in zt_df.columns), None)
        if code_col is None:
            return {'score': 0.0, 'detail': 'Limit-up pool column mismatch'}

        row_df = zt_df[zt_df[code_col] == code6]
        if row_df.empty:
            return {'score': 0.0, 'detail': 'Not limit-up today'}

        row = row_df.iloc[0]

        # ① First seal time (earlier is better, opt=[0,30] minutes)
        time_col = next((c for c in ['First seal time', 'First limit time', '首次封板时间', '首次涨停时间'] if c in zt_df.columns), None)
        time_s = 0.5
        if time_col:
            try:
                t = str(row[time_col]).strip()
                parts = t.replace(':', '').replace('：', '')
                h, m = int(parts[:2]), int(parts[2:4])
                minutes_from_open = (h - 9) * 60 + m - 30
                time_s = range_score(minutes_from_open, 0, 30, left_k=0, right_k=0.015)
            except Exception:
                time_s = 0.5

        # ② Seal amount as percentage of float market cap (larger = more solid)
        fund_col = next((c for c in ['Seal amount', 'Seal funds', '封板资金', '封单金额'] if c in zt_df.columns), None)
        mv_col   = next((c for c in ['Float market cap', '流通市值'] if c in zt_df.columns), None)
        fund_s = 0.5
        if fund_col:
            try:
                fund = float(row[fund_col])
                if mv_col:
                    mv = float(row[mv_col])
                    ratio = fund / mv if mv > 0 else 0.0
                    fund_s = range_score(ratio, 0.02, 0.15, left_k=30, right_k=3)
                else:
                    fund_s = sigmoid_up(np.log10(max(fund, 1e3)), 7.0, k=1.2)
            except Exception:
                fund_s = 0.5

        # ③ Breakout count (0 times = full score, more times = worse)
        break_col = next((c for c in ['Breakout count', 'Open count', '炸板次数', '开板次数'] if c in zt_df.columns), None)
        break_s = 0.5
        if break_col:
            try:
                breaks = int(row[break_col])
                if breaks == 0:     break_s = 1.00
                elif breaks == 1:   break_s = 0.65
                elif breaks == 2:   break_s = 0.35
                else:               break_s = max(0.10, 0.35 - (breaks - 2) * 0.08)
            except Exception:
                break_s = 0.5

        # ④ Today's low / limit price (closer to 1 = more stable, opt=[0.97,1.0])
        low_s = 0.5
        if not hist_df.empty and 'Low' in hist_df.columns and len(hist_df) >= 2:
            try:
                today_low   = float(hist_df.iloc[-1]['Low'])
                prev_close  = float(hist_df.iloc[-2]['Close'])
                limit_price = round(prev_close * 1.1, 2)
                low_ratio   = today_low / limit_price if limit_price > 0 else 1.0
                low_s = range_score(low_ratio, 0.97, 1.01, left_k=20, right_k=5)
            except Exception:
                low_s = 0.5

        combined = time_s * 0.30 + fund_s * 0.20 + break_s * 0.30 + low_s * 0.20
        score = round(w * combined, 2)
        detail = (f"First seal={row.get(time_col, '?') if time_col else '?'}, "
                  f"Breakout={row.get(break_col, '?') if break_col else '?'} times, "
                  f"Seal amount={row.get(fund_col, '?') if fund_col else '?'}")
        return {
            'score': score, 'detail': detail,
            'time_score': round(time_s, 3), 'break_count': row.get(break_col, None) if break_col else None,
        }

    # ── 9. Sector strength score ────────────────────────────────────────────
    def score_sector_strength(self, code, sector_name):
        """
        Max 8 points
        - Sector limit count 50%: ≥5 stocks full, 3-4 stocks 80%, 1-2 stocks 50%, 0 stocks 0%
        - Sector leader strength 30%: max return in sector (limit-up = full score)
        - Sector 5-day return 20%: opt=[3,15]%
        """
        w = self.WEIGHTS['sector_strength']
        if not sector_name:
            return {'score': round(w * 0.3, 2), 'detail': 'No sector info', 'zt_count': 0}

        zt_df = self.fetcher.get_limit_up_pool()
        ind_col = None
        zt_count = 0
        if zt_df is not None and not zt_df.empty:
            ind_col = next((c for c in ['Industry', 'Sector', '所属行业', '行业'] if c in zt_df.columns), None)
            if ind_col:
                zt_count = int((zt_df[ind_col] == sector_name).sum())

        # Sector limit count score
        if zt_count >= 5:    cnt_s = 1.00
        elif zt_count >= 3:  cnt_s = 0.80
        elif zt_count >= 1:  cnt_s = 0.50
        else:                cnt_s = 0.00

        # Sector leader strength (highest return stock in sector, limit-up ≥9.5% → full)
        lead_s = 0.3
        if zt_df is not None and not zt_df.empty and ind_col:
            sec_zt = zt_df[zt_df[ind_col] == sector_name]
            if not sec_zt.empty:
                pct_col = next((c for c in ['Change', 'Return', '涨跌幅', '涨幅'] if c in sec_zt.columns), None)
                if pct_col:
                    try:
                        max_pct = float(pd.to_numeric(sec_zt[pct_col], errors='coerce').max())
                        lead_s = range_score(max_pct, 9.5, 20.0, left_k=0.30, right_k=0)
                    except Exception:
                        lead_s = 0.3

        # Sector 5-day return score
        sector_s = 0.3
        sec_df = self.fetcher.get_sector_hist(sector_name, days=10)
        if not sec_df.empty:
            sc = 'Close' if 'Close' in sec_df.columns else sec_df.columns[-1]
            if sc in sec_df.columns and len(sec_df) >= 6:
                sv = sec_df[sc].values
                sector_5d = (sv[-1] - sv[-6]) / sv[-6] * 100 if sv[-6] > 0 else 0.0
                sector_s = range_score(sector_5d, 3.0, 15.0, left_k=0.20, right_k=0.04)

        combined = cnt_s * 0.50 + lead_s * 0.30 + sector_s * 0.20
        score = round(w * combined, 2)
        detail = f"Sector={sector_name}, limit {zt_count} stocks, leader={round(lead_s,2)}"
        return {'score': score, 'detail': detail, 'zt_count': zt_count}

    # ── 10. Consecutive limit score ───────────────────────────────────────────
    def score_consec_limit(self, code, hist_df):
        """
        Max 11 points (consecutive limit tier)
        - Consecutive count from limit-up pool, fallback to hist calculation
        - opt=[2,4] boards (starting and strong), single board=55%, decreasing for high boards
        - ≥3 boards: count same-tier competition, more opponents = lower success probability (up to -35%)
        - ≥5 boards: calculate high-level suppression (more higher boards = more suppression)
        - ≥7 boards: regulatory pressure decay (additional -15% per board)
        """
        w = self.WEIGHTS['consec_limit']
        code6 = str(code).zfill(6)

        # Priority: get consecutive count from limit-up pool
        consec = 0
        zt_df = self.fetcher.get_limit_up_pool()
        lb_col = None
        if zt_df is not None and not zt_df.empty:
            code_col = next((c for c in ['Code', '代码'] if c in zt_df.columns), None)
            lb_col   = next((c for c in ['Consecutive boards', 'Consecutive limit', '连板数', '连续涨停数', '连板'] if c in zt_df.columns), None)
            if code_col and lb_col:
                row_df = zt_df[zt_df[code_col] == code6]
                if not row_df.empty:
                    try:
                        consec = int(row_df.iloc[0][lb_col])
                    except Exception:
                        pass

        # Fallback: calculate from hist_df recent consecutive limit days (return ≥9.5%)
        if consec == 0 and not hist_df.empty and 'Change' in hist_df.columns:
            for pct in reversed(hist_df['Change'].values[-10:]):
                try:
                    if float(pct) >= 9.5:
                        consec += 1
                    else:
                        break
                except Exception:
                    break

        if consec == 0:
            return {'score': 0.0, 'detail': 'Not limit-up today', 'consec': 0}

        # Base score (opt=[2,4], first board=0.55)
        if consec == 1:
            base_s = 0.55
        else:
            base_s = range_score(consec, 2, 4, left_k=0.7, right_k=0.35)

        # Competition suppression (for ≥3 boards, more same-tier opponents = harder to advance)
        compete_f = 1.0
        if consec >= 3 and zt_df is not None and not zt_df.empty and lb_col:
            try:
                same_count = int((zt_df[lb_col].astype(int) == consec).sum())
                if same_count >= 8:    compete_f = 0.65
                elif same_count >= 5:  compete_f = 0.80
                elif same_count >= 3:  compete_f = 0.90
            except Exception:
                pass

        # High-level suppression (for ≥5 boards, higher-tier stocks attract funds, creating pressure)
        high_f = 1.0
        if consec >= 5 and zt_df is not None and not zt_df.empty and lb_col:
            try:
                higher_count = int((zt_df[lb_col].astype(int) > consec).sum())
                if higher_count >= 3:   high_f = 0.70
                elif higher_count >= 1: high_f = 0.85
            except Exception:
                pass

        # Regulatory pressure (for ≥7 boards, additional -15% per board, minimum 0.15)
        reg_f = 1.0
        if consec >= 7:
            reg_f = max(0.15, 1.0 - (consec - 6) * 0.15)

        final_s = min(1.0, base_s * compete_f * high_f * reg_f)
        score = round(w * final_s, 2)
        detail = (f"Consecutive={consec}, competition factor={compete_f:.2f}, "
                  f"high factor={high_f:.2f}, regulatory factor={reg_f:.2f}")
        return {'score': score, 'detail': detail, 'consec': consec}

    # ── Single stock comprehensive scoring entry ───────────────────────────────────────
    def score_stock(self, code):
        """
        Complete scoring for a single stock
        :param code: 6-digit stock code (string or int)
        :return: dict containing total, scores, each dimension's detail
        """
        code = str(code).zfill(6)

        # Data fetching
        hist_df       = self.fetcher.get_hist(code, days=80)  # data everyday
        ts_code = self.fetcher.analyzer.normal_ts_code(code)
        _tick = self.fetcher.analyzer.get_realtime_tick(ts_code)
        current_price = float(_tick['PRICE'].iloc[0]) if _tick is not None and not _tick.empty else None
        if current_price is not None and np.isnan(current_price):
            current_price = None
        if current_price is None and not hist_df.empty:
            current_price = float(hist_df['Close'].iloc[-1])
        if current_price is None:
            current_price = self.fetcher.get_spot_price(code)

        sh_df       = self.fetcher.get_sh_index(days=80)
        sector_name = self.fetcher.get_sector_name(code)

        # Each dimension scoring
        scores = {
            'breakout':          self.score_breakout(hist_df, current_price),
            'recent_change':     self.score_recent_change(hist_df),
            'relative_strength': self.score_relative_strength(hist_df, sh_df, sector_name),
            'fund_flow':         self.score_fund_flow(code),
            'chip':              self.score_chip(code),
            'technical':         self.score_technical(hist_df),
            'hot_rank':          self.score_hot_rank(code),
            'seal_quality':      self.score_seal_quality(code, hist_df),
            'sector_strength':   self.score_sector_strength(code, sector_name),
            'consec_limit':      self.score_consec_limit(code, hist_df),
        }

        total = round(sum(v['score'] for v in scores.values()), 2)
        return {
            'code': code,
            'current_price': current_price,
            'sector': sector_name,
            'total': total,
            'scores': scores,
        }

    # ── Batch scoring entry ───────────────────────────────────────────────
    def score_list(self, stock_list, interval=1.0, save=True):
        """
        Batch score a list of stocks, return DataFrame sorted by total score descending
        :param stock_list:  List of stock codes (pure 6-digit strings)
        :param interval:    Seconds to wait between scoring each stock
        :param save:        Whether to automatically write to results/scores/ CSV
        """
        print(">> Preloading realtime quotes / Shanghai index / Hot list / Limit-up pool...")
        self.fetcher.prefetch_realtime(stock_list)
        self.fetcher.get_sh_index()
        self.fetcher.get_hot_rank()
        self.fetcher.get_limit_up_pool()
        print(f">> Starting to score {len(stock_list)} stocks\n")

        rows = []
        for i, code in enumerate(stock_list):
            print(f"  [{i+1:>2}/{len(stock_list)}] {code}", end='  ', flush=True)
            try:
                r = self.score_stock(code)
                s = r['scores']
                row = {
                    'Code':          r['code'],
                    'Price':        r.get('current_price', '-'),
                    'Sector':          r.get('sector', '-'),
                    'Total':          r['total'],
                    'Breakout':      s['breakout']['score'],
                    'Recent Return':      s['recent_change']['score'],
                    'Relative Strength':      s['relative_strength']['score'],
                    'Fund Flow':      s['fund_flow']['score'],
                    'Chip':      s['chip']['score'],
                    'Technical':      s['technical']['score'],
                    'Hot Rank':          s['hot_rank']['score'],
                    'Seal Quality':      s['seal_quality']['score'],
                    'Sector Strength':      s['sector_strength']['score'],
                    'Consec Limit':      s['consec_limit']['score'],
                    'RSI':           s['technical'].get('rsi', '-'),
                    'Volume Ratio':          s['technical'].get('volume_ratio', '-'),
                    '1D Return %':        s['recent_change'].get('chg1', '-'),
                    '3D Return %':        s['recent_change'].get('chg3', '-'),
                    '5D Return %':        s['recent_change'].get('chg5', '-'),
                    'Days Since 60D High':  s['breakout'].get('days_ago', '-'),
                    'Current/High %':    s['breakout'].get('price_ratio', '-'),
                    'Winner Ratio %':       s['chip'].get('winner_rate', '-'),
                    '90 Concentration':      s['chip'].get('conc_90', '-'),
                    'Consecutive Boards':        s['consec_limit'].get('consec', '-'),
                    'Sector Limit Count':    s['sector_strength'].get('zt_count', '-'),
                    '5D Main Net (10k)': round(s['fund_flow'].get('flow_5d', 0) / 1e4, 0)
                                     if isinstance(s['fund_flow'].get('flow_5d'), float) else '-',
                }
                rows.append(row)
                print(f"Total={r['total']:>5.1f}  "
                      f"Breakout={s['breakout']['score']:>4.1f}  "
                      f"Return={s['recent_change']['score']:>4.1f}  "
                      f"Strength={s['relative_strength']['score']:>4.1f}  "
                      f"Fund={s['fund_flow']['score']:>4.1f}")
            except Exception as e:
                print(f"Scoring failed: {e}")
                rows.append({'Code': str(code).zfill(6), 'Total': -1.0})
            time.sleep(interval)

        df = pd.DataFrame(rows)
        if not df.empty and 'Total' in df.columns:
            df = df.sort_values('Total', ascending=False).reset_index(drop=True)
        if save:
            save_scores_csv(df)
        return df


# ─────────────────────────────────────────────────────────────────
# Real-time Monitoring Loop
# ─────────────────────────────────────────────────────────────────

def run_monitor(stock_list, refresh_minutes=10):
    """
    Real-time monitoring mode: rescore and print rankings every refresh_minutes minutes
    Ctrl-C to exit
    """
    scorer = StockScorer()
    round_num = 0
    try:
        while True:
            round_num += 1
            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'='*72}")
            print(f"  Round {round_num}  |  {now_str}")
            print(f"{'='*72}")

            scorer.fetcher._sh_cache    = None
            scorer.fetcher._realtime_map = {}
            scorer.fetcher._zt_cache    = None

            result_df = scorer.score_list(stock_list, interval=0.8)

            display_cols = [
                'Code', 'Price', 'Sector', 'Total',
                'Breakout', 'Recent Return', 'Relative Strength', 'Fund Flow', 'Chip',
                'Technical', 'Hot Rank', 'Seal Quality', 'Sector Strength', 'Consec Limit',
                'RSI', 'Volume Ratio', '1D Return %', '3D Return %', '5D Return %', 'Consecutive Boards', 'Sector Limit Count',
            ]
            show = [c for c in display_cols if c in result_df.columns]

            print(f"\n{'─'*72}")
            print("  Comprehensive Ranking (High score priority)")
            print(f"{'─'*72}")
            print(result_df[show].to_string(index=True))
            print(f"\n  Next update: {refresh_minutes} minutes later  (Ctrl-C to exit)")
            time.sleep(refresh_minutes * 60)
    except KeyboardInterrupt:
        print("\n  Exited monitoring")


# ─────────────────────────────────────────────────────────────────
# Example Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TEST_CODE = '600396'   # 修改此处切换调试股票

    scorer  = StockScorer()
    fetcher = scorer.fetcher

    print(f"\n{'='*64}")
    print(f"  调试模式 | 代码: {TEST_CODE}")
    print(f"{'='*64}\n")

    # ── 预加载公共数据（各函数复用）──────────────────────────────
    print(">> 加载公共数据...")
    # hist_df     = fetcher.get_hist(TEST_CODE, days=80)
    # sh_df       = fetcher.get_sh_index(days=80)
    sector_name = fetcher.get_sector_name(TEST_CODE)
    ts_code     = fetcher.analyzer.normal_ts_code(TEST_CODE)
    _tick       = fetcher.analyzer.get_realtime_tick(ts_code)
    current_price = (float(_tick['PRICE'].iloc[0])
                     if _tick is not None and not _tick.empty else None)
    if current_price is not None and np.isnan(current_price):
        current_price = None
    # if current_price is None and not hist_df.empty:
        # current_price = float(hist_df['Close'].iloc[-1])
    fetcher.get_hot_rank()
    fetcher.get_limit_up_pool()
    # print(f"   hist rows={len(hist_df)}, price={current_price}, sector={sector_name}\n")

    def _show(name, result):
        score  = result.get('score', '?')
        detail = result.get('detail', '')
        extras = {k: v for k, v in result.items() if k not in ('score', 'detail')}
        print(f"  score = {score}")
        print(f"  detail: {detail}")
        if extras:
            print(f"  extras: {extras}")
        print()

    # ── 1. 突破强度 ──────────────────────────────────────────────
    print("─── 1. score_breakout ───────────────────────────────────")
    _show('breakout', scorer.score_breakout(hist_df, current_price))

    # ── 2. 近期涨幅 ──────────────────────────────────────────────
    print("─── 2. score_recent_change ──────────────────────────────")
    # _show('recent_change', scorer.score_recent_change(hist_df))

    # ── 3. 相对强度 ──────────────────────────────────────────────
    print("─── 3. score_relative_strength ──────────────────────────")
    # _show('relative_strength', scorer.score_relative_strength(hist_df, sh_df, sector_name))

    # ── 4. 资金流向 ──────────────────────────────────────────────
    print("─── 4. score_fund_flow ──────────────────────────────────")
    _show('fund_flow', scorer.score_fund_flow(TEST_CODE))

    # ── 5. 筹码分布 ──────────────────────────────────────────────
    print("─── 5. score_chip ───────────────────────────────────────")
    _show('chip', scorer.score_chip(TEST_CODE))

    # ── 6. 技术指标 ──────────────────────────────────────────────
    print("─── 6. score_technical ──────────────────────────────────")
    # _show('technical', scorer.score_technical(hist_df))

    # ── 7. 热榜排名 ──────────────────────────────────────────────
    print("─── 7. score_hot_rank ───────────────────────────────────")
    _show('hot_rank', scorer.score_hot_rank(TEST_CODE))

    # ── 8. 封板质量（仅涨停股有效，非涨停返回 0）────────────────
    print("─── 8. score_seal_quality ───────────────────────────────")
    # _show('seal_quality', scorer.score_seal_quality(TEST_CODE, hist_df))

    # ── 9. 板块强度 ──────────────────────────────────────────────
    print("─── 9. score_sector_strength ────────────────────────────")
    # _show('sector_strength', scorer.score_sector_strength(TEST_CODE, sector_name))

    # ── 10. 连板层级（仅涨停股有效，非涨停返回 0）───────────────
    print("─── 10. score_consec_limit ──────────────────────────────")
    # _show('consec_limit', scorer.score_consec_limit(TEST_CODE, hist_df))

    # ── 汇总 ─────────────────────────────────────────────────────
    print(f"{'='*64}")
    print("  各维度得分汇总（完整评分请用 scorer.score_stock(code)）")
    print(f"{'='*64}")
    # scorer.score_list([TEST_CODE])   # 取消注释可跑完整批量评分