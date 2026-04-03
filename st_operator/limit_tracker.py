# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : gaolei
# @FileName: limit_tracker.py
"""
Tracks A-share main-board non-ST stocks over a configurable window of trading days.
Collects daily limit-up / broken-limit-up / limit-down counts, then measures
next-day closing performance for first-board and broken-limit-up events.

Typical usage from another module:
    from limit_tracker import LimitTracker
    tracker = LimitTracker(days=15)
    result  = tracker.run()          # returns structured dict
    df      = result['daily_stats']  # pd.DataFrame
    tracker.print_summary()          # formatted console output
"""

import akshare as ak
import pandas as pd
import datetime
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from security_data import ChipDistributionAnalyzer


# ─────────────────────────── Module-level helpers ────────────────────────────

def _is_main_board(code: str) -> bool:
    """Shanghai 60xxxx or Shenzhen 000/002/003xxxx; excludes STAR/ChiNext/BSE."""
    code = str(code).strip().zfill(6)
    if code.startswith(('688', '689', '300', '301')):
        return False
    if code[0] in ('8', '4'):
        return False
    return code[0] in ('6', '0')


def _is_non_st(name: str) -> bool:
    return 'ST' not in str(name).upper()


def _to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    return code + '.SH' if code.startswith('6') else code + '.SZ'


# ─────────────────────────── LimitTracker ────────────────────────────

class LimitTracker:
    """
    Analyzes limit-up / broken-limit-up / limit-down events on A-share main board.

    Attributes
    ----------
    days         : number of recent trading days to analyse (default 15)
    trading_days : list of YYYYMMDD strings, descending (set after run())
    daily_stats  : list of per-day stat dicts (set after run())
    first_zt_perfs : list of next-day pct_chg for first-board limit-ups
    crush_perfs    : list of next-day pct_chg for broken limit-ups
    """

    def __init__(self, days: int = 15):
        self.days = days
        self._analyzer: ChipDistributionAnalyzer | None = None
        self.trading_days: list = []
        self.daily_stats: list = []
        self.first_zt_perfs: list = []
        self.crush_perfs: list = []

    # ── lazy property ──────────────────────────────────────────────
    @property
    def analyzer(self) -> ChipDistributionAnalyzer:
        if self._analyzer is None:
            self._analyzer = ChipDistributionAnalyzer()
        return self._analyzer

    # ── public API ────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> dict:
        """
        Fetch all data and compute statistics.

        Parameters
        ----------
        verbose : print progress to stdout if True

        Returns
        -------
        dict with keys:
            'daily_stats'       : pd.DataFrame
            'first_board_perfs' : list[float]  (next-day pct_chg)
            'broken_perfs'      : list[float]
        """
        self.trading_days = self._get_trading_days(self.days)
        if len(self.trading_days) < 2:
            print("Insufficient trading day data, aborting")
            return {}

        if verbose:
            print("=" * 60)
            print("  A-Share Main Board (non-ST) — Limit Event Tracker")
            print("=" * 60)
            print(f"Period: {self.trading_days[-1]} ~ {self.trading_days[0]}\n")

        self.daily_stats = []
        first_zt_pairs: list = []
        crush_pairs: list = []

        for i, date in enumerate(self.trading_days):
            if verbose:
                print(f"[{date}]", end=' ', flush=True)

            lu, first_lu_codes = self._fetch_limit_up(date)
            crush = self._fetch_broken(date)
            ld = self._fetch_limit_down(date)

            lu_count    = len(lu)
            crush_count = len(crush)
            ld_count    = len(ld)
            total       = lu_count + crush_count
            crush_rate  = crush_count / total * 100 if total > 0 else 0.0

            self.daily_stats.append({
                'Date':       date,
                'LimitUp':    lu_count,
                'FirstBoard': len(first_lu_codes),
                'Broken':     crush_count,
                'BrokenRate': f"{crush_rate:.1f}%",
                'LimitDown':  ld_count,
            })

            if verbose:
                print(f"LimitUp={lu_count}(first={len(first_lu_codes)})  "
                      f"Broken={crush_count}({crush_rate:.1f}%)  "
                      f"LimitDown={ld_count}")

            # i=0 is the newest day — no next-day data yet, skip
            if i >= 1:
                for code in first_lu_codes:
                    first_zt_pairs.append((code, i))
                if not crush.empty and '代码' in crush.columns:
                    for code in crush['代码'].tolist():
                        crush_pairs.append((code, i))

        # Batch-fetch next-day performance
        if verbose:
            print("\nFetching next-day market data in batch...")
        all_codes = list({c for c, _ in first_zt_pairs + crush_pairs})
        perf_map = self._batch_fetch_daily(
            codes=all_codes,
            start_date=self.trading_days[-1],
            end_date=self.trading_days[0],
        )

        self.first_zt_perfs = self._extract_perfs(first_zt_pairs, perf_map)
        self.crush_perfs    = self._extract_perfs(crush_pairs,    perf_map)

        return {
            'daily_stats':       self.get_summary_df(),
            'first_board_perfs': self.first_zt_perfs,
            'broken_perfs':      self.crush_perfs,
        }

    def get_summary_df(self) -> pd.DataFrame:
        """Return daily stats as a DataFrame sorted ascending by date."""
        if not self.daily_stats:
            return pd.DataFrame()
        return (pd.DataFrame(self.daily_stats)
                .sort_values('Date')
                .reset_index(drop=True))

    def print_summary(self):
        """Print formatted summary table and performance distributions."""
        print("\n" + "=" * 60)
        print("  Daily Summary (ascending by date)")
        print("=" * 60)
        print(self.get_summary_df().to_string(index=False))

        self.print_perf_stats(
            self.first_zt_perfs,
            "First-Board Limit-Up -> Next-Day Close Performance"
        )
        self.print_perf_stats(
            self.crush_perfs,
            "Broken Limit-Up      -> Next-Day Close Performance"
        )
        print("\n" + "=" * 60)
        print("  Done")
        print("=" * 60)

    # ── static / class helpers ─────────────────────────────────────

    @staticmethod
    def print_perf_stats(perfs: list, title: str):
        """Print next-day performance statistics with a histogram distribution."""
        sep = '─' * 55
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        if not perfs:
            print("  No valid data (next-day close unavailable or data missing)")
            return
        s = pd.Series(perfs, dtype=float)
        print(f"  Sample size     : {len(s)}")
        print(f"  Avg change      : {s.mean():+.2f}%")
        print(f"  Median          : {s.median():+.2f}%")
        print(f"  Positive return : {(s > 0).mean() * 100:.1f}%")
        print(f"  Max gain        : {s.max():+.2f}%")
        print(f"  Max loss        : {s.min():+.2f}%")

        bins   = [-float('inf'), -5, -2, 0, 2, 5, float('inf')]
        labels = ['<-5%', '-5~-2%', '-2~0%', '0~2%', '2~5%', '>5%']
        cats   = pd.cut(s, bins=bins, labels=labels)
        dist   = cats.value_counts().reindex(labels).fillna(0).astype(int)
        print(f"\n  {'Range':<9} {'Count':>5}  {'Pct':>6}  Distribution")
        for label, cnt in dist.items():
            pct = cnt / len(s) * 100
            bar = '█' * int(pct / 4)
            print(f"  {label:<9} {cnt:>5}  {pct:>5.1f}%  {bar}")

    @staticmethod
    def filter_pool(df: pd.DataFrame,
                    code_col: str = '代码',
                    name_col: str = '名称') -> pd.DataFrame:
        """Filter a pool DataFrame to main-board non-ST stocks."""
        if df is None or df.empty:
            return pd.DataFrame()
        if code_col not in df.columns or name_col not in df.columns:
            print(f"  [Warning] Column mismatch, available: {df.columns.tolist()}")
            return pd.DataFrame()
        mask = df[code_col].apply(_is_main_board) & df[name_col].apply(_is_non_st)
        return df[mask].reset_index(drop=True)

    # ── private methods ────────────────────────────────────────────

    @staticmethod
    def _get_trading_days(n: int) -> list:
        try:
            df = ak.tool_trade_date_hist_sina()
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            today = datetime.date.today()
            df = df[df[date_col].dt.date <= today].sort_values(date_col, ascending=False)
            return df[date_col].dt.strftime('%Y%m%d').tolist()[:n]
        except Exception as e:
            print(f"  Failed to fetch trading days: {e}")
            return []

    @staticmethod
    def _safe_fetch(func, sleep: float = 1.5, **kwargs):
        try:
            result = func(**kwargs)
            time.sleep(sleep)
            return result
        except Exception as e:
            print(f"  [Failed] {func.__name__}: {e}")
            time.sleep(sleep * 2)
            return None

    def _fetch_limit_up(self, date: str):
        """Return (filtered_df, first_board_codes)."""
        raw = self._safe_fetch(ak.stock_zt_pool_em, date=date)
        lu  = self.filter_pool(raw)
        first_codes = []
        if not lu.empty and '连板数' in lu.columns:
            first_codes = lu[lu['连板数'] == 1]['代码'].tolist()
        return lu, first_codes

    def _fetch_broken(self, date: str) -> pd.DataFrame:
        raw = self._safe_fetch(ak.stock_zt_pool_zbgc_em, date=date)
        return self.filter_pool(raw)

    def _fetch_limit_down(self, date: str) -> pd.DataFrame:
        raw = self._safe_fetch(ak.stock_zt_pool_dtgc_em, date=date)
        return self.filter_pool(raw)

    def _batch_fetch_daily(self, codes: list,
                           start_date: str, end_date: str) -> dict:
        """
        Fetch pct_chg via tushare pro.daily, 25 stocks per batch.
        Returns {(6-digit code, 'YYYYMMDD'): pct_chg}.
        """
        if not codes:
            return {}
        perf_map   = {}
        ts_codes   = [_to_ts_code(c) for c in codes]
        batch_size = 25
        total      = (len(ts_codes) + batch_size - 1) // batch_size
        print(f"  {len(ts_codes)} stocks, {total} batches...", flush=True)

        for i in range(0, len(ts_codes), batch_size):
            batch    = ','.join(ts_codes[i:i + batch_size])
            batch_no = i // batch_size + 1
            for attempt in range(3):
                try:
                    df = self.analyzer.pro.daily(ts_code=batch,
                                                 start_date=start_date,
                                                 end_date=end_date)
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            c = row['ts_code'].split('.')[0]
                            perf_map[(c, row['trade_date'])] = row['pct_chg']
                    time.sleep(1.2)
                    break
                except Exception as e:
                    print(f"\n  Batch {batch_no} attempt {attempt + 1}/3 failed: {e}")
                    time.sleep(3 * (attempt + 1))
            else:
                print(f"  Batch {batch_no} all attempts failed, skipping")

        print(f"  Total records fetched: {len(perf_map)}")
        return perf_map

    def _extract_perfs(self, pairs: list, perf_map: dict) -> list:
        result = []
        for code, day_idx in pairs:
            next_date = self.trading_days[day_idx - 1]
            val = perf_map.get((code, next_date))
            if val is not None:
                result.append(float(val))
        return result


# ─────────────────────────── Public module-level aliases ────────────────────────────
# Expose helpers so other modules can do:
#   from limit_tracker import is_main_board, is_non_st, get_trading_days

is_main_board   = _is_main_board
is_non_st       = _is_non_st
get_trading_days = LimitTracker._get_trading_days


# ─────────────────────────── CLI entry point ────────────────────────────

if __name__ == '__main__':
    tracker = LimitTracker(days=15)
    tracker.run(verbose=True)
    tracker.print_summary()
