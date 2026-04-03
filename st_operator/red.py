# -*- coding: utf-8 -*-
# @Time    : 2026/3/17
# @Author  : gaolei
# @FileName: red.py
"""
Market breadth data collector (EastMoney legu API).

Polling schedule (market hours 09:30 – 15:00, lunch 11:30 – 13:00):
  Before 10:00 → every 2 minutes
  10:00 onwards → every 5 minutes

Each tick appends one row to results/RED/em{YYYY-MM-DD}.csv and
pushes a dict into RedMonitor.cache for in-process consumers.

CSV column order (matches ak.stock_market_activity_legu value column):
  0  rising        上涨
  1  limit_up      涨停
  2  true_limit_up 真实涨停
  3  st_limit_up   ST涨停
  4  falling       下跌
  5  limit_dn      跌停
  6  true_limit_dn 真实跌停
  7  st_limit_dn   ST跌停
  8  flat          平盘
  9  suspended     停牌
  10 sentiment     活跃度 (e.g. "13.44%")
  11 ts            统计日期 (datetime string)
"""

import asyncio
import datetime
import math
import os

import akshare as ak
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'RED')

# Market session windows
_OPEN   = datetime.time(9, 30)
_LUNCH1 = datetime.time(11, 30)
_LUNCH2 = datetime.time(13, 0)
_CLOSE  = datetime.time(15, 1, 8)
_CUTOFF = datetime.time(10, 0)   # threshold: 2-min → 5-min interval


class RedMonitor:
    """
    Async market-breadth monitor.

    Usage
    -----
    monitor = RedMonitor()
    asyncio.create_task(monitor.run())   # starts background polling

    monitor.cache   → list of dicts, each: {ts, rising, falling, sentiment}
    """

    def __init__(self):
        self.cache: list[dict] = []   # in-memory store for today's ticks
        self._running = False

    # ── public ──────────────────────────────────────────────────────

    async def run(self):
        """Background polling loop. Runs until market close or cancelled."""
        self._running = True
        today = datetime.date.today()
        filename = os.path.join(RESULTS_DIR, today.strftime("em%Y-%m-%d.csv"))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if not os.path.exists(filename):
            open(filename, 'w').close()

        next_run = self._first_next_run(today)
        if next_run is None:
            print("[RedMonitor] outside market hours, exiting")
            return

        print(f"[RedMonitor] started, next fetch at {next_run.strftime('%H:%M:%S')}")
        try:
            while self._running:
                now = datetime.datetime.now()
                end_dt = datetime.datetime.combine(today, _CLOSE)

                if now >= end_dt:
                    print("[RedMonitor] market closed")
                    break

                wait = (next_run - now).total_seconds()
                if wait > 0:
                    await asyncio.sleep(wait)

                now = datetime.datetime.now()
                if now >= end_dt:
                    print("[RedMonitor] market closed")
                    break

                # Fetch
                row = await asyncio.get_event_loop().run_in_executor(None, self._fetch)
                if row and row['ts'] >= next_run:
                    self._append_csv(filename, row)
                    self.cache.append(row)
                    print(f"[RedMonitor] {row['ts'].strftime('%H:%M')}  "
                          f"up={row['rising']}  dn={row['falling']}  "
                          f"sentiment={row['sentiment']}")

                # Advance next_run, skip lunch
                interval = self._interval(next_run)
                next_run += datetime.timedelta(minutes=interval)
                next_run = self._skip_lunch(next_run, today)

                if next_run > end_dt:
                    print("[RedMonitor] all ticks collected for today")
                    break

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    def get_plot_data(self) -> pd.DataFrame | None:
        """
        Return today's data as a DataFrame.
        Falls back to today's CSV if cache is empty.
        Columns: ts (datetime), rising (int), falling (int), sentiment (float).
        """
        if self.cache:
            return pd.DataFrame(self.cache)

        today = datetime.date.today()
        filename = os.path.join(RESULTS_DIR, today.strftime("em%Y-%m-%d.csv"))
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return None

        df = pd.read_csv(filename, header=None)
        if df.shape[1] < 12:
            return None

        result = pd.DataFrame({
            'ts':        pd.to_datetime(df.iloc[:, 11]),
            'rising':    pd.to_numeric(df.iloc[:, 0],  errors='coerce'),
            'falling':   pd.to_numeric(df.iloc[:, 4],  errors='coerce'),
            'sentiment': df.iloc[:, 10].astype(str).str.replace('%', '').pipe(
                             pd.to_numeric, errors='coerce'),
        })
        return result.dropna()

    # ── private ─────────────────────────────────────────────────────

    @staticmethod
    def _fetch() -> dict | None:
        """Call akshare and parse one tick."""
        try:
            df = ak.stock_market_activity_legu()
            vals = df['value'].tolist()
            ts_str = str(vals[11])
            return {
                'ts':        datetime.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S'),
                'rising':    float(vals[0]),
                'falling':   float(vals[4]),
                'sentiment': float(str(vals[10]).replace('%', '')),
            }
        except Exception as e:
            print(f"[RedMonitor] fetch error: {e}")
            return None

    @staticmethod
    def _append_csv(filename: str, row: dict):
        """Append one row to the CSV (raw values, no header)."""
        try:
            raw = ak.stock_market_activity_legu()
            first_col = raw.iloc[:, [1]].T
            first_col.to_csv(filename, mode='a', index=False, header=False)
        except Exception as e:
            print(f"[RedMonitor] csv write error: {e}")

    @staticmethod
    def _interval(dt: datetime.datetime) -> int:
        """Return fetch interval in minutes based on time of day."""
        return 2 if dt.time() < _CUTOFF else 5

    @staticmethod
    def _skip_lunch(dt: datetime.datetime, today: datetime.date) -> datetime.datetime:
        """If dt falls in lunch break, advance to 13:00."""
        lunch1 = datetime.datetime.combine(today, _LUNCH1)
        lunch2 = datetime.datetime.combine(today, _LUNCH2)
        if lunch1 < dt < lunch2:
            return lunch2
        return dt

    @staticmethod
    def _first_next_run(today: datetime.date) -> datetime.datetime | None:
        """Calculate the first upcoming fetch time from now."""
        now = datetime.datetime.now()
        start = datetime.datetime.combine(today, _OPEN)
        end   = datetime.datetime.combine(today, _CLOSE)

        if now >= end:
            return None
        if now < start:
            return start

        # Find the next scheduled tick from 09:30
        elapsed = (now - start).total_seconds() / 60
        # Use 2-min slots up to 10:00, then 5-min slots
        cutoff_min = (datetime.datetime.combine(today, _CUTOFF) - start).total_seconds() / 60

        if elapsed < cutoff_min:
            n = math.ceil(elapsed / 2)
            candidate = start + datetime.timedelta(minutes=n * 2)
        else:
            n = math.ceil((elapsed - cutoff_min) / 5)
            candidate = datetime.datetime.combine(today, _CUTOFF) + datetime.timedelta(minutes=n * 5)

        candidate = RedMonitor._skip_lunch(candidate, today)
        return candidate if candidate <= end else None


# ── standalone entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    async def _main():
        monitor = RedMonitor()
        await monitor.run()

    asyncio.run(_main())
