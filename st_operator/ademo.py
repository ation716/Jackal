# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : gaolei
# @FileName: ademo.py
"""
A股主板非ST股 近15个交易日统计
1. 每日涨停数量
2. 炸板数量与炸板率  （炸板率 = 炸板数 / (涨停数+炸板数)）
3. 跌停数量
4. 首版涨停后次日收盘表现（含平均涨幅、分布）
5. 炸板后次日收盘表现
"""

import akshare as ak
import pandas as pd
import datetime
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from security_data import ChipDistributionAnalyzer


# ─────────────────────────── 过滤工具 ────────────────────────────

def is_main_board(code: str) -> bool:
    """
    主板判断：沪市60xxxx、深市000/001/002/003xxxx
    排除科创板(688/689)、创业板(300/301)、北交所(8/4开头)
    """
    code = str(code).strip().zfill(6)
    if code.startswith(('688', '689', '300', '301')):
        return False
    if code[0] in ('8', '4'):
        return False
    return code[0] in ('6', '0')


def is_non_st(name: str) -> bool:
    return 'ST' not in str(name).upper()


def filter_pool(df: pd.DataFrame,
                code_col: str = '代码',
                name_col: str = '名称') -> pd.DataFrame:
    """过滤出主板非ST股"""
    if df is None or df.empty:
        return pd.DataFrame()
    if code_col not in df.columns or name_col not in df.columns:
        print(f"  [警告] 列名不匹配，可用列: {df.columns.tolist()}")
        return pd.DataFrame()
    mask = df[code_col].apply(is_main_board) & df[name_col].apply(is_non_st)
    return df[mask].reset_index(drop=True)


# ─────────────────────────── 数据获取工具 ────────────────────────────

def get_trading_days(n: int = 15) -> list:
    """获取最近 n 个交易日，YYYYMMDD 格式，降序（最新在前）"""
    try:
        df = ak.tool_trade_date_hist_sina()
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        today = datetime.date.today()
        df = df[df[date_col].dt.date <= today].sort_values(date_col, ascending=False)
        return df[date_col].dt.strftime('%Y%m%d').tolist()[:n]
    except Exception as e:
        print(f"  获取交易日失败: {e}")
        return []


def safe_fetch(func, sleep: float = 1.5, **kwargs):
    """带延迟与异常处理的 akshare 接口调用"""
    try:
        result = func(**kwargs)
        time.sleep(sleep)
        return result
    except Exception as e:
        print(f"  [失败] {func.__name__}: {e}")
        time.sleep(sleep * 2)
        return None


def to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    return code + '.SH' if code.startswith('6') else code + '.SZ'


def batch_fetch_daily(codes: list, start_date: str, end_date: str,
                      analyzer: ChipDistributionAnalyzer) -> dict:
    """
    批量获取日线数据（tushare pro.daily），每批 25 只
    返回 {(code_6位, 'YYYYMMDD'): pct_chg}
    """
    if not codes:
        return {}
    perf_map = {}
    ts_codes = [to_ts_code(c) for c in codes]
    batch_size = 25
    total_batches = (len(ts_codes) + batch_size - 1) // batch_size
    print(f"  共 {len(ts_codes)} 只股票，分 {total_batches} 批获取...", flush=True)

    for i in range(0, len(ts_codes), batch_size):
        batch = ','.join(ts_codes[i:i + batch_size])
        batch_no = i // batch_size + 1
        for attempt in range(3):
            try:
                df = analyzer.pro.daily(ts_code=batch,
                                        start_date=start_date,
                                        end_date=end_date)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        c = row['ts_code'].split('.')[0]
                        perf_map[(c, row['trade_date'])] = row['pct_chg']
                time.sleep(1.2)
                break
            except Exception as e:
                print(f"\n  批次 {batch_no} 第 {attempt + 1}/3 次失败: {e}")
                time.sleep(3 * (attempt + 1))
        else:
            print(f"  批次 {batch_no} 全部失败，跳过")

    print(f"  累计获得 {len(perf_map)} 条日线记录")
    return perf_map


# ─────────────────────────── 统计输出 ────────────────────────────

def print_perf_stats(perfs: list, title: str):
    """打印次日表现统计（含分布）"""
    sep = '─' * 55
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    if not perfs:
        print("  暂无有效数据（最近一日尚无次日收盘，或数据缺失）")
        return
    s = pd.Series(perfs, dtype=float)
    print(f"  样本数       : {len(s)}")
    print(f"  平均涨跌幅   : {s.mean():+.2f}%")
    print(f"  中位数       : {s.median():+.2f}%")
    print(f"  正收益占比   : {(s > 0).mean() * 100:.1f}%")
    print(f"  最大涨幅     : {s.max():+.2f}%")
    print(f"  最大跌幅     : {s.min():+.2f}%")

    bins = [-float('inf'), -5, -2, 0, 2, 5, float('inf')]
    labels = ['<-5%', '-5~-2%', '-2~0%', '0~2%', '2~5%', '>5%']
    cats = pd.cut(s, bins=bins, labels=labels)
    dist = cats.value_counts().reindex(labels).fillna(0).astype(int)
    print(f"\n  {'区间':<9} {'数量':>4}  {'占比':>6}  分布")
    for label, cnt in dist.items():
        pct = cnt / len(s) * 100
        bar = '█' * int(pct / 4)
        print(f"  {label:<9} {cnt:>4}  {pct:>5.1f}%  {bar}")


# ─────────────────────────── 主程序 ────────────────────────────

def main():
    print("=" * 60)
    print("  A股主板非ST股 — 近15个交易日涨停/炸板/跌停统计")
    print("=" * 60)

    analyzer = ChipDistributionAnalyzer()

    trading_days = get_trading_days(n=15)
    if len(trading_days) < 2:
        print("交易日数据不足，退出")
        return

    print(f"统计区间: {trading_days[-1]} ～ {trading_days[0]}\n")

    daily_stats = []
    first_zt_pairs = []   # [(code, day_idx), ...]  首版涨停
    crush_pairs = []      # [(code, day_idx), ...]  炸板
    # day_idx: 在 trading_days 中的位置，0=最新，1=前一日，以此类推

    for i, date in enumerate(trading_days):
        print(f"[{date}]", end=' ', flush=True)

        # 1. 涨停池
        lu_df = safe_fetch(ak.stock_zt_pool_em, date=date)
        lu = filter_pool(lu_df)
        lu_count = len(lu)

        # 首版涨停：连板数 == 1
        first_lu_codes = []
        if not lu.empty and '连板数' in lu.columns:
            first_lu_codes = lu[lu['连板数'] == 1]['代码'].tolist()

        # 2. 炸板池
        crush_df = safe_fetch(ak.stock_zt_pool_zbgc_em, date=date)
        crush = filter_pool(crush_df)
        crush_count = len(crush)

        total = lu_count + crush_count
        crush_rate = crush_count / total * 100 if total > 0 else 0.0

        # 3. 跌停池
        ld_df = safe_fetch(ak.stock_zt_pool_dtgc_em, date=date)
        ld = filter_pool(ld_df)
        ld_count = len(ld)

        daily_stats.append({
            '日期': date,
            '涨停数': lu_count,
            '首版涨停': len(first_lu_codes),
            '炸板数': crush_count,
            '炸板率': f"{crush_rate:.1f}%",
            '跌停数': ld_count,
        })

        print(f"涨停={lu_count}(首版{len(first_lu_codes)})  "
              f"炸板={crush_count}({crush_rate:.1f}%)  "
              f"跌停={ld_count}")

        # 收集次日对：i=0 为最新一天，无"次日"数据可查，跳过
        # i>=1 时，次日 = trading_days[i-1]（时间上更新的那天）
        if i >= 1:
            for code in first_lu_codes:
                first_zt_pairs.append((code, i))
            if not crush.empty and '代码' in crush.columns:
                for code in crush['代码'].tolist():
                    crush_pairs.append((code, i))

    # ── 批量获取次日数据 ──────────────────────────────
    print(f"\n正在批量获取次日行情数据...")
    all_need_codes = list({c for c, _ in first_zt_pairs + crush_pairs})
    perf_map = batch_fetch_daily(
        codes=all_need_codes,
        start_date=trading_days[-1],   # 最早的统计日（作为次日查询起点足够）
        end_date=trading_days[0],      # 最新日
        analyzer=analyzer
    )

    def extract_perfs(pairs: list) -> list:
        result = []
        for code, day_idx in pairs:
            next_date = trading_days[day_idx - 1]   # 次日（时间上更新的日期）
            val = perf_map.get((code, next_date))
            if val is not None:
                result.append(float(val))
        return result

    first_zt_perfs = extract_perfs(first_zt_pairs)
    crush_perfs = extract_perfs(crush_pairs)

    # ── 输出汇总 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  每日统计汇总（按日期升序）")
    print("=" * 60)
    stat_df = (pd.DataFrame(daily_stats)
               .sort_values('日期')
               .reset_index(drop=True))
    print(stat_df.to_string(index=False))

    print_perf_stats(first_zt_perfs, "首版涨停 → 次日收盘表现（基于涨停日收盘价）")
    print_perf_stats(crush_perfs,    "炸板    → 次日收盘表现（基于炸板日收盘价）")

    print("\n" + "=" * 60)
    print("  统计完毕")
    print("=" * 60)


if __name__ == '__main__':
    main()
