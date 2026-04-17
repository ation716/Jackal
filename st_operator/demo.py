# Always leave yourself outs
# There's a lot more opportunities in the market than I was expecting.

import datetime
import akshare as ak
import time
import csv
from collections import deque, defaultdict
from security_data import ChipDistributionAnalyzer
from limit_tracker import is_main_board, is_non_st, get_trading_days

"""
在原有的基础上，我想监控实时涨停的股票
只统计主板非ST
1.需要统计涨停股票的行业，输出各行业涨停数，在 9:40 之前，每 10 s输出一次，之后和平均成交量一起，每 120s输出一次
格式为
行业 涨停数
2.一旦涨停的股票，就写入缓存，持续跟踪，
3.在 9:40 之前，输出在 9:31 之前涨停的股票，输出格式（参考 scurity_data get_limit_up）
股票名 涨停时间 当前价 涨停统计 封板资金（万）
4.最高标开盘，在 9.50 之前，自动加入 important，
昨日最高标前三名和前日最高标前三名（剔除重合）
5. 统计炸板，每 5s 检查一次， 输出封单资金迅速减少的股票，与上次相比至少减少了 1/10
"""

# ── 输入格式 ──────────────────────────────────────────────────────────────────
# stock_groups = {
#     'important': ['600203', '600776'],
#     'normal':    ['603687', '002309'],
# }
# ─────────────────────────────────────────────────────────────────────────────

stock_groups = {
    # 'important': ['600250', '000978','600754'],
    'important': ['000678'],
    'normal':    [],
}

VOLUME_CHECK_INTERVAL      = 15   # 秒：成交量检测周期
AVG_VOLUME_PRINT_INTERVAL  = 120  # 秒：平均每分成交量输出周期
PRICE_PRINT_INTERVAL       = 2    # 秒：价格输出周期
SURGE_THRESHOLD            = 2.5  # 成交量突增倍数阈值
VOLUME_THRESHOLD           = 100  # 最小成交量阈值（手）
SPOT_SCAN_INTERVAL_EARLY   = 10   # 9:40前全市场扫描间隔（秒）
SPOT_SCAN_INTERVAL_LATE    = 30   # 9:40后全市场扫描间隔（秒）
ZHABAN_CHECK_INTERVAL      = 5    # 炸板检测间隔（秒）
LIMIT_PRINT_INTERVAL_EARLY = 10   # 9:40前行业涨停统计输出间隔（秒）


# ── 板块信息 ──────────────────────────────────────────────────────────────────
def _load_codes():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from config.codes import codes
    return codes

def _save_codes(codes: dict):
    lines = ['# 股票板块/名称配置\n',
             '# 格式: { code: { \'industry\': ind, \'name\': name } }\n',
             '# 运行时如果遇到未收录的股票，会自动写入\n\n',
             'codes = {\n']
    for k, v in codes.items():
        lines.append(f"    '{k}': {{'industry': '{v['industry']}', 'name': '{v['name']}'}},\n")
    lines.append('}\n')
    import os
    path = os.path.join(os.path.dirname(__file__), 'config', 'codes.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def ensure_code_info(code: str, codes: dict) -> dict:
    """如果 code 不在 codes 里，通过 akshare 查询并写入"""
    if code in codes:
        return codes
    try:
        info = ak.stock_individual_info_em(symbol=code)
        name = info.loc[info['item'] == '股票简称', 'value'].values[0] if '股票简称' in info['item'].values else code
        industry = info.loc[info['item'] == '行业', 'value'].values[0] if '行业' in info['item'].values else '未知'
    except Exception as e:
        code_t='SH'+code if code.startswith('6') else 'SZ'+code
        info =  ak.stock_individual_basic_info_xq(code_t)

        name, industry = code, '未知'
    codes[code] = {'industry': industry, 'name': name}
    _save_codes(codes)
    print(f"[codes] 新增 {code}: {name} / {industry}")
    return codes


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def mask_name(name: str) -> str:
    """只显示第一个字，其余用 * 填充"""
    if not name:
        return '*'
    return name[0] + '*' * (len(name) - 1)

def get_limit_pct(code: str) -> float:
    """返回涨跌停幅度（科创688/创业300/301为20%，主板10%）"""
    if code.startswith('688') or code.startswith('300') or code.startswith('301'):
        return 20.0
    return 10.0

def gen_symbols(id_list: list) -> str:
    parts = []
    for code in id_list:
        if code.startswith('60') or code.startswith('68'):
            parts.append(f'{code}.SH')
        else:
            parts.append(f'{code}.SZ')
    return ','.join(parts)

def popup_alert(title: str, msg: str):
    """Windows 弹窗告警"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x40 | 0x1000)
    except Exception:
        print(f"[ALERT] {title}: {msg}")


# ── 最高连板股 ────────────────────────────────────────────────────────────────
def get_top_board_codes(top_n: int = 3, codes: dict = None) -> list:
    """昨日+前日最高连板主板非ST股，去重后返回 code 列表，同时更新 codes"""
    days = get_trading_days(2)
    result, seen = [], set()
    for d in days:
        try:
            df = ak.stock_zt_pool_em(date=d)
            if df is None or df.empty:
                continue
            df = df[df['代码'].apply(is_main_board) & df['名称'].apply(is_non_st)]
            if df.empty:
                continue
            df = df.sort_values('连板数', ascending=False)
            for _, r in df.head(top_n).iterrows():
                code = str(r['代码']).zfill(6)
                if code not in seen:
                    seen.add(code)
                    result.append(code)
                    if codes is not None and code not in codes:
                        codes[code] = {'industry': r.get('所属行业', '未知'), 'name': r.get('名称', code)}
        except Exception as e:
            print(f"[top_boards] {d} 获取失败: {e}")
    return result


# ── 全市场涨停扫描 ────────────────────────────────────────────────────────────
def scan_limit_up(spot_df, limit_cache: dict, codes: dict,
                  now: datetime.datetime) -> tuple:
    """
    扫描涨停池，识别主板非ST涨停股，更新 limit_cache 和 codes。
    返回 (industry_counts, new_codes)
    """
    if spot_df is None or spot_df.empty:
        return {}, []

    industry_counts = defaultdict(int)
    new_codes = []

    for _, row in spot_df.iterrows():
        code = str(row.get('代码', '')).zfill(6)
        name = str(row.get('名称', ''))
        if not is_main_board(code) or not is_non_st(name):
            continue

        industry = codes.get(code, {}).get('industry') or row.get('所属行业', '未知')
        if code not in codes:
            codes[code] = {'industry': industry, 'name': name}

        industry_counts[industry] += 1

        if code not in limit_cache:
            try:
                price = float(row.get('最新价', 0))
            except (ValueError, TypeError):
                price = 0.0
            limit_cache[code] = {
                'name': name,
                'first_time': now,
                'industry': industry,
                'price': price,
                'stats': '',
                'last_limit_time': str(row.get('最后封板时间', '')),
            }
            new_codes.append(code)
        else:
            limit_cache[code]['last_limit_time'] = str(row.get('最后封板时间', ''))

    return dict(industry_counts), new_codes


# ── 主监控逻辑 ────────────────────────────────────────────────────────────────
def run(groups: dict):
    all_codes     = groups.get('important', []) + groups.get('normal', [])
    important_set = set(groups.get('important', []))

    codes = _load_codes()
    for code in all_codes:
        if code not in codes:
            codes = ensure_code_info(code, codes)

    analyzer = ChipDistributionAnalyzer()
    now   = datetime.datetime.now()
    today = now.date()
    filename = today.strftime("../results/history/hu%Y-%m-%d.cxv")

    # 时间边界
    middle_early = datetime.datetime.combine(today, datetime.time(9, 14))
    middle_com   = datetime.datetime.combine(today, datetime.time(9, 45))
    middle_start = datetime.datetime.combine(today, datetime.time(11, 30))
    middle_end   = datetime.datetime.combine(today, datetime.time(13, 0))
    t_931 = datetime.datetime.combine(today, datetime.time(9, 31))
    t_940 = datetime.datetime.combine(today, datetime.time(22, 40))
    t_950 = datetime.datetime.combine(today, datetime.time(9, 50))

    # 成交量状态
    vol_last = {c: 0   for c in all_codes}
    vol_dq   = {c: deque(maxlen=60) for c in all_codes}
    vol_avg  = {c: 0.0 for c in all_codes}

    # 涨停相关状态
    limit_cache      = {}          # code -> {name, first_time, industry, price, stats}
    prev_bid_capital = {}          # code -> 上次封单资金(万)，炸板检测用
    zhaban_cache     = set()       # 已炸板股票代码集合
    important_inited = False       # 是否已完成最高标自动加入
    tracked_codes    = list(all_codes)  # 动态扩展，含涨停股
    industry_counts  = {}          # 最新一次扫描结果
    spot_df          = None

    last_vol_check    = time.time()
    last_avg_print    = time.time()
    last_price_print  = time.time()
    last_spot_scan    = 0.0
    last_limit_print  = 0.0
    last_zhaban_check = 0.0

    # 启动时加载最高连板股
    print("[init] 加载最高标...")
    top_board_codes = get_top_board_codes(top_n=3, codes=codes)
    print(f"[init] 最高标候选: {top_board_codes}")

    with open(filename, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        while True:
            now = datetime.datetime.now()
            # 午休跳过
            if middle_start <= now < middle_end:
                sleep_sec = (middle_end - now).total_seconds()
                print(f"午休，等待 {sleep_sec:.0f}s ...")
                time.sleep(min(sleep_sec, 30))
                continue

            t_now = time.time()

            # ── 9:50前自动加入最高标 ──────────────────────────────────────────
            if not important_inited and now < t_950:
                for c in top_board_codes:
                    if c not in important_set:
                        important_set.add(c)
                        if c not in all_codes:
                            all_codes.append(c)
                        if c not in tracked_codes:
                            tracked_codes.append(c)
                            vol_last[c] = 0
                            vol_dq[c]   = deque(maxlen=60)
                            vol_avg[c]  = 0.0
                        name = codes.get(c, {}).get('name', c)
                        print(f"[important] 自动加入最高标: {name}({c})")
                important_inited = True

            # ── 全市场扫描（涨停识别）────────────────────────────────────────
            scan_interval = SPOT_SCAN_INTERVAL_EARLY if now < t_940 else SPOT_SCAN_INTERVAL_LATE
            if t_now - last_spot_scan >= scan_interval:
                last_spot_scan = t_now
                try:
                    date_str = today.strftime('%Y%m%d')
                    spot_df = ak.stock_zt_pool_em(date=date_str)
                    industry_counts, new_codes = scan_limit_up(spot_df, limit_cache, codes, now)
                    for c in new_codes:
                        if c not in tracked_codes:
                            tracked_codes.append(c)
                            vol_last[c] = 0
                            vol_dq[c]   = deque(maxlen=60)
                            vol_avg[c]  = 0.0
                        name = limit_cache[c]['name']
                        last_time = limit_cache[c].get('last_limit_time', '')
                        print(f"[ZT] {name:<4}|{c:6}  last_limit_up:{last_time}")
                except Exception as e:
                    print(f"[spot_scan] 失败: {e}")

            # 获取 tracked_codes 实时 tick
            symbols = gen_symbols(tracked_codes)
            df = analyzer.get_realtime_tick(ts_code=symbols)
            if df is None:
                print("can not get data")
                time.sleep(PRICE_PRINT_INTERVAL)
                continue

            # 建立 code -> row 映射
            row_map = {}
            for i in range(len(df)):
                raw  = df.iloc[i, 1]
                code = raw.split('.')[0]
                row_map[code] = df.iloc[i]

            # ── 每 2s 按板块输出价格 ──────────────────────────────────────────
            if t_now - last_price_print >= PRICE_PRINT_INTERVAL:
                last_price_print = t_now
                industry_map = defaultdict(list)
                for code in all_codes:
                    ind = codes.get(code, {}).get('industry', '未知')
                    industry_map[ind].append(code)

                print(f"\n{'='*40}  {now.strftime('%H:%M:%S')}")
                for ind, ind_codes in industry_map.items():
                    if now < middle_com:
                        print(f"{ind}:")
                    for code in ind_codes:
                        if code not in row_map:
                            continue
                        row   = row_map[code]
                        name  = codes.get(code, {}).get('name', code)
                        price = row.iloc[6]
                        pre   = row.iloc[5]
                        chg   = (price - pre) / pre * 100 if pre else 0
                        tag   = '-' if code in important_set else ' '

                        limit_pct   = get_limit_pct(code)
                        near_thresh = limit_pct * 0.95
                        at_thresh   = limit_pct * 0.99
                        if chg >= at_thresh:
                            bid_vol1  = row.iloc[13]
                            limit_tag = f"  [涨停封单:{bid_vol1:.0f}手]"
                        elif chg >= near_thresh:
                            bid_vol1  = row.iloc[13]
                            limit_tag = f"  (接近涨停 封单:{bid_vol1:.0f}手)"
                        elif chg <= -at_thresh:
                            ask_vol1  = row.iloc[23]
                            limit_tag = f"  [跌停封单:{ask_vol1:.0f}手]"
                        elif chg <= -near_thresh:
                            ask_vol1  = row.iloc[23]
                            limit_tag = f"  (接近跌停 封单:{ask_vol1:.0f}手)"
                        else:
                            limit_tag = ""
                        print(f"  {tag:<2} {mask_name(name):<4}  {price:<5.2f}  {chg:+.2f}{limit_tag}")

            # ── 行业涨停统计（9:40前每10s；9:40后随120s块）────────────────────
            if now < t_940 and t_now - last_limit_print >= LIMIT_PRINT_INTERVAL_EARLY:
                last_limit_print = t_now
                if industry_counts:
                    print(f"\n[行业涨停]  {now.strftime('%H:%M:%S')}")
                    for ind, cnt in sorted(industry_counts.items(), key=lambda x: -x[1]):
                        print(f"  {ind:<10} {cnt}")
                # 9:31前涨停的股票列表
                early_birds = {c: v for c, v in limit_cache.items()
                               if v['first_time'] < t_931}
                if early_birds:
                    print(f"\n[早盘涨停 <9:31]")
                    for code, info in early_birds.items():
                        if code in row_map:
                            price = row_map[code].iloc[6]
                            bid_v = row_map[code].iloc[14]
                            cap_wan = bid_v * price / 100
                        else:
                            price, cap_wan = info['price'], 0.0
                        stats = info.get('stats', '')
                        print(f"  {mask_name(info['name']):<4}  "
                              f"{info['first_time'].strftime('%H:%M:%S')}  "
                              f"{price:<5.2f}  {stats:<8}  {cap_wan:.0f}万")

            # ── 炸板检测（每5s）──────────────────────────────────────────────
            if t_now - last_zhaban_check >= ZHABAN_CHECK_INTERVAL:
                last_zhaban_check = t_now
                try:
                    date_str = today.strftime('%Y%m%d')
                    zhaban_df = ak.stock_zt_pool_zbgc_em(date=date_str)
                    if zhaban_df is not None and not zhaban_df.empty:
                        for _, row in zhaban_df.iterrows():
                            code = str(row.get('代码', '')).zfill(6)
                            name = str(row.get('名称', ''))
                            if not is_main_board(code) or not is_non_st(name):
                                continue
                            if code not in codes:
                                codes[code] = {'industry': row.get('所属行业', '未知'), 'name': name}
                            if code not in zhaban_cache:
                                zhaban_cache.add(code)
                                zb_time = row.get('炸板时间', '')
                                zb_count = row.get('炸板次数', 0)
                                chg = row.get('涨跌幅', 0)
                                print(f"\n[炸板] {mask_name(name)}({code})  "
                                      f"时间:{zb_time}  次数:{zb_count}  涨幅:{chg:.2f}")
                except Exception as e:
                    pass  # 炸板接口可能无数据，静默处理

            # ── 每 15s 检测成交量突增 ─────────────────────────────────────────
            if t_now - last_vol_check >= VOLUME_CHECK_INTERVAL:
                last_vol_check = t_now
                for code in tracked_codes:
                    if code not in row_map:
                        continue
                    row     = row_map[code]
                    vol_now = row.iloc[11]
                    delta   = vol_now - vol_last[code]
                    vol_last[code] = vol_now
                    vol_dq[code].append(delta)

                    if vol_avg[code] == 0:
                        if len(vol_dq[code]) >= 4:
                            vol_avg[code] = sum(vol_dq[code]) / len(vol_dq[code])
                    else:
                        vol_avg[code] = sum(vol_dq[code]) / len(vol_dq[code])
                        if delta > SURGE_THRESHOLD * vol_avg[code] and delta > VOLUME_THRESHOLD:
                            ratio = delta / vol_avg[code]
                            name  = codes.get(code, {}).get('name', code)
                            msg   = f"{mask_name(name)} 成交量突增 {ratio:.1f}x  (delta={delta:.0f}手)"
                            print(f"\n[量增] {msg}")
                            row = row_map[code]
                            buy_list = [row.iloc[j] for j in [13, 15, 17, 19, 21]]
                            sel_list = [row.iloc[j] for j in [23, 25, 27, 29, 31]]
                            writer.writerow([f'{code}', now.strftime('%Y-%m-%d %H:%M:%S'),
                                             row.iloc[6], *buy_list, *sel_list, f'surge {ratio:.2f}'])
                            f.flush()

            # ── 每 120s 输出平均每分成交量 + 行业涨停统计（9:40后）────────────
            if t_now - last_avg_print >= AVG_VOLUME_PRINT_INTERVAL:
                last_avg_print = t_now
                print(f"\n[平均每分成交量]  {now.strftime('%H:%M:%S')}")
                for code in tracked_codes:
                    if vol_avg[code] > 0:
                        name = codes.get(code, {}).get('name', code)
                        print(f"  {mask_name(name)}({code})  {vol_avg[code]*4:.0f} 手/分")
                # 9:40后与此块一起输出行业涨停统计
                if now >= t_940 and industry_counts:
                    print(f"\n[行业涨停]  {now.strftime('%H:%M:%S')}")
                    for ind, cnt in sorted(industry_counts.items(), key=lambda x: -x[1]):
                        print(f"  {ind:<10} {cnt}")

            if now < middle_early:
                time.sleep(10)
            time.sleep(PRICE_PRINT_INTERVAL)


if __name__ == '__main__':
    run(stock_groups)
