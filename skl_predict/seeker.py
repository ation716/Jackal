# Always leave yourself outs
# There's a lot more opportunities in the market than I was expecting.

import os
import time
import datetime
import threading
from collections import deque, defaultdict

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from basic_s import ChipDistributionAnalyzer


stock_groups = {
    '人形机器人': {
        '002031': '巨轮智能',
        '603278': '大业股份',
        '002050': '三花智控',
        '600580': '卧龙电驱',
        '000678': '襄阳轴承',
        '6003007': '顺景科技',
    },
    '电力储能': {
        '601991': '大唐发电',
        '600396': '华电辽能',
        '603618': '杭电股份',
        '600726': '华电能源',
        '601016': '节能风电',
        '601179': '中国西电',
        '000179': '甘肃能源',
    },
    '光纤': {
        '002491': '通鼎互联',
        '600869': '远东股份',
        '002281': '光迅科技',
        '600522': '中天科技',
        '002008': '大族激光',
        '002384': '东山精密',
        '600206': '有研新材',
    },
    '半导体': {
        '600379': '宝光股份',
        '002081': '金螳螂',
        '603738': '泰晶科技',
        '600770': '综艺股份',
        '603773': '沃格光电',
        '000066': '中国长城',
        '002156': '通富微电',
        '600172': '黄河旋风',
        '000062': '深圳华强',
    },
    '航天': {
        '002297': '博云新材',
        '002081': '金螳螂',
        '002361': '神剑股份',
        '002565': '顺灏股份',
        '002149': '西部材料',
        '002342': '巨力索具',
        '300265': '通光线缆',
        '603601': '再升科技',
    },
    '算力': {
        '603629': '利通电子',
        '002929': '润建股份',
        '000815': '美丽云',
        '600666': '奥瑞德',
        '000925': '众合科技',
        '601138': '工业富联',
        '600410': '华胜天成',
        '600589': '大位科技',
        '000636': '风华高科',
    },
}


# ------------------------- 工具 -------------------------

def _visible_len(s):
    """中文字符按 2 宽度计算"""
    return sum(2 if ord(c) > 127 else 1 for c in str(s))


def _pad_left(s, w):
    s = str(s)
    return s + ' ' * max(0, w - _visible_len(s))


def _fmt_row(fields, widths):
    return ' | '.join(_pad_left(f, w) for f, w in zip(fields, widths))


def _fmt_pct(x):
    try:
        return f'{float(x):.4f}%'
    except Exception:
        return '-'


def _to_float(x, default=0.0):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _code_to_group(code):
    for industry, members in stock_groups.items():
        if code in members:
            return industry, members[code]
    return None, None


def _is_st(name):
    return 'ST' in str(name).upper()


# ------------------------- 实时行情（统一走 analyzer.get_realtime_tick）-------------------------

def _all_group_codes():
    out = []
    for members in stock_groups.values():
        out.extend(members.keys())
    return out


def get_group_ticks(analyzer, codes=None):
    """
    批量获取实时行情（仅走 analyzer.get_realtime_tick）。
    tushare realtime_quote 支持逗号分隔多 ts_code。

    返回: {code6: {name, price, pre_close, pct, volume, b1v, a1v, b1p, a1p}}
    """
    if codes is None:
        codes = _all_group_codes()
    if not codes:
        return {}

    ts_codes = ','.join(analyzer.normal_ts_code(c) for c in codes)
    try:
        df = analyzer.get_realtime_tick(ts_codes)
    except Exception as e:
        print(f'[warn] get_realtime_tick failed: {e}')
        return {}
    if df is None or df.empty:
        return {}

    out = {}
    cols = {c.upper(): c for c in df.columns}  # 原始列名大小写可能不稳定

    def col(r, *names):
        for n in names:
            if n in cols:
                return r[cols[n]]
        return None

    for _, r in df.iterrows():
        ts_code = str(col(r, 'TS_CODE', 'CODE') or '').upper()
        code6 = ts_code.split('.')[0].zfill(6) if ts_code else ''
        if not code6:
            continue
        price = _to_float(col(r, 'PRICE'))
        pre_close = _to_float(col(r, 'PRE_CLOSE'))
        pct = (price - pre_close) / pre_close * 100 if pre_close > 0 else 0.0
        out[code6] = {
            'name': str(col(r, 'NAME') or ''),
            'price': price,
            'pre_close': pre_close,
            'pct': pct,
            'volume': _to_float(col(r, 'VOLUME')),
            'b1v': _to_float(col(r, 'B1_V', 'BID_VOL1')),
            'a1v': _to_float(col(r, 'A1_V', 'ASK_VOL1')),
            'b1p': _to_float(col(r, 'B1_P', 'BID1')),
            'a1p': _to_float(col(r, 'A1_P', 'ASK1')),
        }
    return out


def get_limit_up_today(analyzer, date_str):
    """今日涨停池"""
    try:
        df = analyzer.get_limit_up(date_str)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['代码'] = df['代码'].astype(str).str.zfill(6)
        return df
    except Exception:
        return pd.DataFrame()


# ------------------------- 涨停行（来自 get_limit_up）-------------------------

def _parse_seal_time(s):
    """
    将 get_limit_up 的「首次/最后封板时间」字段解析为当日秒数。
    akshare 常见为 6 位字符串 'HHMMSS'，也兼容 'HH:MM:SS' 与数值。
    无法解析返回 None。
    """
    if s is None:
        return None
    txt = str(s).strip()
    if not txt or txt.lower() == 'nan':
        return None
    txt = txt.replace(':', '').replace('-', '')
    # 形如 '92500' 补零
    if txt.isdigit():
        txt = txt.zfill(6)
        try:
            h, m, sec = int(txt[:2]), int(txt[2:4]), int(txt[4:6])
            return h * 3600 + m * 60 + sec
        except Exception:
            return None
    return None


def _fmt_seal_time(s):
    """打印用：'092500' → '09:25:00'"""
    if s is None:
        return '-'
    txt = str(s).strip().replace(':', '')
    if not txt.isdigit():
        return str(s)
    txt = txt.zfill(6)
    return f'{txt[:2]}:{txt[2:4]}:{txt[4:6]}'


def rows_from_limit_pool(zt_pool_df):
    """
    涨停池 DataFrame → 打印行。过滤 ST。
    封单量(手) = 封板资金 / 最新价 / 100
    包含 首次封板时间，便于一字板判定（first_seal_seconds < 9:30）
    """
    if zt_pool_df is None or zt_pool_df.empty:
        return []

    rows = []
    for _, r in zt_pool_df.iterrows():
        name = str(r.get('名称', ''))
        if _is_st(name):
            continue
        code = str(r.get('代码', '')).zfill(6)
        price = _to_float(r.get('最新价', 0))
        pct = _to_float(r.get('涨跌幅', 0))
        industry = r.get('所属行业', '-') or '-'
        seal_cap = _to_float(r.get('封板资金', 0))
        seal = int(seal_cap / max(price, 0.01) / 100) if price > 0 else 0
        try:
            broken = int(r.get('炸板次数', 0) or 0)
        except Exception:
            broken = 0
        first_seal_raw = r.get('首次封板时间', '')
        rows.append({
            'code': code, 'name': name, 'price': price, 'pct': pct,
            'seal': seal, 'industry': industry, 'broken': broken,
            'first_seal_raw': first_seal_raw,
            'first_seal_seconds': _parse_seal_time(first_seal_raw),
        })
    return rows


_OPEN_930_SECONDS = 9 * 3600 + 30 * 60


def is_yizi(row):
    """一字板判定：首次封板时间 < 09:30 且未炸板"""
    sec = row.get('first_seal_seconds')
    if sec is None:
        return False
    return sec < _OPEN_930_SECONDS and row.get('broken', 0) == 0


def print_limit_up_block(title, rows):
    """代码 | 价格 | 涨幅 | 封单量 | 首封 | 行业 | 名称"""
    widths = [8, 8, 12, 12, 10, 12, 10]
    print(f'--- {title} ---')
    print(_fmt_row(['代码', '价格', '涨幅', '封单量(手)', '首封', '行业', '名称'], widths))
    for r in rows:
        print(_fmt_row(
            [r['code'], f"{r['price']:.2f}", _fmt_pct(r['pct']),
             r['seal'], _fmt_seal_time(r.get('first_seal_raw')),
             r['industry'], r['name']],
            widths,
        ))

    industry_cnt = defaultdict(int)
    for r in rows:
        industry_cnt[r['industry']] += 1
    parts = ', '.join(f'{k}:{v}' for k, v in sorted(industry_cnt.items(), key=lambda x: -x[1]))
    print(f'涨停总数: {len(rows)}  行业分布: {parts}')


# ------------------------- stock_groups 输出 -------------------------

def print_groups_bidding(ticks):
    """9:20-9:30：按行业打印买一/卖一"""
    widths = [8, 8, 12, 14, 14]
    for industry, members in stock_groups.items():
        print(f'[{industry}]')
        print(_fmt_row(['代码', '价格', '涨幅', '买一(手)', '卖一(手)'], widths))
        for code in members:
            t = ticks.get(code, {})
            price = t.get('price', 0)
            pct = t.get('pct', 0)
            b1v = t.get('b1v', 0)
            a1v = t.get('a1v', 0)
            price_s = f'{price:.2f}' if price else '-'
            print(_fmt_row(
                [code, price_s, _fmt_pct(pct), f'{b1v:.0f}', f'{a1v:.0f}'],
                widths,
            ))


def print_groups_trading(ticks, vol_state):
    """9:30 之后：stock_groups 全部，返回涨幅最高股用作最高标"""
    widths = [8, 8, 12, 10, 12, 10]
    print(_fmt_row(['代码', '价格', '涨幅', '买一量', '行业', '名称'], widths))

    top = None
    for industry, members in stock_groups.items():
        for code, name in members.items():
            t = ticks.get(code, {})
            price = t.get('price', 0)
            pct = t.get('pct', 0)
            b1v = t.get('b1v', 0)
            price_s = f'{price:.2f}' if price else '-'
            b1v_s = f'{b1v:.0f}' if b1v else '-'

            line = _fmt_row([code, price_s, _fmt_pct(pct), b1v_s, industry, name], widths)

            # 放量标记
            vs = vol_state.get(code)
            if vs and vs['avg'] > 0 and vs['last_delta'] is not None:
                ratio = vs['last_delta'] / vs['avg']
                if ratio > 3:
                    line += f'  放量({ratio:.2f}x)'

            print(line)

            if top is None or pct > top['pct']:
                top = {'code': code, 'price': price, 'pct': pct, 'name': name,
                       'b1v': b1v}
    return top


def print_top_and_yizi(yizi_codes, top_stock):
    """最高标（价格/盘口从 ticks）+ 一字板（全量来自 get_limit_up）"""
    if top_stock is not None:
        widths = [8, 8, 12, 12, 10]
        print('--- 最高标 ---')
        print(_fmt_row(['代码', '价格', '涨幅', '封单量(手)', '名称'], widths))
        seal = int(top_stock.get('b1v', 0) or 0)
        price_s = f"{top_stock['price']:.2f}" if top_stock['price'] else '-'
        print(_fmt_row(
            [top_stock['code'], price_s, _fmt_pct(top_stock['pct']),
             seal, top_stock['name']],
            widths,
        ))

    if yizi_codes:
        widths = [8, 8, 12, 12, 10, 12, 10]
        print('--- 一字板 ---')
        print(_fmt_row(['代码', '价格', '涨幅', '封单量(手)', '首封', '行业', '名称'], widths))
        for code, payload in yizi_codes.items():
            price_s = f"{payload.get('price', 0):.2f}"
            print(_fmt_row(
                [code, price_s, _fmt_pct(payload.get('pct', 0)),
                 payload.get('seal', 0),
                 _fmt_seal_time(payload.get('first_seal_raw')),
                 payload.get('industry', '-'),
                 payload.get('name', '')],
                widths,
            ))


# ------------------------- 绘图（独立线程） -------------------------

_plot_lock = threading.Lock()


def _plot_industry_job(snapshot, out_dir, date_str):
    """snapshot: {industry: {code: [(ts_label, price, pct), ...]}}  y 轴为涨跌幅(%)"""
    try:
        os.makedirs(out_dir, exist_ok=True)
        for industry, series_map in snapshot.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            max_len = 0
            time_labels = []
            for code, points in series_map.items():
                if not points:
                    continue
                # 兼容旧记录 (ts, price) 与新记录 (ts, price, pct)
                ys = [p[2] if len(p) >= 3 else 0.0 for p in points]
                _, name = _code_to_group(code)
                ax.plot(range(len(ys)), ys, label=f'{code} {name or ""}')
                if len(points) > max_len:
                    max_len = len(points)
                    time_labels = [p[0] for p in points]

            if max_len == 0:
                plt.close(fig)
                continue

            step = max(1, max_len // 8)
            ticks = list(range(0, max_len, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([time_labels[i] for i in ticks],
                               rotation=30, ha='right', fontsize=8)
            ax.set_title(industry)
            ax.set_xlabel('time')
            ax.set_ylabel('change %')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            fig.tight_layout()
            path = os.path.join(out_dir, f'{date_str}-{industry}.png')
            with _plot_lock:
                fig.savefig(path, dpi=100)
            plt.close(fig)
    except Exception as e:
        print(f'[warn] plot failed: {e}')


def schedule_plot(price_history, out_dir, date_str):
    snapshot = {
        industry: {code: list(dq) for code, dq in series.items()}
        for industry, series in price_history.items()
    }
    threading.Thread(
        target=_plot_industry_job, args=(snapshot, out_dir, date_str), daemon=True,
    ).start()


# ------------------------- 主循环 -------------------------

def in_bidding_before_920(now):
    return now.time() < datetime.time(9, 20)


def in_bidding_920_930(now):
    return datetime.time(9, 20) <= now.time() < datetime.time(9, 30)


def in_trading(now):
    return datetime.time(9, 30) <= now.time() <= datetime.time(15, 0)


def run_seeker(period_seconds=3):
    analyzer = ChipDistributionAnalyzer()

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'pict')
    )

    price_history = {
        industry: {code: deque(maxlen=2000) for code in members}
        for industry, members in stock_groups.items()
    }

    vol_state = {
        code: {'last_cum': None, 'last_delta': None, 'avg': 0.0, 't': 0}
        for members in stock_groups.values() for code in members
    }

    yizi_map = {}

    cycle = 0
    while True:
        cycle_start = time.time()
        now = datetime.datetime.now()
        date_str = now.strftime('%Y%m%d')
        date_label = now.strftime('%Y-%m-%d')
        time_label = now.strftime('%H:%M:%S')

        if now.time() > datetime.time(15, 0):
            print(f'[{time_label}] 收盘，退出')
            # break

        try:
            ticks = get_group_ticks(analyzer)
            zt_pool = get_limit_up_today(analyzer, date_str)
            zt_rows = rows_from_limit_pool(zt_pool)

            print(f'\n=========== cycle #{cycle}  {time_label} ===========')

            if in_bidding_before_920(now):
                # 1. 评分跳过
                # 2. 竞价涨停（全部来自 get_limit_up）
                print_limit_up_block('竞价涨停 (<=9:20)', zt_rows)

            elif in_bidding_920_930(now):
                # 3. 竞价涨停 + stock_groups 买一卖一
                print_limit_up_block('竞价涨停 (9:20-9:30)', zt_rows)
                print_groups_bidding(ticks)

            elif in_trading(now):
                # 一字板：按首次封板时间 < 09:30 且未炸板筛选（每周期重算，支持晚启动）
                yizi_map = {r['code']: r for r in zt_rows if is_yizi(r)}

                # 4. stock_groups 全部 + 放量标记
                top = print_groups_trading(ticks, vol_state)
                print_top_and_yizi(yizi_map, top)

            # ---- 采样 price_history（每周期）----
            for industry, members in stock_groups.items():
                for code in members:
                    t = ticks.get(code)
                    if not t:
                        continue
                    price = t.get('price')
                    if not price:
                        continue
                    pct = _to_float(t.get('pct', 0))
                    price_history[industry][code].append((time_label, float(price), pct))

            # ---- 放量统计：每两个周期（9:30 之后）----
            if in_trading(now) and cycle % 2 == 0:
                for code, vs in vol_state.items():
                    t = ticks.get(code)
                    if not t:
                        continue
                    cum_hands = _to_float(t.get('volume', 0)) / 100.0
                    if vs['last_cum'] is None:
                        vs['last_cum'] = cum_hands
                        continue
                    delta = max(0.0, cum_hands - vs['last_cum'])
                    vs['last_cum'] = cum_hands
                    vs['last_delta'] = delta
                    if vs['avg'] == 0:
                        vs['avg'] = delta
                        vs['t'] = 1
                    else:
                        t_cnt = min(vs['t'], 30)
                        vs['avg'] = (vs['avg'] * t_cnt + delta) / (t_cnt + 1)
                        vs['t'] = min(vs['t'] + 1, 30)

            # 6. 每 3 个周期画一次图
            if cycle > 0 and cycle % 3 == 0:
                schedule_plot(price_history, out_dir, date_label)

        except Exception as e:
            print(f'[error] cycle {cycle} failed: {e}')

        cycle += 1
        elapsed = time.time() - cycle_start
        time.sleep(max(0, period_seconds - elapsed))


if __name__ == '__main__':
    """
    证券检测系统（周期 3 秒）
      1. 9:20 前：输出竞价涨停（来自 get_limit_up，非 ST）并统计行业分布
      2. 9:20-9:30：竞价涨停 + stock_groups 各股买一/卖一
      3. 9:30 之后：stock_groups 全部、最高标、一字板；放量时附加 放量(xN)
    行情统一从 analyzer.get_realtime_tick 批量获取；涨停/一字板信息统一从 analyzer.get_limit_up 获取。
    每 3 个周期独立线程按行业绘图到 data/pict/y-m-d-industry.png
    """


    run_seeker(period_seconds=3)
