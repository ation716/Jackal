# Always leave yourself outs
# There's a lot more opportunities in the market than I was expecting.

import os
import time
import datetime
import threading
import logging
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


PRICE_PRINT_INTERVAL = 60
LIMIT_COUNT_PRINT_INTERVAL = 60
LIMIT_SEAL_CHECK_INTERVAL = 5
BIG_MOVE_WINDOW_SECONDS = 5 * 60
BIG_MOVE_THRESHOLD_PCT = 2.0
POPUP_SECONDS = 5


stock_groups = {
    # '人形机器人': {
    #     '002031': {'name': '巨轮智能', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    #     '603278': {'name': '大业股份', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    #     '002050': {'name': '三花智控', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    #     '600580': {'name': '卧龙电驱', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    #     '000678': {'name': '襄阳轴承', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    #     '002896': {'name': '中大力德', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    #     '000559': {'name': '万向钱潮', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    # },
    '电力储能': {
        '601991': {'name': '大唐发电', 'reasonable_r': [10,18], 'overvalue_r': [18,30], 'imagine_r': []},
        '600396': {'name': '华电辽能', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
        '600863': {'name': '华能蒙电', 'reasonable_r': [3.32,4.13], 'overvalue_r': [], 'imagine_r': []},
        '600726': {'name': '华电能源', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
        '601016': {'name': '节能风电', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
        '001896': {'name': '豫能控股', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
        '600578': {'name': '京能电力', 'reasonable_r': [], 'overvalue_r': [], 'imagine_r': []},
    },
    # '光纤': {
    #     '002491': '通鼎互联',
    #     '600869': '远东股份',
    #     '002281': '光迅科技',
    #     '600522': '中天科技',
    #     '002008': '大族激光',
    #     '601869': '长飞光纤',
    #     '603618': '杭电股份',
    #     '002428':'云南锗业',
    # },
    # '半导体硅片': {
    #     '600379': '宝光股份',
    #     '002081': '金螳螂',
    #     '603738': '泰晶科技',
    #     '600770': '综艺股份',
    #     '603773': '沃格光电',
    #     '000066': '中国长城',
    #     '600172': '黄河旋风',
    #     '000062': '深圳华强',
    # },
    # '航天': {
    #     '002297': '博云新材',
    #     '002361': '神剑股份',
    #     '002565': '顺灏股份',
    #     '002149': '西部材料',
    #     '002342': '巨力索具',
    #     '603601': '再升科技',
    # },
    # '英伟达': {
    #     '603629': '利通电子',
    #     '002929': '润建股份',
    #     '000815': '美丽云',
    #     '600666': '奥瑞德',
    #     '601138': '工业富联',
    #     '600589': '大位科技',
    # },
    # 'MLCC':{
    #     '000636': '风华高科',
    #     '002552': '宝鼎科技',
    #     '002859': '洁美科技',
    #     '605376': '博迁新材',
    #     '002975': '博杰股份',
    # },
    # '液冷':{
    #     '002837':'英维克',
    #     '002272':'川润股份',
    #     '603757':'大元泵业',
    #     '000811':'冰轮环境',
    #     '603516':'淳中科技'
    # },
    # 'PCB':{
    #     '002579':'中京电子',
    #     '603890':'春秋电子',
    #     '002426':'胜利精密'
    # },
    # '煤炭':{
    #     '600403':'大有能源',
    #     '601666':'平煤发展',
    #     '601001':'晋控煤业',
    # },
    # 'pet 铜':{
    #     '002741':'光华科技',
    #     '002585':'双星新材',
    #     '601208':'东材科技',
    #     '600110':'诺德股份'
    # },
    # '培育钻石':{
    #     '002171':'楚江新材', # 精密铜，培育钻石 12.45-12.53
    #     '600172':'黄河旋风',
    #     '000519':'中兵红箭',
    #     '002046':'国机精工',
    # },
    # '医药创新药': {
    #     '000566': '海南海药',
    #     '603538': '美诺华',
    #     '603222': '济民健康',
    # },
    #
    # '氟化工': {
    #     '002407': '多氟多',
    # },
    # '玻璃基板': {
    #     '000725': '京东方A',
    #     '603773': '沃格光电'
    # },
    # "有色(涨价)":{
    #     '002167':'东方锆业',
    #     '000962':'东方钽业',
    #     '603407':'长裕集团',  # 钨
    #     '600397': '江钨装备',
    #     '002378': '章源钨业',
    #     '001257': '盛龙股份',
    #     '603993': '洛阳钼业',
    #     '601958': '金钼股份',
    #     '600378': '昊华科技',
    #     '600641':'先导基电',  # 铋
    #     '600206':'有研新材',
    #     '002674':'兴业科技',  # 磷化铟衬底，半导体材料
    # },
    # '长鑫科技': {
    #         '000021': '深科技',
    #         '603986': '兆易创新',
    #         '603650': '彤程新材',
    #         '002409': '雅克科技',
    #         '600667': '太极实业',
    #         '002156': '通富微电',
    #     },
    # 't 概念': {
    #     '002185': '华天科技',
    #     '002156': '通富微电',
    #     '600584': '长电科技'
    # },
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


def _stock_cfg(code):
    for industry, members in stock_groups.items():
        if code in members:
            payload = members[code]
            if isinstance(payload, dict):
                return industry, payload
            return industry, {
                'name': str(payload),
                'reasonable_r': [],
                'overvalue_r': [],
                'imagine_r': [],
            }
    return None, {}


def _stock_name(code, fallback=''):
    _, payload = _stock_cfg(code)
    return payload.get('name') or fallback or code


def _code_to_group(code):
    industry, payload = _stock_cfg(code)
    return industry, payload.get('name') if payload else None


def _is_st(name):
    return 'ST' in str(name).upper()


def _range_pair(values):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    low = _to_float(values[0], None)
    high = _to_float(values[1], None)
    if low is None or high is None:
        return None
    return (min(low, high), max(low, high))


def _fmt_range(values):
    pair = _range_pair(values)
    if pair is None:
        return '-'
    return f'{pair[0]:.2f}-{pair[1]:.2f}'


def _is_in_range(price, values):
    pair = _range_pair(values)
    if pair is None:
        return False
    low, high = pair
    return low <= price <= high


def _entered_range(price, prev_price, values, direction):
    pair = _range_pair(values)
    if pair is None or price <= 0:
        return False
    low, high = pair
    if not low <= price <= high:
        return False
    if prev_price is None:
        return True
    if direction == 'down':
        return prev_price > high
    if direction == 'up':
        return prev_price < low
    return not low <= prev_price <= high


def setup_logger():
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"seeker-{datetime.datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger('seeker')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    return logger


def popup_alert(title, message, seconds=POPUP_SECONDS):
    """非阻塞弹窗：确认立即关闭，未确认 5 秒后自动消失。"""
    def job():
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            win = tk.Toplevel(root)
            win.title(title)
            win.attributes('-topmost', True)

            label = tk.Label(
                win, text=message, padx=24, pady=16, justify='left',
                font=('Microsoft YaHei', 11),
            )
            label.pack()
            btn = tk.Button(win, text='确认', width=10, command=root.destroy)
            btn.pack(pady=(0, 16))
            win.protocol('WM_DELETE_WINDOW', root.destroy)

            win.update_idletasks()
            x = max(0, win.winfo_screenwidth() - win.winfo_width() - 40)
            y = max(0, win.winfo_screenheight() - win.winfo_height() - 80)
            win.geometry(f'+{x}+{y}')
            root.after(int(seconds * 1000), root.destroy)
            root.mainloop()
        except Exception as e:
            print(f'[warn] popup failed: {title} {message} ({e})')

    threading.Thread(target=job, daemon=True).start()


def emit_event(logger, title, message, popup=False):
    line = f'[{title}] {message}'
    print(line)
    logger.info('%s | %s', title, message)
    if popup:
        popup_alert(title, message)


# ------------------------- 实时行情（统一走 analyzer.get_realtime_tick）-------------------------

def _all_group_codes():
    out = []
    for members in stock_groups.values():
        out.extend(members.keys())
    return out


def _normal_realtime_code(code):
    code = str(code).zfill(6)
    if code.startswith(('60', '68')):
        return code + '.SH'
    if code.startswith(('00', '30')):
        return code + '.SZ'
    return code


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

    ts_codes = ','.join(_normal_realtime_code(c) for c in codes)
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


def get_limit_down_today(analyzer, date_str):
    """今日跌停池"""
    try:
        df = analyzer.get_limit_down(date_str)
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


def rows_from_limit_pool(zt_pool_df, side='up'):
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
        seal_cap = _to_float(
            r.get('封板资金', r.get('封单资金', r.get('跌停封单资金', 0)))
        )
        seal = int(seal_cap / max(price, 0.01) / 100) if price > 0 else 0
        try:
            broken = int(r.get('炸板次数', 0) or 0)
        except Exception:
            broken = 0
        first_seal_raw = r.get('首次封板时间', '')
        rows.append({
            'code': code, 'name': name, 'price': price, 'pct': pct,
            'seal': seal, 'industry': industry, 'broken': broken,
            'side': side,
            'seal_cap': seal_cap,
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
        for code, payload in members.items():
            name = payload.get('name', code) if isinstance(payload, dict) else str(payload)
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


# ------------------------- 事件检测 -------------------------

def check_valuation_events(ticks, prev_prices, range_alerted, logger):
    checks = [
        ('reasonable_r', '合理估值', '进入合理估值，建议买入', 'down'),
        ('overvalue_r', '高估', '进入高估区间', 'up'),
        ('imagine_r', '想象空间', '进入想象空间，建议卖出', 'up'),
    ]

    for code in _all_group_codes():
        t = ticks.get(code)
        if not t:
            continue
        price = _to_float(t.get('price', 0))
        if price <= 0:
            continue

        _, payload = _stock_cfg(code)
        name = payload.get('name') or t.get('name') or code
        prev_price = prev_prices.get(code)

        for field, title, action, direction in checks:
            key = (code, field)
            if key in range_alerted:
                continue
            values = payload.get(field, [])
            if _entered_range(price, prev_price, values, direction):
                msg = f'{name}({code}) 当前价 {price:.2f} {action}，区间 {_fmt_range(values)}'
                emit_event(logger, title, msg, popup=True)
                range_alerted.add(key)


def update_big_move_state(ticks, move_history, big_move_active, now, logger):
    cutoff = now - datetime.timedelta(seconds=BIG_MOVE_WINDOW_SECONDS)

    for code in _all_group_codes():
        t = ticks.get(code)
        if not t:
            continue
        price = _to_float(t.get('price', 0))
        if price <= 0:
            continue
        volume_hands = _to_float(t.get('volume', 0)) / 100.0
        dq = move_history[code]
        dq.append((now, price, volume_hands))
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        if len(dq) < 2:
            continue

        start_time, start_price, start_volume = dq[0]
        prices = [p[1] for p in dq]
        low_price = min(prices)
        high_price = max(prices)
        amplitude = (high_price - low_price) / low_price * 100 if low_price else 0
        if amplitude > BIG_MOVE_THRESHOLD_PCT and code not in big_move_active:
            cur_price = dq[-1][1]
            cur_volume = dq[-1][2]
            diff = cur_price - start_price
            pct = diff / start_price * 100 if start_price else 0
            direction = '上涨' if diff >= 0 else '下跌'
            vol_delta = max(0.0, cur_volume - start_volume)
            name = _stock_name(code, t.get('name', ''))
            msg = (
                f'{name}({code}) 5分钟内大幅变动: {direction} '
                f'{diff:+.2f} ({pct:+.2f}%), 振幅 {amplitude:.2f}%, '
                f'区间成交量变化 {vol_delta:.0f}手 '
                f'[{start_time.strftime("%H:%M:%S")} - {now.strftime("%H:%M:%S")}]'
            )
            print(f'[大幅变动] {msg}')
            logger.info('大幅变动 | %s', msg)
            big_move_active.add(code)
        elif amplitude <= BIG_MOVE_THRESHOLD_PCT:
            big_move_active.discard(code)


def handle_limit_events(rows, side, seen, seal_state, logger, now_ts):
    side_name = '涨停' if side == 'up' else '跌停'
    for row in rows:
        code = row['code']
        name = row.get('name') or _stock_name(code)
        seal = _to_float(row.get('seal', 0))

        if code not in seen:
            seen.add(code)
            msg = (
                f'{name}({code}) 刚{side_name}: 当前价 {row.get("price", 0):.2f}, '
                f'涨幅 {row.get("pct", 0):+.2f}%, 封单 {seal:.0f}手'
            )
            emit_event(logger, side_name, msg, popup=True)

        key = (side, code)
        state = seal_state.get(key)
        if state is None:
            seal_state[key] = {'seal': seal, 'last_check': now_ts}
            continue
        if now_ts - state['last_check'] < LIMIT_SEAL_CHECK_INTERVAL:
            continue

        prev_seal = state['seal']
        diff = seal - prev_seal
        pct = diff / prev_seal * 100 if prev_seal else (100.0 if seal else 0.0)
        if prev_seal > 0 and abs(pct) > 5:
            msg = f'{name}({code}) {side_name}封单变化 {diff:+.0f}手 ({pct:+.2f}%), 当前 {seal:.0f}手'
            print(f'[封单变化] {msg}')
            logger.info('封单变化 | %s', msg)

        seal_state[key] = {'seal': seal, 'last_check': now_ts}


def print_price_summary(ticks, now):
    widths = [12, 12]
    print(f'\n[股价摘要] {now.strftime("%H:%M:%S")}')
    print(_fmt_row(['名称', '涨幅'], widths))
    for industry, members in stock_groups.items():
        print(f'[{industry}]')
        for code in members:
            t = ticks.get(code, {})
            name = _stock_name(code, t.get('name', ''))
            print(_fmt_row([name, _fmt_pct(t.get('pct', 0))], widths))


def print_limit_count(rows, now):
    industry_cnt = defaultdict(int)
    for row in rows:
        industry_cnt[row.get('industry', '未知') or '未知'] += 1

    print(f'\n[行业涨停] {now.strftime("%H:%M:%S")}')
    print(f'涨停总数: {len(rows)}')
    for industry, count in sorted(industry_cnt.items(), key=lambda x: -x[1]):
        print(f'  {industry:<10} {count}')


# ------------------------- 绘图（独立线程） -------------------------

_plot_lock = threading.Lock()


def _plot_industry_job(snapshot, out_dir, date_str):
    """snapshot: {industry: {code: [(ts_label, price, pct), ...]}}  y 轴为涨跌幅(%)

    午休断点：上一个点在 11:30 前、下一个点在 13:00 后时，在中间插一个 NaN，
    让 matplotlib 自动断线，避免一条横跨 1.5 小时的直线连接早盘尾和午后头。
    """
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
                ts = [p[0] for p in points]
                ys_raw = [p[2] if len(p) >= 3 else 0.0 for p in points]

                # 在午休断点处插 NaN
                ys = []
                ts_final = []
                for i, t in enumerate(ts):
                    if i > 0:
                        prev = _parse_seal_time(ts[i - 1])
                        cur = _parse_seal_time(t)
                        if prev is not None and cur is not None \
                                and prev < 11 * 3600 + 30 * 60 <= cur:
                            ys.append(float('nan'))
                            ts_final.append('')
                    ys.append(ys_raw[i])
                    ts_final.append(t)

                _, name = _code_to_group(code)
                ax.plot(range(len(ys)), ys, label=f'{code} {name or ""}')
                if len(ts_final) > max_len:
                    max_len = len(ts_final)
                    time_labels = ts_final

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


def in_lunch_break(now):
    """午间休市 11:30-13:00（含两端）"""
    return datetime.time(11, 30) <= now.time() < datetime.time(13, 0)


def run_seeker(period_seconds=3):
    analyzer = ChipDistributionAnalyzer()
    logger = setup_logger()

    prev_prices = {}
    range_alerted = set()
    move_history = {code: deque() for code in _all_group_codes()}
    big_move_active = set()
    limit_up_seen = set()
    limit_down_seen = set()
    seal_state = {}

    last_price_summary = 0.0
    last_limit_count = 0.0

    cycle = 0
    while True:
        cycle_start = time.time()
        now = datetime.datetime.now()
        date_str = now.strftime('%Y%m%d')
        time_label = now.strftime('%H:%M:%S')

        if now.time() > datetime.time(15, 0):
            print(f'[{time_label}] 收盘，退出')
            break

        try:
            ticks = get_group_ticks(analyzer)
            zt_rows = rows_from_limit_pool(get_limit_up_today(analyzer, date_str), side='up')
            dt_rows = rows_from_limit_pool(get_limit_down_today(analyzer, date_str), side='down')

            if not in_lunch_break(now):
                check_valuation_events(ticks, prev_prices, range_alerted, logger)
                update_big_move_state(ticks, move_history, big_move_active, now, logger)

            now_ts = time.time()
            handle_limit_events(zt_rows, 'up', limit_up_seen, seal_state, logger, now_ts)
            handle_limit_events(dt_rows, 'down', limit_down_seen, seal_state, logger, now_ts)

            if now_ts - last_price_summary >= PRICE_PRINT_INTERVAL:
                last_price_summary = now_ts
                print_price_summary(ticks, now)

            if now_ts - last_limit_count >= LIMIT_COUNT_PRINT_INTERVAL:
                last_limit_count = now_ts
                print_limit_count(zt_rows, now)

            for code, t in ticks.items():
                price = _to_float(t.get('price', 0))
                if price > 0:
                    prev_prices[code] = price

        except Exception as e:
            print(f'[error] cycle {cycle} failed: {e}')
            logger.exception('cycle %s failed', cycle)

        cycle += 1
        elapsed = time.time() - cycle_start
        time.sleep(max(0, period_seconds - elapsed))


if __name__ == '__main__':
    """
    证券检测系统（周期 3 秒）
      1. stock_groups 使用 name/reasonable_r/overvalue_r/imagine_r 配置。
      2. 首次进入合理估值、高估、想象空间区间时弹窗并写 logs/seeker-YYYY-MM-DD.log。
      3. 5 分钟内振幅超过 2% 时输出大幅变动、方向、变动值和区间成交量变化。
      4. 刚涨停/跌停弹窗并写日志；每 5 秒检查封单，变化超过 5% 时打印并写日志。
      5. 每分钟只输出一次 stock_groups 名称和涨幅。
      6. 每分钟按行业统计涨停数量，输出格式参考 st_operator/duanxian.py。
    """


    run_seeker(period_seconds=3)
