# -*- coding: utf-8 -*-
# @Time    : 2026/4/2 11:50
# @Author  : gaolei
# @FileName: big_fund.py
# @Software: PyCharm
"""Large-order report built on SinaBigFundCrawler."""

import csv
import datetime
import logging
import os
from collections import defaultdict

try:
    from st_operator.get_big_fund import SinaBigFundCrawler
except ModuleNotFoundError:
    from get_big_fund import SinaBigFundCrawler


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'big_fund')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
DEFAULT_PLOT_PATH = os.path.join(PLOT_DIR, 'big_orders.png')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
PAGE_SIZE = 60
P90_EXCLUDE_TIMES = {'09:25:00', '09:30:00', '15:00:00'}

PRICE_TIERS = [
    (6, 5000, 'Tier-1(<6)'),
    (12, 3000, 'Tier-2(6-12)'),
    (24, 1500, 'Tier-3(12-24)'),
    (48, 800, 'Tier-4(24-48)'),
    (float('inf'), 500, 'Tier-5(>=48)'),
]

CSV_COLUMNS = [
    'time',
    'change_pct',
    'price',
    'volume_lots',
    'order_type',
]


def _get_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    log_path = os.path.abspath(os.path.join(LOG_DIR, f'big_fund-{today}.log'))
    logger = logging.getLogger('big_fund')
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == log_path
        for handler in logger.handlers
    )
    if not exists:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    return logger


def _emit(message: str) -> None:
    print(message)
    _get_logger().info(message)


def _stock_code(ts_code: str) -> str:
    return str(ts_code).split('.')[0].strip()


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(',', '').strip())
    except (TypeError, ValueError):
        return default


def _tier_for(price: float):
    for upper, min_lots, label in PRICE_TIERS:
        if price < upper:
            return min_lots, label
    return PRICE_TIERS[-1][1], PRICE_TIERS[-1][2]


def _format_order(record: dict) -> dict:
    price = _to_float(record.get('成交价'))
    volume = _to_float(record.get('成交量(手)'))
    _, tier_label = _tier_for(price)
    side = record.get('买卖盘性质') or 'Unknown'
    return {
        'time': record.get('发生时间', ''),
        'change_pct': 'N/A',
        'price': f'{price:.2f}',
        'volume_lots': f'{volume:.2f}',
        'amount_wan': record.get('成交额(万元)', ''),
        'side': side,
        'order_type': f'{tier_label}-{side}',
    }


def _filter_by_price_tier(records: list[dict]) -> list[dict]:
    big_orders = []
    for record in records:
        price = _to_float(record.get('成交价'))
        volume = _to_float(record.get('成交量(手)'))
        min_lots, _ = _tier_for(price)
        if volume >= min_lots:
            big_orders.append(record)
    return big_orders


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        key = (
            record.get('发生时间', ''),
            record.get('成交价', ''),
            record.get('成交量(手)', ''),
            record.get('买卖盘性质', ''),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _is_p90_detail(record: dict, p90_volume: float) -> bool:
    if record.get('发生时间') in P90_EXCLUDE_TIMES:
        return False
    return _to_float(record.get('成交量(手)')) > p90_volume


def _row_side(row: dict) -> str:
    side = row.get('side') or row.get('买卖盘性质') or ''
    if side:
        return side

    order_type = str(row.get('order_type', ''))
    if '买盘' in order_type:
        return '买盘'
    if '卖盘' in order_type:
        return '卖盘'
    return ''


def _row_price(row: dict) -> float:
    return _to_float(row.get('price') or row.get('成交价'))


def _row_volume_lots(row: dict) -> float:
    return _to_float(row.get('volume_lots') or row.get('成交量(手)'))


def _price_volume_distribution(rows: list[dict]) -> dict[str, dict[float, float]]:
    distribution = {
        '买盘': defaultdict(float),
        '卖盘': defaultdict(float),
    }
    for row in rows:
        side = _row_side(row)
        if side not in distribution:
            continue

        price = _row_price(row)
        volume = _row_volume_lots(row)
        if price <= 0 or volume <= 0:
            continue
        distribution[side][price] += volume
    return distribution


def _weighted_price_interval(
        price_volumes: dict[float, float], coverage: float = 0.9) -> dict:
    total_volume = sum(price_volumes.values())
    if total_volume <= 0:
        return {}

    lower_target = total_volume * (1 - coverage) / 2
    upper_target = total_volume * (1 + coverage) / 2
    cumulative = 0.0
    lower_price = None
    upper_price = None
    for price, volume in sorted(price_volumes.items()):
        cumulative += volume
        if lower_price is None and cumulative >= lower_target:
            lower_price = price
        if upper_price is None and cumulative >= upper_target:
            upper_price = price
            break

    amount = sum(price * volume for price, volume in price_volumes.items())
    return {
        'lower_price': lower_price,
        'upper_price': upper_price,
        'avg_price': amount / total_volume,
        'total_volume_lots': total_volume,
    }


def _print_price_weight_intervals(ts_code: str, rows: list[dict]) -> None:
    distribution = _price_volume_distribution(rows)
    for side in ('买盘', '卖盘'):
        interval = _weighted_price_interval(distribution[side])
        if not interval:
            _emit(f"{ts_code}: no {side} volume for 90% weighted price interval")
            continue

        _emit(
            f"{ts_code}: {side} 90% weighted price interval "
            f"{interval['lower_price']:.2f}-{interval['upper_price']:.2f}  "
            f"weighted avg {interval['avg_price']:.3f}  "
            f"total volume {interval['total_volume_lots']:.2f} lots"
        )


def _side_stats(records: list[dict], side: str) -> dict:
    side_records = [r for r in records if r.get('买卖盘性质') == side]
    volumes = [_to_float(r.get('成交量(手)')) for r in side_records]
    total_volume = sum(volumes)
    if not total_volume:
        return {
            'side': side,
            'count': 0,
            'avg_price': None,
            'avg_volume_lots': None,
            'p90_volume_lots': None,
            'p90_records': [],
        }

    amount = sum(
        _to_float(r.get('成交价')) * _to_float(r.get('成交量(手)'))
        for r in side_records
    )
    p90_volume = _percentile(volumes, 0.9)
    return {
        'side': side,
        'count': len(side_records),
        'avg_price': amount / total_volume,
        'avg_volume_lots': total_volume / len(side_records),
        'p90_volume_lots': p90_volume,
        'p90_records': [
            r for r in side_records
            if _is_p90_detail(r, p90_volume)
        ],
    }


def _print_side_stats(ts_code: str, stats: dict) -> None:
    side = stats['side']
    if not stats['count']:
        _emit(f"{ts_code}: no {side} large orders")
        return

    _emit(
        f"{ts_code}: {stats['count']} {side} large orders  "
        f"avg price {stats['avg_price']:.3f}  "
        f"avg order volume {stats['avg_volume_lots']:.2f} lots  "
        f"p90 volume {stats['p90_volume_lots']:.2f} lots"
    )

    if not stats['p90_records']:
        _emit(
            f"{ts_code}: no {side} orders above p90 volume "
            f"(excluded times: {', '.join(sorted(P90_EXCLUDE_TIMES))})"
        )
        return

    _emit(
        f"===== {ts_code} {side} orders above p90 volume "
        f"(excluded times: {', '.join(sorted(P90_EXCLUDE_TIMES))}) ====="
    )
    for record in stats['p90_records']:
        row = _format_order(record)
        _emit(
            f"{row['time']}  price={row['price']}  "
            f"volume={row['volume_lots']}  amount={row['amount_wan']}  "
            f"type={row['order_type']}"
        )


def _append_csv(filepath: str, rows: list[dict], buy_stats: dict, sell_stats: dict) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0

    with open(filepath, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        for prefix, stats in (('buy', buy_stats), ('sell', sell_stats)):
            if stats['avg_price'] is None:
                continue
            f.write(f"{prefix}_avg_price,{stats['avg_price']:.3f}\n")
            f.write(f"{prefix}_avg_volume_lots,{stats['avg_volume_lots']:.2f}\n")
            f.write(f"{prefix}_p90_volume_lots,{stats['p90_volume_lots']:.2f}\n")


def _plot_rows(rows: list[dict]):
    parsed = []
    for row in rows:
        time_text = str(row.get('time', '')).strip()
        try:
            tick_time = datetime.datetime.strptime(time_text, '%H:%M:%S')
        except ValueError:
            continue

        side = row.get('side', '')
        if not side:
            order_type = str(row.get('order_type', ''))
            if '买盘' in order_type:
                side = '买盘'
            elif '卖盘' in order_type:
                side = '卖盘'
            else:
                side = '中性盘'

        parsed.append({
            'time': tick_time,
            'time_text': time_text,
            'price': _to_float(row.get('price')),
            'volume_lots': _to_float(row.get('volume_lots')),
            'side': side,
        })

    parsed.sort(key=lambda item: item['time'])
    cumulative = 0.0
    for item in parsed:
        if item['side'] == '买盘':
            cumulative += item['volume_lots']
        elif item['side'] == '卖盘':
            cumulative -= item['volume_lots']
        item['net_volume_lots'] = cumulative
    return parsed


def plot_big_orders(rows: list[dict], ts_code: str = '', date=None,
                    save_path: str = None, show: bool = False) -> str:
    """绘制大单价格和累积净买量，返回图片路径。

    rows 使用 get_big_orders 的返回值。
    累积净买量单位为手：买盘加量，卖盘减量，中性盘不改变。
    """
    if not rows:
        raise ValueError('rows 不能为空')

    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            '绘图需要 matplotlib，请先安装：pip install matplotlib'
        ) from exc

    plot_data = _plot_rows(rows)
    if not plot_data:
        raise ValueError('rows 中没有可绘制的有效时间数据')

    stock_code = _stock_code(ts_code) if ts_code else 'unknown'
    plot_date = str(date or datetime.datetime.now().strftime('%Y-%m-%d'))
    if save_path is None:
        os.makedirs(PLOT_DIR, exist_ok=True)
        save_path = os.path.join(
            PLOT_DIR,
            f'{stock_code}{plot_date}_big_orders.png',
        )
    else:
        save_path = os.path.abspath(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    times = [item['time'] for item in plot_data]
    prices = [item['price'] for item in plot_data]
    net_volumes = [item['net_volume_lots'] for item in plot_data]

    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    fig, price_ax = plt.subplots(figsize=(15, 8))
    net_ax = price_ax.twinx()

    price_line = price_ax.plot(
        times, prices, color='tab:blue', linewidth=1.8, label='价格'
    )[0]
    net_line = net_ax.plot(
        times, net_volumes, color='tab:red', linewidth=1.8,
        label='累积净买量(手)'
    )[0]

    price_ax.set_xlabel('成交时间')
    price_ax.set_ylabel('价格', color='tab:blue')
    net_ax.set_ylabel('累积净买量(手)', color='tab:red')
    price_ax.tick_params(axis='y', labelcolor='tab:blue')
    net_ax.tick_params(axis='y', labelcolor='tab:red')
    price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    price_ax.grid(True, linestyle='--', alpha=0.25)
    price_ax.set_title(f'{stock_code} 大单价格与累积净买量')

    price_ax.legend(
        [price_line, net_line],
        [price_line.get_label(), net_line.get_label()],
        loc='best',
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_big_order_price_distribution(
        rows: list[dict], ts_code: str = '', date=None,
        save_path: str = None, show: bool = False) -> str:
    """绘制买卖盘按价格聚合的成交量分布，返回图片路径。"""
    if not rows:
        raise ValueError('rows 不能为空')

    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as exc:
        raise RuntimeError(
            '绘图需要 matplotlib，请先安装：pip install matplotlib'
        ) from exc

    distribution = _price_volume_distribution(rows)
    price_levels = sorted(
        set(distribution['买盘'].keys()) | set(distribution['卖盘'].keys())
    )
    if not price_levels:
        raise ValueError('rows 中没有可绘制的买盘/卖盘价格数据')

    stock_code = _stock_code(ts_code) if ts_code else 'unknown'
    plot_date = str(date or datetime.datetime.now().strftime('%Y-%m-%d'))
    if save_path is None:
        os.makedirs(PLOT_DIR, exist_ok=True)
        save_path = os.path.join(
            PLOT_DIR,
            f'{stock_code}{plot_date}_price_distribution.png',
        )
    else:
        save_path = os.path.abspath(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    diffs = [
        price_levels[index + 1] - price_levels[index]
        for index in range(len(price_levels) - 1)
    ]
    positive_diffs = [diff for diff in diffs if diff > 0]
    bar_width = min(positive_diffs) * 0.72 if positive_diffs else 0.01

    buy_volumes = [distribution['买盘'].get(price, 0.0) for price in price_levels]
    sell_volumes = [
        -distribution['卖盘'].get(price, 0.0) for price in price_levels
    ]

    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(
        price_levels, buy_volumes, width=bar_width, color='red',
        alpha=0.72, label='买盘成交量(手)'
    )
    ax.bar(
        price_levels, sell_volumes, width=bar_width, color='green',
        alpha=0.72, label='卖盘成交量(手)'
    )
    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_xlabel('成交价格')
    ax.set_ylabel('成交量(手)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _pos: f'{abs(value):.0f}')
    )
    ax.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax.set_title(f'{stock_code} 买卖盘价格-数量分布')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def get_big_orders(ts_code: str, page_count: int = 60, date=None,
                   lots: int = None, plot: bool = False,
                   plot_path: str = DEFAULT_PLOT_PATH,
                   price_distribution_plot_path: str = None):
    """查询大单明细，写入 CSV，并计算买盘/卖盘加权平均价格。

    page_count 表示要拉取的页数，最多拉到第 60 页。
    lots 为空时沿用原价格分档阈值；传入后按固定手数阈值查询。
    plot 为 True 时，查询完成后同时生成价格/累积净买量图和价格-数量分布图。
    """
    crawler = SinaBigFundCrawler()
    stock_code = _stock_code(ts_code)
    query_lots = lots or PRICE_TIERS[-1][1]

    page_count = max(1, min(int(page_count or 1), 60))
    records = []
    for page in range(1, page_count + 1):
        page_rows = crawler.query_big_bill_detail(
            stock_code=stock_code,
            date=date,
            num=PAGE_SIZE,
            page=page,
            lots=query_lots,
        )
        if not page_rows:
            break
        records.extend(page_rows)

    records = _dedupe_records(records)

    if not records:
        _emit(f"{ts_code}: no large orders returned")
        return []

    big_orders = records if lots is not None else _filter_by_price_tier(records)
    if not big_orders:
        _emit(f"{ts_code}: no large orders found under price-tiered thresholds")
        return []

    rows = [_format_order(record) for record in big_orders]

    _emit(f"\n===== {ts_code} large orders =====")
    for row in rows:
        _emit(
            f"{row['time']}  price={row['price']}  "
            f"volume={row['volume_lots']}  amount={row['amount_wan']}  "
            f"type={row['order_type']}"
        )

    buy_stats = _side_stats(big_orders, '买盘')
    sell_stats = _side_stats(big_orders, '卖盘')

    filepath = os.path.join(RESULTS_DIR, f"{stock_code}{date or ''}.csv")
    _append_csv(filepath, rows, buy_stats, sell_stats)

    _print_side_stats(ts_code, buy_stats)
    _print_side_stats(ts_code, sell_stats)
    _print_price_weight_intervals(ts_code, rows)

    _emit(f"{ts_code}: {len(rows)} large orders total, appended to {filepath}")
    if plot:
        image_path = plot_big_orders(
            rows,
            ts_code=ts_code,
            date=date,
            save_path=plot_path,
        )
        _emit(f"{ts_code}: plot saved to {image_path}")
        distribution_image_path = plot_big_order_price_distribution(
            rows,
            ts_code=ts_code,
            date=date,
            save_path=price_distribution_plot_path,
        )
        _emit(
            f"{ts_code}: price distribution plot saved to "
            f"{distribution_image_path}"
        )
    return rows


if __name__ == '__main__':
    # get_big_orders('603330.SH', date='2026-08-19',lots=400,plot=True)
    get_big_orders('002580.SZ', date='2026-08-13',lots=400,plot=True)
