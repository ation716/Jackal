# -*- coding: utf-8 -*-
# @Time    : 2026/8/13 15:51
# @Author  : gaolei
# @FileName: get_big_fund.py
# @Software: PyCharm

import re
import html
import json
import datetime
import math
from collections import defaultdict
from html.parser import HTMLParser
from urllib import parse, request


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._table_depth = 0
        self._current_rows = None
        self._current_row = None
        self._current_cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == 'table':
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_rows = []
        elif tag == 'tr' and self._table_depth:
            self._current_row = []
        elif tag in ('th', 'td') and self._table_depth and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_entityref(self, name):
        if self._current_cell is not None:
            self._current_cell.append(html.unescape(f'&{name};'))

    def handle_charref(self, name):
        if self._current_cell is not None:
            self._current_cell.append(html.unescape(f'&#{name};'))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ('th', 'td') and self._current_cell is not None:
            text = ''.join(self._current_cell)
            text = re.sub(r'\s+', ' ', text).strip()
            self._current_row.append(text)
            self._current_cell = None
        elif tag == 'tr' and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self._current_rows.append(self._current_row)
            self._current_row = None
        elif tag == 'table' and self._table_depth:
            if self._table_depth == 1 and self._current_rows is not None:
                self.tables.append(self._current_rows)
                self._current_rows = None
            self._table_depth -= 1


class SinaBigFundCrawler:
    PRICE_HISTORY_URL = 'https://market.finance.sina.com.cn/iframe/pricehis.php'
    PRICE_HISTORY_FALLBACK_PAGE_SIZE = 500
    BIG_BILL_API_URL = (
        'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php'
    )
    BIG_BILL_KIND_MAP = {
        'U': '买盘',
        'D': '卖盘',
        'E': '中性盘',
    }

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Referer": "https://market.finance.sina.com.cn/",
        }

    @staticmethod
    def normalize_symbol(stock_code: str) -> str:
        stock_code = str(stock_code).strip().lower()
        if stock_code.startswith(('sh', 'sz')):
            return stock_code
        code = stock_code.zfill(6)
        if code.startswith(('6', '9')):
            return f'sh{code}'
        return f'sz{code}'

    @staticmethod
    def _parse_date(value: str) -> datetime.date:
        return datetime.datetime.strptime(str(value), '%Y-%m-%d').date()

    @classmethod
    def _iter_dates(cls, startdate: str, enddate: str):
        start = cls._parse_date(startdate)
        end = cls._parse_date(enddate)
        if end < start:
            raise ValueError('enddate 不能早于 date')

        current = start
        while current <= end:
            yield current.strftime('%Y-%m-%d')
            current += datetime.timedelta(days=1)

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(str(value).replace(',', '').strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _format_number(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f'{value:.2f}'

    def build_price_history_url(self, stock_code: str, startdate: str, enddate: str) -> str:
        params = parse.urlencode({
            'symbol': self.normalize_symbol(stock_code),
            'startdate': startdate,
            'enddate': enddate,
        })
        return f'{self.PRICE_HISTORY_URL}?{params}'

    def _build_big_bill_api_url(self, method: str, stock_code: str, date: str = None,
                                num: int = 60, page: int = 1,
                                sort: str = 'ticktime', asc: int = 0,
                                volume: int = 40000, amount: int = 0,
                                bill_type: int = 0, lots: int = None) -> str:
        if lots is not None:
            volume = int(lots) * 100

        params = {
            'symbol': self.normalize_symbol(stock_code),
            'num': int(num),
            'page': int(page),
            'sort': sort,
            'asc': int(asc),
            'volume': int(volume),
            'amount': int(amount),
            'type': int(bill_type),
        }
        if date:
            params['day'] = str(date)
        return f'{self.BIG_BILL_API_URL}/{method}?{parse.urlencode(params)}'

    def fetch_html(self, url: str) -> tuple[str, str]:
        headers = dict(self.headers)
        if 'vip.stock.finance.sina.com.cn' in url:
            headers['Referer'] = 'https://vip.stock.finance.sina.com.cn/'
        req = request.Request(url, headers=headers)
        with request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            match = re.search(r"charset=([\w-]+)", content_type, re.I)
            encoding = match.group(1) if match else "utf-8"
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                encoding = "gb18030"
                text = raw.decode(encoding, errors="replace")
            return text, resp.url

    def fetch_json(self, url: str):
        text, _ = self.fetch_html(url)
        text = text.strip()
        return json.loads(text) if text else None

    def parse_tables(self, text: str) -> list[list[list[str]]]:
        parser = _TableParser()
        parser.feed(text)
        return parser.tables

    def table_to_records(self, table: list[list[str]]) -> list[dict]:
        if not table:
            return []
        header = table[0]
        records = []
        for row in table[1:]:
            if len(row) != len(header):
                continue
            if row == header:
                continue
            records.append(dict(zip(header, row)))
        return records

    def find_price_history_table(self, tables: list[list[list[str]]]) -> list[list[str]]:
        fallback = []
        for table in tables:
            if not table:
                continue
            if len(table) > len(fallback):
                fallback = table
            header_text = ''.join(table[0])
            if '价格' in header_text and (
                    '成交' in header_text or '占比' in header_text or '买卖盘' in header_text):
                return table
        return fallback

    def build_price_history_from_bill_detail(
            self, stock_code: str, startdate: str, enddate: str) -> list[dict]:
        """在新浪历史分价页返回空表时，用逐笔成交按价格聚合分价表。"""
        price_volumes = defaultdict(float)

        for trade_date in self._iter_dates(startdate, enddate):
            count = self.query_big_bill_count(
                stock_code=stock_code,
                date=trade_date,
                volume=0,
            )
            if not count:
                continue

            page_count = int(math.ceil(count / self.PRICE_HISTORY_FALLBACK_PAGE_SIZE))
            for page in range(1, page_count + 1):
                rows = self.query_big_bill_detail(
                    stock_code=stock_code,
                    date=trade_date,
                    num=self.PRICE_HISTORY_FALLBACK_PAGE_SIZE,
                    page=page,
                    volume=0,
                )
                if not rows:
                    break

                for row in rows:
                    price = str(row.get('成交价') or '').strip()
                    if not price:
                        continue
                    volume_lots = self._to_float(row.get('成交量(手)'))
                    price_volumes[price] += volume_lots * 100

        total_volume = sum(price_volumes.values())
        if not total_volume:
            return []

        records = []
        for price, volume in sorted(
                price_volumes.items(),
                key=lambda item: self._to_float(item[0]),
                reverse=True):
            records.append({
                '成交价(元)': price,
                '成交量(股)': self._format_number(volume),
                '占比': f'{volume / total_volume * 100:.2f}%',
                '占比图': '',
            })
        return records

    def format_big_bill_records(self, rows: list[dict]) -> list[dict]:
        records = []
        for row in rows or []:
            price = float(row.get('price') or 0)
            volume = float(row.get('volume') or 0)
            records.append({
                '股票代码': row.get('symbol', ''),
                '名称': row.get('name', ''),
                '发生时间': row.get('ticktime', ''),
                '成交价': f'{price:.2f}',
                '成交量(手)': f'{volume / 100:.2f}',
                '成交额(万元)': f'{price * volume / 10000:.2f}',
                '买卖盘性质': self.BIG_BILL_KIND_MAP.get(
                    row.get('kind'), row.get('kind', '')
                ),
            })
        return records

    def query_history_price_table(self, stock_code: str, date: str, enddate: str = None) -> list[dict]:
        """查询历史分价表，返回表格记录。

        stock_code 支持 '002580'、'sz002580'、'sh600000'。
        date 为开始日期；enddate 为空时查询单日，非空时查询 date 到 enddate 的区间。
        """
        startdate = str(date)
        enddate = str(enddate or date)
        url = self.build_price_history_url(stock_code, startdate, enddate)
        text, _ = self.fetch_html(url)
        tables = self.parse_tables(text)
        table = self.find_price_history_table(tables)
        records = self.table_to_records(table)
        if records:
            return records
        return self.build_price_history_from_bill_detail(
            stock_code, startdate, enddate)

    def query_price_history(self, symbol: str, startdate: str, enddate: str) -> list[dict]:
        """兼容旧入口：查询历史分价表，返回表格记录。"""
        return self.query_history_price_table(symbol, startdate, enddate)

    def query_big_bill_detail(self, stock_code: str, date: str = None, num: int = 60,
                              page: int = 1, sort: str = 'ticktime', asc: int = 0,
                              volume: int = 40000, amount: int = 0,
                              bill_type: int = 0, lots: int = None) -> list[dict]:
        """查询大单明细，返回页面表格对应的记录。

        stock_code 支持 '002580'、'sz002580'、'sh600000'。
        date 支持 'YYYY-MM-DD'；为空时使用新浪页面默认日期。
        num 控制单页笔数；volume 控制大单股数阈值；lots 可用手数表达阈值。
        bill_type 使用新浪原始筛选类型：0 为全部，其他值跟页面筛选项保持一致。
        """
        url = self._build_big_bill_api_url(
            method='CN_Bill.GetBillList',
            stock_code=stock_code,
            date=date,
            num=num,
            page=page,
            sort=sort,
            asc=asc,
            volume=volume,
            amount=amount,
            bill_type=bill_type,
            lots=lots,
        )
        rows = self.fetch_json(url)
        return self.format_big_bill_records(rows)

    def query_big_bill_count(self, stock_code: str, date: str = None,
                             volume: int = 40000, amount: int = 0,
                             bill_type: int = 0, lots: int = None) -> int:
        """查询符合条件的大单总笔数。"""
        url = self._build_big_bill_api_url(
            method='CN_Bill.GetBillListCount',
            stock_code=stock_code,
            date=date,
            volume=volume,
            amount=amount,
            bill_type=bill_type,
            lots=lots,
        )
        data = self.fetch_json(url)
        return int(data or 0)

    def query_big_bill_sum(self, stock_code: str, date: str = None,
                           volume: int = 40000, amount: int = 0,
                           bill_type: int = 0, lots: int = None) -> dict:
        """查询符合条件的大单汇总信息。"""
        url = self._build_big_bill_api_url(
            method='CN_Bill.GetBillSum',
            stock_code=stock_code,
            date=date,
            volume=volume,
            amount=amount,
            bill_type=bill_type,
            lots=lots,
        )
        data = self.fetch_json(url)
        if isinstance(data, list):
            return data[0] if data else {}
        return data or {}

    def query_big_orders(self, stock_code: str, date: str = None, num: int = 60,
                         page: int = 1, volume: int = 40000,
                         lots: int = None) -> list[dict]:
        """大单明细查询的简化入口。"""
        return self.query_big_bill_detail(
            stock_code=stock_code,
            date=date,
            num=num,
            page=page,
            volume=volume,
            lots=lots,
        )
if __name__ == '__main__':
    fc=SinaBigFundCrawler()
    data=fc.query_history_price_table('600613','2026-08-14','2026-08-17')
    if data:
        import csv

        # 指定输出文件路径（可自行修改）
        output_file = 'history_price.csv'
        # 写入 CSV，使用 utf-8-sig 编码使 Excel 正常显示中文
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            # 从第一条记录获取所有字段名作为表头
            fieldnames = data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"数据已写入 {output_file}，共 {len(data)} 条记录。")
    else:
        print("未查询到数据，CSV 文件未生成。")