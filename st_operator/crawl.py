# -*- coding: utf-8 -*-
# @Time    : 2026/3/17 17:38
# @Author  : gaolei
# @FileName: craw.py
# @Software: PyCharm
import json
import datetime
import requests
import numpy as np
import pandas as pd

class Crawler(object):
    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────────────
    # Chip distribution (pure-Python CYQ + EastMoney direct API, no py_mini_racer)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_cyq(records, idx, cyq_range=120):
        """
        Pure-Python translation of the JS CYQCalculator inside akshare stock_cyq_em.
        records : list of dict with keys: open/close/high/low/hsl (turnover rate 0-100)
        idx     : target K-line index
        Returns dict or None.
        """
        factor = 150
        start = max(0, idx - cyq_range + 1)
        kdata = records[start: idx + 1]
        if not kdata:
            return None

        maxprice = max(d['high'] for d in kdata)
        minprice = min(d['low'] for d in kdata)
        accuracy = max(0.01, (maxprice - minprice) / (factor - 1))

        xdata = np.zeros(factor, dtype=np.float64)

        for eles in kdata:
            o, c, h, l = eles['open'], eles['close'], eles['high'], eles['low']
            avg = (o + c + h + l) / 4.0
            tr = min(1.0, (eles.get('hsl') or 0) / 100.0)

            # decay existing chips
            xdata *= (1.0 - tr)

            H = min(int((h - minprice) / accuracy), factor - 1)
            L = max(int(np.ceil((l - minprice) / accuracy)), 0)

            if h == l:
                g_idx = min(int((avg - minprice) / accuracy), factor - 1)
                xdata[g_idx] += (factor - 1) * tr / 2.0
            else:
                g_coef = 2.0 / (h - l)
                j_arr = np.arange(L, H + 1)
                cur_prices = minprice + accuracy * j_arr

                mask_low = cur_prices <= avg
                if mask_low.any():
                    denom = avg - l
                    if abs(denom) < 1e-8:
                        xdata[j_arr[mask_low]] += g_coef * tr
                    else:
                        xdata[j_arr[mask_low]] += (cur_prices[mask_low] - l) / denom * g_coef * tr

                mask_high = ~mask_low
                if mask_high.any():
                    denom = h - avg
                    if abs(denom) < 1e-8:
                        xdata[j_arr[mask_high]] += g_coef * tr
                    else:
                        xdata[j_arr[mask_high]] += (h - cur_prices[mask_high]) / denom * g_coef * tr

        current_price = records[idx]['close']
        total_chips = float(xdata.sum())
        if total_chips == 0:
            return None

        # Profit ratio: share of chips at or below the current price
        max_idx = min(int((current_price - minprice) / accuracy), factor - 1)
        benefit_part = float(xdata[:max_idx + 1].sum() / total_chips)

        # Cumulative chips for getCostByChip
        cum = np.cumsum(xdata)

        def get_cost_by_chip(target):
            s = 0.0
            for i in range(factor):
                s += xdata[i]
                if s > target:
                    return minprice + i * accuracy
            return minprice + (factor - 1) * accuracy

        avg_cost = round(get_cost_by_chip(total_chips * 0.5), 2)

        def compute_percent_chips(percent):
            pr0 = get_cost_by_chip(total_chips * (1 - percent) / 2)
            pr1 = get_cost_by_chip(total_chips * (1 + percent) / 2)
            conc = 0.0 if (pr0 + pr1) == 0 else (pr1 - pr0) / (pr0 + pr1)
            return {'priceRange': [round(pr0, 2), round(pr1, 2)], 'concentration': round(conc, 6)}

        return {
            'benefitPart': benefit_part,
            'avgCost': avg_cost,
            'percentChips': {
                '90': compute_percent_chips(0.9),
                '70': compute_percent_chips(0.7),
            }
        }

    def get_chip_em(self, code, lmt=210):
        """
        Call EastMoney push2his API with pure-Python CYQ, bypassing py_mini_racer.
        Returns a DataFrame in the same format as ak.stock_cyq_em:
        日期, 获利比例, 平均成本, 90成本-低, 90成本-高, 90集中度,
        70成本-低, 70成本-高, 70集中度
        """
        code = str(code).zfill(6)
        market_code = 1 if code.startswith('6') else 0
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"{market_code}.{code}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "0",
            "end": datetime.date.today().strftime("%Y%m%d"),
            "lmt": str(lmt),
            "cb": "quote_jp1",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://quote.eastmoney.com/",
        }
        r = requests.get(url, params=params, headers=headers, timeout=15)
        data_str = r.text.strip("quote_jp1(").strip(");")
        data_json = json.loads(data_str)
        klines_raw = data_json.get("data", {}).get("klines", [])
        if not klines_raw:
            return pd.DataFrame()

        records = []
        for item in klines_raw:
            p = item.split(',')
            records.append({
                'date': p[0],
                'open': float(p[1]), 'close': float(p[2]),
                'high': float(p[3]), 'low':   float(p[4]),
                'volume': float(p[5]), 'amount': float(p[6]),
                'zf': float(p[7]), 'zdf': float(p[8]),
                'zde': float(p[9]), 'hsl': float(p[10]),
            })

        rows = []
        for i in range(len(records)):
            res = self._calc_cyq(records, i)
            if res is None:
                continue
            rows.append({
                '日期':    records[i]['date'],
                '获利比例': res['benefitPart'],
                '平均成本': res['avgCost'],
                '90成本-低': res['percentChips']['90']['priceRange'][0],
                '90成本-高': res['percentChips']['90']['priceRange'][1],
                '90集中度':  res['percentChips']['90']['concentration'],
                '70成本-低': res['percentChips']['70']['priceRange'][0],
                '70成本-高': res['percentChips']['70']['priceRange'][1],
                '70集中度':  res['percentChips']['70']['concentration'],
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce').dt.date
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.iloc[-90:].reset_index(drop=True)

    def get_ths_hot_rank(self, list_type='normal', time_type='hour'):
        """
        Fetch THS (Tonghuashun) popularity ranking.

        Parameters
        ----------
        list_type : 'normal' (most viewed), 'skyrocket' (fast rising), 'tech' (technical),
                    'value' (value investing), 'trend' (trend following)
        time_type : 'hour' (hourly), 'day' (daily)
        """
        url = 'https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock'

        type_map = {'hour': 'hour', 'day': 'day'}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        params = {
            'stock_type': 'a',
            'type': type_map[time_type],
            'list_type': list_type,
        }

        try:
            res = requests.get(url, params=params, headers=headers)
            data = res.json()

            if data['status_code'] == 0:
                stock_list = data['data']['stock_list']
                df = pd.DataFrame(stock_list)

                # Extract concept and popularity tag fields
                def extract_tag_info(row):
                    if 'tag' in row and isinstance(row['tag'], dict):
                        return {
                            'concept_tag': ', '.join(row['tag'].get('concept_tag', [])),
                            'popularity_tag': row['tag'].get('popularity_tag', '')
                        }
                    return {'concept_tag': '', 'popularity_tag': ''}

                tag_df = df.apply(extract_tag_info, axis=1, result_type='expand')
                df = pd.concat([df, tag_df], axis=1)

                return df
            else:
                print(f"Request failed: {data}")
                return None
        except Exception as e:
            print(f"Exception: {e}")
            return None

    def get_stock_history_simple(self, symbol, scale=240, datalen=10000):
        """
        Fetch stock historical data from Sina Finance API.

        Parameters
        ----------
        symbol  : str  stock code with market prefix, e.g. 'sz000001'
        scale   : int  data granularity; 240 = daily, 60 = 60-minute bars
        datalen : int  number of bars to retrieve (max covers full history)

        Returns
        -------
        pandas.DataFrame with columns: date, open, high, close, low, volume
        """
        # Sina Finance K-line data endpoint
        url = "http://money.finance.sina.com.cn/quotes_service/api/jsonp_v2.php/var=/CN_MarketData.getKLineData"

        params = {
            'symbol': symbol,
            'scale': scale,    # 240 = daily bars
            'ma': 'no',        # omit moving-average data
            'datalen': datalen
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.encoding = 'utf-8'

            # Strip JSONP wrapper and extract the JSON array
            text = response.text
            json_str = text[text.find('['):text.rfind(']') + 1]

            data = eval(json_str)

            df = pd.DataFrame(data)
            df.rename(columns={
                'day': 'date',
                'open': 'open',
                'high': 'high',
                'close': 'close',
                'low': 'low',
                'volume': 'volume'
            }, inplace=True)

            # Type conversion
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'close', 'low']:
                df[col] = df[col].astype(float)
            df['volume'] = df['volume'].astype(float)

            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            print(f"Failed to fetch data: {e}")
            return None


if __name__ == '__main__':
    cc = Crawler()

    # Example usage:
    # df_hot = cc.get_ths_hot_rank(list_type='normal', time_type='hour')
    # print(df_hot[['code', 'name', 'rise_and_fall', 'rate', 'concept_tag']].head())

    # Fetch full daily history for a stock
    # df = cc.get_stock_history_simple('sz000001', datalen=100)
    # if df is not None:
    #     print(df.head())
    #     print(f"Total records: {len(df)}")
