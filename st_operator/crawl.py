# -*- coding: utf-8 -*-
# @Time    : 2026/3/17 17:38
# @Author  : gaolei
# @FileName: craw.py
# @Software: PyCharm
import requests
import pandas as pd

class Crawler(object):
    def __init__(self):
        pass
    def get_ths_hot_rank(self,list_type='normal', time_type='hour'):
        """
        获取同花顺人气榜
        :param list_type: normal(大家都在看), skyrocket(快速飙升中), tech(技术交易派), value(价值投资派), trend(趋势投资派)
        :param time_type: hour(小时榜), day(日榜)
        """
        url = 'https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock'

        # 参数映射
        type_map = {'hour': 'hour', 'day': 'day'}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        params = {
            'stock_type': 'a',  # A股
            'type': type_map[time_type],  # 时间维度
            'list_type': list_type  # 榜单类型
        }

        try:
            res = requests.get(url, params=params, headers=headers)
            data = res.json()

            if data['status_code'] == 0:
                stock_list = data['data']['stock_list']
                df = pd.DataFrame(stock_list)

                # 提取标签信息（概念标签、人气标签）
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
                print(f"请求失败: {data}")
                return None
        except Exception as e:
            print(f"异常: {e}")
            return None

    def get_stock_history_simple(self, symbol, scale=240, datalen=10000):
        """
        从新浪财经API获取股票历史数据

        Parameters
        ----------
        symbol : str
            股票代码，需要包含市场前缀，如 'sz000001' (平安银行)
        scale : int
            数据粒度，240代表日K线，60代表60分钟线，默认240
        datalen : int
            获取的数据条数，最大可获取全部历史数据，默认10000

        Returns
        -------
        pandas.DataFrame
            包含日期、开盘、最高、收盘、最低、成交量的DataFrame
        """
        # 新浪财经K线数据接口
        url = f"http://money.finance.sina.com.cn/quotes_service/api/jsonp_v2.php/var=/CN_MarketData.getKLineData"

        params = {
            'symbol': symbol,
            'scale': scale,  # 240为日线
            'ma': 'no',  # 不返回均线数据
            'datalen': datalen  # 获取的数据条数
        }

        try:
            # 发送请求
            response = requests.get(url, params=params, timeout=10)
            response.encoding = 'utf-8'

            # 处理返回的JSONP数据，提取JSON部分
            text = response.text
            json_str = text[text.find('['):text.rfind(']') + 1]

            # 解析JSON数据
            data = eval(json_str)  # 或者使用 json.loads()

            # 转换为DataFrame并重命名字段
            df = pd.DataFrame(data)
            df.rename(columns={
                'day': 'date',
                'open': 'open',
                'high': 'high',
                'close': 'close',
                'low': 'low',
                'volume': 'volume'
            }, inplace=True)

            # 数据类型转换
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'close', 'low']:
                df[col] = df[col].astype(float)
            df['volume'] = df['volume'].astype(float)

            # 按日期排序
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            print(f"获取数据失败: {e}")
            return None





if __name__ == '__main__':
    cc=Crawler()

    # 使用示例
    # df_hot = cc.get_ths_hot_rank(list_type='normal', time_type='hour')
    # print(df_hot[['code', 'name', 'rise_and_fall', 'rate', 'concept_tag']].head())
    # 获取平安银行全部历史日线数据
    # df = cc.get_stock_history_simple('sz000001', datalen=100)
    # if df is not None:
    #     print(df.head())
    #     print(f"共获取 {len(df)} 条数据")