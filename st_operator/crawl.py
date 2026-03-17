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
    def get_ths_hot_rank(list_type='normal', time_type='hour'):
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

if __name__ == '__main__':
    cc=Crawler()
    # 使用示例
    df_hot = cc.get_ths_hot_rank(list_type='normal', time_type='hour')
    print(df_hot[['code', 'name', 'rise_and_fall', 'rate', 'concept_tag']].head())