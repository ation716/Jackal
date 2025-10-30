import akshare as ak
import time
import pandas as pd
from datetime import datetime, timedelta


class StockMonitor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data_history = []  # 存储历史数据
        self.last_volume = None  # 上一次的成交量
        self.position = 0  # 持仓状态 (0: 无持仓, 1: 持仓)

    def get_realtime_data(self):
        """获取实时数据并处理"""
        try:
            data = ak.stock_bid_ask_em(symbol=self.symbol)
            current_time = datetime.now()

            # 提取关键数据
            current_price = float(data['最新价'])
            change_percent = float(data['涨跌幅'])
            total_volume = int(data['成交量'])

            # 计算增量成交量
            if self.last_volume is not None:
                volume_5s = total_volume - self.last_volume
            else:
                volume_5s = 0

            self.last_volume = total_volume

            # 存储历史数据
            self.data_history.append({
                'time': current_time,
                'price': current_price,
                'change': change_percent,
                'volume_5s': volume_5s
            })

            # 保留最近30秒数据 (约6条记录)
            self.data_history = [d for d in self.data_history
                                 if d['time'] > current_time - timedelta(seconds=30)]

            return {
                'time': current_time,
                'price': current_price,
                'change': change_percent,
                'volume_5s': volume_5s
            }
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def analyze_volume_price(self, current_data):
        """分析量价关系"""
        if len(self.data_history) < 6:  # 至少需要6个5秒周期(30秒)
            print("数据不足，等待更多数据...")
            return

        # 计算成交量指标
        volumes = [d['volume_5s'] for d in self.data_history]
        avg_volume_30s = sum(volumes) / len(volumes)