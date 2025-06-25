import random

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import random

# 配置参数
API_URL = "https://example.com/api/prices"  # 替换为实际API地址
SAVE_PATH = "p_data.csv"  # 数据存储路径
INTERVAL = 5  # 采集间隔(秒)


# def fetch_price_vector():
#     """获取价格向量"""
    # try:
    #     response = requests.get(API_URL, timeout=3)
    #     response.raise_for_status()
    #     return response.json()['prices']  # 假设返回格式: {"prices": [1.23, 4.56, ...]}
    # except Exception as e:
    #     print(f"请求失败: {e}")
    #     return None

def fetch_price_vector():
    """获取价格向量"""
    or_prices=[12.83,19.98,34.95]
    new_prices=[]
    def generate_price():
        nonlocal new_prices,new_prices
        random.gauss(0,2)
        return random.uniform(100, 1000)  # 随机生成价格
    # try:
    #     response = requests.get(API_URL, timeout=3)


def save_to_csv(prices, timestamp):
    """保存数据到CSV"""
    df = pd.DataFrame([prices], columns=[f"product_{i}" for i in range(len(prices))])
    df['timestamp'] = timestamp
    df.to_csv(SAVE_PATH, mode='a', header=not pd.io.common.file_exists(SAVE_PATH), index=False)


def data_collection():
    """持续采集数据"""
    while True:
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        prices = fetch_price_vector()
        if prices:
            save_to_csv(prices, timestamp)
            print(f"{timestamp} 数据已保存 | 商品数: {len(prices)}")

        # 精确间隔控制
        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    # data_collection()
    for i in range(20):
        print(random.gauss(0,0.01))