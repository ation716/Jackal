import pandas as pd
import numpy as np
from datetime import datetime
import json
import akshare as ak

# 示例数据结构
data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "stock_A": {"price": 10.2, "change": 0.02},
    "stock_B": {"price": 15.5, "change": -0.01},
    # ...其他股票
    "sector_index": 1250.75  # 板块指数
}