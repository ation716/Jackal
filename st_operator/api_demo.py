# -*- coding: utf-8 -*-
# @Time    : 2026/3/16 10:16
# @Author  : gaolei
# @FileName: api_demo.py
# @Software: PyCharm
import sys
import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

import akshare as ak
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from security_data import ChipDistributionAnalyzer


# data=ak.stock_individual_info_em('000537',5)
data=ak.stock_individual_basic_info_xq('SZ000537')
time.sleep(1)