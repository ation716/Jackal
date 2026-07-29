# -*- coding: utf-8 -*-
# @Time    : 2026/4/24 10:29
# @Author  : gaolei
# @FileName: cal_info.py
# @Software: PyCharm
import os
import time
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from basic_s import ChipDistributionAnalyzer as analyzer


class CalSecurity:

    def cal_peak_trough(self, df: pd.DataFrame, period: int = 5) -> dict:
        """
        计算波峰、波谷及其涟漪。

        波峰：当天 close 在左右各 period 天内均为最大值。
        波谷：当天 close 在左右各 period 天内均为最小值。
        涟漪：波峰/波谷邻近 ±5 日内，high > 波峰价 或 low < 波谷价 的日期和价格。

        :param df: 含 date/open/high/low/close/volume 列的 DataFrame
        :param period: 判断波峰波谷的左右窗口大小，默认 5
        :return: {"peak": {date: {"price": float, "ripple": [(date, price)]}},
                  "trough": {date: {"price": float, "ripple": [(date, price)]}}}
        """
        df = df.copy().reset_index(drop=True)
        peaks = {}
        troughs = {}
        ripple_window = 5

        for i in range(period, len(df) - period):
            close = df['close'].iloc[i]
            date_str = str(df['date'].iloc[i])
            left_close = df['close'].iloc[i - period:i]
            right_close = df['close'].iloc[i + 1:i + period + 1]

            is_peak = close > left_close.max() and close > right_close.max()
            is_trough = close < left_close.min() and close < right_close.min()

            if not is_peak and not is_trough:
                continue

            r_start = max(0, i - ripple_window)
            r_end = min(len(df), i + ripple_window + 1)

            if is_peak:
                ripple = [
                    (str(df['date'].iloc[j]), float(df['high'].iloc[j]))
                    for j in range(r_start, r_end)
                    if j != i and df['high'].iloc[j] > close
                ]
                peaks[date_str] = {'price': float(close), 'ripple': ripple}

            if is_trough:
                ripple = [
                    (str(df['date'].iloc[j]), float(df['low'].iloc[j]))
                    for j in range(r_start, r_end)
                    if j != i and df['low'].iloc[j] < close
                ]
                troughs[date_str] = {'price': float(close), 'ripple': ripple}

        return {'peak': peaks, 'trough': troughs}


if __name__ == '__main__':
    df = pd.read_csv('predict_tes_data/sh.600396.csv')
    cal = CalSecurity()
    result = cal.cal_peak_trough(df, period=10)
    print(result)
