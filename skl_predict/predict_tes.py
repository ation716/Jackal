# -*- coding: utf-8 -*-
# @Time    : 2026/4/15 13:18
# @Author  : gaolei
# @FileName: predict_tes.py
# @Software: PyCharm
import os
import sys
import pickle
from pathlib import Path
from basic_s import ChipDistributionAnalyzer as Analyzer
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

"""
我现在要根据一只A股主板股票最近 90 天的数据 拟合出次日 的涨幅概率分布函数
获得的历史数据有
1. 90 日每日收盘价，开盘价，当日最高价，当日最低价
2. 所在板块最近 10 日板块指数数据，同板块涨停个股数量，跌停个股数量
3. 上证指数 15 日数据
需要给出次日涨幅概率分布函数，打印并保存到文件，次日数据更新后根据新的历史数据（时间窗口需要你推荐，或者推荐一个动态时间窗口方法）持续修正分布函数
"""

def fit_distribution(price_series, sector_ret_series, index_ret_series,
                     half_life=20, bandwidth='silverman'):
    """
    price_series: 最近M个交易日的收盘价 (Series, index为日期)
    sector_ret_series: 对应日期的板块日涨幅 (float, pct_change)
    index_ret_series: 对应日期的上证指数日涨幅
    返回: dict 包含 'kde', 'regressor', 'residuals', 'cond_mean', 'last_date'
    """
    # 1. 计算个股日涨幅
    ret = price_series.pct_change().dropna().values  # shape (N,)
    # 对齐数据（删除第一天，因为没有涨幅）
    sector = sector_ret_series.iloc[1:].values
    index_ret = index_ret_series.iloc[1:].values
    N = len(ret)

    # 2. 回归：个股涨幅 ~ 指数涨幅 + 板块涨停比
    X = np.column_stack([index_ret, sector])
    reg = LinearRegression().fit(X, ret)
    resid = ret - reg.predict(X)

    # 3. 计算指数衰减权重（半衰期half_life）
    lambda_ = np.log(2) / half_life
    ages = np.arange(N - 1, -1, -1)  # 最近样本权重最高
    weights = np.exp(-lambda_ * ages)
    weights /= weights.sum()

    # 4. 加权KDE拟合残差分布
    # 计算加权标准差和IQR (用于带宽)
    weighted_mean = np.average(resid, weights=weights)
    weighted_var = np.average((resid - weighted_mean) ** 2, weights=weights)
    weighted_std = np.sqrt(weighted_var)
    # 加权分位数 (简化: 用numpy percentile模拟)
    # 更精确可用 weighted quantile 函数，此处近似
    q25 = np.percentile(resid, 25)
    q75 = np.percentile(resid, 75)
    iqr = q75 - q25
    h = 0.9 * min(weighted_std, iqr / 1.34) * N ** (-0.2)
    # 使用 gaussian_kde 但需要传入样本权重（scipy 1.2+支持）
    kde = gaussian_kde(resid, bw_method=h)
    # 注意：gaussian_kde 不支持直接加权，需手动构造加权样本：重复采样？效率低。
    # 替代：使用 sklearn.neighbors.KernelDensity with sample_weight
    from sklearn.neighbors import KernelDensity
    kde_sk = KernelDensity(bandwidth=h, kernel='gaussian')
    kde_sk.fit(resid.reshape(-1, 1), sample_weight=weights)

    # 5. 记录最新条件均值（用于次日预测）
    last_sector = sector_ret_series.iloc[-1]
    last_index_ret = index_ret_series.iloc[-1]
    cond_mean = reg.predict([[last_index_ret, last_sector]])[0]

    return {
        'kde': kde_sk,
        'regressor': reg,
        'residuals': resid,
        'cond_mean': cond_mean,
        'last_date': price_series.index[-1]
    }


def predict_distribution(model, n_points=1000):
    """
    返回次日涨幅的PDF和CDF在网格上的值
    """
    # 残差分布范围 (取残差样本的min/max，向外扩30%)
    r_min, r_max = model['residuals'].min(), model['residuals'].max()
    eps_grid = np.linspace(r_min - 0.03, r_max + 0.03, n_points)
    # 计算残差密度
    log_dens = model['kde'].score_samples(eps_grid.reshape(-1, 1))
    pdf_eps = np.exp(log_dens)
    # 平移：次日涨幅 = 条件均值 + 残差
    cond_mean = model['cond_mean']
    ret_grid = eps_grid + cond_mean
    pdf_ret = pdf_eps  # 平移不变
    # 计算CDF (数值积分)
    cdf_ret = np.cumsum(pdf_ret) * (ret_grid[1] - ret_grid[0])
    cdf_ret = np.clip(cdf_ret, 0, 1)

    result = pd.DataFrame({
        'return': ret_grid,
        'pdf': pdf_ret, # 概率密度
        'cdf': cdf_ret # 累计分布
    })
    return result


def save_distribution(result, filepath='dist_next_day.csv'):
    result.to_csv(filepath, index=False)
    print(f"分布函数已保存至 {filepath}")


def update_model(new_price, new_sector_ratio, new_index_ret, old_model,
                 max_history_days=200, half_life=20):
    """
    每日收盘后调用，将新数据追加到历史序列并重新拟合模型。

    new_price:        当日收盘价（float）
    new_sector_ratio: 当日板块涨幅（float，pct_change 格式）
    new_index_ret:    当日上证指数涨幅（float，pct_change 格式）
    old_model:        之前 fit_distribution 返回的模型字典，
                      需额外携带 '_price_series'、'_sector_series'、'_index_series' 三个历史 Series
    max_history_days: 滑动窗口上限，超出时丢弃最早数据
    返回: 新模型字典（同 fit_distribution 格式，同样携带三个历史 Series）
    """
    # 1. 取出历史序列
    price_s  = old_model['_price_series'].copy()
    sector_s = old_model['_sector_series'].copy()
    index_s  = old_model['_index_series'].copy()

    # 2. 推断新日期（上一个日期 +1 个交易日，简单用 +1 天；实际可传入日期参数）
    last_date = price_s.index[-1]
    new_date  = last_date + pd.Timedelta(days=1)

    # 3. 追加新数据
    price_s  = pd.concat([price_s,  pd.Series([new_price],        index=[new_date])])
    sector_s = pd.concat([sector_s, pd.Series([new_sector_ratio], index=[new_date])])
    index_s  = pd.concat([index_s,  pd.Series([new_index_ret],    index=[new_date])])

    # 4. 滑动窗口裁剪（保留最近 max_history_days 条）
    if len(price_s) > max_history_days:
        price_s  = price_s.iloc[-max_history_days:]
        sector_s = sector_s.iloc[-max_history_days:]
        index_s  = index_s.iloc[-max_history_days:]

    # 5. 重新拟合
    new_model = fit_distribution(price_s, sector_s, index_s, half_life=half_life)

    # 6. 将历史序列挂回新模型，供下次调用
    new_model['_price_series']  = price_s
    new_model['_sector_series'] = sector_s
    new_model['_index_series']  = index_s

    return new_model







if __name__ == '__main__':
    """这个预测效果有点后置，第二天的实际值很大程度会偏离预测区间 25概率~75 概率"""