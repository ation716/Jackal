# -*- coding: utf-8 -*-
# @Time    : 2026/4/15 13:18
# @Author  : gaolei
# @FileName: predict_tes.py
# @Software: PyCharm
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression


def fit_distribution(price_series, sector_ratio_series, index_ret_series,
                     half_life=20, bandwidth='silverman'):
    """
    price_series: 最近M个交易日的收盘价 (Series, index为日期)
    sector_ratio_series: 对应日期的板块涨停比 (0~1)
    index_ret_series: 对应日期的上证指数日涨幅
    返回: dict 包含 'kde', 'beta', 'residuals', 'last_cond_mean' 等
    """
    # 1. 计算个股日涨幅
    ret = price_series.pct_change().dropna().values  # shape (N,)
    # 对齐数据（删除第一天，因为没有涨幅）
    sector = sector_ratio_series.iloc[1:].values
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
    last_sector = sector_ratio_series.iloc[-1]
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
        'pdf': pdf_ret,
        'cdf': cdf_ret
    })
    return result


def save_distribution(result, filepath='dist_next_day.csv'):
    result.to_csv(filepath, index=False)
    print(f"分布函数已保存至 {filepath}")


def update_model(new_price, new_sector_ratio, new_index_ret, old_model,
                 max_history_days=200, half_life=20):
    """
    每日收盘后调用，更新模型
    new_price: 当日收盘价
    new_sector_ratio: 当日板块涨停比
    new_index_ret: 当日上证指数涨幅
    old_model: 之前保存的模型字典
    max_history_days: 最多保留多少天数据（防止无限增长）
    """
    # 将新数据追加到历史序列（需要维护外部序列，这里仅示意逻辑）
    # 实际使用中，你需要维护三个全局Series，每次追加新数据
    # 然后重新调用 fit_distribution
    # 此处简化为重新拟合的伪代码
    pass


# ─── 测试数据管理 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TEST_STOCKS = ['000678', '600396']
_DATA_DIR = _HERE / 'test_data'
_DATA_FILE = _DATA_DIR / 'fixture.pkl'
_START = '20240101'
_END = '20241231'


def _fetch_data():
    """从网络拉取实验数据并持久化到 test_data/fixture.pkl"""
    import akshare as ak
    from skl_predict.basic_s import ChipDistributionAnalyzer

    _DATA_DIR.mkdir(exist_ok=True)
    ana = ChipDistributionAnalyzer()
    ds = {}

    for sym in TEST_STOCKS:
        print(f"[fetch] {sym} 日线...")
        df = ana.get_daily_ak(sym, _START, _END)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期').sort_index()
        ds[sym] = df
        print(f"  {sym}: {len(df)} 条  {df.index[0].date()} ~ {df.index[-1].date()}")

    print("[fetch] 上证指数...")
    idx = ak.stock_zh_index_daily(symbol='sh000001')
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx.set_index('date').sort_index().loc[_START:_END]
    ds['index'] = idx
    print(f"  上证: {len(idx)} 条")

    for sym in TEST_STOCKS:
        print(f"[fetch] {sym} 筹码...")
        try:
            chip = ana.get_stock_chip_akshare(sym)
            chip['日期'] = pd.to_datetime(chip['日期'])
            chip = chip.set_index('日期').sort_index()
            ds[f'{sym}_chip'] = chip
            print(f"  {sym} 筹码: {len(chip)} 条")
        except Exception as e:
            print(f"  {sym} 筹码失败: {e}")
            ds[f'{sym}_chip'] = pd.DataFrame()

    with open(_DATA_FILE, 'wb') as f:
        pickle.dump(ds, f)
    print(f"[save] → {_DATA_FILE}")
    return ds


def load_fixture(force_refresh=False):
    """加载测试数据；本地有缓存则直接读取，否则重新拉取"""
    if not force_refresh and _DATA_FILE.exists():
        print(f"[load] 使用缓存 {_DATA_FILE}")
        with open(_DATA_FILE, 'rb') as f:
            return pickle.load(f)
    return _fetch_data()


def _build_inputs(ds, symbol, lookback=120):
    """
    构造 fit_distribution 所需的三个 Series（日期对齐）
    sector_ratio_series: 测试用个股5日均涨幅代替板块涨停比
    """
    stock = ds[symbol].tail(lookback).copy()
    idx = ds['index']
    dates = stock.index.intersection(idx.index)
    stock = stock.loc[dates]
    idx_a = idx.loc[dates]

    price_s = stock['收盘'].astype(float)
    index_ret_s = idx_a['close'].pct_change().fillna(0)
    # 简化板块涨停比：用个股涨跌幅5日滚动均值作为代理
    sector_s = (stock['涨跌幅'].astype(float) / 100.0).rolling(5, min_periods=1).mean().fillna(0)
    return price_s, sector_s, index_ret_s


if __name__ == '__main__':
    # ── 1. 加载/获取数据 ──────────────────────────────────────────────────────
    # 改 force_refresh=True 可强制重新拉取网络数据
    ds = load_fixture(force_refresh=False)

    for sym in TEST_STOCKS:
        print(f"\n{'─' * 50}")
        print(f"▶ {sym}")
        price_s, sector_s, index_s = _build_inputs(ds, sym)
        print(f"  日期范围: {price_s.index[0].date()} ~ {price_s.index[-1].date()}  ({len(price_s)} 天)")

        # ── 2. 拟合分布 ───────────────────────────────────────────────────────
        model = fit_distribution(price_s, sector_s, index_s)
        print(f"  条件均值: {model['cond_mean']:.4f}")
        print(f"  残差 std: {model['residuals'].std():.4f}")

        # ── 3. 预测次日分布 ───────────────────────────────────────────────────
        dist = predict_distribution(model)
        p_up3 = 1 - dist.loc[dist['return'] <= 0.03, 'cdf'].max()
        p_dn3 = dist.loc[dist['return'] <= -0.03, 'cdf'].max()
        idx_med = (dist['cdf'] - 0.5).abs().argsort().iloc[0]
        median = dist.iloc[idx_med]['return']
        print(f"  P(>+3%): {p_up3:.2%}  |  P(<-3%): {p_dn3:.2%}  |  中位数: {median:.2%}")

        # ── 4. 筹码信息 ───────────────────────────────────────────────────────
        chip = ds.get(f'{sym}_chip')
        if chip is not None and not chip.empty:
            latest = chip.iloc[-1]
            win = latest.get('获利比例', None)
            avg = latest.get('平均成本', None)
            print(f"  筹码(最新 {chip.index[-1].date()}): "
                  f"获利比例={win:.2%}  平均成本={avg}" if win is not None else "  筹码数据字段异常")