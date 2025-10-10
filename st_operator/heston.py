# coding:utf8
import time
from guass_smoother import AdaptiveForwardGaussianSmoother
import matplotlib.pyplot as plt
import numpy as np

"""
GBM 模型
特点：模拟的价格路径连续且对数收益率服从正态分布。价格本身服从对数正态分布，保证了价格不会为负。
局限：
-假设波动率 $\sigma$ 恒定，这与实际市场中波动率聚簇（volatility clustering）的现象不符。
-假设收益率服从正态分布，忽略了实际金融时间序列中经常出现的尖峰厚尾（Fat Tails）特性。
-无法捕捉股价的跳跃和暴跌。    
"""


"""
改进
使用 Heston 模型模拟随机波动率 
"""


def heston_simulate_euler(S0, v0, mu, kappa, theta, sigma, rho, T, N, num_simulations=1):
    """
    使用欧拉方法模拟Heston模型路径

    Parameters:
    S0 : float - 初始价格
    v0 : float - 初始方差
    mu : float - 漂移率（年化）
    kappa : float - 均值回归速度
    theta : float - 长期平均方差
    sigma : float - 方差波动率
    rho : float - 价格与方差的相关系数
    T : float - 总时间（年）
    N : int - 时间步数
    num_simulations : int - 模拟路径数量

    Returns:
    t : array - 时间点
    S_paths : ndarray - 模拟的价格路径 (num_simulations x (N+1))
    v_paths : ndarray - 模拟的方差路径 (num_simulations x (N+1))
    g_paths : ndarray - 涨幅 (num_simulations x (N+1))
    """
    dt = T / N  # 每个周期有多少个步长
    t = np.linspace(0, T, N + 1)

    # 初始化数组
    S_paths = np.zeros((num_simulations, N + 1))
    v_paths = np.zeros((num_simulations, N + 1))
    g_paths = np.zeros((num_simulations, N + 1))  # 涨幅
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    g_paths[:, 0] = 0

    # 生成相关的随机数
    for i in range(num_simulations):
        # 生成两组独立的标准正态随机数
        Z1 = np.random.standard_normal(N)
        Z2 = np.random.standard_normal(N)
        # 通过Cholesky分解构造相关随机数
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2  # 与W1相关系数为rho的随机数

        for j in range(1, N + 1):
            # 对方差过程应用欧拉离散，确保方差非负（采用反射或吸收等更复杂方法会更稳健）
            v_current = v_paths[i, j - 1]
            # 防止方差为负，简单处理为取绝对值，更复杂的方法可使用完全截断等
            v_next = v_current + kappa * (theta - v_current) * dt + sigma * np.sqrt(v_current) * np.sqrt(dt) * W2[j - 1]
            v_paths[i, j] = max(v_next, 1e-6)  # 设置一个小的下限，防止计算问题

            # 对价格过程应用欧拉离散 (使用对数形式通常更稳定，这里使用标准形式)
            S_current = S_paths[i, j - 1]
            # 振幅
            g_paths[i, j] = mu * dt + np.sqrt(v_current) * np.sqrt(dt) * W1[j - 1]
            if g_paths[i, j]>0.1:
                pass
            S_next = S_current + mu * S_current * dt + np.sqrt(v_current) * S_current * np.sqrt(dt) * W1[j - 1]
            S_paths[i, j] = S_next


    return t, S_paths, v_paths, g_paths


def shift_right_fill(arr, shift, fill_value=0):
    """
    将数组右移并在左侧用固定值填充

    参数:
    arr: 输入数组
    shift: 右移位数
    fill_value: 左侧填充值（默认为0）

    返回:
    移位并填充后的数组
    """
    result = np.full_like(arr, fill_value)
    result[shift:] = arr[:-shift]
    return result

def test_adaptive_smoothing(arr,s=3):
    # 应用自适应平滑
    smoother = AdaptiveForwardGaussianSmoother(
        min_sigma=0.3,
        max_sigma=2.0,
        base_window_size=7,
        sensitivity=s
    )

    array_B, array_Bo,sigmas, windows = smoother.smooth_array(arr)
    return array_B


def weighted_moving_average(arr, n, weights=None):
    """
    对一维数组进行前n个位置的加权平均，生成新数组。

    参数:
        arr (list or np.ndarray): 输入的一维数组
        n (int): 考虑前n个位置的窗口大小
        weights (list or np.ndarray, optional): 加权权重，若为None则使用均匀权重

    返回:
        np.ndarray: 加权平均后的新数组
    """
    # 转换为numpy数组以便操作
    arr = np.array(arr, dtype=float)
    result = np.zeros_like(arr)

    # 如果未提供权重，生成均匀权重
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights)  # 归一化权重

    # 确保权重长度与窗口大小一致
    if len(weights) != n:
        raise ValueError("权重长度必须等于窗口大小n")

    # 对每个位置计算加权平均
    for i in range(len(arr)):
        # 确定窗口范围：从 max(0, i-n+1) 到 i
        start = max(0, i - n + 1)
        window = arr[start:i + 1]
        window_weights = weights[-len(window):]  # 取对应的权重
        # 计算加权平均
        result[i] = np.sum(window * window_weights) / np.sum(window_weights)

    return result


