"""
使用 Heston 模型模拟随机波动率
"""

import time
from test_demo2 import AdaptiveForwardGaussianSmoother
import matplotlib.pyplot as plt
import numpy as np



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

def test_adaptive_smoothing(arr):
    # # 生成测试数据：混合平稳段和波动段
    # np.random.seed(42)
    # n_points = 200
    # t = np.linspace(0, 4 * np.pi, n_points)
    #
    # # 基础信号 + 噪声 + 突发波动
    # base_signal = np.sin(t) * 2
    # noise = np.random.normal(0, 0.5, n_points)
    #
    # # 添加一些突发波动
    # spikes = np.zeros(n_points)
    # spikes[50:55] = 4.0  # 第一个尖峰
    # spikes[120:125] = -3.0  # 第二个尖峰
    # spikes[180:185] = 5.0  # 第三个尖峰

    # array_A = base_signal + noise + spikes

    # 应用自适应平滑
    smoother = AdaptiveForwardGaussianSmoother(
        min_sigma=0.3,
        max_sigma=2.0,
        base_window_size=7,
        sensitivity=2
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



    # # 绘制结果
    # plt.figure(figsize=(12, 10))
    #
    # plt.subplot(3, 1, 1)
    # plt.plot(array_A, 'b-', alpha=0.7, label='原始数据 A')
    # plt.plot(array_B, 'r-', linewidth=2, label='平滑后数据 B')
    # plt.legend()
    # plt.title('自适应前向高斯平滑结果')
    # plt.grid(True)
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(sigmas, 'g-')
    # plt.title('自适应调整的 Sigma 参数')
    # plt.ylabel('Sigma')
    # plt.grid(True)
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(windows, 'm-')
    # plt.title('自适应调整的窗口大小')
    # plt.ylabel('窗口大小')
    # plt.xlabel('数据点索引')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()



#
# # # 使用示例
# # arr = np.array([1, 2, 3, 4, 5])
# # shifted = shift_right_fill(arr, 2)
#
#
# # Heston 模型参数
# S0 = 100.0  # 初始股价
# v0 = 0.3  # 初始方差 (对应初始波动率20%，因为 sqrt(0.04)=0.2) 初始波动率可以参考 vix 恐慌指数, 越大越恐慌,0.04 为波动率 20% 的均衡状态
# mu = 0.03  # 期望年收益率 (5%)
# kappa = 2.0  # 均值回归速度  "κ < 1": "慢速回归 - 波动率冲击持续较长时间","1 < κ < 3": "中等速度 - 典型市场情况", "κ > 3": "快速回归 - 波动率迅速回到正常水平"
# theta = 0.09  # 长期平均方差 (对应长期平均波动率20%),θ=0.0100 → 长期波动率=10.00% → 低波动环境 (平静市场),θ=0.0400 → 长期波动率=20.00% → 正常波动环境,θ=0.0625 → 长期波动率=25.00% → 高波动环境,θ=0.0900 → 长期波动率=30.00% → 极高波动环境 (危机模式), 不同板块的波动率不一样
# sigma = 0.3  # 方差波动率
# rho = -0.6 # 价格与波动的相关系数 (通常为负，体现杠杆效应), "范围": (-0.8, -0.4),
# T = 1.0  # 模拟1年
# N = 100 # 时间步数
# num_sims = 1  # 模拟路径数
#
# # 模拟
# t, S_paths, v_paths, g_path = heston_simulate_euler(S0, v0, mu, kappa, theta, sigma, rho, T, N, num_sims)
# features1=shift_right_fill(S_paths[0],1,100)
# features2=shift_right_fill(S_paths[0],2,100)
# features3=shift_right_fill(S_paths[0],3,100)
# features4=shift_right_fill(S_paths[0],4,100)
#
# weights = np.array([0.1, 0.15, 0.25, 0.5])  # 权重：i-5, i-4, i-3, i-1(仅取4个点的均值)
# weights /= weights.sum()  #
# a=np.array([features1,features2,features3,features4])
# a_transposed = a.transpose(1, 0)
# c=np.dot(a_transposed,weights) # 根据权重滑动
# gauss_d=test_adaptive_smoothing(S_paths[0])  # 高斯前向平滑
# pass
# time.sleep(1)
# # 绘图 - 价格路径
# fig,(ax1,ax2)=plt.subplots(2, 1, figsize=(12, 8))
# # plt.figure(figsize=(12, 8))
# # plt.subplot(3, 1, 1)
# for i in range(num_sims):
#     ax1.plot(t, S_paths[0], lw=1,label='S_paths')
# # plt.title('Heston Model')
# # plt.ylabel('pr')
# plt.grid(True)
# #
# # # 绘图 - 方差路径
#
# for i in range(num_sims):
#     ax1.plot(t, c, lw=1,label='c')
# # plt.title('Heston Model - Simulated Variance Paths')
# # plt.xlabel('Time (Years)')
# # plt.ylabel('Variance')
# plt.grid(True)
#
# # plt.subplot(3, 1, 3)
# for i in range(num_sims):
#     ax1.plot(t, gauss_d, lw=1,label='gauss_d')
#
# for i in range(num_sims):
#     ax2.plot(t, g_path[i], lw=1,label='S_paths')
# plt.title('gauss')
# plt.xlabel('Time (Years)')
# plt.ylabel('Variance')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
