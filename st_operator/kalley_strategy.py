# 基于凯利公式的投资策略

import numpy as np
import matplotlib.pyplot as plt
from heston import *
import akshare as ak
import pandas as pd


stock_code = "603686"

# 获取最近 1 年的日线数据（足够覆盖 200 个交易日）
# stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20240101", end_date="20250929", adjust="qfq")

# 提取收盘价并取最近 200 个交易日
# close_prices = stock_zh_a_hist_df['收盘'].tail(200).tolist()
close_prices=[9.73, 9.69, 9.62, 9.77, 9.84, 9.56, 9.52, 9.26, 9.26, 9.18, 9.69, 9.73, 10.72, 11.81, 13.01, 12.67, 11.39, 10.24, 9.85, 9.18, 8.96, 9.26, 9.18, 9.32, 9.42, 9.26, 9.92, 10.35, 10.24, 9.89, 10.52, 10.61, 10.33, 10.44, 11.08, 10.53, 10.7, 10.97, 10.89, 11.01, 11.39, 11.41, 10.99, 10.78, 10.96, 10.75, 11.22, 11.28, 11.3, 11.25, 11.17, 11.77, 11.83, 12.19, 12.82, 12.54, 13.24, 13.22, 12.41, 12.41, 12.3, 12.24, 11.78, 11.95, 12.17, 12.27, 12.58, 12.54, 13.81, 14.01, 13.29, 13.43, 14.4, 13.53, 14.9, 14.66, 14.28, 12.84, 11.54, 10.67, 11.3, 11.64, 11.85, 11.81, 11.67, 11.3, 11.29, 11.29, 11.6, 11.42, 11.77, 11.61, 11.79, 11.34, 11.91, 12.59, 13.12, 13.0, 13.23, 12.93, 13.25, 13.15, 13.14, 13.04, 13.12, 14.45, 14.41, 13.87, 13.69, 13.48, 13.61, 14.46, 14.55, 16.02, 17.64, 17.45, 15.75, 15.28, 15.44, 15.52, 15.14, 15.43, 15.64, 15.04, 14.74, 14.64, 14.34, 14.18, 14.18, 14.78, 15.2, 15.79, 15.63, 16.57, 16.34, 16.08, 15.85, 16.11, 15.87, 16.14, 16.27, 15.8, 15.94, 15.76, 16.46, 16.42, 16.39, 16.57, 17.08, 16.9, 16.4, 16.09, 16.44, 16.44, 16.57, 16.24, 16.47, 15.82, 16.85, 17.31, 17.35, 17.74, 17.61, 17.43, 17.83, 18.0, 18.32, 17.92, 19.71, 19.55, 19.35, 19.05, 19.3, 19.26, 20.01, 19.88, 19.14, 18.07, 18.57, 18.5, 19.06, 19.42, 18.97, 19.17, 18.94, 19.36, 19.15, 18.89, 18.54, 18.53, 19.92, 19.11, 21.02, 23.12, 25.43, 27.97, 27.1, 24.39, 21.95, 21.8,21.86]
# print(f"{stock_code} 最近 200 个交易日的收盘价数组：")
print(close_prices)

# Heston 模型参数
S0 = close_prices[0]  # 初始股价
v0 = 0.09  # 初始方差 (对应初始波动率20%，因为 sqrt(0.04)=0.2) 初始波动率可以参考 vix 恐慌指数, 越大越恐慌,0.04 为波动率 20% 的均衡状态
mu = 0.03  # 期望年收益率 (一般为5%)
kappa = 2.0  # 均值回归速度  "κ < 1": "慢速回归 - 波动率冲击持续较长时间","1 < κ < 3": "中等速度 - 典型市场情况", "κ > 3": "快速回归 - 波动率迅速回到正常水平"
theta = 0.09  # 长期平均方差 (对应长期平均波动率20%),θ=0.0100 → 长期波动率=10.00% → 低波动环境 (平静市场),θ=0.0400 → 长期波动率=20.00% → 正常波动环境,θ=0.0625 → 长期波动率=25.00% → 高波动环境,θ=0.0900 → 长期波动率=30.00% → 极高波动环境 (危机模式), 不同板块的波动率不一样d
sigma = 0.3  # 方差波动率
rho = -0.6 # 价格与波动的相关系数 (通常为负，体现杠杆效应), "范围": (-0.8, -0.4),
T = 1.5  # 模拟1年
N = len(close_prices) # 时间步数
num_sims = 1  # 模拟路径数

# f(p) = [ p * r_win(p) + (1-p) * r_loss(p) ] / [ - r_win(p) * r_loss(p) ]

def kellly_strategy_continues(p_matrix,b_matrix):
    """ 凯莉策略 """
    # 转换为numpy数组以便操作
    p_matrix = np.array(p_matrix, dtype=float)
    result = np.zeros_like(p_matrix)

    # 对每个位置计算加权平均
    for i in range(len(p_matrix)):
        if i==0 or b_matrix[i]<0:
            result[i] = 0
        else:
            tem = kalley1(p_matrix[i],b_matrix[i]*10)
            if tem<0:
                result[i] = 0
            elif tem<1:
                result[i] = tem
            else:
                result[i] = 1
    return result

def kellly_strategy_continues2(p_matrix,b_matrix):
    """ 凯莉策略 """
    # 转换为numpy数组以便操作
    p_matrix = np.array(p_matrix, dtype=float)
    result = np.zeros_like(p_matrix)

    # 对每个位置计算加权平均
    for i in range(len(p_matrix)):
        if i==0 or b_matrix[i]<0:
            result[i] = 0
        else:
            tem = kalley2(p_matrix[i],b_matrix[i]*10)
            if tem<0:
                result[i] = 0
            elif tem<1:
                result[i] = tem
            else:
                result[i] = 1
    return result

def kalley1(p,b):
    """ 经典凯莉  p - (1-p)/b """
    return p-(1-p)/b

def kalley2(p,r):
    """ 在 f(p) = [ p * r_win(p) + (1-p) * r_loss(p) ] / [ - r_win(p) * r_loss(p) ]，
    零和博弈的凯莉 (2p-1)/r
    """
    return (2*p-1)/r

def kalley3(p,r):
    """"""




def continuse_gain(kelly_weight,path,s0):
    """"""
    result = np.zeros_like(kelly_weight)
    for i in range(len(kelly_weight)):
        if i==0:
            result[i]=s0*kelly_weight[i]*path[i]+s0*(1-kelly_weight[i])
        else:
            result[i]=result[i-1]*kelly_weight[i]*(1+path[i])+result[i-1]*(1-kelly_weight[i])
    return result




if __name__ == "__main__":
    # # 测试数据
    # input_array = [1, 2, 3, 4, 5, 6, 7, 8]
    # window_size = 3
    # custom_weights = [0.2, 0.3, 0.5]  # 自定义权重
    #
    # # 计算加权平均
    # result = weighted_moving_average(input_array, window_size, custom_weights)
    # print("输入数组:", input_array)
    # print("窗口大小:", window_size)
    # print("权重:", custom_weights)
    # print("加权平均结果:", result)

    # 模拟
    # t, S_paths, v_paths, g_path = heston_simulate_euler(S0, v0, mu, kappa, theta, sigma, rho, T, N, num_sims)
    t = np.linspace(0, T, N)
    S_paths=[[],]
    S_paths[0]=close_prices
    S_array=np.array(S_paths[0])
    mean_slip_price=weighted_moving_average(S_paths[0],4, [0.1, 0.15, 0.25, 0.5]) # 移动加权平滑

    gauss_d=test_adaptive_smoothing(S_paths[0])  # 高斯前向平滑预测

    mean_acc = np.ones_like(gauss_d[:-1])-np.abs(gauss_d[:-1] - S_array[1:])*5 / S_array[1:]
    mean_slip_accurance = weighted_moving_average(mean_acc, 4, [0.2, 0.2, 0.25, 0.35]) # 准确率估算
    mean_slip_accurance=np.clip(mean_slip_accurance,0,1) # 准确率范围修正
    # shift_s=shift_right_fill(S_paths[0],1,S0)
    gain_loss_p=(gauss_d-S_paths[0]) / S_paths[0] # 预估涨幅计算
    gain_loss_p=np.clip(gain_loss_p,-0.1,0.1)
    kelly_weights=kellly_strategy_continues2(mean_slip_accurance,gain_loss_p)
    gain_loss_r=(S_array[1:]-S_array[:-1])/ S_array[:-1]  # 实际涨幅计算
    assets=continuse_gain(kelly_weights,gain_loss_r,S0)

    with open('kelly_weights2.txt', 'w') as f:
        f.write("kelly_weight\tgain_loss_r\tmean_slip_accurance\tgain_loss_p\tS_paths[0]\tassets\n")
        for i in range(len(kelly_weights)):
            if i==0:
                f.write("{:<10s}{:<10s}{:<10s}{:<10s}{:<10.5f}{:<10.5f}\n".format(
                    "---", "---", "---", '---', S_paths[0][i], assets[i]))
            else:
                f.write("{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}\n".format(
                    kelly_weights[i], gain_loss_r[i-1], mean_slip_accurance[i-1], gain_loss_p[i-1], S_paths[0][i], assets[i]))
        f.write("{:<10.5f}{:<10s}{:<10.5f}{:<10.5f}{:<10s}{:<10s}\n".format(
                kelly_weights[i], '---', mean_slip_accurance[i], gain_loss_p[i], '---', '---'))




    # # # 绘图 - 价格路径
    # fig,(ax1,ax2)=plt.subplots(2, 1, figsize=(12, 8))
    # # plt.figure(figsize=(12, 8))
    # # plt.subplot(3, 1, 1)
    # for i in range(num_sims):
    #     ax1.plot(t, S_paths[0], lw=1,label='S_paths')
    # ax1.set_title('Heston Model')
    # # ax1.set_ylabel('Price')
    # ax1.grid(True)
    # #
    # # # 绘图 - 方差路径
    #
    # # for i in range(num_sims):
    # #     ax1.plot(t, c, lw=1,label='c')
    # # # plt.title('Heston Model - Simulated Variance Paths')
    # # # plt.xlabel('Time (Years)')
    # # # plt.ylabel('Variance')
    # # plt.grid(True)
    #
    # # plt.subplot(3, 1, 3)
    # for i in range(num_sims):
    #     ax1.plot(t, gauss_d, lw=1,label='gauss_d')
    #
    # for i in range(num_sims):
    #     ax2.plot(t, assets, lw=1,label='assets')
    # plt.title('gauss')
    # plt.xlabel('Time (Years)')
    # plt.ylabel('Variance')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()










#
# # 6. 绘图
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))
# fig.suptitle('Heston Model Simulation with Kelly Criterion Strategy (Continuous Compounding)')
#
# # 子图1: 股票价格路径
# ax1.plot(S, color='b', lw=1.5)
# ax1.set_ylabel('Stock Price ($)')
# ax1.grid(True)
# ax1.set_title('Simulated Stock Price Path (Heston Model)')
#
# # 子图2: 瞬时波动率路径
# ax2.plot(np.sqrt(v), color='r', lw=1.5)
# ax2.set_ylabel('Volatility')
# ax2.grid(True)
# ax2.set_title('Instantaneous Volatility Path')
#
# # 子图3: 投资组合价值路径（连续复利）
# ax3.plot(V, color='g', lw=2.5, label='Portfolio Value')
# ax3.set_ylabel('Portfolio Value ($)')
# ax3.set_xlabel('Time (Days)')
# ax3.grid(True)
# ax3.set_title('Portfolio Value Evolution with Continuous Compounding (Kelly Criterion)')
# ax3.legend()
#
# # 在组合价值图上标注最终价值
# final_value = V[-1]
# ax3.annotate(f'Final Value: ${final_value:,.2f}',
#              xy=(N - 1, final_value),
#              xytext=(N - 50, final_value * 0.8),  # 动态调整标注位置
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
#              fontsize=12,
#              bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
#
# # 子图4: 凯利投资比例
# ax4.plot(f_star_arr, color='purple', lw=1.5)
# ax4.set_ylabel('Kelly Fraction (f*)')
# ax4.set_xlabel('Time (Days)')
# ax4.grid(True)
# ax4.set_title('Optimal Investment Fraction Over Time')
# ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='100% Allocation')
# ax4.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 打印一些关键结果
# print(f"Initial Portfolio Value: ${V[0]:,.2f}")
# print(f"Final Portfolio Value: ${final_value:,.2f}")
# print(f"Total Return: {((final_value / V[0]) - 1) * 100:.2f}%")
# print(f"Final Kelly Fraction (f*): {f_star_arr[-1]:.4f} ({f_star_arr[-1] * 100:.1f}% allocation)")
# print(f"Annualized Log Return: {cumulative_growth[-1] * 100:.2f}%")  # 年化对数收益率