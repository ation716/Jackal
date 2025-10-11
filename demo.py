import numpy as np
import matplotlib.pyplot as plt


def geometric_brownian_motion(S0, mu, sigma, T, N):
    """
    模拟几何布朗运动路径
    Parameters:
    S0 : float - 初始价格
    mu : float - 漂移率（年化）
    sigma : float - 波动率（年化）
    T : float - 总时间（年）
    N : int - 时间步数
    Returns:
    t : array - 时间点
    S : array - 模拟的价格路径
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    # 生成布朗路径 W_t: W_{t+dt} = W_t + sqrt(dt) * N(0,1)
    W = np.cumsum(np.random.standard_normal(N + 1) * np.sqrt(dt))
    W[0] = 0  # 初始维纳过程为0

    # 代入解析解公式
    drift = (mu - 0.5 * sigma ** 2) * t
    diffusion = sigma * W
    S = S0 * np.exp(drift + diffusion)

    return t, S


# 参数设置
S0 = 100.0  # 初始股价
mu = 0.05  # 期望年收益率 (5%)
sigma = 0.2  # 年波动率 (20%)
T = 1.0  # 模拟1年
N = 252  # 模拟252个交易日（假设一年有252个交易日）

# 模拟一条路径
t, S = geometric_brownian_motion(S0, mu, sigma, T, N)
#
# # 绘图
# plt.figure(figsize=(10, 6))
# plt.plot(t, S)
# plt.title('Geometric Brownian Motion - Stock Price Simulation')
# plt.xlabel('Time (Years)')
# plt.ylabel('Stock Price ($)')
# plt.grid(True)
# plt.show()


def mc_stock_price(S0, mu, sigma, T, N, num_simulations):
    """
    使用几何布朗运动和蒙特卡罗方法模拟多条股票价格路径
    Parameters:
    ... (同GBM参数)
    num_simulations : int - 模拟路径数量
    Returns:
    t : array - 时间点
    all_paths : ndarray - 所有模拟路径 (num_simulations x (N+1))
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    # 初始化所有路径 [模拟数量, 时间步数+1]
    all_paths = np.zeros((num_simulations, N + 1))
    all_paths[:, 0] = S0  # 所有路径的起点都是S0

    # 生成随机数并计算每条路径
    for i in range(num_simulations):
        # 生成每个时间步的随机冲击 - 标准正态分布
        Z = np.random.standard_normal(N)
        # 计算每个时间步的收益率 (对数空间)
        daily_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        # 计算累计收益 (从S0开始)
        all_paths[i, 1:] = S0 * np.exp(np.cumsum(daily_returns))

    return t, all_paths


# 参数设置
num_simulations = 1000  # 模拟1000次
t, paths = mc_stock_price(S0, mu, sigma, T, N, num_simulations)

# 绘制前100条路径
plt.figure(figsize=(12, 6))
for i in range(100):
    plt.plot(t, paths[i], lw=1, alpha=0.7)
plt.title(f'Monte Carlo Simulation of Stock Price ({num_simulations} Paths) - First 100 Shown')
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price ($)')
plt.grid(True)
plt.show()

# 计算在时间T的股价分布，并计算其均值和95%置信区间
final_prices = paths[:, -1]
mean_final_price = np.mean(final_prices)
ci_low = np.percentile(final_prices, 2.5)
ci_high = np.percentile(final_prices, 97.5)

print(f"模拟 {num_simulations} 次后，在时间 T={T} 年时：")
print(f"股票价格均值: ${mean_final_price:.2f}")
print(f"95% 置信区间: [${ci_low:.2f}, ${ci_high:.2f}]")

# 绘制T时刻的价格分布直方图
plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, density=True, alpha=0.75, edgecolor='black')
plt.axvline(mean_final_price, color='r', linestyle='--', label=f'Mean (${mean_final_price:.2f})')
plt.axvline(ci_low, color='g', linestyle=':', label=f'95% CI Lower (${ci_low:.2f})')
plt.axvline(ci_high, color='g', linestyle=':', label=f'95% CI Upper (${ci_high:.2f})')
plt.title('Distribution of Stock Price at Time T')
plt.xlabel('Stock Price ($)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()