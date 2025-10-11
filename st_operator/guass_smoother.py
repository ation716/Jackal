import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class AdaptiveForwardGaussianSmoother:
    def __init__(self, min_sigma=0.5, max_sigma=3.0, base_window_size=5, sensitivity=1.0):
        """
        自适应前向高斯平滑器
        该平滑器根据最近数据的波动性自适应调整高斯核的参数 sigma 和 窗口大小，因果性高斯加权平均得到最新数据的平滑值


        参数:
        min_sigma: 最小标准差，用于平稳数据
        max_sigma: 最大标准差，用于波动数据
        base_window_size: 基础窗口大小
        sensitivity: 波动敏感度 (0.1-5)，值越大对波动越敏感
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.base_window_size = base_window_size
        self.sensitivity = sensitivity
        self.history = []  # 存储历史数据用于自适应计算

    def compute_local_variance(self, new_value, lookback=10):
        """计算局部方差来评估数据波动性"""
        if len(self.history) < lookback:
            return 0

        recent_data = self.history[-lookback:]
        recent_data.append(new_value)
        return np.var(recent_data)

    def adapt_parameters(self, current_value):
        """根据数据波动性自适应调整参数"""
        # 计算局部波动性
        local_variance = self.compute_local_variance(current_value)

        # 归一化波动性 (0-1范围)
        # 这里使用经验值，可以根据实际数据调整
        max_expected_variance = 100.0  # 假设的最大方差
        normalized_variance = min(local_variance / max_expected_variance, 1.0)

        # 根据波动性调整sigma：波动越大，sigma越小（保持细节）
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * (normalized_variance ** self.sensitivity)

        # 动态调整窗口大小：确保足够的采样点但不过度
        # 在高斯分布中：
        #     ±1σ 范围内包含约68%的数据
        #     ±2σ 范围内包含约95%的数据 (此处采取的策略)
        #     ±3σ 范围内包含约99.7%的数据
        window_size = max(3, min(self.base_window_size, int(2 * np.ceil(2 * sigma) + 1))) # +1 表示包含中心点

        return sigma, window_size

    def create_causal_gaussian_kernel(self, sigma, window_size):
        """创建因果性（前向）高斯核"""
        # 只使用当前和之前的位置
        x = np.arange(0, window_size)
        kernel = stats.norm.pdf(x, loc=0, scale=sigma)

        # 归一化确保权重和为1
        kernel = kernel / np.sum(kernel)
        return kernel

    def smooth_value(self, new_value):
        """平滑单个新数据点"""
        # 添加到历史
        self.history.append(new_value)

        # 自适应调整参数
        sigma, window_size = self.adapt_parameters(new_value)

        # 获取当前可用的数据窗口
        available_data = self.history[-window_size:]

        # 如果数据不足，使用较小的窗口
        if len(available_data) < window_size:
            actual_window = len(available_data)
            sigma_adjusted = sigma * (actual_window / window_size)  # 按比例调整sigma
            kernel = self.create_causal_gaussian_kernel(sigma_adjusted, actual_window)
        else:
            kernel = self.create_causal_gaussian_kernel(sigma, window_size)

        # 应用卷积（实际上是加权平均）
        smoothed_value = np.dot(available_data, kernel[::-1])

        return smoothed_value, sigma, window_size

    def predict_next(self):
        """预测下一个值"""
        if len(self.history) < 3:  # 至少需要3个点
            return np.mean(self.history)  # 初始阶段用均值

        # 使用最后一个值计算自适应参数
        sigma, window_size = self.adapt_parameters(self.history[-1])
        available_data = self.history[-window_size:]

        # 调整窗口和sigma
        if len(available_data) < window_size:
            actual_window = len(available_data)
            sigma_adjusted = sigma * (actual_window / window_size)
            kernel = self.create_causal_gaussian_kernel(sigma_adjusted, actual_window)
        else:
            kernel = self.create_causal_gaussian_kernel(sigma, window_size)

        # 预测下一个值：用加权平均
        predicted_value = np.dot(available_data, kernel[::-1])
        return predicted_value

    def smooth_array(self, array_A):
        """平滑整个数组"""
        predict_B=np.zeros_like(array_A)
        array_B = np.zeros_like(array_A)
        sigmas = np.zeros_like(array_A)
        windows = np.zeros_like(array_A, dtype=int)

        for i, value in enumerate(array_A):
            # 对于前几个点，使用渐进式平滑
            if i < 3:
                # 初始阶段使用简单平均避免过度平滑
                array_B[i] = np.mean(array_A[:i + 1])
                sigmas[i] = self.min_sigma
                windows[i] = i + 1
                self.history.append(value)
                if i==2:
                    predict_B[0]=array_B[0]
                    predict_B[1]=array_B[1]
                    predict_B[2]=array_B[2]
                    predict_B[3]=array_B[3]
                else:
                    continue
                # predict_B[i+1] = array_B[i]
            else:
                array_B[i], sigmas[i], windows[i] = self.smooth_value(value)
                # print(len(array_A))
                if i==len(array_A)-1:
                    print("predict",self.predict_next())
                #     continue
                predict_B[i] = self.predict_next() # predict_B[i] 是 i-1 的预测值

        return predict_B,array_B, sigmas, windows

    def predict_array(self,array_A):
        """"""
        array_B = np.zeros_like(array_A)
        for i, value in enumerate(array_A):
            # 对于前几个点，使用渐进式平滑
            array_B[i]= self.predict_next()

        return array_B






# 运行测试
if __name__ == "__main__":
    pass