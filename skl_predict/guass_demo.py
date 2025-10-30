# -*- coding: utf-8 -*-
# @Time    : 2025/9/29 14:12
# @Author  : gaolei
# @FileName: guass_demo.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

# 设置随机种子以保证结果可重现
np.random.seed(42)


class GPRRevenuePredictor:
    def __init__(self, kernel=None):
        """
        初始化高斯过程回归预测器

        Parameters:
        kernel: 核函数，如果为None则使用默认组合核
        """
        if kernel is None:
            # 默认使用组合核：常数核 * RBF核 + 白噪声核
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=0.1)
        else:
            self.kernel = kernel

        self.gpr = GaussianProcessRegressor(kernel=self.kernel,
                                            n_restarts_optimizer=10,
                                            alpha=1e-10,
                                            random_state=42)
        self.is_fitted = False

    def generate_sample_data(self, n_samples=100, noise=0.1):
        """
        生成模拟数据：假设收益w和概率p之间存在非线性关系

        Parameters:
        n_samples: 样本数量
        noise: 噪声水平

        Returns:
        p: 概率值数组 (n_samples, 1)
        w: 收益值数组 (n_samples,)
        """
        # 生成概率p，在0到1之间均匀分布（排除0和1附近极端值）
        p = np.linspace(0.05, 0.95, n_samples).reshape(-1, 1)

        # 定义真实的非线性关系：高概率对应低收益，低概率对应高收益
        # 使用指数衰减函数模拟这种关系
        true_w = 5.0 * np.exp(-3 * p.flatten()) + 0.5

        # 添加噪声
        noise = np.random.normal(0, noise, n_samples)
        w = true_w + noise

        return p, w, true_w

    def fit(self, p, w):
        """
        训练高斯过程回归模型

        Parameters:
        p: 概率值数组 (n_samples, 1)
        w: 收益值数组 (n_samples,)
        """
        print("开始训练高斯过程回归模型...")
        print(f"初始核函数: {self.kernel}")

        self.gpr.fit(p, w)
        self.is_fitted = True

        print(f"优化后的核函数: {self.gpr.kernel_}")
        print(f"对数边际似然: {self.gpr.log_marginal_likelihood_value_:.3f}")

    def predict(self, p, return_std=True):
        """
        使用训练好的模型进行预测

        Parameters:
        p: 要预测的概率值数组 (n_samples, 1)
        return_std: 是否返回标准差

        Returns:
        如果return_std为True: 返回 (均值, 标准差)
        否则: 返回均值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        if return_std:
            w_pred, w_std = self.gpr.predict(p, return_std=True)
            return w_pred, w_std
        else:
            return self.gpr.predict(p, return_std=False)

    def calculate_prediction_intervals(self, p, confidence=0.95):
        """
        计算预测区间

        Parameters:
        p: 概率值数组
        confidence: 置信水平

        Returns:
        w_pred: 预测均值
        lower_bound: 预测区间下界
        upper_bound: 预测区间上界
        """
        w_pred, w_std = self.predict(p, return_std=True)

        # 计算z值
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        lower_bound = w_pred - z * w_std
        upper_bound = w_pred + z * w_std

        return w_pred, lower_bound, upper_bound

    def plot_results(self, p_train, w_train, p_test=None, w_test=None, true_w=None):
        """
        可视化训练结果和预测

        Parameters:
        p_train: 训练集概率
        w_train: 训练集收益
        p_test: 测试集概率（可选）
        w_test: 测试集收益（可选）
        true_w: 真实函数值（可选）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        # 创建用于绘图的密集点
        p_plot = np.linspace(0.01, 0.99, 200).reshape(-1, 1)

        # 预测
        w_pred, w_std = self.predict(p_plot)
        w_pred_plot, lower_bound, upper_bound = self.calculate_prediction_intervals(p_plot)

        plt.figure(figsize(12, 8))

        # 绘制训练数据
        plt.scatter(p_train, w_train, c='blue', alpha=0.6, label='训练数据', s=50)

        # 绘制真实函数（如果已知）
        if true_w is not None:
            plt.plot(p_train, true_w, 'r-', linewidth=2, label='真实函数', alpha=0.8)

        # 绘制预测均值
        plt.plot(p_plot, w_pred, 'g-', linewidth=2, label='GPR预测均值')

        # 绘制置信区间
        plt.fill_between(p_plot.flatten(), lower_bound, upper_bound,
                         alpha=0.3, color='green', label=f'95%置信区间')

        # 绘制测试数据（如果存在）
        if p_test is not None and w_test is not None:
            plt.scatter(p_test, w_test, c='red', alpha=0.6, label='测试数据', s=50, marker='s')

        plt.xlabel('概率 p', fontsize=12)
        plt.ylabel('收益 w', fontsize=12)
        plt.title('高斯过程回归：收益 vs 概率', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, p_test, w_test):
        """
        评估模型性能

        Parameters:
        p_test: 测试集概率
        w_test: 测试集真实收益
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        w_pred, w_std = self.predict(p_test)

        mse = mean_squared_error(w_test, w_pred)
        mae = mean_absolute_error(w_test, w_pred)

        print("\n模型性能评估:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"预测标准差范围: [{w_std.min():.4f}, {w_std.max():.4f}]")

        return mse, mae


def main():
    """
    主函数：演示完整的工作流程
    """
    # 1. 初始化预测器
    predictor = GPRRevenuePredictor()

    # 2. 生成样本数据
    print("生成样本数据...")
    p, w, true_w = predictor.generate_sample_data(n_samples=80, noise=0.2)

    # 3. 分割数据为训练集和测试集
    p_train, p_test, w_train, w_test = train_test_split(
        p, w, test_size=0.2, random_state=42
    )

    # 4. 训练模型
    predictor.fit(p_train, w_train)

    # 5. 评估模型
    predictor.evaluate_model(p_test, w_test)

    # 6. 可视化结果
    predictor.plot_results(p_train, w_train, p_test, w_test, true_w)

    # 7. 对新概率点进行预测示例
    print("\n新概率点预测示例:")
    new_p = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])

    for p_val in new_p:
        w_pred, w_std = predictor.predict(p_val.reshape(1, -1))
        lower = w_pred - 1.96 * w_std
        upper = w_pred + 1.96 * w_std
        print(f"p = {p_val[0]:.1f}: 预测收益 = {w_pred[0]:.3f} ± {w_std[0]:.3f} "
              f"(95% CI: [{lower[0]:.3f}, {upper[0]:.3f}])")

    # 8. 演示如何使用预测结果进行凯利公式计算
    print("\n凯利公式应用示例:")
    p_kelly = 0.6
    w_pred_kelly, w_std_kelly = predictor.predict(np.array([[p_kelly]]))
    w_kelly = w_pred_kelly[0]

    # 使用保守估计（均值 - 1倍标准差）
    w_conservative = w_kelly - w_std_kelly[0]

    # 假设损失率固定为 -1
    r_loss = -1.0

    # 经典凯利公式
    kelly_fraction = p_kelly - (1 - p_kelly) / w_kelly
    kelly_conservative = p_kelly - (1 - p_kelly) / w_conservative

    print(f"对于 p = {p_kelly}:")
    print(f"  预测收益: {w_kelly:.3f} ± {w_std_kelly[0]:.3f}")
    print(f"  标准凯利比例: {kelly_fraction:.3f}")
    print(f"  保守凯利比例 (使用均值-1σ): {kelly_conservative:.3f}")


if __name__ == "__main__":
    main()