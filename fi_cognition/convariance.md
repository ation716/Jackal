# 协方差（Covariance）

本文基于以下文档整理:
[协方差 维基百科](https://en.wikipedia.org/wiki/Covariance)

什么是协方差？

协方差(Covariance) 是统计学中衡量两个随机变量联合变动关系的核心指标。它量化了当其中一个变量变化时，另一个变量如何随之变化：

    正协方差：一个变量增加时，另一个变量也倾向于增加（正相关）

    负协方差：一个变量增加时，另一个变量倾向于减少（负相关）

    零协方差：两个变量之间没有线性关系（不相关）

协方差的概念最早由英国统计学家卡尔·皮尔逊在19世纪末提出，是现代相关分析和回归分析的基础。理解协方差对于数据科学、金融分析和机器学习等领域至关重要。

[Khan Academy: Covariance Intuition](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/covariance-and-the-regression-line)

数学定义与公式
总体协方差


样本协方差（更常用）

$\text{Cov}(X,Y) = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu_X)(y_i - \mu_Y)$


变量说明：

    $X,Y$：两个随机变量

    $\mu_X, \mu_Y$：总体均值

    $\bar{x}, \bar{y}$：样本均值

    $N$：总体数据点数量

    $n$：样本数据点数量

    $E$：期望运算符

    公式解释：Stat Trek: Covariance Formula

协方差的计算方法

```python
import numpy as np
import pandas as pd

# 样本数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

# 使用NumPy计算
cov_np = np.cov(x, y, ddof=0)[0, 1]  # ddof=0表示总体协方差
print(f"NumPy计算总体协方差: {cov_np:.2f}")

# 使用Pandas计算
data = pd.DataFrame({'X': x, 'Y': y})
cov_pd = data.cov().iloc[0, 1]  # 获取协方差矩阵的X-Y元素
print(f"Pandas计算总体协方差: {cov_pd:.2f}")

# 手动计算
def manual_cov(x, y, is_sample=True):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    divisor = n - 1 if is_sample else n
    return cov / divisor

print(f"手动计算样本协方差: {manual_cov(x, y):.2f}")
```

相关系数解决了协方差的量纲问题，使其在不同变量间具有可比性。当需要比较不同变量对的关系强度时，相关系数更为合适。