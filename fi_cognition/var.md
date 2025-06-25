# VAR

Value at Risk is a statistic that quantifies the extent of posssible financial losses within a firm,portfolio,or position over a specific time frame.
Vector Autoregression (VAR) is a statistical method used to model the relationship between multiple time series variables.
VAR模型描述在同一样本期间内的n个变量（内生变量）可以作为它们过去值的线性函数。 
一个VAR(p)模型可以写成为：


$y_t=c+A_1y_{t-1}+A_2y_{t-2}+...+A_py_{t-p}+e_t$



其中：$c$ 是 $n × 1$ 常数向量，$A_i$ 是 $n × n$ 矩阵。$e_t$ 是 $n × 1$ 误差向量，满足：

1. $E(e_t) = 0$  误差项的均值为0
2. $E(e_te^{'}_t) = Ω$ 误差项的协方差矩阵为Ω（一个n×n半正定矩阵）
3. $E(e_te^{'}_{t−k}) = 0$  （对于所有不为0的k都满足）—误差项不存在自我相关

[more](https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F%E8%87%AA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B)




