Kelly Criterion 推到过程

你以概率 p 赢得赌注，回报率为 b（即下注1元，赢时获得 b 元，净收益为 b 元）；以概率 q=1−p 输掉赌注，损失全部下注金额（即回报率为 -1）

一次下注后的资金变化：

win:  $ V_1 = V_0 (1 + fb) $

loss: $ V_1 = V_0 (1 - f) $


n次下注后的资金：

假设赢 W 次，输 L 次（W+L=n），则：
$$ V_n = V_0 (1 + fb)^W (1 - f)^L $$

增长率（对数收益）为：

$$ log(\frac{V_n}{V_0}) = W log(1 + fb) + L log(1 - f) $$

期望增长率（每局）：

$$ G(f) = E[\frac{1}{n} log(\frac{V_n}{V_0})] = p loa(1+fb) + q log(1-f)$$

这里用了大数定律：$ \frac{W}{n} → p, \frac{L}{n} → q$

最大化 G(f)：

对 f 求导并令导数为零：

$ \frac{dG}{df} = p \frac{b}{1 + fb} + q \frac{-1}{1 - f} = 0 $

即：

$\frac{pb}{1+fb} = \frac{q}{1-f} $

解出：

$ f^* = p - \frac{q}{b} $

经典的凯利公式认为 概率和赔率都是独立的, 然而现实更常见的是 b=g(p)，其中 g(p) 是一个函数，例如 g(p)=p 时， Kelly Criterion 退化为普通的伯努利赌博策略。

这样凯利公式将演化为

$ f^* = p - \frac{1-p}{g(p)} $






