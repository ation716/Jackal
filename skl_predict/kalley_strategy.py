# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 13:19
# @Author  : gaolei
# @FileName: kally_strategy.py
# @Software: PyCharm
"""
 本文件主要功能 是将 由 fi_cognition/kalley.md 第 54 行公式 由代码实现，封装到 类 KellyCR 类中，
 支持输入离散得预测数据，  格式为 dataframe ，返回最佳比列 和 次最佳比列（即最佳比列对应的p0的左右临近的 p的 f 值）
 在 if main 中 通过 skl_predict/result.csv 测试输出f
"""
import os
import numpy as np
import pandas as pd


class KellyCR:
    """
    广义 Kelly 仓位公式（fi_cognition/kalley.md 第54行）：

        f* = g(b1)/b2 - (1-g(b1))/b1

    其中：
        b1  = 以阈值 r0 为分界，return > r0 时的条件期望收益
        b2  = return <= r0 时的条件期望亏损幅度（正数）
        g(b1) = P(return > r0)，即赢的概率

    对离散分布的处理：
        遍历所有离散 return 值作为分界阈值 r0，
        对每个 r0 计算对应的 f，取 f 最大时的结果为最佳仓位比例，
        同时返回相邻左右两个阈值对应的 f 作为次最佳。
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 含 ['return', 'pdf', 'cdf'] 列的 DataFrame
        """
        self.df = df.reset_index(drop=True).copy()
        # 用 pdf * Δr 近似每个离散点的概率，再归一化
        dr = self.df['return'].diff().abs().median()
        self.df['prob'] = self.df['pdf'] * dr
        self.df['prob'] /= self.df['prob'].sum()

    def _f_at(self, idx: int) -> float:
        """以第 idx 行的 return 值为阈值，计算 Kelly 仓位比例。"""
        threshold = self.df.loc[idx, 'return']
        win_mask = self.df['return'] > threshold
        loss_mask = ~win_mask

        p = self.df.loc[win_mask, 'prob'].sum()   # 赢的概率 g(b1)
        q = self.df.loc[loss_mask, 'prob'].sum()  # 输的概率 1-g(b1)

        if p <= 0 or q <= 0:
            return np.nan

        # 条件期望收益 b1（正数）
        b1 = (self.df.loc[win_mask, 'return'] * self.df.loc[win_mask, 'prob']).sum() / p
        # 条件期望亏损幅度 b2（正数）
        b2 = -(self.df.loc[loss_mask, 'return'] * self.df.loc[loss_mask, 'prob']).sum() / q

        if b1 <= 0 or b2 <= 0:
            return np.nan

        # fi_cognition/kalley.md 第54行公式
        return p / b2 - q / b1

    def compute(self) -> dict:
        """
        遍历所有阈值，找到最佳 Kelly 仓位比例及相邻次最佳。

        Returns:
            dict:
                best_f        最佳仓位比例
                left_f        左邻阈值对应的 f
                right_f       右邻阈值对应的 f
                best_threshold  最佳阈值（return 值）
                best_idx      最佳阈值在 df 中的行索引
                p_win         最佳阈值下的赢概率
        """
        n = len(self.df)
        f_values = np.array([self._f_at(i) for i in range(n)])

        valid = np.where(~np.isnan(f_values))[0]
        if len(valid) == 0:
            raise ValueError("未找到有效的 Kelly 仓位比例，请检查输入数据")

        best_local = valid[np.argmax(f_values[valid])]
        best_f = float(f_values[best_local])

        left_idx = best_local - 1 if best_local > 0 else None
        right_idx = best_local + 1 if best_local < n - 1 else None

        left_f = float(f_values[left_idx]) if left_idx is not None else None
        right_f = float(f_values[right_idx]) if right_idx is not None else None

        threshold = float(self.df.loc[best_local, 'return'])
        p_win = float(self.df.loc[self.df['return'] > threshold, 'prob'].sum())

        return {
            'best_f': best_f,
            'left_f': left_f,
            'right_f': right_f,
            'best_threshold': threshold,
            'best_idx': int(best_local),
            'p_win': p_win,
        }


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'result2.csv')
    df = pd.read_csv(csv_path)

    kelly = KellyCR(df)
    result = kelly.compute()

    print("=== Kelly 仓位计算结果 ===")
    print(f"最佳仓位比例  best_f    : {result['best_f']:.6f}")
    print(f"左邻次最佳    left_f    : {result['left_f']:.6f}")
    print(f"右邻次最佳    right_f   : {result['right_f']:.6f}")
    print(f"最佳分界阈值  threshold : {result['best_threshold']:.6f}")
    print(f"赢的概率      p_win     : {result['p_win']:.4%}")
