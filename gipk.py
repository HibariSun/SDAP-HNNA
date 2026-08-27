# Author:Hibari
# 2025年11月18日17时30分28秒
# gipk.py
import numpy as np


# GIPK计算模块
class GIPKCalculator:
    """高斯相互作用谱核(GIPK)计算器"""

    def __init__(self, association_matrix, gamma_ratio=1.0):
        """
        初始化GIPK计算器

        参数:
            association_matrix: 关联矩阵 (m×n) m: snoRNA的数量 n: disease的数量
            gamma_ratio: 带宽控制参数 γ'参数，通常设为1 控制高斯核的宽度 影响相似性的"衰减速度" 较大的γ' → 更快衰减 → 只有非常相似的实体才有高相似度 较小的γ' → 更慢衰减 → 更多实体被认为相似
        """
        self.association_matrix = association_matrix
        self.gamma_ratio = gamma_ratio

    def compute_gipk_snorna(self):  # 计算snoRNA的GIPK相似性
        """
        计算snoRNA的GIPK相似性矩阵

        公式: GIPK_S(si, sj) = exp(-γs × ||R(si) - R(sj)||²)
              γs = γ's × m / Σ||R(si)||²

        返回: GIPK相似性矩阵 (m×m)
        """
        m, n = self.association_matrix.shape  # 获取矩阵维度

        # 计算归一化的带宽参数 γs
        sum_norm_squared = np.sum(np.linalg.norm(self.association_matrix, axis=1) ** 2)  # 计算每个snoRNA关联向量的L2范数平方和
        gamma_s = self.gamma_ratio * m / sum_norm_squared  # 计算归一化的γs

        print(f"\n[GIPK计算] snoRNA:")
        print(f"  - 关联矩阵形状: {self.association_matrix.shape}")
        print(f"  - γs (gamma): {gamma_s:.6f}")

        # 计算GIPK相似性矩阵
        GIPK_matrix = np.zeros((m, m), dtype=np.float32)  # 初始化GIPK矩阵

        # 双重循环计算相似性
        for i in range(m):
            for j in range(m):
                # 计算两个snoRNA关联向量的差
                diff = self.association_matrix[i, :] - self.association_matrix[j, :]
                # 计算差向量的欧氏距离平方
                norm_squared = np.sum(diff ** 2)
                # 应用高斯核函数
                GIPK_matrix[i, j] = np.exp(-gamma_s * norm_squared)

        print(f"  - 计算完成: ({m}, {m})")
        print(f"  - 对角线值: {GIPK_matrix[0, 0]:.6f} (应为1.0)")  # 验证对角线

        return GIPK_matrix

    def compute_gipk_disease(self):
        """
        计算disease的GIPK相似性矩阵

        使用关联矩阵的转置（每行代表一个disease与所有snoRNA的关联）

        返回: GIPK相似性矩阵 (n×n)
        """
        m, n = self.association_matrix.shape
        A_T = self.association_matrix.T  # 转置关联矩阵

        # 计算归一化的带宽参数 γd
        sum_norm_squared = np.sum(np.linalg.norm(A_T, axis=1) ** 2)
        gamma_d = self.gamma_ratio * n / sum_norm_squared

        print(f"\n[GIPK计算] Disease:")
        print(f"  - 关联矩阵转置形状: {A_T.shape}")
        print(f"  - γd (gamma): {gamma_d:.6f}")

        # 计算GIPK相似性矩阵
        GIPK_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                diff = A_T[i, :] - A_T[j, :]
                norm_squared = np.sum(diff ** 2)
                GIPK_matrix[i, j] = np.exp(-gamma_d * norm_squared)

        print(f"  - 计算完成: ({n}, {n})")
        print(f"  - 对角线值: {GIPK_matrix[0, 0]:.6f} (应为1.0)")

        return GIPK_matrix
