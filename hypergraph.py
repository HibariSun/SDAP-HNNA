# Author:Hibari
# 2025年11月18日17时31分50秒
# hypergraph.py
import numpy as np
import torch
from utils import device


class HypergraphConstructor:
    """超图构造器（回归“三超图”版本）

    输出三个超图：
      - H_all: 统一超图（sno+dis 节点），包含 sno/dis 的 KNN 相似性超边 +（可选）星型关联超边
      - H_sno: snoRNA-only 超图（列数=num_snorna），每个 sno 一条 KNN 超边
      - H_dis: disease-only 超图（列数=num_disease），每个 disease 一条 KNN 超边

    稀疏数据集建议：
      - KNN 超边加入中心节点自身（自环），避免节点“挂空边”
      - 关联超边使用星型/邻域超边（每个 sno 一条 + 每个 disease 一条），比“一对一两点边”更适合稀疏高阶传播
    """

    def __init__(self, association_matrix, snorna_sim, disease_sim):
        self.association_matrix = np.asarray(association_matrix)
        self.snorna_sim = np.asarray(snorna_sim)
        self.disease_sim = np.asarray(disease_sim)

        self.num_snorna = self.association_matrix.shape[0]
        self.num_disease = self.association_matrix.shape[1]

    def construct_hypergraph(self, k_snorna=15, k_disease=15,
                             use_association_edges=True, association_weight=0.5):
        """构造统一超图 H_all（兼容旧接口）"""
        H_all, _, _ = self.construct_multi_hypergraph(
            k_snorna=k_snorna,
            k_disease=k_disease,
            use_association_edges=use_association_edges,
            association_weight=association_weight
        )
        return H_all

    def construct_multi_hypergraph(self,
                                  k_snorna=15,
                                  k_disease=15,
                                  use_association_edges=True,
                                  association_weight=0.5):
        """构建三超图：H_all、H_sno、H_dis"""
        print(f"\n[Hypergraph] 构建三超图...")
        print(f"  - k_snorna={k_snorna}, k_disease={k_disease}")
        use_association_edges = bool(use_association_edges and association_weight > 0)
        print(f"  - use_association_edges={use_association_edges}, association_weight={association_weight}")

        # 1) KNN 图
        snorna_knn = self._build_knn_graph(self.snorna_sim, k_snorna)
        disease_knn = self._build_knn_graph(self.disease_sim, k_disease)

        num_nodes = self.num_snorna + self.num_disease

        # 2) H_all（统一超图）
        num_sim_edges = self.num_snorna + self.num_disease
        if use_association_edges:
            sno_deg = int((self.association_matrix.sum(axis=1) > 0).sum())
            dis_deg = int((self.association_matrix.sum(axis=0) > 0).sum())
            num_assoc_edges = sno_deg + dis_deg
        else:
            num_assoc_edges = 0
        num_hyperedges_all = num_sim_edges + num_assoc_edges

        H_all = np.zeros((num_nodes, num_hyperedges_all), dtype=np.float32)
        edge_idx = 0

        # 2.1 snoRNA 相似性超边（每个 i 一条：{i}+KNN(i)）
        for i in range(self.num_snorna):
            H_all[i, edge_idx] = 1.0  # ★中心节点自环
            neighbors = np.where(snorna_knn[i] > 0)[0]
            if neighbors.size > 0:
                H_all[neighbors, edge_idx] = snorna_knn[i, neighbors]
            edge_idx += 1

        # 2.2 disease 相似性超边（每个 j 一条：{j}+KNN(j)）
        for j in range(self.num_disease):
            H_all[self.num_snorna + j, edge_idx] = 1.0  # ★中心节点自环（注意偏移）
            neighbors = np.where(disease_knn[j] > 0)[0]
            if neighbors.size > 0:
                H_all[self.num_snorna + neighbors, edge_idx] = disease_knn[j, neighbors]
            edge_idx += 1

        # 2.3 关联超边（星型/邻域超边）
        if use_association_edges:
            A = self.association_matrix

            # sno 星型超边：{sno_i} ∪ N(i)
            for i in range(self.num_snorna):
                js = np.where(A[i] > 0)[0]
                if js.size == 0:
                    continue
                H_all[i, edge_idx] = 1.0
                H_all[self.num_snorna + js, edge_idx] = association_weight
                edge_idx += 1

            # disease 星型超边：{dis_j} ∪ N(j)
            for j in range(self.num_disease):
                is_ = np.where(A[:, j] > 0)[0]
                if is_.size == 0:
                    continue
                H_all[self.num_snorna + j, edge_idx] = 1.0
                H_all[is_, edge_idx] = association_weight
                edge_idx += 1

        # 3) H_sno：snoRNA-only 超图（列数=num_snorna）
        H_sno = np.zeros((num_nodes, self.num_snorna), dtype=np.float32)
        for i in range(self.num_snorna):
            H_sno[i, i] = 1.0  # ★中心节点自环
            neighbors = np.where(snorna_knn[i] > 0)[0]
            if neighbors.size > 0:
                H_sno[neighbors, i] = snorna_knn[i, neighbors]

        # 4) H_dis：disease-only 超图（列数=num_disease）
        H_dis = np.zeros((num_nodes, self.num_disease), dtype=np.float32)
        for j in range(self.num_disease):
            H_dis[self.num_snorna + j, j] = 1.0  # ★中心节点自环
            neighbors = np.where(disease_knn[j] > 0)[0]
            if neighbors.size > 0:
                H_dis[self.num_snorna + neighbors, j] = disease_knn[j, neighbors]

        print(f"  ✓ 三超图构建完成:")
        print(f"    - H_all: {H_all.shape} (sim_edges={num_sim_edges}, assoc_edges={num_assoc_edges})")
        print(f"    - H_sno: {H_sno.shape}")
        print(f"    - H_dis: {H_dis.shape}")

        return (
            torch.as_tensor(H_all, dtype=torch.float32, device=device),
            torch.as_tensor(H_sno, dtype=torch.float32, device=device),
            torch.as_tensor(H_dis, dtype=torch.float32, device=device),
        )

    @staticmethod
    def _build_knn_graph(similarity_matrix, k):
        """从相似度矩阵构建 KNN 图（返回对称权重邻接矩阵）"""
        similarity_matrix = np.asarray(similarity_matrix)
        n = similarity_matrix.shape[0]
        knn_graph = np.zeros((n, n), dtype=np.float32)

        k = int(k)
        k = max(1, min(k, n - 1))  # 防止越界/空

        for i in range(n):
            sim_scores = similarity_matrix[i].copy()
            sim_scores[i] = -1  # 排除自己
            top_k_indices = np.argsort(sim_scores)[-k:]
            knn_graph[i, top_k_indices] = similarity_matrix[i, top_k_indices]

        # 对称化（无向 KNN）
        knn_graph = (knn_graph + knn_graph.T) / 2.0
        return knn_graph
