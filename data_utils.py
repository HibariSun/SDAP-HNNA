# Author:Hibari
# 2025年11月18日17时35分33秒

# data_utils.py
import os
import numpy as np
import pandas as pd

from gipk import GIPKCalculator


# 数据加载类
class DataLoader:
    """改进的数据加载类 - 自己计算GIPK并融合外部相似性
       支持 snoRNA / disease 使用不同的 α，并做网格搜索时复用 GIPK 计算结果
    """

    def __init__(self, data_path='./data/',
                 alpha_snorna=None, alpha_disease=None):
        """
        alpha_snorna: snoRNA 相似性中 GIPK 所占比例 (0~1)，网格搜索时可覆盖
        alpha_disease: disease 相似性中 GIPK 所占比例 (0~1)，网格搜索时可覆盖
        """
        self.data_path = data_path
        self.alpha_snorna = alpha_snorna
        self.alpha_disease = alpha_disease

        # 缓存基础矩阵，避免在网格搜索时重复计算 GIPK
        self._association_matrix = None
        self._snorna_gipk = None
        self._disease_gipk = None
        self._sno_external = None
        self._disease_external = None
        self._external_available = False

        # 额外外部Embedding（用于后融合，不参与相似性构图）
        self._sno_rnaernie_emb = None   # shape: [num_snorna, d_sno_emb]
        self._disease_llm_emb = None   # shape: [num_disease, d_dis_emb]
        self._sno_ids = None
        self._disease_ids = None

    def load_all_data(self, alpha_snorna=None, alpha_disease=None):
        """
        统一的数据加载接口

        返回:
            association_matrix: [num_snorna, num_disease] 的 0/1 关联矩阵
            snorna_sim_fused:   [num_snorna, num_snorna] 的 snoRNA 相似性
            disease_sim_fused:  [num_disease, num_disease] 的 disease 相似性

        当前设计:
            - snoRNA: 使用 sno_p2p_smith.csv 作为外部相似性（Smith-Waterman 序列相似性）
            - disease: 使用 disease_do_similarity.csv 作为外部相似性（DO 语义相似性）

            - RNAErnie / LLM embedding 不参与相似性融合与构图，而是用于模型后融合(late fusion)

            - 然后用 SWF 做:
                snorna_sim_fused  = alpha_snorna * GIPK + (1 - alpha_snorna) * 外部(SNF)
                disease_sim_fused = alpha_disease * GIPK + (1 - alpha_disease) * 外部
              如果你把 alpha_snorna / alpha_disease 设为 0，就是「完全不用 GIPK」。
        """
        print("\n[步骤 1] 加载数据并计算相似性...")

        # ========== 第一次调用时，真正加载 & 计算 ==========
        if self._association_matrix is None:
            # 1. 加载关联矩阵
            adj_df = pd.read_csv(os.path.join(self.data_path, "adj_index.csv"), index_col=0)
            self._association_matrix = adj_df.values.astype(np.float32)
            self._sno_ids = list(adj_df.index.astype(str))
            self._disease_ids = list(adj_df.columns.astype(str))

            print(f"  ✓ 关联矩阵加载完成:")
            print(f"    - snoRNA数量: {self._association_matrix.shape[0]}")
            print(f"    - Disease数量: {self._association_matrix.shape[1]}")
            print(f"    - 已知关联数量: {int(self._association_matrix.sum())}")

            # 2. 基于关联矩阵计算 GIPK（你现在如果不想用，只要后面 alpha=0 即可）
            gipk_calculator = GIPKCalculator(self._association_matrix, gamma_ratio=1.0)
            snorna_gipk = gipk_calculator.compute_gipk_snorna()
            disease_gipk = gipk_calculator.compute_gipk_disease()

            self._snorna_gipk = self._normalize_similarity(snorna_gipk)
            self._disease_gipk = self._normalize_similarity(disease_gipk)

            print("\n  ✓ GIPK 计算和归一化完成")

            
            # 3. 加载外部相似性矩阵（用于与每折 GIPK 线性融合，从而参与构图/初始相似性特征）
            #    你当前的设定：
            #      - snoRNA 外部相似性：Smith-Waterman 序列相似性（sno_p2p_smith.csv）
            #      - disease 外部相似性：DO 语义相似性（disease_do_similarity.csv）
            sno_smith_path = os.path.join(self.data_path, "sno_p2p_smith.csv")
            dis_do_path = os.path.join(self.data_path, "disease_do_similarity.csv")

            # 4. 加载外部 embedding（用于“后融合/late fusion”，不参与构图）
            sno_rna_emb_path = os.path.join(self.data_path, "sno_rnaernie_emb.npy")
            dis_llm_emb_path = os.path.join(self.data_path, "disease_feat_bgeicl.csv")

            self._external_available = (
                    os.path.exists(sno_smith_path)
                    and os.path.exists(dis_do_path)
            )

            if not self._external_available:
                print("\n[警告] 未找到以下一个或多个外部相似性文件：")
                print("       - sno_p2p_smith.csv")
                print("       - disease_do_similarity.csv")
                print("       将仅使用每折 GIPK 相似性进行构图。")
            else:
                print("\n  ✓ 外部相似性文件已找到, 开始加载（不再进行SNF）:")

                sno_smith = pd.read_csv(sno_smith_path, index_col=0)
                disease_do = pd.read_csv(dis_do_path, index_col=0)

                # 保存 id 顺序，用于对齐外部 embedding
                self._sno_ids = list(sno_smith.index.astype(str))
                self._disease_ids = list(disease_do.index.astype(str))

                sno_smith = sno_smith.values.astype(np.float32)
                disease_do = disease_do.values.astype(np.float32)

                # 形状检查（必须与 GIPK 对齐）
                if sno_smith.shape != self._snorna_gipk.shape:
                    raise ValueError(
                        f"snoRNA 外部相似性矩阵(smith)形状为 {sno_smith.shape}, "
                        f"但 GIPK 相似性为 {self._snorna_gipk.shape}, 请检查行列顺序是否一致。"
                    )
                if disease_do.shape != self._disease_gipk.shape:
                    raise ValueError(
                        f"disease 外部相似性矩阵(DO)形状为 {disease_do.shape}, "
                        f"但 GIPK 相似性为 {self._disease_gipk.shape}, 请检查行列顺序是否一致。"
                    )

                # 归一化
                self._sno_external = self._normalize_similarity(sno_smith)
                self._disease_external = self._normalize_similarity(disease_do)

                print("  ✓ 外部相似性加载完成:")
                print("    - snoRNA 外部: Smith-Waterman 序列相似性")
                print("    - disease 外部: DO 语义相似性")

            # ====== 5. 外部Embedding（late fusion）加载与对齐 ======
            # 即使相似性文件缺失，也尝试加载 embedding，便于后续做 late fusion
            if os.path.exists(sno_rna_emb_path):
                try:
                    self._sno_rnaernie_emb = np.load(sno_rna_emb_path).astype(np.float32)
                except Exception as e:
                    print(f"[警告] 读取 sno_rnaernie_emb.npy 失败: {e}")
                    self._sno_rnaernie_emb = None
            else:
                self._sno_rnaernie_emb = None

            if os.path.exists(dis_llm_emb_path):
                try:
                    df_dis = pd.read_csv(dis_llm_emb_path)
                    # 默认格式：第一列是 DOID（列名可能是 'DOI' / 'DOID' / 'id'）
                    id_col = df_dis.columns[0]
                    df_dis = df_dis.set_index(id_col)
                    # 对齐 disease id 顺序
                    if self._disease_ids is not None:
                        df_dis = df_dis.loc[self._disease_ids]
                    self._disease_llm_emb = df_dis.values.astype(np.float32)
                except Exception as e:
                    print(f"[警告] 读取 disease_feat_bgeicl.csv 失败: {e}")
                    self._disease_llm_emb = None
            else:
                self._disease_llm_emb = None

            # 对齐 snoRNA embedding 的顺序：默认认为与 sno_p2p_smith.csv / adj_index.csv 顺序一致
            if self._sno_rnaernie_emb is not None and self._association_matrix is not None:
                if self._sno_rnaernie_emb.shape[0] != self._association_matrix.shape[0]:
                    print(
                        f"[警告] sno_rnaernie_emb.npy 行数={self._sno_rnaernie_emb.shape[0]} "
                        f"与 snoRNA 数量={self._association_matrix.shape[0]} 不一致，已禁用该 embedding。"
                    )
                    self._sno_rnaernie_emb = None

            if self._disease_llm_emb is not None and self._association_matrix is not None:
                if self._disease_llm_emb.shape[0] != self._association_matrix.shape[1]:
                    print(
                        f"[警告] disease_feat_bgeicl.csv 行数={self._disease_llm_emb.shape[0]} "
                        f"与 disease 数量={self._association_matrix.shape[1]} 不一致，已禁用该 embedding。"
                    )
                    self._disease_llm_emb = None

        else:
            print("  ✓ 使用缓存的关联矩阵与相似性（GIPK + 外部矩阵）")

        # ========== 每次调用可以传入不同的 alpha ==========
        if alpha_snorna is None:
            alpha_snorna = self.alpha_snorna
        if alpha_disease is None:
            alpha_disease = self.alpha_disease

        # 当前你的需求：想完全不用 GIPK，就在 main.py / grid_search 里把 alpha 设为 0.0
        if (not self._external_available) or (self._sno_external is None) or (self._disease_external is None):
            snorna_sim_fused = self._snorna_gipk
            disease_sim_fused = self._disease_gipk
            print("\n  ✓ 使用 GIPK 相似性（未进行外部融合）")
        else:
            snorna_sim_fused = self._swf_fusion(
                self._snorna_gipk, self._sno_external,
                name="snoRNA", alpha=alpha_snorna
            )
            disease_sim_fused = self._swf_fusion(
                self._disease_gipk, self._disease_external,
                name="Disease", alpha=alpha_disease
            )

            print("\n  ✓ 相似性融合完成:")
            print(f"    - snoRNA 相似性矩阵形状: {snorna_sim_fused.shape}")
            print(f"    - Disease 相似性矩阵形状: {disease_sim_fused.shape}")

        return (
            self._association_matrix.astype(np.float32),
            snorna_sim_fused.astype(np.float32),
            disease_sim_fused.astype(np.float32)
        )

    def _snf_two_similarities(self, S1, S2, K=20, t=20, mu=0.5):
        """
        使用一个简化版的 SNF 算法，把两个相似性矩阵 S1 / S2 融合成一个矩阵。

        参数:
            S1, S2: 形状 [N, N] 的相似性矩阵（建议先做过 _normalize_similarity）
            K: 每个样本保留的近邻数
            t: 迭代次数
            mu: 对弱边的抑制程度 (0<mu<=1, 越小越强化大边)

        返回:
            fused: 形状 [N, N] 的融合后相似性矩阵，已归一化到 [0, 1]，对角线为 1
        """

        def _make_affinity(S):
            """从相似性 S 构造 KNN 的转移矩阵 P"""
            N = S.shape[0]
            # 去掉自环
            S = S - np.diag(np.diag(S))

            W = np.zeros_like(S, dtype=np.float32)
            K_eff = min(K, N - 1)
            for i in range(N):
                # 取每一行最大的 K 个近邻
                idx = np.argsort(S[i])[-K_eff:]
                W[i, idx] = S[i, idx]

            # 对称化
            W = (W + W.T) / 2.0

            # 变成转移矩阵 P
            row_sum = W.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            P = W / (2.0 * row_sum)
            np.fill_diagonal(P, 0.5)
            return P

        # 先构造两个网络的初始转移矩阵
        P1 = _make_affinity(S1)
        P2 = _make_affinity(S2)
        P_list = [P1, P2]

        # 迭代信息扩散
        for _ in range(t):
            new_list = []
            for i in range(2):
                other = P_list[1 - i]  # 只有两个网络时，另一个就是 "other"
                P_bar = other

                # 信息扩散
                P_new = P_list[i] @ P_bar @ P_list[i].T
                if mu != 1.0:
                    P_new = np.power(P_new, mu)

                # 重新构造 KNN + 归一化
                P_new = _make_affinity(P_new)
                new_list.append(P_new)

            P_list = new_list

        fused = (P_list[0] + P_list[1]) / 2.0
        fused = self._normalize_similarity(fused)
        return fused

    def _normalize_similarity(self, sim_matrix):
        """归一化相似性矩阵: 对称化 + 对角线设为 1 + 截断到 [0,1]"""
        sim_matrix = (sim_matrix + sim_matrix.T) / 2.0
        np.fill_diagonal(sim_matrix, 1.0)
        sim_matrix = np.clip(sim_matrix, 0, 1)
        return sim_matrix

    def _swf_fusion(self, sim_gipk, sim_ext, name="", alpha=None):
        """
        SWF 融合:
            如果 alpha 不为空:
                S_fused = alpha * sim_gipk + (1 - alpha) * sim_ext
            如果 alpha 为 None:
                使用“自动权重模式”（按均值算权重）
        """
        if sim_gipk.shape != sim_ext.shape:
            raise ValueError(
                f"SWF 融合失败: 两个矩阵形状不一致: {sim_gipk.shape} vs {sim_ext.shape}"
            )

        # 先做一次归一化和对称化
        sim_gipk = self._normalize_similarity(sim_gipk)
        sim_ext = self._normalize_similarity(sim_ext)

        if alpha is None:
            # 兼容旧逻辑: 按平均值自动算权重
            w_gipk = float(np.mean(sim_gipk))
            w_ext = float(np.mean(sim_ext))
            denom = w_gipk + w_ext + 1e-8
            alpha = w_gipk / denom
            print(f"    - [{name}] 自动计算 alpha(GIPK)={alpha:.4f} (基于均值)")
        else:
            alpha = float(alpha)
            alpha = max(0.0, min(1.0, alpha))

        print(
            f"    - [{name}] 最终使用 alpha(GIPK)={alpha:.4f}, "
            f"1-alpha(EXT)={1 - alpha:.4f}"
        )

        fused = alpha * sim_gipk + (1.0 - alpha) * sim_ext
        fused = self._normalize_similarity(fused)
        return fused


# 样本准备
def prepare_samples(association_matrix, train_indices, test_indices):  # 从关联矩阵中提取正负样本
    """从关联矩阵中提取正负样本"""
    """
    association_matrix：snoRNA-disease关联矩阵，形状 [num_snorna, num_disease]
    train_indices：训练集的正样本索引（来自K折交叉验证）
    test_indices：测试集的正样本索引（来自K折交叉验证）
    """
    num_snorna, num_disease = association_matrix.shape  # 获取矩阵维度

    # 找出所有正样本
    pos_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_samples.append((i, j))

    pos_samples = np.array(pos_samples)

    # 划分正样本
    train_pos = pos_samples[train_indices]
    test_pos = pos_samples[test_indices]

    # 提取负样本
    neg_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 0:
                neg_samples.append((i, j))

    neg_samples = np.array(neg_samples)
    np.random.shuffle(neg_samples)  # 随机打乱负样本

    # 平衡采样负样本
    train_neg = neg_samples[:len(train_pos)]
    test_neg = neg_samples[len(train_pos):len(train_pos) + len(test_pos)]

    return train_pos, train_neg, test_pos, test_neg


def sample_negatives_mixed(
        assoc_full,  # 原始 association_matrix（含 test_pos=1，避免把 test_pos 抽成负）
        train_assoc,  # 当前折 train_assoc_matrix（test_pos 已置0，用于做“可能为正”的过滤评分）
        pos_pairs,  # train_pos: np.array shape [n_pos, 2]
        snorna_sim, disease_sim,  # 当前折相似度（已融合GIPK+外部）
        num_negs=None,
        hard_ratio=0.5,
        hard_k=50,
        avoid_top_quantile=0.90,  # 过滤掉“最像正样本”的那一部分0
        seed=42
):
    rng = np.random.default_rng(seed)
    pos_pairs = np.asarray(pos_pairs)
    if num_negs is None:
        num_negs = len(pos_pairs)

    # -------- 1) 用相似性传播做一个“伪打分”，过滤疑似真阳性的0 --------
    # score = S_sno @ A_train @ S_dis
    # 直觉：如果 i 与某些正连接 sno 很相似，且 j 与那些 disease 也相似，那 (i,j) 更可能为正
    score = snorna_sim @ train_assoc @ disease_sim  # (n_sno, n_dis)

    zero_mask = (assoc_full == 0)
    zero_scores = score[zero_mask]
    if zero_scores.size > 0:
        thr = np.quantile(zero_scores, avoid_top_quantile)
        candidate_mask = zero_mask & (score <= thr)
    else:
        candidate_mask = zero_mask

    # 预取候选0坐标
    cand_coords = np.argwhere(candidate_mask)  # [n_cand, 2]
    # 去重用
    neg_set = set()

    # -------- 2) hard negatives：围绕每个正样本找“相似但未连接”的负 --------
    n_hard = int(num_negs * hard_ratio)
    if n_hard > 0:
        # 预先取每个 disease 的 topK 相似 disease；每个 sno 的 topK 相似 sno
        top_dis = np.argsort(-disease_sim, axis=1)[:, 1:hard_k + 1]
        top_sno = np.argsort(-snorna_sim, axis=1)[:, 1:hard_k + 1]

        for _ in range(n_hard):
            i, j = pos_pairs[rng.integers(len(pos_pairs))]

            if rng.random() < 0.5:
                # 固定 sno i，找相似 disease
                for j2 in top_dis[j]:
                    if candidate_mask[i, j2]:
                        neg_set.add((int(i), int(j2)))
                        break
            else:
                # 固定 disease j，找相似 sno
                for i2 in top_sno[i]:
                    if candidate_mask[i2, j]:
                        neg_set.add((int(i2), int(j)))
                        break

            if len(neg_set) >= n_hard:
                break

    # -------- 3) uniform negatives：从候选0里补齐 --------
    n_uniform = num_negs - len(neg_set)
    if n_uniform > 0:
        # 从 cand_coords 里无放回抽样
        pick = rng.choice(len(cand_coords), size=min(n_uniform, len(cand_coords)), replace=False)
        for (i, j) in cand_coords[pick]:
            neg_set.add((int(i), int(j)))

    neg_pairs = np.array(list(neg_set), dtype=int)
    # 如果因为过滤太强导致不够，退化补充到 assoc_full==0 的全集
    if len(neg_pairs) < num_negs:
        all_zero = np.argwhere(assoc_full == 0)
        pick = rng.choice(len(all_zero), size=(num_negs - len(neg_pairs)), replace=False)
        neg_pairs = np.vstack([neg_pairs, all_zero[pick]])

    rng.shuffle(neg_pairs)
    return neg_pairs[:num_negs]
