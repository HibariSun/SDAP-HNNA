# Author:Hibari
# Streamlined model — 基于消融实验结果精简
# 移除: DualAttentionModule, Multi-Scale分支, Early Fusion
# 保留: 三超图 + CrossAttentionFusion + StrongPairwisePredictor + Late Fusion
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础组件 ====================

class EnhancedHypergraphConvolution(nn.Module):
    """超图卷积层。

    超图传播矩阵为
        P_tilde = D_v^(-1/2) H W_e D_e^(-1) H^T D_v^(-1/2),
    其中所有超边采用相同权重，因此 W_e = I。为避免实际构造稠密单位阵，
    代码只保存 W_e 的对角线（全 1 向量）。
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.3,
                 use_w2=True):
        super(EnhancedHypergraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_w2 = bool(use_w2)

        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_features)
        if self.use_w2:
            self.linear2 = nn.Linear(in_features, out_features, bias=bias)
            self.bn2 = nn.BatchNorm1d(out_features)
        else:
            self.linear2 = None
            self.bn2 = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):
        # W_e = diag(W_e_diag) = I_|E|：每条超边的权重均为 1。
        # 使用对角线向量等价表示单位阵，避免 |E| x |E| 的无效存储和乘法。
        W_e_diag = torch.ones(H.size(1), dtype=H.dtype, device=H.device)

        # d(v) = sum_e W_e(e, e) H(v, e)，d(e) = sum_v H(v, e)。
        D_v = torch.sum(H * W_e_diag.unsqueeze(0), dim=1, keepdim=True).clamp(min=1)
        D_e = torch.sum(H, dim=0, keepdim=True).clamp(min=1)
        D_v_inv_sqrt = D_v.rsqrt()
        D_e_inv = D_e.reciprocal().T

        X1 = self.linear1(X)

        # 按从右到左的顺序计算 P_tilde X，避免显式构造节点传播矩阵 P_tilde。
        X1 = D_v_inv_sqrt * X1
        X1 = H.T @ X1
        X1 = D_e_inv * X1
        X1 = W_e_diag.unsqueeze(1) * X1
        X1 = H @ X1
        X1 = D_v_inv_sqrt * X1
        X1 = self.bn1(X1)
        X1 = self.dropout(X1)

        if not self.use_w2:
            return F.elu(X1)

        X2 = self.linear2(X)
        X2 = self.bn2(X2)
        X2 = self.dropout(X2)
        return F.elu(X1 + X2)


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, dim, num_heads=4, dropout=0.3):
        super(CrossAttentionFusion, self).__init__()

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

    def forward(self, feat_all, feat_sno, feat_dis):
        stacked = torch.stack([feat_all, feat_sno, feat_dis], dim=1)

        q = self.q_proj(feat_all).unsqueeze(1)
        k = self.k_proj(stacked)
        v = self.v_proj(stacked)

        attn_out, _ = self.cross_attention(q, k, v)
        attn_out = attn_out.squeeze(1)

        gate_input = torch.cat([feat_all, feat_sno, feat_dis], dim=1)
        gate_weight = self.gate(gate_input)

        output = self.norm(feat_all + gate_weight * self.dropout(attn_out))
        return output


class TripleHypergraphBlock(nn.Module):
    """三超图卷积块（精简版：去掉 DualAttention）"""

    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3,
                 use_w2=True):
        super(TripleHypergraphBlock, self).__init__()

        self.hgc_all = EnhancedHypergraphConvolution(
            in_features, out_features, dropout=dropout, use_w2=use_w2
        )
        self.hgc_sno = EnhancedHypergraphConvolution(
            in_features, out_features, dropout=dropout, use_w2=use_w2
        )
        self.hgc_dis = EnhancedHypergraphConvolution(
            in_features, out_features, dropout=dropout, use_w2=use_w2
        )

        self.fusion = CrossAttentionFusion(out_features, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(out_features)
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, X, H_all, H_sno, H_dis):
        identity = self.residual_proj(X)

        feat_all = self.hgc_all(X, H_all)
        feat_sno = self.hgc_sno(X, H_sno)
        feat_dis = self.hgc_dis(X, H_dis)

        X = self.fusion(feat_all, feat_sno, feat_dis)
        X = self.norm(X + identity)

        X = X + self.ffn(X)
        return X


class PostStackCrossAttentionEncoder(nn.Module):
    """先独立堆叠多层三路超图卷积，最后只做一次交叉注意力。

    三个分支分别沿 H_all、H_sno 和 H_dis 传播。在所有超图层完成前，
    分支之间不进行信息融合；这对应“先做三层 HGCN，再做多头交叉注意力”
    的实验顺序。
    """

    def __init__(self, hidden_dims, num_heads=8, dropout=0.3, use_w2=True):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims 至少需要包含一个维度")

        dims = [hidden_dims[0]] + list(hidden_dims)
        self.hgc_all = nn.ModuleList()
        self.hgc_sno = nn.ModuleList()
        self.hgc_dis = nn.ModuleList()
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            self.hgc_all.append(EnhancedHypergraphConvolution(
                in_features, out_features, dropout=dropout, use_w2=use_w2
            ))
            self.hgc_sno.append(EnhancedHypergraphConvolution(
                in_features, out_features, dropout=dropout, use_w2=use_w2
            ))
            self.hgc_dis.append(EnhancedHypergraphConvolution(
                in_features, out_features, dropout=dropout, use_w2=use_w2
            ))

        final_dim = hidden_dims[-1]
        self.fusion = CrossAttentionFusion(final_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(final_dim, final_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim * 4, final_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(final_dim)

    def forward(self, X, H_all, H_sno, H_dis):
        feat_all = feat_sno = feat_dis = X
        for layer_all, layer_sno, layer_dis in zip(
                self.hgc_all, self.hgc_sno, self.hgc_dis):
            feat_all = layer_all(feat_all, H_all)
            feat_sno = layer_sno(feat_sno, H_sno)
            feat_dis = layer_dis(feat_dis, H_dis)

        X = self.fusion(feat_all, feat_sno, feat_dis)
        return self.norm(X + self.ffn(X))


class FeatureEnhancementModule(nn.Module):
    """特征增强模块"""

    def __init__(self, in_features, out_features, dropout=0.3):
        super(FeatureEnhancementModule, self).__init__()

        self.enhance = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.BatchNorm1d(out_features * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.enhance(x)


class StrongPairwisePredictor(nn.Module):
    def __init__(self, dim, dropout=0.2, pair_feature_mode="full"):
        super().__init__()
        if pair_feature_mode not in {"full", "concat"}:
            raise ValueError(
                "pair_feature_mode 必须是 'full'（四项特征）或 'concat'（仅左右嵌入）"
            )
        self.pair_feature_mode = pair_feature_mode
        self.bilinear = nn.Bilinear(dim, dim, dim) if pair_feature_mode == "full" else None
        in_dim = dim * (4 if pair_feature_mode == "full" else 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, left, right):
        if self.pair_feature_mode == "concat":
            feat = torch.cat([left, right], dim=-1)
            return self.mlp(feat)

        prod = left * right
        diff = torch.abs(left - right)
        bilin = self.bilinear(left, right)
        feat = torch.cat([left, right, diff, prod + bilin], dim=-1)
        return self.mlp(feat)


class TripleHypergraphNN(nn.Module):
    """精简版三超图神经网络

    相比原版移除了:
      - Early Fusion:  外部嵌入不再拼入初始节点特征，仅通过 Late Fusion 引入
      - DualAttention: 块内和全局的双重注意力模块均已移除
      - Multi-Scale:   多尺度特征初始化分支已移除，仅保留 FeatureEnhancementModule

    保留:
      - 三超图卷积 (H_all, H_sno, H_dis) + CrossAttentionFusion
      - Late Fusion (外部嵌入的后融合)
      - Pairwise predictor（默认四项特征，也支持仅 [z_sno || z_disease]）

    消融/顺序实验开关:
      - use_hgc_w2=False: 移除超图卷积公式中的 X W2 项
      - cross_attention_stage="after_stack": 所有 HGCN 层后只融合一次
      - pair_feature_mode="concat": 配对特征从 4d 简化为 2d
    """

    def __init__(self, num_snorna, num_disease, snorna_sim, disease_sim, assoc_matrix,
                 hidden_dims=None, num_heads=8, dropout=0.2,
                 sno_external_emb=None, disease_external_emb=None,
                 enable_late_fusion=True,
                 use_assoc_profile_features=True,
                 use_early_fusion_external=False,       # ← 默认关闭（精简）
                 use_strong_predictor=True,              # ← 保留接口兼容性
                 use_hgc_w2=True,
                 cross_attention_stage="per_layer",
                 pair_feature_mode="full"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]
        if cross_attention_stage not in {"per_layer", "after_stack"}:
            raise ValueError(
                "cross_attention_stage 必须是 'per_layer' 或 'after_stack'"
            )
        self.num_snorna = num_snorna
        self.num_disease = num_disease
        self.enable_late_fusion = enable_late_fusion
        self.use_hgc_w2 = bool(use_hgc_w2)
        self.cross_attention_stage = cross_attention_stage
        self.pair_feature_mode = pair_feature_mode

        assoc_tensor = torch.as_tensor(assoc_matrix, dtype=torch.float32)
        snorna_sim_tensor = torch.as_tensor(snorna_sim, dtype=torch.float32)
        disease_sim_tensor = torch.as_tensor(disease_sim, dtype=torch.float32)

        ext_sno = torch.as_tensor(sno_external_emb, dtype=torch.float32) if sno_external_emb is not None else None
        ext_dis = torch.as_tensor(disease_external_emb, dtype=torch.float32) if disease_external_emb is not None else None
        self.has_external_emb = ext_sno is not None and ext_dis is not None

        # ---- 初始节点特征（不再拼接外部嵌入） ----
        if use_assoc_profile_features:
            snorna_feat = torch.cat([assoc_tensor, snorna_sim_tensor], dim=1)
            disease_feat = torch.cat([assoc_tensor.t(), disease_sim_tensor], dim=1)
        else:
            snorna_feat = snorna_sim_tensor
            disease_feat = disease_sim_tensor

        self.snorna_features = nn.Parameter(snorna_feat, requires_grad=True)
        self.disease_features = nn.Parameter(disease_feat, requires_grad=True)

        snorna_in_dim = self.snorna_features.size(1)
        disease_in_dim = self.disease_features.size(1)

        # ---- 特征增强（仅保留单路 FeatureEnhancementModule，无 Multi-Scale） ----
        self.snorna_enhance = FeatureEnhancementModule(snorna_in_dim, hidden_dims[0], dropout)
        self.disease_enhance = FeatureEnhancementModule(disease_in_dim, hidden_dims[0], dropout)

        # ---- 三超图编码器 ----
        if self.cross_attention_stage == "per_layer":
            self.hg_blocks = nn.ModuleList()
            dims = [hidden_dims[0]] + hidden_dims
            for i in range(len(hidden_dims)):
                self.hg_blocks.append(TripleHypergraphBlock(
                    dims[i], dims[i + 1], num_heads, dropout,
                    use_w2=self.use_hgc_w2,
                ))
            self.post_stack_encoder = None
        else:
            self.hg_blocks = nn.ModuleList()
            self.post_stack_encoder = PostStackCrossAttentionEncoder(
                hidden_dims=hidden_dims,
                num_heads=num_heads,
                dropout=dropout,
                use_w2=self.use_hgc_w2,
            )

        final_dim = hidden_dims[-1]

        # ---- 外部嵌入仅通过 Late Fusion 引入 ----
        if self.has_external_emb:
            self.register_buffer("sno_external_emb", ext_sno)
            self.register_buffer("disease_external_emb", ext_dis)
        else:
            self.sno_external_emb = None
            self.disease_external_emb = None

        if self.enable_late_fusion and self.has_external_emb:
            self.sno_ext_proj = nn.Sequential(nn.Linear(ext_sno.size(1), final_dim), nn.LayerNorm(final_dim), nn.ELU())
            self.dis_ext_proj = nn.Sequential(nn.Linear(ext_dis.size(1), final_dim), nn.LayerNorm(final_dim), nn.ELU())
            self.sno_fuse = nn.Sequential(nn.Linear(final_dim * 2, final_dim), nn.LayerNorm(final_dim), nn.ELU(), nn.Dropout(dropout))
            self.dis_fuse = nn.Sequential(nn.Linear(final_dim * 2, final_dim), nn.LayerNorm(final_dim), nn.ELU(), nn.Dropout(dropout))

        # ---- 预测器 ----
        self.predictor = StrongPairwisePredictor(
            final_dim, dropout, pair_feature_mode=self.pair_feature_mode
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def _drop_edge(self, H, rate):
        if H is None:
            return None
        mask = torch.rand_like(H) > rate
        return H * mask.float()

    def _pair_scores(self, snorna_embed, disease_embed, predictor):
        snorna_expanded = snorna_embed.unsqueeze(1).expand(-1, self.num_disease, -1)
        disease_expanded = disease_embed.unsqueeze(0).expand(self.num_snorna, -1, -1)
        left = snorna_expanded.reshape(-1, snorna_expanded.size(-1))
        right = disease_expanded.reshape(-1, disease_expanded.size(-1))
        scores = predictor(left, right)
        return scores.view(self.num_snorna, self.num_disease)

    def forward(self, H_all, H_sno, H_dis, drop_edge_rate=0.0, return_reps=False):
        if self.training and drop_edge_rate > 0:
            H_all = self._drop_edge(H_all, drop_edge_rate)
            H_sno = self._drop_edge(H_sno, drop_edge_rate)
            H_dis = self._drop_edge(H_dis, drop_edge_rate)

        # 特征增强（单路，无 Multi-Scale）
        snorna_feat = self.snorna_enhance(self.snorna_features)
        disease_feat = self.disease_enhance(self.disease_features)

        # 三超图卷积（逐层融合，或全部堆叠完成后只融合一次）
        X = torch.cat([snorna_feat, disease_feat], dim=0)
        if self.cross_attention_stage == "per_layer":
            for hg_block in self.hg_blocks:
                X = hg_block(X, H_all, H_sno, H_dis)
        else:
            X = self.post_stack_encoder(X, H_all, H_sno, H_dis)

        # 直接取嵌入（无全局 DualAttention）
        snorna_embed = X[:self.num_snorna]
        disease_embed = X[self.num_snorna:]

        # Late Fusion: 外部嵌入在这里引入
        if self.enable_late_fusion and self.has_external_emb:
            sno_ext = self.sno_ext_proj(self.sno_external_emb)
            dis_ext = self.dis_ext_proj(self.disease_external_emb)
            snorna_embed = self.sno_fuse(torch.cat([snorna_embed, sno_ext], dim=1))
            disease_embed = self.dis_fuse(torch.cat([disease_embed, dis_ext], dim=1))
        else:
            sno_ext = dis_ext = None

        scores = self._pair_scores(snorna_embed, disease_embed, self.predictor)

        if return_reps:
            reps = {
                'sno_int': snorna_embed,
                'sno_ext': sno_ext,
                'dis_int': disease_embed,
                'dis_ext': dis_ext,
            }
            return scores, reps
        return scores
