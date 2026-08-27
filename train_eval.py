# Author:Hibari
# train_eval.py — 纯10折CV版本（无独立测试集、无辅助监督）
import os
import pandas as pd
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from tqdm import tqdm
import torch.nn as nn

from gipk import GIPKCalculator
from hypergraph import HypergraphConstructor
from model import TripleHypergraphNN
from utils import device
from entity_cv import sample_negatives_uniform


# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


# ==================== 加权BCE ====================
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        loss = -self.pos_weight * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return loss.mean()


# ==================== 学习率调度 ====================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ==================== 动态阈值 ====================
def find_optimal_threshold(y_true, y_scores, metric='f1'):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_score = -float('inf')
    best_threshold = 0.5

    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f'Unsupported metric: {metric}')

        if score > best_score:
            best_score = float(score)
            best_threshold = float(thresh)

    return best_threshold, best_score


# ==================== 指标计算与汇总 ====================
def compute_metrics_from_scores(y_true, y_scores, threshold=None, threshold_strategy='fixed_0.5'):
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if len(np.unique(y_true)) < 2:
        auc_score = float('nan')
    else:
        auc_score = roc_auc_score(y_true, y_scores)
    aupr_score = average_precision_score(y_true, y_scores)

    if threshold is None:
        if threshold_strategy == 'fixed_0.5':
            threshold = 0.5
        elif threshold_strategy == 'optimal_mcc':
            threshold, _ = find_optimal_threshold(y_true, y_scores, metric='mcc')
        elif threshold_strategy == 'optimal_recall':
            threshold, _ = find_optimal_threshold(y_true, y_scores, metric='recall')
        elif threshold_strategy == 'optimal_precision':
            threshold, _ = find_optimal_threshold(y_true, y_scores, metric='precision')
        else:
            threshold, _ = find_optimal_threshold(y_true, y_scores, metric='f1')

    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "auc": auc_score,
        "aupr": aupr_score,
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
    }


def summarize_fold_results(fold_results):
    metrics = ["auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc"]
    summary = {}
    for m in metrics:
        vals = np.array([r[m] for r in fold_results], dtype=float)
        summary[f"mean_{m}"] = float(np.nanmean(vals))
        summary[f"std_{m}"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary


def print_fold_table(fold_results, title="各折结果"):
    print(f"\n  {title}:")
    print("  " + "-" * 86)
    print("  fold |   AUC   |  AUPR  |  ACC   |  Prec  |  Rec   |   F1   |  MCC")
    print("  " + "-" * 86)
    for r in fold_results:
        print(
            f"  {r['fold']:>4d} | {r['auc']:.4f} | {r['aupr']:.4f} | {r['accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['mcc']:.4f}"
        )
    print("  " + "-" * 86)


def print_cv_summary(fold_results, prefix="CV"):
    s = summarize_fold_results(fold_results)
    print(f"\n  {prefix} 汇总 (mean±std):")
    print(
        f"    AUC: {s['mean_auc']:.4f} ± {s['std_auc']:.4f} | "
        f"AUPR: {s['mean_aupr']:.4f} ± {s['std_aupr']:.4f}"
    )
    print(
        f"    ACC: {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f} | "
        f"Prec: {s['mean_precision']:.4f} ± {s['std_precision']:.4f} | "
        f"Rec: {s['mean_recall']:.4f} ± {s['std_recall']:.4f}"
    )
    print(
        f"    F1:  {s['mean_f1']:.4f} ± {s['std_f1']:.4f} | "
        f"MCC: {s['mean_mcc']:.4f} ± {s['std_mcc']:.4f}"
    )
    return s


# ==================== 训练器 ====================
class TripleHypergraphTrainer:
    def __init__(self, model, device, label_smoothing=0.05,
                 contrastive_lambda=0.0, contrastive_temp=0.2):
        self.model = model
        self.device = device
        self.label_smoothing = label_smoothing
        self.contrastive_lambda = float(contrastive_lambda)
        self.contrastive_temp = float(contrastive_temp)

    @staticmethod
    def _info_nce(z1, z2, temp=0.2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = (z1 @ z2.t()) / temp
        labels = torch.arange(z1.size(0), device=z1.device)
        loss12 = F.cross_entropy(logits, labels)
        loss21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss12 + loss21)

    def train_epoch(self, H_all, H_sno, H_dis, train_pos, train_neg, optimizer, criterion, drop_edge_rate=0.1):
        self.model.train()
        if self.contrastive_lambda > 0:
            predictions, reps = self.model(H_all, H_sno, H_dis, drop_edge_rate=drop_edge_rate, return_reps=True)
        else:
            predictions = self.model(H_all, H_sno, H_dis, drop_edge_rate=drop_edge_rate)
            reps = None

        loss = 0
        count = 0

        pos_label = 1.0 - self.label_smoothing
        neg_label = self.label_smoothing

        for i, j in train_pos:
            loss += criterion(predictions[i, j].unsqueeze(0),
                              torch.tensor([pos_label]).to(self.device))
            count += 1

        for i, j in train_neg:
            loss += criterion(predictions[i, j].unsqueeze(0),
                              torch.tensor([neg_label]).to(self.device))
            count += 1

        loss = loss / count

        if self.contrastive_lambda > 0 and reps is not None and reps.get("sno_ext", None) is not None:
            cl_sno = self._info_nce(reps["sno_int"], reps["sno_ext"], temp=self.contrastive_temp)
            cl_dis = self._info_nce(reps["dis_int"], reps["dis_ext"], temp=self.contrastive_temp)
            loss = loss + self.contrastive_lambda * (cl_sno + cl_dis)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    def evaluate(self, H_all, H_sno, H_dis, test_pos, test_neg):
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(H_all, H_sno, H_dis)

        y_true = []
        y_scores = []

        for i, j in test_pos:
            y_true.append(1)
            y_scores.append(predictions[i, j].cpu().item())

        for i, j in test_neg:
            y_true.append(0)
            y_scores.append(predictions[i, j].cpu().item())

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        auc_score = roc_auc_score(y_true, y_scores)
        aupr_score = average_precision_score(y_true, y_scores)

        return auc_score, aupr_score, y_true, y_scores


# ==================== 获取所有正样本 ====================
def get_all_pos_samples(association_matrix):
    """从关联矩阵中提取所有正样本 (i, j) 坐标"""
    pos_indices = np.argwhere(association_matrix == 1)
    return pos_indices


# ==================== 保存/加载10折CV划分 ====================
def save_cv_splits(all_pos, n_splits, seed, save_path):
    """保存10折CV的每一折的 train_pos / test_pos / test_neg 划分，
    方便其他方法使用完全一致的划分做公平比较。
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(range(len(all_pos)))):
        splits[f'fold{fold}_train_idx'] = train_idx.astype(np.int64)
        splits[f'fold{fold}_test_idx'] = test_idx.astype(np.int64)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savez(save_path,
             all_pos=all_pos,
             n_splits=n_splits,
             seed=seed,
             **splits)
    print(f"  ✓ 10折CV划分已保存到: {save_path}")
    print(f"    - 总正样本数: {len(all_pos)}, n_splits={n_splits}, seed={seed}")


def load_cv_splits(load_path):
    """加载已保存的10折CV划分"""
    data = np.load(load_path, allow_pickle=True)
    all_pos = data['all_pos']
    n_splits = int(data['n_splits'])
    seed = int(data['seed'])
    splits = []
    for fold in range(n_splits):
        train_idx = data[f'fold{fold}_train_idx']
        test_idx = data[f'fold{fold}_test_idx']
        splits.append((train_idx, test_idx))
    print(f"  ✓ 从文件加载10折CV划分: {load_path}")
    print(f"    - 总正样本数: {len(all_pos)}, n_splits={n_splits}, seed={seed}")
    return all_pos, n_splits, seed, splits


# ==================== 10折交叉验证（全数据） ====================
def cross_validation(association_matrix, snorna_sim, disease_sim,
                     all_pos,
                     cv_splits=None,
                     n_splits=10, epochs=300, lr=0.0005,
                     patience=30,
                     use_association_edges=True, association_weight=0.5,
                     k_snorna=20, k_disease=20,
                     hidden_dims=None,
                     num_heads=8,
                     dropout=0.2,
                     train_neg_ratio=2,
                     test_neg_ratio=1,
                     loss_type='focal',
                     pos_weight=2.0,
                     data_loader=None,
                     alpha_snorna=None,
                     alpha_disease=None,
                     contrastive_lambda=0.0,
                     contrastive_temp=0.2,
                     enable_late_fusion=True,
                     disjoint_train_ratio=0.2,
                     focal_alpha=0.75,
                     focal_gamma=2.0,
                     weight_decay=5e-4,
                     use_assoc_profile_features=True,
                     use_early_fusion_external=False,
                     use_strong_predictor=True,
                     use_hgc_w2=True,
                     cross_attention_stage="per_layer",
                     pair_feature_mode="full",
                     cv_level="pair",
                     cv_fold_entities=None,
                     validation_ratio=0.0,
                     seed=42):
    """
    在全部数据上进行10折交叉验证。

    口径：
      - disjoint training：loss 只在 supervision/disjoint 正边上算
      - message passing 图只使用 message 边
      - early stopping 按 val MCC@0.5
      - cv_splits: 预先划分好的折列表 [(train_idx, test_idx), ...]，若为 None 则现场划分
      - cv_level: pair / snorna / disease
      - cv_fold_entities: 实体级CV时，每折留出的 snoRNA 或 disease 编号
      - validation_ratio > 0 时，从外层训练正样本中留出内部验证集，避免用测试集早停
    """
    if hidden_dims is None:
        hidden_dims = [512, 384, 256, 128, 64]

    use_association_edges = bool(use_association_edges and association_weight > 0)

    cv_level = str(cv_level).strip().lower()
    if cv_level not in {"pair", "snorna", "disease"}:
        raise ValueError("cv_level 必须是 'pair'、'snorna' 或 'disease'。")
    if not 0.0 <= float(validation_ratio) < 1.0:
        raise ValueError("validation_ratio 必须在 [0, 1) 范围内。")

    print(f"\n[CV] 在全部数据上进行 {n_splits} 折交叉验证...")
    print("=" * 80)
    print(f"  超图参数: k_snorna={k_snorna}, k_disease={k_disease}, association_weight={association_weight}")
    print(f"  use_association_edges={use_association_edges}")
    print(f"  训练参数: lr={lr}, loss_type={loss_type}, dropout={dropout}")
    print(f"  Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    print(f"  Disjoint Training: ratio={disjoint_train_ratio}")
    print(f"  CV level: {cv_level}")
    if validation_ratio > 0:
        print(f"  内部验证集: 外层训练正样本的 {validation_ratio:.1%}（测试折不参与早停）")
    else:
        print("  内部验证集: 未启用（保持原实现，使用测试折早停）")
    print(
        f"  架构: W2={'on' if use_hgc_w2 else 'off'}, "
        f"cross_attention={cross_attention_stage}, pair_features={pair_feature_mode}"
    )

    num_snorna, num_disease = association_matrix.shape
    num_all_pos = len(all_pos)
    print(f"  总正样本数: {num_all_pos}")

    # 使用预先划分好的折，或现场划分
    if cv_splits is not None:
        fold_splits = cv_splits
        print(f"  使用预加载的CV划分 ({len(fold_splits)} 折)")
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_splits = list(kfold.split(range(num_all_pos)))

    if len(fold_splits) != n_splits:
        raise ValueError(
            f"CV划分包含 {len(fold_splits)} 折，但 n_splits={n_splits}。"
        )
    if cv_level != "pair":
        if cv_fold_entities is None or len(cv_fold_entities) != len(fold_splits):
            raise ValueError("实体级CV必须为每一折提供 cv_fold_entities。")

    fold_results = []
    all_y_true = []
    all_y_scores = []

    sno_ext = dis_ext = None
    external_available = False
    if data_loader is not None:
        sno_ext = getattr(data_loader, "_sno_external", None)
        dis_ext = getattr(data_loader, "_disease_external", None)
        external_available = getattr(data_loader, "_external_available", False) and sno_ext is not None and dis_ext is not None
        sno_emb = getattr(data_loader, "_sno_rnaernie_emb", None)
        dis_emb = getattr(data_loader, "_disease_llm_emb", None)
    else:
        sno_emb = None
        dis_emb = None

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"\n{'─' * 60}")
        print(f"CV 折 {fold + 1}/{n_splits}")
        print(f"{'─' * 60}")

        fold_train_pos = np.asarray(all_pos[train_idx], dtype=np.int64)
        fold_test_pos = np.asarray(all_pos[test_idx], dtype=np.int64)
        rng = np.random.default_rng(seed + fold)

        heldout_entities = np.array([], dtype=np.int64)
        train_allowed_snorna = test_allowed_snorna = None
        train_allowed_disease = test_allowed_disease = None
        if cv_level == "snorna":
            heldout_entities = np.asarray(cv_fold_entities[fold], dtype=np.int64)
            train_entities = np.setdiff1d(
                np.arange(num_snorna, dtype=np.int64), heldout_entities
            )
            train_allowed_snorna = train_entities
            test_allowed_snorna = heldout_entities
            if np.isin(fold_train_pos[:, 0], heldout_entities).any():
                raise ValueError(f"第 {fold + 1} 折训练正样本包含留出 snoRNA。")
            if not np.isin(fold_test_pos[:, 0], heldout_entities).all():
                raise ValueError(f"第 {fold + 1} 折测试正样本包含非留出 snoRNA。")
        elif cv_level == "disease":
            heldout_entities = np.asarray(cv_fold_entities[fold], dtype=np.int64)
            train_entities = np.setdiff1d(
                np.arange(num_disease, dtype=np.int64), heldout_entities
            )
            train_allowed_disease = train_entities
            test_allowed_disease = heldout_entities
            if np.isin(fold_train_pos[:, 1], heldout_entities).any():
                raise ValueError(f"第 {fold + 1} 折训练正样本包含留出 disease。")
            if not np.isin(fold_test_pos[:, 1], heldout_entities).all():
                raise ValueError(f"第 {fold + 1} 折测试正样本包含非留出 disease。")

        # 外层测试负样本只从测试实体一侧抽取。
        test_neg = sample_negatives_uniform(
            association_matrix,
            num_negs=len(fold_test_pos) * test_neg_ratio,
            seed=seed + fold,
            allowed_snorna=test_allowed_snorna,
            allowed_disease=test_allowed_disease,
        )
        if len(test_neg) != len(fold_test_pos) * test_neg_ratio:
            raise ValueError(
                f"第 {fold + 1} 折可用测试负样本不足：需要 "
                f"{len(fold_test_pos) * test_neg_ratio}，实际 {len(test_neg)}。"
            )
        test_neg_flat = set(test_neg[:, 0] * num_disease + test_neg[:, 1])
        test_neg_flat_arr = np.array(list(test_neg_flat), dtype=np.int64) if len(test_neg_flat) > 0 else np.array([], dtype=np.int64)

        # 可选的内部验证集。它只来自外层训练实体，因此外层测试集不参与早停。
        if validation_ratio > 0:
            num_validation = max(1, int(len(fold_train_pos) * validation_ratio))
            if num_validation >= len(fold_train_pos) - 1:
                raise ValueError(
                    f"第 {fold + 1} 折训练正样本太少，无法按 validation_ratio="
                    f"{validation_ratio} 划分内部验证集。"
                )
            validation_idx = rng.choice(
                len(fold_train_pos), size=num_validation, replace=False
            )
            train_pool_idx = np.setdiff1d(
                np.arange(len(fold_train_pos)), validation_idx
            )
            validation_pos = fold_train_pos[validation_idx]
            train_pool_pos = fold_train_pos[train_pool_idx]
            validation_neg = sample_negatives_uniform(
                association_matrix,
                num_negs=len(validation_pos) * test_neg_ratio,
                exclude_indices=test_neg_flat_arr,
                seed=seed + 5000 + fold,
                allowed_snorna=train_allowed_snorna,
                allowed_disease=train_allowed_disease,
            )
            if len(validation_neg) != len(validation_pos) * test_neg_ratio:
                raise ValueError(f"第 {fold + 1} 折可用内部验证负样本不足。")
            early_stop_pos = validation_pos
            early_stop_neg = validation_neg
        else:
            validation_pos = np.empty((0, 2), dtype=np.int64)
            validation_neg = np.empty((0, 2), dtype=np.int64)
            train_pool_pos = fold_train_pos
            early_stop_pos = fold_test_pos
            early_stop_neg = test_neg

        # Disjoint split: supervision 用于 loss, message 用于构图。
        num_train_pool = len(train_pool_pos)
        num_disjoint = max(1, int(num_train_pool * disjoint_train_ratio))
        num_disjoint = min(num_disjoint, num_train_pool)
        disjoint_idx = rng.choice(num_train_pool, size=num_disjoint, replace=False)
        message_idx = np.setdiff1d(np.arange(num_train_pool), disjoint_idx)
        disjoint_pos = train_pool_pos[disjoint_idx]
        message_pos = train_pool_pos[message_idx]

        validation_neg_flat_arr = (
            validation_neg[:, 0] * num_disease + validation_neg[:, 1]
            if len(validation_neg) > 0 else np.array([], dtype=np.int64)
        )
        excluded_train_neg_flat = np.union1d(
            test_neg_flat_arr, validation_neg_flat_arr
        ).astype(np.int64)

        # 构图用关联矩阵：移除 test_pos、内部 validation_pos 和 disjoint_pos。
        train_assoc_matrix = association_matrix.copy().astype(np.float32)
        for (i, j) in fold_test_pos:
            train_assoc_matrix[i, j] = 0.0
        for (i, j) in validation_pos:
            train_assoc_matrix[i, j] = 0.0
        for (i, j) in disjoint_pos:
            train_assoc_matrix[i, j] = 0.0

        print(
            f"  样本: train_pos={len(fold_train_pos)}, supervision_pos={len(disjoint_pos)}, "
            f"validation_pos={len(validation_pos)}, test_pos={len(fold_test_pos)}, "
            f"test_neg={len(test_neg)}"
        )
        if cv_level != "pair":
            print(f"  留出 {cv_level} 实体数: {len(heldout_entities)}")

        # 每折重新计算 GIPK / 融合相似度
        if data_loader is not None:
            gipk_calculator = GIPKCalculator(train_assoc_matrix, gamma_ratio=1.0)
            snorna_gipk_fold = gipk_calculator.compute_gipk_snorna()
            disease_gipk_fold = gipk_calculator.compute_gipk_disease()

            if external_available:
                snorna_sim_fold = data_loader._swf_fusion(
                    snorna_gipk_fold, sno_ext, name=f"snoRNA-CVFold{fold + 1}", alpha=alpha_snorna
                )
                disease_sim_fold = data_loader._swf_fusion(
                    disease_gipk_fold, dis_ext, name=f"Disease-CVFold{fold + 1}", alpha=alpha_disease
                )
            else:
                snorna_sim_fold = data_loader._normalize_similarity(snorna_gipk_fold)
                disease_sim_fold = data_loader._normalize_similarity(disease_gipk_fold)
        else:
            snorna_sim_fold = snorna_sim
            disease_sim_fold = disease_sim

        hg_constructor = HypergraphConstructor(train_assoc_matrix, snorna_sim_fold, disease_sim_fold)
        H_all, H_sno, H_dis = hg_constructor.construct_multi_hypergraph(
            k_snorna=k_snorna,
            k_disease=k_disease,
            use_association_edges=use_association_edges,
            association_weight=association_weight,
        )

        model = TripleHypergraphNN(
            num_snorna=num_snorna,
            num_disease=num_disease,
            snorna_sim=snorna_sim_fold,
            disease_sim=disease_sim_fold,
            assoc_matrix=train_assoc_matrix,
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            sno_external_emb=sno_emb,
            disease_external_emb=dis_emb,
            enable_late_fusion=enable_late_fusion,
            use_assoc_profile_features=use_assoc_profile_features,
            use_early_fusion_external=use_early_fusion_external,
            use_strong_predictor=use_strong_predictor,
            use_hgc_w2=use_hgc_w2,
            cross_attention_stage=cross_attention_stage,
            pair_feature_mode=pair_feature_mode,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'weighted_bce':
            criterion = WeightedBCELoss(pos_weight=pos_weight)
        elif loss_type == 'bce':
            criterion = nn.BCELoss()
        else:
            raise ValueError(f'Unsupported loss_type: {loss_type}')
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs)

        trainer = TripleHypergraphTrainer(
            model, device, label_smoothing=0.05,
            contrastive_lambda=contrastive_lambda, contrastive_temp=contrastive_temp,
        )

        best_mcc = -1.0
        best_state = None
        best_epoch = -1
        patience_counter = 0

        pbar = tqdm(range(epochs), desc=f"CV Fold {fold + 1}", leave=False)
        for epoch in pbar:
            _ = scheduler.step(epoch)

            train_neg_need = len(disjoint_pos) * train_neg_ratio
            fold_train_neg = sample_negatives_uniform(
                association_matrix,
                num_negs=train_neg_need,
                exclude_indices=excluded_train_neg_flat,
                seed=seed + 10000 + fold * 1000 + epoch,
                allowed_snorna=train_allowed_snorna,
                allowed_disease=train_allowed_disease,
            )
            if len(fold_train_neg) != train_neg_need:
                raise ValueError(
                    f"第 {fold + 1} 折可用训练负样本不足：需要 "
                    f"{train_neg_need}，实际 {len(fold_train_neg)}。"
                )

            train_loss = trainer.train_epoch(
                H_all, H_sno, H_dis,
                disjoint_pos, fold_train_neg,
                optimizer, criterion,
            )

            if epoch % 10 == 0:
                _, _, y_true_val, y_scores_val = trainer.evaluate(
                    H_all, H_sno, H_dis, early_stop_pos, early_stop_neg
                )
                val_metrics_fixed = compute_metrics_from_scores(
                    y_true_val, y_scores_val, threshold=0.5, threshold_strategy='fixed_0.5'
                )
                val_mcc = val_metrics_fixed['mcc']
                val_rec = val_metrics_fixed['recall']

                pbar.set_postfix({
                    "Loss": f"{train_loss:.4f}",
                    "MCC@0.5": f"{val_mcc:.4f}",
                    "Rec@0.5": f"{val_rec:.4f}"
                })

                if val_mcc > best_mcc:
                    best_mcc = val_mcc
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        _, _, y_true, y_scores = trainer.evaluate(H_all, H_sno, H_dis, fold_test_pos, test_neg)
        m_05 = compute_metrics_from_scores(y_true, y_scores, threshold=0.5, threshold_strategy='fixed_0.5')
        m_opt = compute_metrics_from_scores(y_true, y_scores, threshold=None, threshold_strategy='optimal_mcc')

        fold_results.append({
            "fold": fold + 1,
            "cv_level": cv_level,
            "num_train_pos": len(fold_train_pos),
            "num_validation_pos": len(validation_pos),
            "num_test_pos": len(fold_test_pos),
            "num_test_neg": len(test_neg),
            "num_heldout_entities": len(heldout_entities),
            "auc": m_05["auc"],
            "aupr": m_05["aupr"],
            "accuracy": m_05["accuracy"],
            "precision": m_05["precision"],
            "recall": m_05["recall"],
            "f1": m_05["f1"],
            "mcc": m_05["mcc"],
        })

        all_y_true.extend(y_true.tolist())
        all_y_scores.extend(y_scores.tolist())

        print(
            f"  Fold {fold + 1}: AUC={m_05['auc']:.4f}, AUPR={m_05['aupr']:.4f}, "
            f"ACC={m_05['accuracy']:.4f}, Prec={m_05['precision']:.4f}, Rec={m_05['recall']:.4f}, "
            f"F1={m_05['f1']:.4f}, MCC={m_05['mcc']:.4f} (thr=0.5 | best_mcc_thr={m_opt['threshold']:.3f} | best_epoch={best_epoch})"
        )

    print_fold_table(fold_results, title=f"CV {n_splits} 折结果")
    summary = print_cv_summary(fold_results, prefix=f"CV {n_splits} 折")

    return {
        "mean_auc": summary["mean_auc"],
        "mean_aupr": summary["mean_aupr"],
        "mean_accuracy": summary["mean_accuracy"],
        "mean_precision": summary["mean_precision"],
        "mean_recall": summary["mean_recall"],
        "mean_f1": summary["mean_f1"],
        "mean_mcc": summary["mean_mcc"],
        "std_auc": summary["std_auc"],
        "std_aupr": summary["std_aupr"],
        "std_accuracy": summary["std_accuracy"],
        "std_precision": summary["std_precision"],
        "std_recall": summary["std_recall"],
        "std_f1": summary["std_f1"],
        "std_mcc": summary["std_mcc"],
        "fold_results": fold_results,
        "all_y_true": np.array(all_y_true),
        "all_y_scores": np.array(all_y_scores),
    }


# ==================== 保存单组网格搜索结果 ====================
def save_single_grid_result_npz(save_dir, comb_id, config, cv_results):
    """把每组网格搜索的完整结果单独保存为 npz，便于后续单独画 ROC/PR 曲线。"""
    os.makedirs(save_dir, exist_ok=True)

    hidden_dims = config.get("hidden_dims", [])
    hidden_dims_str = "-".join(map(str, hidden_dims)) if len(hidden_dims) > 0 else "none"

    file_name = (
        f"comb_{int(comb_id):03d}"
        f"_as{float(config['alpha_snorna']):.2f}"
        f"_ad{float(config['alpha_disease']):.2f}"
        f"_ks{int(config['k_snorna'])}"
        f"_kd{int(config['k_disease'])}"
        f"_aw{float(config['association_weight']):.3f}"
        f"_lr{float(config['lr']):.0e}"
        f"_wd{float(config['weight_decay']):.0e}"
        f"_dp{float(config['dropout']):.2f}"
        f"_{str(config['loss_type'])}"
        f"_hd{hidden_dims_str}.npz"
    )
    save_path = os.path.join(save_dir, file_name)

    fold_results = cv_results.get("fold_results", [])
    all_y_true = np.asarray(cv_results.get("all_y_true", []), dtype=np.int64)
    all_y_scores = np.asarray(cv_results.get("all_y_scores", []), dtype=np.float32)

    np.savez_compressed(
        save_path,
        comb_id=np.int64(comb_id),

        alpha_snorna=np.float32(config["alpha_snorna"]),
        alpha_disease=np.float32(config["alpha_disease"]),
        k_snorna=np.int64(config["k_snorna"]),
        k_disease=np.int64(config["k_disease"]),
        association_weight=np.float32(config["association_weight"]),
        lr=np.float32(config["lr"]),
        weight_decay=np.float32(config["weight_decay"]),
        loss_type=np.array(config["loss_type"], dtype=object),
        hidden_dims=np.asarray(config["hidden_dims"], dtype=np.int64),
        focal_alpha=np.float32(config["focal_alpha"]),
        focal_gamma=np.float32(config["focal_gamma"]),
        dropout=np.float32(config["dropout"]),
        contrastive_lambda=np.float32(config["contrastive_lambda"]),
        contrastive_temp=np.float32(config["contrastive_temp"]),
        enable_late_fusion=np.bool_(config["enable_late_fusion"]),
        disjoint_train_ratio=np.float32(config["disjoint_train_ratio"]),
        use_assoc_profile_features=np.bool_(config["use_assoc_profile_features"]),
        use_early_fusion_external=np.bool_(config["use_early_fusion_external"]),
        use_strong_predictor=np.bool_(config["use_strong_predictor"]),

        mean_auc=np.float32(cv_results["mean_auc"]),
        mean_aupr=np.float32(cv_results["mean_aupr"]),
        mean_accuracy=np.float32(cv_results["mean_accuracy"]),
        mean_precision=np.float32(cv_results["mean_precision"]),
        mean_recall=np.float32(cv_results["mean_recall"]),
        mean_f1=np.float32(cv_results["mean_f1"]),
        mean_mcc=np.float32(cv_results["mean_mcc"]),
        std_auc=np.float32(cv_results["std_auc"]),
        std_aupr=np.float32(cv_results["std_aupr"]),
        std_accuracy=np.float32(cv_results["std_accuracy"]),
        std_precision=np.float32(cv_results["std_precision"]),
        std_recall=np.float32(cv_results["std_recall"]),
        std_f1=np.float32(cv_results["std_f1"]),
        std_mcc=np.float32(cv_results["std_mcc"]),

        all_y_true=all_y_true,
        all_y_scores=all_y_scores,
        fold_results=np.array(fold_results, dtype=object),
    )

    return save_path


# ==================== 网格搜索（纯CV） ====================
def grid_search(
        data_loader,
        all_pos,
        cv_splits,
        alpha_snorna_list,
        alpha_disease_list,
        k_snorna_list,
        k_disease_list,
        association_weight_list,
        lr_list=None,
        focal_alpha_list=None,
        focal_gamma_list=None,
        loss_type_list=None,
        hidden_dims_candidates=None,
        weight_decay_list=None,
        n_splits=10,
        epochs=300,
        patience=30,
        hidden_dims=None,
        num_heads=8,
        dropout_list=None,
        contrastive_lambda=0.0,
        contrastive_temp=0.2,
        enable_late_fusion=True,
        disjoint_train_ratio=0.2,
        use_assoc_profile_features=True,
        use_early_fusion_external=False,
        use_strong_predictor=True,
        mcc_floor=0.62,
        per_combination_npz_dir="results/grid_npz",
        output_dir="results",
):
    """
    网格搜索（纯10折CV版本）：
      1. 在全部数据上做10折CV调参
      2. 同时输出四套最优配置：
         - max_mcc
         - max_auc
         - max_aupr
         - safe_lexi: 仅在 MCC >= mcc_floor 中按 AUPR -> AUC -> MCC 排序
      3. 默认返回 safe_lexi 对应的 best_config / best_cv_results / comparison_df
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]
    if hidden_dims_candidates is None:
        hidden_dims_candidates = [hidden_dims]
    if lr_list is None:
        lr_list = [0.0005]
    if focal_alpha_list is None:
        focal_alpha_list = [0.75]
    if focal_gamma_list is None:
        focal_gamma_list = [2.0]
    if loss_type_list is None:
        loss_type_list = ['focal']
    if weight_decay_list is None:
        weight_decay_list = [5e-4]
    if dropout_list is None:
        dropout_list = [0.3]

    results = []
    cv_results_store = {}
    config_store = {}

    total_combinations = (
        len(alpha_snorna_list) * len(alpha_disease_list) *
        len(k_snorna_list) * len(k_disease_list) *
        len(association_weight_list) * len(lr_list) *
        len(focal_alpha_list) * len(focal_gamma_list) * len(dropout_list) *
        len(loss_type_list) * len(hidden_dims_candidates) * len(weight_decay_list)
    )
    comb_id = 0
    print(f"\n[网格搜索] 总组合数: {total_combinations}")
    print(f"[网格搜索] 选择策略: 只从 MCC >= {mcc_floor:.4f} 的配置中，按 AUPR -> AUC -> MCC 排序")

    for alpha_s in alpha_snorna_list:
        for alpha_d in alpha_disease_list:
            association_matrix, snorna_sim, disease_sim = data_loader.load_all_data(
                alpha_snorna=alpha_s, alpha_disease=alpha_d
            )

            for k_s in k_snorna_list:
                for k_d in k_disease_list:
                    for assoc_w in association_weight_list:
                        for lr in lr_list:
                            for wd in weight_decay_list:
                                for loss_type in loss_type_list:
                                    for hidden_dims_cur in hidden_dims_candidates:
                                        for f_alpha in focal_alpha_list:
                                            for f_gamma in focal_gamma_list:
                                                for dp in dropout_list:
                                                    comb_id += 1
                                                    print(f"\n>>> 组合 {comb_id}/{total_combinations}:")
                                                    print(f"    alpha_s={alpha_s}, alpha_d={alpha_d}")
                                                    print(f"    k_snorna={k_s}, k_disease={k_d}, assoc_w={assoc_w}, dropout={dp}")
                                                    print(f"    loss_type={loss_type}, lr={lr}, weight_decay={wd}, hidden_dims={hidden_dims_cur}")
                                                    print(f"    focal_alpha={f_alpha}, focal_gamma={f_gamma}")

                                                    cv_results = cross_validation(
                                                        association_matrix=association_matrix,
                                                        snorna_sim=snorna_sim,
                                                        disease_sim=disease_sim,
                                                        all_pos=all_pos,
                                                        cv_splits=cv_splits,
                                                        n_splits=n_splits,
                                                        epochs=epochs,
                                                        lr=lr,
                                                        patience=patience,
                                                        use_association_edges=(assoc_w > 0),
                                                        association_weight=assoc_w,
                                                        k_snorna=k_s,
                                                        k_disease=k_d,
                                                        hidden_dims=hidden_dims_cur,
                                                        num_heads=num_heads,
                                                        dropout=dp,
                                                        loss_type=loss_type,
                                                        data_loader=data_loader,
                                                        alpha_snorna=alpha_s,
                                                        alpha_disease=alpha_d,
                                                        contrastive_lambda=contrastive_lambda,
                                                        contrastive_temp=contrastive_temp,
                                                        enable_late_fusion=enable_late_fusion,
                                                        disjoint_train_ratio=disjoint_train_ratio,
                                                        focal_alpha=f_alpha,
                                                        focal_gamma=f_gamma,
                                                        weight_decay=wd,
                                                        use_assoc_profile_features=use_assoc_profile_features,
                                                        use_early_fusion_external=use_early_fusion_external,
                                                        use_strong_predictor=use_strong_predictor,
                                                    )

                                                    row = {
                                                        "comb_id": comb_id,
                                                        "alpha_snorna": alpha_s,
                                                        "alpha_disease": alpha_d,
                                                        "k_snorna": k_s,
                                                        "k_disease": k_d,
                                                        "association_weight": assoc_w,
                                                        "lr": lr,
                                                        "weight_decay": wd,
                                                        "loss_type": loss_type,
                                                        "hidden_dims": str(hidden_dims_cur),
                                                        "focal_alpha": f_alpha,
                                                        "focal_gamma": f_gamma,
                                                        "dropout": dp,
                                                        **{
                                                            f"cv_{k}": v
                                                            for k, v in cv_results.items()
                                                            if k not in ("fold_results", "all_y_true", "all_y_scores")
                                                        }
                                                    }
                                                    results.append(row)
                                                    cv_results_store[comb_id] = cv_results
                                                    config_store[comb_id] = {
                                                        "alpha_snorna": alpha_s,
                                                        "alpha_disease": alpha_d,
                                                        "k_snorna": int(k_s),
                                                        "k_disease": int(k_d),
                                                        "association_weight": float(assoc_w),
                                                        "lr": float(lr),
                                                        "weight_decay": float(wd),
                                                        "loss_type": loss_type,
                                                        "hidden_dims": list(hidden_dims_cur),
                                                        "focal_alpha": float(f_alpha),
                                                        "focal_gamma": float(f_gamma),
                                                        "dropout": float(dp),
                                                        "contrastive_lambda": contrastive_lambda,
                                                        "contrastive_temp": contrastive_temp,
                                                        "enable_late_fusion": enable_late_fusion,
                                                        "disjoint_train_ratio": disjoint_train_ratio,
                                                        "use_assoc_profile_features": use_assoc_profile_features,
                                                        "use_early_fusion_external": use_early_fusion_external,
                                                        "use_strong_predictor": use_strong_predictor,
                                                    }

                                                    npz_path = save_single_grid_result_npz(
                                                        save_dir=per_combination_npz_dir,
                                                        comb_id=comb_id,
                                                        config=config_store[comb_id],
                                                        cv_results=cv_results,
                                                    )

                                                    print(
                                                        f"    => CV AUPR={cv_results['mean_aupr']:.4f}, "
                                                        f"AUC={cv_results['mean_auc']:.4f}, "
                                                        f"MCC={cv_results['mean_mcc']:.4f}"
                                                    )
                                                    print(f"    => 单组结果已保存: {npz_path}")

    os.makedirs(output_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    cv_results_path = os.path.join(output_dir, "HGNN_cv_results.csv")
    df_results.to_csv(cv_results_path, index=False)
    print(f"\n[网格搜索] CV结果已保存到 {cv_results_path}")

    metric_cols = {
        "mcc": "cv_mean_mcc",
        "auc": "cv_mean_auc",
        "aupr": "cv_mean_aupr",
        "acc": "cv_mean_accuracy",
        "precision": "cv_mean_precision",
        "recall": "cv_mean_recall",
        "f1": "cv_mean_f1",
    }

    rank_df = df_results.copy()
    for col in metric_cols.values():
        rank_df[col] = pd.to_numeric(rank_df[col], errors='coerce').fillna(-np.inf)

    def pick_best_row(df, sort_cols, ascending):
        df_sorted = df.sort_values(by=sort_cols, ascending=ascending, kind='mergesort')
        return df_sorted.iloc[0]

    row_best_mcc = pick_best_row(
        rank_df,
        sort_cols=[metric_cols["mcc"], metric_cols["aupr"], metric_cols["auc"]],
        ascending=[False, False, False],
    )
    row_best_auc = pick_best_row(
        rank_df,
        sort_cols=[metric_cols["auc"], metric_cols["aupr"], metric_cols["mcc"]],
        ascending=[False, False, False],
    )
    row_best_aupr = pick_best_row(
        rank_df,
        sort_cols=[metric_cols["aupr"], metric_cols["auc"], metric_cols["mcc"]],
        ascending=[False, False, False],
    )

    df_safe = rank_df[rank_df[metric_cols["mcc"]] >= mcc_floor].copy()
    if len(df_safe) == 0:
        print(f"\n[警告] 没有配置满足 MCC >= {mcc_floor:.4f}，safe_lexi 自动回退到 max_mcc")
        row_safe_lexi = row_best_mcc
        safe_pool_size = 0
    else:
        row_safe_lexi = pick_best_row(
            df_safe,
            sort_cols=[metric_cols["aupr"], metric_cols["auc"], metric_cols["mcc"]],
            ascending=[False, False, False],
        )
        safe_pool_size = len(df_safe)

    comparison_rows = []
    for strategy_name, row in [
        ("max_mcc", row_best_mcc),
        ("max_auc", row_best_auc),
        ("max_aupr", row_best_aupr),
        ("safe_lexi", row_safe_lexi),
    ]:
        comparison_rows.append({
            "strategy": strategy_name,
            "comb_id": int(row["comb_id"]),
            "cv_mean_aupr": float(row[metric_cols["aupr"]]),
            "cv_mean_auc": float(row[metric_cols["auc"]]),
            "cv_mean_mcc": float(row[metric_cols["mcc"]]),
            "cv_mean_accuracy": float(row[metric_cols["acc"]]),
            "cv_mean_precision": float(row[metric_cols["precision"]]),
            "cv_mean_recall": float(row[metric_cols["recall"]]),
            "cv_mean_f1": float(row[metric_cols["f1"]]),
            "alpha_snorna": row["alpha_snorna"],
            "alpha_disease": row["alpha_disease"],
            "k_snorna": int(row["k_snorna"]),
            "k_disease": int(row["k_disease"]),
            "association_weight": float(row["association_weight"]),
            "lr": float(row["lr"]),
            "weight_decay": float(row["weight_decay"]),
            "loss_type": row["loss_type"],
            "hidden_dims": row["hidden_dims"],
            "focal_alpha": float(row["focal_alpha"]),
            "focal_gamma": float(row["focal_gamma"]),
            "dropout": float(row["dropout"]),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(output_dir, "HGNN_selection_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\n[多标准对比] safe 配置池大小: {safe_pool_size}")
    print(f"[多标准对比] 已保存到 {comparison_path}")
    print(comparison_df.to_string(index=False))

    selected_row = row_safe_lexi
    selected_comb_id = int(selected_row["comb_id"])
    best_cv_results = cv_results_store[selected_comb_id]
    best_config = config_store[selected_comb_id].copy()
    best_config.update({
        "selection_strategy": "safe_lexi",
        "mcc_floor": float(mcc_floor),
        "selected_comb_id": selected_comb_id,
    })

    print("\n[网格搜索] 默认返回策略: safe_lexi")
    print(f"[网格搜索] 最优CV配置（MCC >= {mcc_floor:.4f} 后按 AUPR -> AUC -> MCC）:")
    for key, val in best_config.items():
        print(f"  {key:25} = {val}")

    return best_config, best_cv_results, comparison_df
