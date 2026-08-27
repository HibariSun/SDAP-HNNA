"""Entity-level cross-validation split utilities.

The project stores positive snoRNA--disease associations as coordinates in an
``[num_snorna, num_disease]`` matrix.  Pair-wise CV splits those coordinates.
Entity-level CV instead splits matrix rows (snoRNA) or columns (disease), then
assigns every positive association of a held-out entity to the test fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CV_LEVEL_TO_AXIS = {"snorna": 0, "disease": 1}


def sample_negatives_uniform(
    association_matrix: np.ndarray,
    num_negs: int,
    exclude_indices: np.ndarray | None = None,
    seed: int = 42,
    allowed_snorna: np.ndarray | None = None,
    allowed_disease: np.ndarray | None = None,
) -> np.ndarray:
    """Uniformly sample unknown pairs, optionally within an entity subset."""
    matrix = np.asarray(association_matrix)
    if matrix.ndim != 2:
        raise ValueError("association_matrix 必须是二维矩阵。")
    if int(num_negs) < 0:
        raise ValueError("num_negs 不能为负数。")

    rng = np.random.default_rng(seed)
    _, num_disease = matrix.shape
    all_neg_flat = np.flatnonzero(matrix.ravel() == 0)

    if allowed_snorna is not None:
        allowed_snorna = np.asarray(allowed_snorna, dtype=np.int64)
        all_neg_flat = all_neg_flat[
            np.isin(all_neg_flat // num_disease, allowed_snorna)
        ]
    if allowed_disease is not None:
        allowed_disease = np.asarray(allowed_disease, dtype=np.int64)
        all_neg_flat = all_neg_flat[
            np.isin(all_neg_flat % num_disease, allowed_disease)
        ]
    if exclude_indices is not None and len(exclude_indices) > 0:
        all_neg_flat = np.setdiff1d(
            all_neg_flat,
            np.asarray(exclude_indices, dtype=np.int64),
            assume_unique=False,
        )

    need = min(int(num_negs), len(all_neg_flat))
    sampled_flat = rng.choice(all_neg_flat, size=need, replace=False)
    return np.column_stack([
        sampled_flat // num_disease,
        sampled_flat % num_disease,
    ]).astype(np.int64)


@dataclass(frozen=True)
class EntityCVPlan:
    """A reproducible entity-level CV plan."""

    cv_level: str
    all_pos: np.ndarray
    cv_splits: list[tuple[np.ndarray, np.ndarray]]
    heldout_entities: list[np.ndarray]
    n_splits: int
    seed: int
    matrix_shape: tuple[int, int]


def _normalize_cv_level(cv_level: str) -> str:
    level = str(cv_level).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "snorna": "snorna",
        "sno": "snorna",
        "row": "snorna",
        "disease": "disease",
        "column": "disease",
        "col": "disease",
    }
    if level not in aliases:
        raise ValueError(
            f"不支持的 cv_level={cv_level!r}；请选择 'snorna' 或 'disease'。"
        )
    return aliases[level]


def make_entity_cv_plan(
    association_matrix: np.ndarray,
    cv_level: str,
    n_splits: int = 10,
    seed: int = 42,
) -> EntityCVPlan:
    """Create K-fold splits over snoRNA rows or disease columns.

    All entities are assigned to exactly one test fold, including entities with
    no known positive associations.  Positive-pair indices remain compatible
    with :func:`train_eval.cross_validation`.
    """
    matrix = np.asarray(association_matrix)
    if matrix.ndim != 2:
        raise ValueError(f"association_matrix 必须是二维矩阵，实际形状为 {matrix.shape}")

    level = _normalize_cv_level(cv_level)
    axis = CV_LEVEL_TO_AXIS[level]
    num_entities = matrix.shape[axis]
    if not 2 <= int(n_splits) <= num_entities:
        raise ValueError(
            f"n_splits 必须在 [2, {num_entities}] 范围内，实际为 {n_splits}。"
        )

    all_pos = np.argwhere(matrix == 1).astype(np.int64)
    if len(all_pos) == 0:
        raise ValueError("关联矩阵中没有值为 1 的正关联。")

    entities = np.arange(num_entities, dtype=np.int64)
    # Equivalent fold-size policy to shuffled KFold, implemented with NumPy so
    # split preparation does not require the full training stack.
    shuffled_entities = entities.copy()
    np.random.RandomState(int(seed)).shuffle(shuffled_entities)
    fold_sizes = np.full(int(n_splits), num_entities // int(n_splits), dtype=int)
    fold_sizes[: num_entities % int(n_splits)] += 1
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    heldout_entities: list[np.ndarray] = []

    current = 0
    for fold_size in fold_sizes:
        heldout = shuffled_entities[current: current + fold_size]
        current += fold_size
        test_mask = np.isin(all_pos[:, axis], heldout)
        train_idx = np.flatnonzero(~test_mask).astype(np.int64)
        test_idx = np.flatnonzero(test_mask).astype(np.int64)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                "至少一折没有训练正样本或测试正样本。请减少 n_splits，或检查是否存在"
                "大量没有已知关联的实体。"
            )
        cv_splits.append((train_idx, test_idx))
        heldout_entities.append(heldout.astype(np.int64))

    plan = EntityCVPlan(
        cv_level=level,
        all_pos=all_pos,
        cv_splits=cv_splits,
        heldout_entities=heldout_entities,
        n_splits=int(n_splits),
        seed=int(seed),
        matrix_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
    )
    validate_entity_cv_plan(plan, association_matrix=matrix)
    return plan


def validate_entity_cv_plan(
    plan: EntityCVPlan,
    association_matrix: np.ndarray | None = None,
) -> None:
    """Raise ``ValueError`` if an entity split contains leakage or is stale."""
    level = _normalize_cv_level(plan.cv_level)
    axis = CV_LEVEL_TO_AXIS[level]
    all_pos = np.asarray(plan.all_pos, dtype=np.int64)

    if all_pos.ndim != 2 or all_pos.shape[1] != 2:
        raise ValueError(f"all_pos 应为 [N, 2]，实际形状为 {all_pos.shape}")
    if len(plan.cv_splits) != plan.n_splits:
        raise ValueError("cv_splits 数量与 n_splits 不一致。")
    if len(plan.heldout_entities) != plan.n_splits:
        raise ValueError("heldout_entities 数量与 n_splits 不一致。")

    if association_matrix is not None:
        matrix = np.asarray(association_matrix)
        if tuple(matrix.shape) != tuple(plan.matrix_shape):
            raise ValueError(
                f"划分对应矩阵形状为 {plan.matrix_shape}，当前矩阵为 {matrix.shape}。"
            )
        current_pos = np.argwhere(matrix == 1).astype(np.int64)
        if not np.array_equal(current_pos, all_pos):
            raise ValueError("划分文件中的正关联与当前关联矩阵不一致。")

    num_entities = plan.matrix_shape[axis]
    test_counts = np.zeros(len(all_pos), dtype=np.int64)
    entity_counts = np.zeros(num_entities, dtype=np.int64)

    for fold, ((train_idx, test_idx), heldout) in enumerate(
        zip(plan.cv_splits, plan.heldout_entities)
    ):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        heldout = np.asarray(heldout, dtype=np.int64)

        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"第 {fold + 1} 折的训练/测试正样本索引有重叠。")
        if np.any((heldout < 0) | (heldout >= num_entities)):
            raise ValueError(f"第 {fold + 1} 折包含越界实体编号。")

        expected_test = np.flatnonzero(np.isin(all_pos[:, axis], heldout))
        expected_train = np.flatnonzero(~np.isin(all_pos[:, axis], heldout))
        if not np.array_equal(np.sort(test_idx), expected_test):
            raise ValueError(f"第 {fold + 1} 折未包含留出实体的全部正关联。")
        if not np.array_equal(np.sort(train_idx), expected_train):
            raise ValueError(f"第 {fold + 1} 折训练集错误地包含了留出实体。")

        test_counts[test_idx] += 1
        entity_counts[heldout] += 1

    if not np.all(test_counts == 1):
        raise ValueError("每个正关联必须恰好作为测试样本出现一次。")
    if not np.all(entity_counts == 1):
        raise ValueError("每个实体必须恰好在一折中被留出。")


def save_entity_cv_plan(plan: EntityCVPlan, save_path: str | Path) -> Path:
    """Save an :class:`EntityCVPlan` as a compressed NPZ file."""
    validate_entity_cv_plan(plan)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "format_version": np.int64(1),
        "cv_level": np.array(plan.cv_level),
        "all_pos": np.asarray(plan.all_pos, dtype=np.int64),
        "n_splits": np.int64(plan.n_splits),
        "seed": np.int64(plan.seed),
        "matrix_shape": np.asarray(plan.matrix_shape, dtype=np.int64),
    }
    for fold, ((train_idx, test_idx), heldout) in enumerate(
        zip(plan.cv_splits, plan.heldout_entities)
    ):
        payload[f"fold{fold}_train_idx"] = np.asarray(train_idx, dtype=np.int64)
        payload[f"fold{fold}_test_idx"] = np.asarray(test_idx, dtype=np.int64)
        payload[f"fold{fold}_heldout_entities"] = np.asarray(heldout, dtype=np.int64)

    np.savez_compressed(path, **payload)
    return path


def load_entity_cv_plan(
    load_path: str | Path,
    association_matrix: np.ndarray | None = None,
) -> EntityCVPlan:
    """Load and validate an entity-level CV plan."""
    path = Path(load_path)
    with np.load(path, allow_pickle=False) as data:
        required = {"cv_level", "all_pos", "n_splits", "seed", "matrix_shape"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"划分文件缺少字段 {missing}: {path}")

        n_splits = int(data["n_splits"])
        cv_splits = []
        heldout_entities = []
        for fold in range(n_splits):
            cv_splits.append((
                np.asarray(data[f"fold{fold}_train_idx"], dtype=np.int64),
                np.asarray(data[f"fold{fold}_test_idx"], dtype=np.int64),
            ))
            heldout_entities.append(
                np.asarray(data[f"fold{fold}_heldout_entities"], dtype=np.int64)
            )

        plan = EntityCVPlan(
            cv_level=str(data["cv_level"].item()),
            all_pos=np.asarray(data["all_pos"], dtype=np.int64),
            cv_splits=cv_splits,
            heldout_entities=heldout_entities,
            n_splits=n_splits,
            seed=int(data["seed"]),
            matrix_shape=tuple(int(x) for x in data["matrix_shape"]),
        )

    validate_entity_cv_plan(plan, association_matrix=association_matrix)
    return plan
