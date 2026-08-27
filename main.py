"""
Author:Hibari
超图神经网络 - snoRNA-Disease关联预测（纯10折CV版本）
"""
import argparse
import os
import numpy as np

from utils import set_seed, init_logging, print_device_info
from data_utils import DataLoader
from train_eval import (
    get_all_pos_samples, save_cv_splits, load_cv_splits,
    cross_validation, grid_search, print_fold_table, print_cv_summary,
    sample_negatives_uniform,
)
from visualization import ResultVisualizer, save_results


def save_comparison_data_npz(
    save_path,
    dataset_name,
    association_matrix,
    all_pos,
    cv_splits,
    n_splits,
    seed,
    sno_ids,
    disease_ids,
    test_neg_ratio=1,
):
    """Save a reusable benchmark package for fair cross-method comparison."""
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload = {
        "dataset_name": np.array(dataset_name),
        "association_matrix": association_matrix.astype(np.float32),
        "all_pos": all_pos.astype(np.int64),
        "n_splits": np.int64(n_splits),
        "seed": np.int64(seed),
        "test_neg_ratio": np.int64(test_neg_ratio),
        "sno_ids": np.asarray(sno_ids, dtype=str),
        "disease_ids": np.asarray(disease_ids, dtype=str),
    }

    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_pos = all_pos[train_idx].astype(np.int64)
        test_pos = all_pos[test_idx].astype(np.int64)
        test_neg = sample_negatives_uniform(
            association_matrix,
            num_negs=len(test_pos) * int(test_neg_ratio),
            seed=42 + fold,
        ).astype(np.int64)

        payload[f"fold{fold}_train_idx"] = train_idx
        payload[f"fold{fold}_test_idx"] = test_idx
        payload[f"fold{fold}_train_pos"] = train_pos
        payload[f"fold{fold}_test_pos"] = test_pos
        payload[f"fold{fold}_test_neg"] = test_neg

    np.savez_compressed(save_path, **payload)
    print(f"  comparison npz saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SDAP-MHNN cross-validation.")
    parser.add_argument("--data_path", default="./data/", help="Directory containing normalized dataset files.")
    parser.add_argument("--dataset_name", default="default", help="Name used for logs and result folders.")
    parser.add_argument("--output_dir", default=None, help="Result directory. Defaults to results/<dataset_name>.")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for splits and training.")
    parser.add_argument("--epochs", type=int, default=300, help="Max epochs per fold.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--mcc_floor", type=float, default=0.62, help="Safe selection MCC floor.")
    parser.add_argument(
        "--mode",
        choices=["fixed", "grid"],
        default="fixed",
        help="fixed runs one model configuration; grid runs the original grid search.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join("results", args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = init_logging(log_dir=os.path.join(output_dir, "logs"))
    print(f"[日志] 当前运行日志文件: {log_path}")

    set_seed(args.seed)
    print_device_info()

    print("\n" + "=" * 80)
    print(f"  SDAP-MHNN（{args.n_splits}折CV版本）")
    print("=" * 80)
    print(f"[数据集] {args.dataset_name}")
    print(f"[数据路径] {args.data_path}")
    print(f"[输出目录] {output_dir}")

    data_loader = DataLoader(data_path=args.data_path)

    association_matrix, _, _ = data_loader.load_all_data(
        alpha_snorna=0.7, alpha_disease=0.7
    )

    n_splits = args.n_splits
    split_path = os.path.join(args.data_path, f"cv_splits_{n_splits}fold_seed{args.seed}.npz")

    all_pos = get_all_pos_samples(association_matrix)
    print(f"\n[数据] 总正样本数: {len(all_pos)}")

    if os.path.exists(split_path):
        all_pos_loaded, n_splits_loaded, seed_loaded, cv_splits = load_cv_splits(split_path)
        assert len(all_pos_loaded) == len(all_pos), (
            f"划分文件中正样本数 ({len(all_pos_loaded)}) 与当前数据不一致 ({len(all_pos)})"
        )
        all_pos = all_pos_loaded
        n_splits = n_splits_loaded
    else:
        save_cv_splits(all_pos, n_splits=n_splits, seed=args.seed, save_path=split_path)
        _, _, _, cv_splits = load_cv_splits(split_path)

    sno_ids = getattr(data_loader, "_sno_ids", None)
    disease_ids = getattr(data_loader, "_disease_ids", None)
    if sno_ids is None:
        sno_ids = np.arange(association_matrix.shape[0]).astype(str)
    if disease_ids is None:
        disease_ids = np.arange(association_matrix.shape[1]).astype(str)

    comparison_npz_name = f"comparison_data_{n_splits}fold_seed{args.seed}.npz"
    comparison_npz_data_path = os.path.join(args.data_path, comparison_npz_name)
    comparison_npz_output_path = os.path.join(output_dir, comparison_npz_name)
    save_comparison_data_npz(
        save_path=comparison_npz_data_path,
        dataset_name=args.dataset_name,
        association_matrix=association_matrix,
        all_pos=all_pos,
        cv_splits=cv_splits,
        n_splits=n_splits,
        seed=args.seed,
        sno_ids=sno_ids,
        disease_ids=disease_ids,
    )
    if os.path.abspath(comparison_npz_output_path) != os.path.abspath(comparison_npz_data_path):
        save_comparison_data_npz(
            save_path=comparison_npz_output_path,
            dataset_name=args.dataset_name,
            association_matrix=association_matrix,
            all_pos=all_pos,
            cv_splits=cv_splits,
            n_splits=n_splits,
            seed=args.seed,
            sno_ids=sno_ids,
            disease_ids=disease_ids,
        )

    # Step 2: 网格搜索参数设置
    # 默认采用“局部结构搜索”，组合数控制在可跑范围；跑完后可围绕最优组合再做二次精调。
    epochs = args.epochs
    patience = args.patience
    mcc_floor = args.mcc_floor

    alpha_snorna_list = [0.65, 0.70, 0.75]
    alpha_disease_list = [0.65, 0.70, 0.75]

    k_snorna_list = [9, 10, 11]
    k_disease_list = [7, 8, 9]

    association_weight_list = [0.03, 0.05, 0.08]
    lr_list = [3e-4]
    dropout_list = [0.30]
    focal_alpha_list = [0.6]
    focal_gamma_list = [2.0]

    loss_type_list = ['bce']
    hidden_dims_candidates = [[128, 128, 64]]
    weight_decay_list = [1e-3]

    use_assoc_profile_features = True
    use_early_fusion_external = False
    use_strong_predictor = True

    enable_late_fusion = True
    contrastive_lambda = 0.1
    contrastive_temp = 0.1
    disjoint_train_ratio = 0.2

    total_combinations = (
        len(alpha_snorna_list) * len(alpha_disease_list) *
        len(k_snorna_list) * len(k_disease_list) *
        len(association_weight_list) * len(lr_list) *
        len(focal_alpha_list) * len(focal_gamma_list) *
        len(dropout_list) * len(loss_type_list) *
        len(hidden_dims_candidates) * len(weight_decay_list)
    )

    print(f"\n[参数设置]")
    print(f"  alpha_snorna: {alpha_snorna_list}")
    print(f"  alpha_disease: {alpha_disease_list}")
    print(f"  k_snorna: {k_snorna_list}")
    print(f"  k_disease: {k_disease_list}")
    print(f"  association_weight: {association_weight_list}")
    print(f"  lr: {lr_list}")
    print(f"  dropout: {dropout_list}")
    print(f"  weight_decay: {weight_decay_list}")
    print(f"  hidden_dims_candidates: {hidden_dims_candidates}")
    print(f"  loss_type_list: {loss_type_list}")
    print(f"  mcc_floor: {mcc_floor}")
    print(f"  总组合数: {total_combinations}")
    print(f"  运行模式: {args.mode}")
    if args.mode == "grid":
        print(f"  每组合 {n_splits} 折CV，预计耗时较长...")

        best_config, best_cv_results, comparison_df = grid_search(
            data_loader=data_loader,
            all_pos=all_pos,
            cv_splits=cv_splits,
            alpha_snorna_list=alpha_snorna_list,
            alpha_disease_list=alpha_disease_list,
            k_snorna_list=k_snorna_list,
            k_disease_list=k_disease_list,
            association_weight_list=association_weight_list,
            lr_list=lr_list,
            focal_alpha_list=focal_alpha_list,
            focal_gamma_list=focal_gamma_list,
            loss_type_list=loss_type_list,
            hidden_dims_candidates=hidden_dims_candidates,
            weight_decay_list=weight_decay_list,
            n_splits=n_splits,
            epochs=epochs,
            patience=patience,
            contrastive_lambda=contrastive_lambda,
            contrastive_temp=contrastive_temp,
            enable_late_fusion=enable_late_fusion,
            disjoint_train_ratio=disjoint_train_ratio,
            dropout_list=dropout_list,
            use_assoc_profile_features=use_assoc_profile_features,
            use_early_fusion_external=use_early_fusion_external,
            use_strong_predictor=use_strong_predictor,
            mcc_floor=mcc_floor,
            per_combination_npz_dir=os.path.join(output_dir, "grid_npz"),
            output_dir=output_dir,
        )

        print("\n[多标准最优配置对比表]")
        print(comparison_df.to_string(index=False))

        print(f"\n默认返回的最优配置:")
        for key, val in best_config.items():
            print(f"  {key:25} = {val}")
    else:
        best_config = {
            "alpha_snorna": 0.70,
            "alpha_disease": 0.70,
            "k_snorna": 9,
            "k_disease": 8,
            "association_weight": 0.03,
            "lr": 3e-4,
            "dropout": 0.30,
            "focal_alpha": 0.6,
            "focal_gamma": 2.0,
            "loss_type": "bce",
            "hidden_dims": [128, 128, 64],
            "weight_decay": 1e-3,
            "contrastive_lambda": contrastive_lambda,
            "contrastive_temp": contrastive_temp,
            "enable_late_fusion": enable_late_fusion,
            "disjoint_train_ratio": disjoint_train_ratio,
            "use_assoc_profile_features": use_assoc_profile_features,
            "use_early_fusion_external": use_early_fusion_external,
            "use_strong_predictor": use_strong_predictor,
        }
        print("\n[固定配置]")
        for key, val in best_config.items():
            print(f"  {key:25} = {val}")

        best_cv_results = cross_validation(
            association_matrix,
            None,
            None,
            all_pos,
            cv_splits=cv_splits,
            n_splits=n_splits,
            epochs=epochs,
            lr=best_config["lr"],
            patience=patience,
            association_weight=best_config["association_weight"],
            k_snorna=best_config["k_snorna"],
            k_disease=best_config["k_disease"],
            hidden_dims=best_config["hidden_dims"],
            dropout=best_config["dropout"],
            loss_type=best_config["loss_type"],
            data_loader=data_loader,
            alpha_snorna=best_config["alpha_snorna"],
            alpha_disease=best_config["alpha_disease"],
            contrastive_lambda=best_config["contrastive_lambda"],
            contrastive_temp=best_config["contrastive_temp"],
            enable_late_fusion=best_config["enable_late_fusion"],
            disjoint_train_ratio=best_config["disjoint_train_ratio"],
            focal_alpha=best_config["focal_alpha"],
            focal_gamma=best_config["focal_gamma"],
            weight_decay=best_config["weight_decay"],
            use_assoc_profile_features=best_config["use_assoc_profile_features"],
            use_early_fusion_external=best_config["use_early_fusion_external"],
            use_strong_predictor=best_config["use_strong_predictor"],
        )
        comparison_df = None

    if best_cv_results is not None and 'fold_results' in best_cv_results:
        print_fold_table(best_cv_results['fold_results'], title='safe_lexi 最优配置的10折CV各项指标')
        print_cv_summary(best_cv_results['fold_results'], prefix='safe_lexi 最优配置 CV')

    print(f"\n  {n_splits}折CV汇总性能 (threshold=0.5):")
    for metric in ['auc', 'aupr', 'accuracy', 'precision', 'recall', 'f1', 'mcc']:
        mean_val = best_cv_results.get(f'mean_{metric}', float('nan'))
        std_val = best_cv_results.get(f'std_{metric}', float('nan'))
        print(f"  {metric.upper():>10s} : {mean_val:.4f} ± {std_val:.4f}")

    try:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        if best_cv_results is not None and 'fold_results' in best_cv_results:
            save_results(best_cv_results['fold_results'], output_dir=output_dir)
            visualizer = ResultVisualizer(output_dir=plots_dir)

            y_true = best_cv_results.get('all_y_true', np.array([]))
            y_scores = best_cv_results.get('all_y_scores', np.array([]))
            visualizer.plot_cv_report(best_cv_results['fold_results'], y_true, y_scores)
        else:
            print('[可视化] 未找到 best_cv_results.fold_results，跳过CV可视化。')
    except Exception as e:
        print(f"[可视化] 生成失败: {e}")

    y_true = best_cv_results.get('all_y_true', np.array([]))
    y_scores = best_cv_results.get('all_y_scores', np.array([]))
    pred_path = os.path.join(output_dir, "cv_predictions.npz")
    np.savez(pred_path, y_true=y_true, y_scores=y_scores)
    print(f"\nCV预测结果已保存到 {pred_path}")
    print(f"[多标准对比表] 已保存到 {os.path.join(output_dir, 'HGNN_selection_comparison.csv')}")
    print(f"[全部CV结果] 已保存到 {os.path.join(output_dir, 'HGNN_cv_results.csv')}")

    print("\n" + "=" * 80)
    print("  训练完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
