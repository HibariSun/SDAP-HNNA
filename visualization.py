# Author:Hibari
# 2025年11月18日17时42分30秒
# visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# 结果可视化
class ResultVisualizer:
    """结果可视化器"""

    def __init__(self, output_dir='./outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_all_results(self, fold_results, all_y_true, all_y_scores, fold_predictions):
        print(f"\n[步骤 4] 生成可视化结果...")

        self._plot_fold_comparison(fold_results)
        self._plot_overall_roc(all_y_true, all_y_scores)
        self._plot_overall_pr(all_y_true, all_y_scores)
        self._plot_comprehensive_panel(fold_results, all_y_true, all_y_scores)

        print(f"\n所有可视化已保存到: {self.output_dir}/")

    def _plot_fold_comparison(self, fold_results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]

        # AUC
        bars1 = axes[0].bar(folds, aucs, color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
        axes[0].axhline(y=np.mean(aucs), color='blue', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(aucs):.4f}')
        axes[0].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('AUC', fontsize=12, fontweight='bold')
        axes[0].set_title('AUC Score across Folds', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        # AUPR
        bars2 = axes[1].bar(folds, auprs, color='coral', alpha=0.7, edgecolor='darkred', linewidth=1.5)
        axes[1].axhline(y=np.mean(auprs), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(auprs):.4f}')
        axes[1].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('AUPR', fontsize=12, fontweight='bold')
        axes[1].set_title('AUPR Score across Folds', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/IMP_01_fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_01_fold_comparison.png")

    def plot_cv_report(self, cv_fold_results, all_y_true, all_y_scores):
        """纯10折CV版本：各折指标 + 汇总ROC/PR曲线"""
        print(f"\n[步骤 4] 生成可视化结果...")
        self._plot_fold_metrics_grid(cv_fold_results)
        if len(all_y_true) > 0 and len(all_y_scores) > 0:
            self._plot_overall_roc(all_y_true, all_y_scores, prefix="CV")
            self._plot_overall_pr(all_y_true, all_y_scores, prefix="CV")
            # 综合面板
            self._plot_comprehensive_panel(cv_fold_results, all_y_true, all_y_scores)
        # CV汇总指标柱状图
        from train_eval import summarize_fold_results
        summary = summarize_fold_results(cv_fold_results)
        cv_metrics = {k.replace("mean_", ""): v for k, v in summary.items() if k.startswith("mean_")}
        self._plot_metrics_bar(cv_metrics, filename="CV_03_metrics_bar.png", title="10-Fold CV Mean Metrics")
        print(f"\n所有可视化已保存到: {self.output_dir}/")

    def _plot_fold_metrics_grid(self, fold_results, metrics=None):
        """把每折的多指标一起画出来。"""
        if metrics is None:
            metrics = [
                ("auc", "AUC"),
                ("aupr", "AUPR"),
                ("accuracy", "Accuracy"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("mcc", "MCC"),
            ]

        folds = [r.get("fold", i + 1) for i, r in enumerate(fold_results)]

        n = len(metrics)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, (key, label) in enumerate(metrics):
            ax = axes[idx // ncols, idx % ncols]
            vals = [r.get(key, np.nan) for r in fold_results]
            ax.bar(folds, vals, alpha=0.8)
            mean_v = np.nanmean(vals)
            ax.axhline(mean_v, linestyle="--", linewidth=2, label=f"Mean: {mean_v:.4f}")
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Score")
            ax.set_xticks(folds)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=9)

        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/IMP_00_fold_metrics_grid.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: IMP_00_fold_metrics_grid.png")

    def _plot_metrics_bar(self, metrics_dict, filename="IMP_metrics_bar.png", title="Metrics"):
        """把一组指标画成柱状图。"""
        keys = ["auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc"]
        labels = ["AUC", "AUPR", "ACC", "Prec", "Rec", "F1", "MCC"]
        vals = [metrics_dict.get(k, np.nan) for k in keys]

        plt.figure(figsize=(10, 5))
        x = np.arange(len(keys))
        plt.bar(x, vals, alpha=0.85)
        plt.xticks(x, labels)
        plt.ylim(0, 1.0)
        plt.title(title, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)

        for xi, v in zip(x, vals):
            if np.isfinite(v):
                plt.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")

    def _plot_overall_roc(self, y_true, y_scores, prefix='IMP'):
        plt.figure(figsize=(10, 8))

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=3,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Overall ROC Curve (10-Fold CV)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)

        plt.savefig(f'{self.output_dir}/IMP_02_overall_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_02_overall_roc_curve.png")

    def _plot_overall_pr(self, y_true, y_scores, prefix='IMP'):
        plt.figure(figsize=(10, 8))

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, color='blue', lw=3,
                 label=f'PR curve (AUPR = {pr_auc:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Overall Precision-Recall Curve (10-Fold CV)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3)

        plt.savefig(f'{self.output_dir}/IMP_03_overall_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_03_overall_pr_curve.png")

    def _plot_comprehensive_panel(self, fold_results, y_true, y_scores):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]

        # 性能对比
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(folds))
        width = 0.35
        ax1.bar(x - width / 2, aucs, width, label='AUC', color='steelblue', alpha=0.8)
        ax1.bar(x + width / 2, auprs, width, label='AUPR', color='coral', alpha=0.8)
        ax1.set_xlabel('Fold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(A) Performance Comparison', fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # ROC曲线
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.4f}')
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlabel('False Positive Rate', fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontweight='bold')
        ax2.set_title('(B) ROC Curve', fontweight='bold', loc='left')
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)

        # PR曲线
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, color='blue', lw=3, label=f'AUPR = {pr_auc:.4f}')
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('(C) Precision-Recall Curve', fontweight='bold', loc='left')
        ax3.legend(loc="lower left")
        ax3.grid(alpha=0.3)

        # 统计表格
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')

        summary_data = [
            ['Metric', 'Mean', 'Std', 'Min', 'Max'],
            ['AUC', f'{np.mean(aucs):.4f}', f'{np.std(aucs):.4f}',
             f'{np.min(aucs):.4f}', f'{np.max(aucs):.4f}'],
            ['AUPR', f'{np.mean(auprs):.4f}', f'{np.std(auprs):.4f}',
             f'{np.min(auprs):.4f}', f'{np.max(auprs):.4f}']
        ]

        table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                          colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax4.set_title('(D) Statistical Summary', fontweight='bold', loc='left', pad=20)

        plt.savefig(f'{self.output_dir}/IMP_04_comprehensive_panel.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_04_comprehensive_panel.png")


# 保存结果
def save_results(fold_results, output_dir='./outputs'):
    os.makedirs(output_dir, exist_ok=True)

    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(f'{output_dir}/IMP_fold_results.csv', index=False)

    metric_values = {
        'AUC':      [r['auc']       for r in fold_results],
        'AUPR':     [r['aupr']      for r in fold_results],
        'Accuracy': [r['accuracy']  for r in fold_results],
        'Precision':[r['precision'] for r in fold_results],
        'Recall':   [r['recall']    for r in fold_results],
        'F1':       [r['f1']        for r in fold_results],
        'MCC':      [r['mcc']       for r in fold_results],
    }

    summary_rows = []
    for metric_name, values in metric_values.items():
        values = np.array(values, dtype=float)
        summary_rows.append({
            'Metric': metric_name,
            'Mean':   np.mean(values),
            'Std':    np.std(values),
            'Min':    np.min(values),
            'Max':    np.max(values),
            'Median': np.median(values),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f'{output_dir}/IMP_summary_statistics.csv', index=False)

    print(f"\n[步骤 5] 结果已保存到: {output_dir}/")
