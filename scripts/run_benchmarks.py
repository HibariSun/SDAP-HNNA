"""Run SDAP-MHNN on the prepared benchmark datasets.

Use this after running scripts/prepare_benchmark_datasets.py.
The script launches main.py once per dataset and keeps each dataset's outputs
under results/benchmarks/<dataset>/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDAP-MHNN on benchmark datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["psnod", "gbdtsvm", "gcnmf_sda"],
        choices=["psnod", "gbdtsvm", "gcnmf_sda"],
    )
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcc_floor", type=float, default=0.62)
    parser.add_argument("--mode", choices=["fixed", "grid"], default="fixed")
    args = parser.parse_args()

    for dataset in args.datasets:
        data_path = ROOT / "benchmarks" / dataset
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} does not exist. Run scripts/prepare_benchmark_datasets.py first."
            )

        output_dir = ROOT / "results" / "benchmarks" / dataset
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "--data_path",
            str(data_path),
            "--dataset_name",
            dataset,
            "--output_dir",
            str(output_dir),
            "--n_splits",
            str(args.n_splits),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--mcc_floor",
            str(args.mcc_floor),
            "--mode",
            args.mode,
        ]

        print("\n" + "=" * 80)
        print(f"[run] {dataset}")
        print(" ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
