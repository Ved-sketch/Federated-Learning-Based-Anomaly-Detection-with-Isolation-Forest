"""Run the paper's ten named local datasets when prepared .npz files exist."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PAPER_DATASETS = ["celeba", "aloi", "skin", "mnist", "fashionmnist", "cifar10", "waveform", "internetads", "letter", "vowel"]
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/local"))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--subsample-size", type=int, default=512)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=100)
    args = parser.parse_args()
    summaries = []
    missing = []
    for name in PAPER_DATASETS:
        path = args.data_dir / f"{name}.npz"
        if not path.exists():
            missing.append(name)
            continue
        output = ROOT / "results" / f"{name}_computed.csv"
        command = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--dataset-file", str(path), "--dataset-name", name, "--repetitions", str(args.repetitions), "--subsample-size", str(args.subsample_size), "--clients", str(args.clients), "--n-trees", str(args.n_trees), "--output", str(output)]
        subprocess.run(command, check=True)
        summaries.append(pd.read_csv(output))
    if not summaries:
        raise SystemExit("No expected .npz files found. See README.md for the required filenames.")
    combined = pd.concat(summaries, ignore_index=True)
    result = ROOT / "results" / "computed_auc_roc.csv"
    combined.to_csv(result, index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_results.py"), "--input", str(result), "--output", str(ROOT / "outputs" / "computed_auc_roc.png"), "--title", "Computed AUC-ROC: iForest vs FLiForest"], check=True)
    if missing:
        print("Skipped missing datasets: " + ", ".join(missing))


if __name__ == "__main__":
    main()
