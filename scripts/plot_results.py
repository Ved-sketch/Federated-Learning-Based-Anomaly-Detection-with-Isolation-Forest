"""Create a grouped dataset-versus-AUC-ROC comparison chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="AUC-ROC comparison: iForest vs FLiForest")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"dataset", "method", "auc_roc_mean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")
    methods = [method for method in ("iForest", "FLiForest") if method in set(frame.method)]
    if len(methods) != 2:
        raise ValueError("input must include both iForest and FLiForest")
    datasets = list(dict.fromkeys(frame.dataset))
    x = np.arange(len(datasets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    colours = {"iForest": "#4C78A8", "FLiForest": "#F58518"}
    for index, method in enumerate(methods):
        part = frame[frame.method == method].set_index("dataset").reindex(datasets)
        means = part.auc_roc_mean.to_numpy(dtype=float)
        # Paper table is percent; computed runner uses unit interval.
        if np.nanmax(means) <= 1.0:
            means *= 100
        errors = part.get("auc_roc_std", pd.Series(0.0, index=part.index)).fillna(0).to_numpy(dtype=float)
        if np.nanmax(errors, initial=0) <= 1.0:
            errors *= 100
        ax.bar(x + (index - 0.5) * width, means, width, yerr=errors, capsize=3, label=method, color=colours[method], alpha=0.92)
    ax.set_title(args.title)
    ax.set_ylabel("AUC-ROC (%)")
    ax.set_xticks(x, datasets, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    source = frame.get("source", pd.Series(["unknown"])).iloc[0]
    ax.text(
        0.99, 0.02, f"Source: {source}", transform=ax.transAxes, ha="right",
        fontsize=9, color="#555555", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
