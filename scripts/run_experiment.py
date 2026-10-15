"""Run the central iForest vs. federated FLiForest comparison for one dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fliforest import FederatedIsolationForest  # noqa: E402
from fliforest.data import load_labeled_matrix  # noqa: E402


def synthetic_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic substitute for verifying the end-to-end pipeline."""
    x, y = make_classification(
        n_samples=2500, n_features=20, n_informative=8, n_redundant=3,
        weights=[0.94, 0.06], class_sep=0.8, random_state=seed,
    )
    # The minority class is made spatially unusual so that an unsupervised
    # detector has a meaningful anomaly target in the smoke test.
    x[y == 1] += 2.5
    return x.astype(float), y.astype(int)


def partition_clients(x: np.ndarray, clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(len(x))
    return [part for part in np.array_split(x[order], clients) if len(part)]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-file", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--subsample-size", type=int, default=512)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/computed_auc_roc.csv"))
    args = parser.parse_args()
    if args.clients < 2 or args.repetitions < 1:
        parser.error("--clients must be >= 2 and --repetitions must be >= 1")

    if args.synthetic:
        x, y = synthetic_dataset(args.seed)
        name = args.dataset_name or "synthetic_smoke"
    else:
        x, y = load_labeled_matrix(args.dataset_file, args.label_column)
        name = args.dataset_name or args.dataset_file.stem

    rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        seed = args.seed + repetition
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, stratify=y, random_state=seed
        )
        scale = StandardScaler().fit(x_train)
        x_train, x_test = scale.transform(x_train), scale.transform(x_test)

        central = IsolationForest(
            n_estimators=args.n_trees,
            max_samples=min(args.subsample_size, len(x_train)),
            random_state=seed,
            n_jobs=-1,
        ).fit(x_train)
        central_auc = roc_auc_score(y_test, -central.score_samples(x_test))

        clients = partition_clients(x_train, args.clients, np.random.default_rng(seed))
        federated = FederatedIsolationForest(
            n_trees=args.n_trees,
            subsample_size=args.subsample_size,
            random_state=seed,
        ).fit(clients)
        federated_auc = roc_auc_score(y_test, federated.score_samples(x_test))
        rows.extend(
            [
                {"dataset": name, "method": "iForest", "repetition": repetition + 1, "auc_roc": central_auc, "source": "computed"},
                {"dataset": name, "method": "FLiForest", "repetition": repetition + 1, "auc_roc": federated_auc, "source": "computed"},
            ]
        )
        print(f"{name} repetition {repetition + 1}/{args.repetitions}: iForest={central_auc:.4f}, FLiForest={federated_auc:.4f}")

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["dataset", "method", "source"], as_index=False)["auc_roc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "auc_roc_mean", "std": "auc_roc_std", "count": "repetitions"})
    )
    # pandas retains a synthetic grouping index after the named aggregation.
    # It is an implementation detail, not a result column.
    summary = summary.drop(columns="index", errors="ignore")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    per_run.to_csv(args.output.with_name(args.output.stem + "_per_run.csv"), index=False)
    print(f"Wrote {args.output} and per-run measurements alongside it.")


if __name__ == "__main__":
    main()
