"""Loading and validation helpers for user-supplied anomaly datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_labeled_matrix(path: str | Path, label_column: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Read X/y from .npz or a numeric CSV; anomalies must be labelled 1."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        if "X" not in archive or "y" not in archive:
            raise ValueError("NPZ input must contain arrays named X and y")
        x, y = archive["X"], archive["y"]
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if label_column not in frame:
            raise ValueError(f"CSV requires a '{label_column}' column (or pass --label-column)")
        y = frame.pop(label_column).to_numpy()
        x = frame.to_numpy(dtype=float)
    else:
        raise ValueError("supported inputs are .npz and .csv")
    x, y = np.asarray(x, dtype=float), np.asarray(y).astype(int)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("X must be 2-D and have the same number of rows as y")
    if not np.isfinite(x).all():
        raise ValueError("remove/impute missing and non-finite feature values before running")
    labels = set(np.unique(y))
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError("labels must contain both 0 (normal) and 1 (anomaly)")
    return x, y
