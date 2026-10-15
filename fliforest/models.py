"""A data-private, layer-synchronised isolation forest.

Only a candidate feature, candidate cut and local node count leave a client.
Raw feature rows never leave the client object.  The edge server uses counts as
weights, matching the client-size weighting described in paper Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Sequence

import numpy as np


def _c_factor(n: int | float) -> float:
    """Expected unsuccessful-BST search length used by isolation forests."""
    n = int(n)
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return float(2.0 * (np.log(n - 1) + np.euler_gamma) - 2.0 * (n - 1) / n)


@dataclass(frozen=True)
class Leaf:
    size: int


@dataclass(frozen=True)
class Node:
    feature: int
    threshold: float
    left: "Tree"
    right: "Tree"


Tree = Leaf | Node


class FederatedIsolationForest:
    """Federated binary isolation forest with a simulated in-process edge server.

    This is intentionally an inspectable implementation rather than a wrapper
    around a centralized model.  A global split is negotiated at every tree
    layer from client-local proposals.  It supports horizontally partitioned
    data with the same feature schema on every client.
    """

    def __init__(
        self,
        n_trees: int = 100,
        subsample_size: int = 512,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if n_trees < 1 or subsample_size < 2:
            raise ValueError("n_trees must be >= 1 and subsample_size must be >= 2")
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, client_data: Sequence[np.ndarray]) -> "FederatedIsolationForest":
        clients = [np.asarray(x, dtype=np.float64) for x in client_data]
        if len(clients) < 2:
            raise ValueError("FLiForest requires at least two clients")
        if any(x.ndim != 2 or x.shape[0] == 0 for x in clients):
            raise ValueError("each client must hold a non-empty 2-D array")
        if len({x.shape[1] for x in clients}) != 1:
            raise ValueError("all clients need the same number of features")
        if not all(np.isfinite(x).all() for x in clients):
            raise ValueError("input contains missing or non-finite values")

        self.n_features_in_ = clients[0].shape[1]
        self.effective_subsample_size_ = min(self.subsample_size, sum(x.shape[0] for x in clients))
        depth = self.max_depth or ceil(log2(self.effective_subsample_size_))
        rng = np.random.default_rng(self.random_state)
        self.trees_: list[Tree] = []
        for _ in range(self.n_trees):
            samples = [
                x[rng.choice(x.shape[0], size=min(x.shape[0], self.subsample_size), replace=False)]
                for x in clients
            ]
            self.trees_.append(self._build_tree(samples, depth, rng))
        return self

    def _build_tree(self, local_rows: list[np.ndarray], depth_remaining: int, rng: np.random.Generator) -> Tree:
        sizes = np.asarray([rows.shape[0] for rows in local_rows], dtype=np.int64)
        total = int(sizes.sum())
        if depth_remaining <= 0 or total <= 1:
            return Leaf(total)

        # Local candidate generation: no feature values are sent to the server.
        proposals: list[tuple[int, float, int]] = []
        for client_id, rows in enumerate(local_rows):
            if len(rows) <= 1:
                continue
            feature = int(rng.integers(self.n_features_in_))
            low, high = float(rows[:, feature].min()), float(rows[:, feature].max())
            if low < high:
                proposals.append((feature, float(rng.uniform(low, high)), client_id))
        if not proposals:
            return Leaf(total)

        # Edge aggregation: select a proposed attribute proportional to the
        # proposer’s local node size, then compute a weighted global cut from
        # proposals for that attribute. This operationalises Algorithm 1's
        # weighted parameter aggregation when attributes are discrete.
        proposal_weights = np.asarray([sizes[c] for _, _, c in proposals], dtype=float)
        selected = proposals[int(rng.choice(len(proposals), p=proposal_weights / proposal_weights.sum()))]
        feature = selected[0]
        same_feature = [(cut, c) for f, cut, c in proposals if f == feature]
        cut_weights = np.asarray([sizes[c] for _, c in same_feature], dtype=float)
        threshold = float(np.average([cut for cut, _ in same_feature], weights=cut_weights))

        left = [rows[rows[:, feature] < threshold] for rows in local_rows]
        right = [rows[rows[:, feature] >= threshold] for rows in local_rows]
        # A non-splitting negotiation cannot be progressed by recursion.
        if sum(len(x) for x in left) == 0 or sum(len(x) for x in right) == 0:
            return Leaf(total)
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(left, depth_remaining - 1, rng),
            right=self._build_tree(right, depth_remaining - 1, rng),
        )

    @staticmethod
    def _path_length(row: np.ndarray, tree: Tree, depth: int = 0) -> float:
        if isinstance(tree, Leaf):
            return depth + _c_factor(tree.size)
        child = tree.left if row[tree.feature] < tree.threshold else tree.right
        return FederatedIsolationForest._path_length(row, child, depth + 1)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "trees_"):
            raise RuntimeError("fit must be called before score_samples")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("x has an incompatible feature shape")
        if not np.isfinite(x).all():
            raise ValueError("input contains missing or non-finite values")
        mean_paths = np.fromiter(
            (np.mean([self._path_length(row, tree) for tree in self.trees_]) for row in x),
            dtype=float,
            count=len(x),
        )
        normalizer = _c_factor(self.effective_subsample_size_)
        return np.power(2.0, -mean_paths / max(normalizer, np.finfo(float).eps))
