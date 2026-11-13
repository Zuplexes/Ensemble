"""CART decision tree (weighted Gini) and decision stump. numpy + stdlib only."""
from __future__ import annotations

import numpy as np


def _weighted_impurity(counts: np.ndarray) -> np.ndarray:
    """Sum-of-weights * Gini for each row of class-weight counts: s - sum(c^2)/s (0 if s == 0)."""
    s = counts.sum(-1)
    return s - (counts ** 2).sum(-1) / np.where(s > 0, s, 1.0)


class DecisionTree:
    """CART classifier with Gini impurity weighted by sample_weight."""

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.n_classes_ = int(y.max()) + 1
        W = np.zeros((len(y), self.n_classes_))
        W[np.arange(len(y)), y] = w  # per-sample weighted one-hot
        # Flat node arrays; feature == -1 marks a leaf.
        self.feature, self.threshold, self.left, self.right, self.value = [], [], [], [], []
        self._build(X, W, depth=0)
        return self

    def _new_node(self, W: np.ndarray) -> int:
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(int(np.argmax(W.sum(0))))  # ties -> lowest class index
        return len(self.feature) - 1

    def _build(self, X: np.ndarray, W: np.ndarray, depth: int) -> int:
        node = self._new_node(W)
        if (self.max_depth is not None and depth >= self.max_depth) or len(X) < self.min_samples_split:
            return node
        split = self._best_split(X, W)
        if split is None:
            return node
        j, t = split
        mask = X[:, j] <= t
        self.feature[node], self.threshold[node] = j, t
        self.left[node] = self._build(X[mask], W[mask], depth + 1)
        self.right[node] = self._build(X[~mask], W[~mask], depth + 1)
        return node

    @staticmethod
    def _best_split(X: np.ndarray, W: np.ndarray) -> tuple[int, float] | None:
        """Vectorized search: per feature, sort then cumsum class weights for every cut point."""
        total = W.sum(0)
        if np.count_nonzero(total) <= 1:  # pure node
            return None
        # Like sklearn: any valid cut of an impure node is allowed, even with zero gain (e.g. XOR).
        best_imp, best = np.inf, None
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            left = np.cumsum(W[order], axis=0)[:-1]
            imp = _weighted_impurity(left) + _weighted_impurity(total - left)
            imp[xs[1:] <= xs[:-1]] = np.inf  # can't cut between equal values
            if len(imp) and imp.min() < best_imp:
                i = int(np.argmin(imp))  # first minimum -> deterministic
                best_imp, best = imp[i], (j, (xs[i] + xs[i + 1]) / 2)
        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X), dtype=int)
        stack = [(0, np.arange(len(X)))]
        while stack:  # route index blocks down the tree instead of looping per sample
            node, idx = stack.pop()
            if self.feature[node] < 0:
                out[idx] = self.value[node]
                continue
            go_left = X[idx, self.feature[node]] <= self.threshold[node]
            stack += [(self.left[node], idx[go_left]), (self.right[node], idx[~go_left])]
        return out


class DecisionStump(DecisionTree):
    """Depth-1 tree."""

    def __init__(self) -> None:
        super().__init__(max_depth=1)
