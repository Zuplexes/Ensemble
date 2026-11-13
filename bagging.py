"""Bagging (bootstrap aggregating) with majority vote. numpy + stdlib only."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def majority_vote(preds: np.ndarray, n_classes: int, weights: np.ndarray | None = None) -> np.ndarray:
    """preds: (n_estimators, n_samples) labels -> per-sample (weighted) vote winner; ties -> lowest label."""
    n = preds.shape[1]
    w = None if weights is None else np.repeat(weights, n)
    flat = (preds + n_classes * np.arange(n)).ravel()  # offset so one bincount covers all samples
    return np.bincount(flat, weights=w, minlength=n * n_classes).reshape(n, n_classes).argmax(1)


class Bagging:
    """Train each base learner on its own bootstrap sample (with replacement, size n)."""

    def __init__(self, base_estimator_factory: Callable[[], object], n_estimators: int = 10,
                 seed: int | None = None) -> None:
        self.base_estimator_factory = base_estimator_factory
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Bagging":
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.n_classes_ = int(y.max()) + 1
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n)
            self.estimators_.append(self.base_estimator_factory().fit(X[idx], y[idx]))
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        preds = np.array([est.predict(X) for est in self.estimators_])
        out = majority_vote(preds, self.n_classes_)
        self.predict_time = time.perf_counter() - t0
        return out
