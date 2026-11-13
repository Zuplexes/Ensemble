"""Multi-class AdaBoost (SAMME). numpy + stdlib only."""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from bagging import majority_vote
from tree import DecisionStump

EPS = 1e-10


class AdaBoost:
    """SAMME: alpha = lr * (ln((1-err)/err) + ln(K-1)); misclassified weights *= exp(alpha)."""

    def __init__(self, base_estimator_factory: Callable[[], object] = DecisionStump, n_estimators: int = 50,
                 learning_rate: float = 1.0) -> None:
        self.base_estimator_factory = base_estimator_factory
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        t0 = time.perf_counter()
        n = len(y)
        K = self.n_classes_ = int(y.max()) + 1
        w = np.full(n, 1.0 / n)
        self.estimators_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            est = self.base_estimator_factory().fit(X, y, sample_weight=w)
            miss = est.predict(X) != y
            err = float(w[miss].sum())
            if err >= 1 - 1 / K:  # no better than random guessing -> stop
                break
            perfect = err <= 0
            err = max(err, EPS)  # perfect learner: finite but dominant alpha
            alpha = self.learning_rate * (math.log((1 - err) / err) + math.log(K - 1))
            self.estimators_.append(est)
            self.alphas_.append(alpha)
            if perfect:  # weights would not change any more
                break
            w = w * np.exp(alpha * miss)
            w /= w.sum()
        if not self.estimators_:
            raise ValueError("first base learner is no better than random guessing")
        self.sample_weight_ = w
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        preds = np.array([est.predict(X) for est in self.estimators_])
        out = majority_vote(preds, self.n_classes_, weights=np.array(self.alphas_))
        self.predict_time = time.perf_counter() - t0
        return out
