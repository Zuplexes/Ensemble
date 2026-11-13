"""Evaluation, cross-validation and timing helpers."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data_utils import kfold_indices


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro precision/recall/F1 and confusion matrix."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}


def cv_accuracy(make_model: Callable[[], object], X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Mean stratified k-fold accuracy on the given (train) data."""
    scores = []
    for tr, va in kfold_indices(y, k):
        m = make_model()
        m.fit(X[tr], y[tr])
        scores.append(np.mean(m.predict(X[va]) == y[va]))
    return float(np.mean(scores))


def time_model(make_model: Callable[[int], object], X_tr: np.ndarray, y_tr: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray, n_runs: int = 10) -> dict:
    """Fit/predict n_runs times (seed = run index); mean/std of times (s) and test accuracy."""
    fit_t, pred_t, acc = [], [], []
    for seed in range(n_runs):
        m = make_model(seed)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        t1 = time.perf_counter()
        pred = m.predict(X_te)
        t2 = time.perf_counter()
        fit_t.append(t1 - t0)
        pred_t.append(t2 - t1)
        acc.append(np.mean(pred == y_te))
    return {"fit_mean": float(np.mean(fit_t)), "fit_std": float(np.std(fit_t)),
            "pred_mean": float(np.mean(pred_t)), "pred_std": float(np.std(pred_t)),
            "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc))}
