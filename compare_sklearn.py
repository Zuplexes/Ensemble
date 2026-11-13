"""Sanity check: hand-written Tree / Bagging / AdaBoost vs scikit-learn on the same split."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from bagging import Bagging
from boosting import AdaBoost
from data_utils import get_split, scale
from metrics import time_model
from tree import DecisionStump, DecisionTree

N_RUNS = 10


def run_comparison(n_estimators: int = 50) -> pd.DataFrame:
    X_tr, X_te, y_tr, y_te = get_split()
    X_tr, X_te = scale(X_tr, X_te)
    n = n_estimators
    pairs = {
        "Decision Tree": (lambda s: DecisionTree(),
                          lambda s: DecisionTreeClassifier(random_state=s)),
        f"Bagging (n={n})": (lambda s: Bagging(DecisionTree, n, seed=s),
                             lambda s: BaggingClassifier(DecisionTreeClassifier(), n_estimators=n, random_state=s)),
        f"AdaBoost SAMME stump (n={n}, lr=1.0)": (
            lambda s: AdaBoost(DecisionStump, n, 1.0),
            lambda s: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n, learning_rate=1.0,
                                         random_state=s)),
    }
    rows = []
    for name, (mine, theirs) in pairs.items():
        for impl, make in (("ours", mine), ("sklearn", theirs)):
            t = time_model(make, X_tr, y_tr, X_te, y_te, N_RUNS)
            rows.append({"model": name, "impl": impl, "test_acc_mean": t["acc_mean"], "test_acc_std": t["acc_std"],
                         "fit_ms": 1e3 * t["fit_mean"], "predict_ms": 1e3 * t["pred_mean"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_comparison().round(4).to_string(index=False))
