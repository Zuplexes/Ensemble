import numpy as np

from bagging import Bagging, majority_vote
from data_utils import get_split
from tree import DecisionTree


def test_majority_vote():
    preds = np.array([[0, 1, 2], [0, 2, 2], [1, 2, 0]])
    assert majority_vote(preds, 3).tolist() == [0, 2, 2]
    assert majority_vote(preds, 3, weights=np.array([1, 1, 5.0])).tolist() == [1, 2, 0]


def test_bagging_reproducible_and_timed():
    X_tr, X_te, y_tr, y_te = get_split()
    a = Bagging(DecisionTree, 10, seed=0).fit(X_tr, y_tr)
    b = Bagging(DecisionTree, 10, seed=0).fit(X_tr, y_tr)
    pa = a.predict(X_te)
    assert np.array_equal(pa, b.predict(X_te))
    assert len(a.estimators_) == 10 and a.fit_time > 0 and a.predict_time > 0
    assert (pa == y_te).mean() >= 0.85
