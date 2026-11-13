import numpy as np

from data_utils import get_split
from tree import DecisionStump, DecisionTree


def test_stump_known_threshold():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    s = DecisionStump().fit(X, y)
    assert s.feature[0] == 0 and s.threshold[0] == 6.5
    assert s.predict(X).tolist() == y.tolist()


def test_picks_informative_feature():
    X = np.array([[5, 0], [1, 0], [4, 1], [2, 1]], dtype=float)  # only feature 1 separates
    y = np.array([0, 0, 1, 1])
    assert DecisionStump().fit(X, y).feature[0] == 1


def test_xor_needs_depth_2():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    assert (DecisionTree().fit(X, y).predict(X) == y).all()


def test_sample_weight_changes_leaf():
    X = np.zeros((3, 1))  # no split possible -> single leaf = weighted majority
    y = np.array([0, 0, 1])
    assert DecisionTree().fit(X, y).predict(X[:1])[0] == 0
    assert DecisionTree().fit(X, y, sample_weight=np.array([1, 1, 5.0])).predict(X[:1])[0] == 1


def test_max_depth_and_determinism():
    X_tr, X_te, y_tr, _ = get_split()
    t = DecisionTree(max_depth=2).fit(X_tr, y_tr)
    assert len(t.feature) <= 7
    assert np.array_equal(t.predict(X_te), DecisionTree(max_depth=2).fit(X_tr, y_tr).predict(X_te))
    assert (DecisionTree().fit(X_tr, y_tr).predict(X_tr) == y_tr).mean() == 1.0
