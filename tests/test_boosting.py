import math

import numpy as np

from boosting import AdaBoost
from data_utils import get_split
from tree import DecisionStump


def test_weights_sum_to_one_after_every_round():
    X_tr, _, y_tr, _ = get_split()
    for m in range(1, 16):
        b = AdaBoost(n_estimators=m).fit(X_tr, y_tr)
        assert math.isclose(b.sample_weight_.sum(), 1.0)
        assert (b.sample_weight_ > 0).all()


def test_first_alpha_matches_samme_formula():
    X_tr, _, y_tr, _ = get_split()
    b = AdaBoost(n_estimators=1, learning_rate=0.5).fit(X_tr, y_tr)
    err = (DecisionStump().fit(X_tr, y_tr).predict(X_tr) != y_tr).mean()
    assert math.isclose(b.alphas_[0], 0.5 * (math.log((1 - err) / err) + math.log(2)))


def test_perfect_learner_stops():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    b = AdaBoost(n_estimators=10).fit(X, y)
    assert len(b.estimators_) == 1 and (b.predict(X) == y).all()


def test_boosting_beats_single_stump():
    X_tr, X_te, y_tr, y_te = get_split()
    stump = (DecisionStump().fit(X_tr, y_tr).predict(X_te) == y_te).mean()
    boost = (AdaBoost(n_estimators=50).fit(X_tr, y_tr).predict(X_te) == y_te).mean()
    assert boost > stump and boost >= 0.85
