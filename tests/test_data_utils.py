import numpy as np

from data_utils import get_split, kfold_indices, load_iris, scale


def test_load():
    X, y = load_iris()
    assert X.shape == (150, 4) and X.dtype == np.float64
    assert np.bincount(y).tolist() == [50, 50, 50]


def test_split_80_20_stratified_no_overlap():
    X_tr, X_te, y_tr, y_te = get_split()
    assert len(y_tr) == 120 and len(y_te) == 30
    assert np.bincount(y_tr).tolist() == [40, 40, 40]
    assert np.bincount(y_te).tolist() == [10, 10, 10]
    idx = np.load("data/split_indices.npz")
    assert not set(idx["train"]) & set(idx["test"])
    assert len(set(idx["train"]) | set(idx["test"])) == 150


def test_split_is_reproducible():
    a, b = get_split(), get_split()
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_scale_uses_train_only():
    X_tr, X_te, _, _ = get_split()
    s_tr, s_te = scale(X_tr, X_te)
    assert np.allclose(s_tr.mean(0), 0) and np.allclose(s_tr.std(0), 1)
    assert not np.allclose(s_te.mean(0), 0)


def test_kfold():
    _, _, y_tr, _ = get_split()
    folds = kfold_indices(y_tr)
    assert len(folds) == 5
    for tr, va in folds:
        assert len(va) == 24 and np.bincount(y_tr[va]).tolist() == [8, 8, 8]
