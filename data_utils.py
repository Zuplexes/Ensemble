"""Data loading, fixed stratified 80/20 split, scaling and k-fold helpers."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "Iris.csv"
SPLIT_PATH = DATA_DIR / "split_indices.npz"
CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
SEED = 42


def load_iris(path: str | Path = CSV_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return X (150x4 float64) and y (0/1/2); the Id column is dropped."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in ("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")]
                  for r in rows])
    y = np.array([CLASSES.index(r["Species"]) for r in rows])
    return X, y


def get_split(seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 split, made once and reloaded from data/split_indices.npz."""
    X, y = load_iris()
    if SPLIT_PATH.exists():
        idx = np.load(SPLIT_PATH)
        tr, te = idx["train"], idx["test"]
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        np.savez(SPLIT_PATH, train=tr, test=te)
    return X[tr], X[te], y[tr], y[te]


def scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using statistics from the train set only."""
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)


def kfold_indices(y: np.ndarray, k: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold (train_idx, val_idx) pairs."""
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
