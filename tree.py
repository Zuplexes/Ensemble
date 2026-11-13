"""Base learner ที่ใช้ร่วมกันทั้ง Bagging และ Boosting (Decision Tree + Gini)

ปรับจากเดิม: หาจุดแบ่งด้วยการ sort + cumulative count (เร็วกว่าเดิมมาก)
เพื่อให้ทดลองกับข้อมูลหลักหมื่นแถวได้ โดยผลลัพธ์ตรรกะเหมือนเดิม
"""
import numpy as np


class SimpleTree:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.n_classes_ = None
        self.tree_ = None

    def fit(self, X, y, n_classes=None):
        y = np.asarray(y, dtype=int)
        self.n_classes_ = n_classes if n_classes is not None else int(y.max()) + 1
        self.tree_ = self._build(X, y, depth=0)
        return self

    # ---------- สร้างต้นไม้ ----------
    def _leaf(self, y):
        return {"value": int(np.bincount(y, minlength=self.n_classes_).argmax())}

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self._leaf(y)
        split = self._best_split(X, y)
        if split is None:
            return self._leaf(y)
        j, thr = split
        mask = X[:, j] <= thr
        return {
            "feature": j,
            "threshold": thr,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def _best_split(self, X, y):
        n, d = X.shape
        onehot = np.eye(self.n_classes_)[y]
        total = onehot.sum(axis=0)
        n_left = np.arange(1, n)
        n_right = n - n_left
        best_score, best = np.inf, None

        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            valid = xs[:-1] != xs[1:]          # แบ่งได้เฉพาะตรงที่ค่าเปลี่ยน
            if not valid.any():
                continue
            left = np.cumsum(onehot[order], axis=0)[:-1]
            right = total - left
            gini_l = 1.0 - np.sum((left / n_left[:, None]) ** 2, axis=1)
            gini_r = 1.0 - np.sum((right / n_right[:, None]) ** 2, axis=1)
            score = (n_left * gini_l + n_right * gini_r) / n   # weighted Gini
            score[~valid] = np.inf
            i = int(np.argmin(score))
            if score[i] < best_score:
                best_score, best = score[i], (j, (xs[i] + xs[i + 1]) / 2.0)
        return best

    # ---------- ทำนาย ----------
    def predict(self, X):
        out = np.empty(len(X), dtype=int)
        self._predict_node(self.tree_, X, np.arange(len(X)), out)
        return out

    def _predict_node(self, node, X, idx, out):
        if "value" in node:
            out[idx] = node["value"]
            return
        mask = X[idx, node["feature"]] <= node["threshold"]
        self._predict_node(node["left"], X, idx[mask], out)
        self._predict_node(node["right"], X, idx[~mask], out)