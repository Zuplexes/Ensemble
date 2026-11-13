"""Run every experiment: hyperparameters via 5-fold CV on train, one final test eval per model,
10-seed timing, tables (CSV/JSON/MD) and plots in results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bagging import Bagging
from boosting import AdaBoost
from compare_sklearn import run_comparison
from data_utils import get_split, scale
from metrics import cv_accuracy, evaluate, time_model
from tree import DecisionStump, DecisionTree

RESULTS = Path(__file__).parent / "results"
N_RUNS = 10
N_ESTIMATORS = [5, 10, 25, 50, 100]
LEARNING_RATES = [0.5, 1.0]
TREE_DEPTHS = [1, 2, 3, 4, 5, None]
CV_SEED = 0  # seed for the ensemble used during CV
COLORS = {"Bagging": "#2a78d6", "Boosting": "#eb6834"}
MODEL_COLORS = ["#1baf7a", "#2a78d6", "#eb6834"]  # Tree, Bagging, Boosting (fixed order)


def sweep(name: str, make, X_tr, y_tr, X_te, y_te) -> list[dict]:
    """CV acc + 10-seed timing/test acc for each n_estimators. make(n, seed) -> model."""
    rows = []
    for n in N_ESTIMATORS:
        cv = cv_accuracy(lambda: make(n, CV_SEED), X_tr, y_tr)
        t = time_model(lambda s: make(n, s), X_tr, y_tr, X_te, y_te, N_RUNS)
        rows.append({"model": name, "n_estimators": n, "cv_acc": cv, **t})
        print(f"  {name:16s} n={n:3d}  CV={cv:.4f}  test={t['acc_mean']:.4f}  fit={t['fit_mean']*1e3:.2f}ms")
    return rows


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean * 1e3:.3f} ± {std * 1e3:.3f} ms"


def plot_sweeps(df: pd.DataFrame, tree: dict) -> None:
    """Plot Bagging + Boosting at the learning rate chosen by CV (all lrs stay in the CSV).
    The single Decision Tree has no n_estimators, so it is a horizontal reference line.
    tree: label, cv_acc, acc_mean, fit_mean, pred_mean."""
    boost = df[df.model != "Bagging"]
    df = df[df.model.isin(["Bagging", boost.loc[boost.cv_acc.idxmax(), "model"]])]
    color = lambda name: COLORS[name.split()[0]]
    ref = {"color": "#8a8a85", "ls": "--", "lw": 1.5, "label": tree["label"]}
    fig, (ax_cv, ax_te) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for name, g in df.groupby("model", sort=False):
        ax_cv.plot(g.n_estimators, g.cv_acc, "-o", color=color(name), lw=2, ms=6, label=name)
        ax_te.errorbar(g.n_estimators, g.acc_mean, yerr=g.acc_std, fmt="-o", color=color(name), lw=2, ms=6,
                       capsize=3, label=name)
    ax_cv.axhline(tree["cv_acc"], **ref)
    ax_te.axhline(tree["acc_mean"], **ref)
    style(ax_cv, "Train: 5-fold CV accuracy (used for model selection)", "accuracy")
    style(ax_te, f"Test accuracy (mean ± std, {N_RUNS} seeds)", "accuracy")
    ax_te.tick_params(labelleft=True)
    fig.savefig(RESULTS / "accuracy_vs_n_estimators.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "fit", "Train time"), (axes[1], "pred", "Predict time")):
        for name, g in df.groupby("model", sort=False):
            ax.errorbar(g.n_estimators, g[f"{key}_mean"] * 1e3, yerr=g[f"{key}_std"] * 1e3, fmt="-o",
                        color=color(name), lw=2, ms=6, capsize=3, label=name)
        ax.axhline(tree[f"{key}_mean"] * 1e3, **ref)
        style(ax, f"{title} vs n_estimators (mean ± std, {N_RUNS} runs)", "milliseconds")
    fig.savefig(RESULTS / "time_vs_n_estimators.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_models(m: pd.DataFrame) -> None:
    """Final models side by side: accuracy (CV hollow, test filled) | train time | predict time."""
    y = list(range(len(m)))[::-1]  # first model on top
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True, gridspec_kw={"wspace": 0.15})
    ax = axes[0]
    for yi, r, c in zip(y, m.itertuples(), MODEL_COLORS):
        ax.plot([r.cv_acc, r.test_acc], [yi, yi], color="#c3c2b7", lw=2, zorder=1)
        ax.scatter(r.cv_acc, yi, s=80, facecolor="white", edgecolor=c, lw=2, zorder=2)
        ax.scatter(r.test_acc, yi, s=80, color=c, zorder=2)
        ax.annotate(f"CV {r.cv_acc:.3f}\ntest {r.test_acc:.3f}", (0.995, yi), ha="right", va="center", fontsize=8)
    ax.set(xlim=(0.9, 1.0), title="Accuracy  (○ CV on train,  ● test)", xlabel="accuracy")
    for ax, key, title in ((axes[1], "fit", "Train time"), (axes[2], "pred", "Predict time")):
        mean, std = m[f"{key}_mean"] * 1e3, m[f"{key}_std"] * 1e3
        ax.barh(y, mean, xerr=std, color=MODEL_COLORS, height=0.55, capsize=3, error_kw={"lw": 1})
        for yi, v, s in zip(y, mean, std):
            ax.text(v + s, yi, f"  {v:.3f} ms", va="center", fontsize=8)
        ax.set_xlim(0, (mean + std).max() * 1.35)
        ax.set(title=f"{title} (mean ± std, {N_RUNS} runs)", xlabel="milliseconds")
    axes[0].set_yticks(y, m.model)
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def style(ax, title: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xticks(N_ESTIMATORS, [str(n) for n in N_ESTIMATORS])
    ax.minorticks_off()
    ax.set(title=title, xlabel="n_estimators (log scale)", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    X_tr_raw, X_te_raw, y_tr, y_te = get_split()
    X_tr, X_te = scale(X_tr_raw, X_te_raw)  # scaler fit on train only

    print("Decision tree: choose max_depth by CV ...")
    tree_cv = {str(d): cv_accuracy(lambda: DecisionTree(max_depth=d), X_tr, y_tr) for d in TREE_DEPTHS}
    best_depth = TREE_DEPTHS[list(tree_cv.values()).index(max(tree_cv.values()))]  # ties -> shallowest
    tree_t = time_model(lambda s: DecisionTree(max_depth=best_depth), X_tr, y_tr, X_te, y_te, N_RUNS)

    print("Bagging / Boosting sweeps ...")
    rows = sweep("Bagging", lambda n, s: Bagging(DecisionTree, n, seed=s), X_tr, y_tr, X_te, y_te)
    for lr in LEARNING_RATES:
        rows += sweep(f"Boosting lr={lr}", lambda n, s, lr=lr: AdaBoost(DecisionStump, n, lr), X_tr, y_tr, X_te, y_te)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "ensemble_sweep.csv", index=False)

    # Model selection uses CV only; ties -> fewest estimators (rows are ordered by n ascending).
    bag = df[df.model == "Bagging"].loc[lambda d: d.cv_acc.idxmax()]
    boo = df[df.model != "Bagging"].loc[lambda d: d.cv_acc.idxmax()]
    best_lr = float(boo.model.split("=")[1])

    finals = {  # model -> (fitted on full train, seed 0) for the single test evaluation
        f"Decision Tree (max_depth={best_depth})": DecisionTree(max_depth=best_depth).fit(X_tr, y_tr),
        f"Bagging (n={bag.n_estimators})": Bagging(DecisionTree, int(bag.n_estimators), seed=0).fit(X_tr, y_tr),
        f"Boosting SAMME (n={boo.n_estimators}, lr={best_lr})":
            AdaBoost(DecisionStump, int(boo.n_estimators), best_lr).fit(X_tr, y_tr),
    }
    test_metrics = {name: evaluate(y_te, m.predict(X_te)) for name, m in finals.items()}

    final_rows = list(zip(finals, [tree_cv[str(best_depth)], bag.cv_acc, boo.cv_acc],
                          [tree_t, bag, boo]))
    summary = pd.DataFrame([
        {"Model": n, "CV Acc (train)": cv, "Test Acc": test_metrics[n]["accuracy"],
         "Test Acc (10 seeds)": f"{t['acc_mean']:.4f} ± {t['acc_std']:.4f}", "Macro F1": test_metrics[n]["f1"],
         "Train time (mean±std)": fmt_ms(t["fit_mean"], t["fit_std"]),
         "Predict time (mean±std)": fmt_ms(t["pred_mean"], t["pred_std"])}
        for n, cv, t in final_rows])
    models = pd.DataFrame([
        {"model": n, "cv_acc": cv, "test_acc": test_metrics[n]["accuracy"],
         **{k: float(t[k]) for k in ("fit_mean", "fit_std", "pred_mean", "pred_std")}}
        for n, cv, t in final_rows])
    models.to_csv(RESULTS / "model_comparison.csv", index=False)
    plot_models(models)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    md = summary.round(4).astype(str)
    (RESULTS / "summary.md").write_text(
        "| " + " | ".join(md.columns) + " |\n|" + "---|" * len(md.columns) + "\n"
        + "\n".join("| " + " | ".join(r) + " |" for r in md.values) + "\n", encoding="utf-8")

    cmp = run_comparison()
    cmp.to_csv(RESULTS / "sklearn_comparison.csv", index=False)

    json.dump({"tree_cv_by_depth": tree_cv, "test_metrics": test_metrics},
              open(RESULTS / "results.json", "w"), indent=2)
    plot_sweeps(df, {"label": f"Decision Tree (depth={best_depth})", "cv_acc": tree_cv[str(best_depth)], **tree_t})

    pd.set_option("display.width", 200)
    print("\n" + summary.round(4).to_string(index=False))
    print("\nsklearn comparison:\n" + cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
