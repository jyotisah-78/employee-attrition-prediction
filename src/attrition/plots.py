"""Charts. Every number plotted comes from the real run, never typed in by hand."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


def plot_model_comparison(table, baseline: float, target: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Model comparison (hold-out test set)", fontsize=14, fontweight="bold")
    for ax, metric in zip(axes.ravel(), ["accuracy", "roc_auc", "f1", "log_loss"]):
        d = table.sort_values(metric, ascending=(metric == "log_loss"))
        bars = ax.barh(d["model"], d[metric], color=sns.color_palette("muted", len(d)))
        ax.set_title(metric)
        for b, v in zip(bars, d[metric]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.3f}", va="center", fontsize=9)
        if metric == "accuracy":
            ax.axvline(baseline, color="grey", ls=":", label=f"always-'stays' baseline {baseline:.2f}")
            ax.axvline(target, color="red", ls="--", label=f"target {target:.2f}")
            ax.set_xlim(max(0.0, baseline - 0.15), 1.02)  # zoom in so small gaps are visible
            ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_roc_curves(results: dict, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_cv_folds(cv: dict, title: str, path: Path) -> None:
    folds = [f"Fold {i + 1}" for i in range(len(cv["fold_accuracy"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, mean_key, color in [(axes[0], "fold_accuracy", "cv_accuracy_mean", "steelblue"),
                                     (axes[1], "fold_roc_auc", "cv_roc_auc_mean", "coral")]:
        ax.bar(folds, cv[key], color=color)
        ax.axhline(cv[mean_key], color="red", ls="--", label=f"mean {cv[mean_key]:.3f}")
        ax.set_ylim(max(0, float(np.min(cv[key])) - 0.05), 1.0)
        ax.set_title(f"{title}: {key.replace('fold_', '')} per fold")
        ax.legend()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
