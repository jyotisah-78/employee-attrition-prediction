"""Metrics: test-set scores, cross-validation and the 'always say stays' baseline."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import Config


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


def majority_baseline_accuracy(y) -> float:
    """Accuracy of a 'model' that always predicts the most common class."""
    share = float(np.mean(y))
    return max(share, 1 - share)


def cross_validate_pipeline(pipeline, X, y, cfg: Config) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1",
               "balanced_accuracy": "balanced_accuracy"}
    s = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=cfg.n_jobs)
    out: Dict[str, object] = {}
    for key in scoring:
        out[f"cv_{key}_mean"] = float(s[f"test_{key}"].mean())
        out[f"cv_{key}_std"] = float(s[f"test_{key}"].std())
    out["fold_accuracy"] = s["test_accuracy"]
    out["fold_roc_auc"] = s["test_roc_auc"]
    return out
