"""The full experiment: load -> features -> split -> train/evaluate -> plots -> save."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import plots
from .config import Config
from .data import load_dataset
from .evaluation import compute_metrics, cross_validate_pipeline, majority_baseline_accuracy
from .features import engineer_features
from .modeling import (IMBLEARN_AVAILABLE, build_pipeline, get_model_factories,
                       should_use_smote)

logger = logging.getLogger("attrition")


def run_pipeline(cfg: Config, mode: str = "merged", model_names: Optional[List[str]] = None,
                 run_cv: bool = True) -> Dict[str, object]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_smote and not IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn is not installed: running WITHOUT SMOTE.")

    df = load_dataset(cfg, mode)
    X, y = engineer_features(df, cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state)
    logger.info("Train %s | Test %s", X_train.shape, X_test.shape)

    baseline = majority_baseline_accuracy(y_test)
    factories = get_model_factories(cfg)
    if model_names:
        unknown = [m for m in model_names if m not in factories]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. Available: {list(factories)}")
        factories = {m: factories[m] for m in model_names}
    logger.info("Models to train: %s", list(factories))

    results: Dict[str, dict] = {}
    for name, make in factories.items():
        try:
            use_smote = should_use_smote(name, y_train, cfg)
            logger.info("Training %s (SMOTE=%s)...", name, use_smote)
            pipe = build_pipeline(make(), X_train, use_smote, cfg)
            cv = cross_validate_pipeline(pipe, X_train, y_train, cfg) if run_cv else None
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = pipe.predict(X_test)
            test = compute_metrics(y_test, pred, proba)
            results[name] = {"pipeline": pipe, "test": test, "cv": cv, "proba": proba,
                             "pred": pred, "smote": use_smote}
            logger.info("%-12s | test acc %.4f | AUC %.4f | recall(left) %.4f", name,
                        test["accuracy"], test["roc_auc"], test["recall"])
        except Exception as exc:  # one broken model should not kill the whole run
            logger.warning("%s failed and was skipped: %s", name, exc)

    if not results:
        raise RuntimeError("No model trained successfully.")

    # Choose the best model WITHOUT looking at the test set (CV on training data only).
    if run_cv:
        best = max(results, key=lambda n: results[n]["cv"]["cv_roc_auc_mean"])
        basis = "cross-validation ROC-AUC on the training set"
    else:
        best = max(results, key=lambda n: results[n]["test"]["roc_auc"])
        basis = "test ROC-AUC (no CV: selection is slightly optimistic)"

    rows = []
    for name, r in results.items():
        row = {"model": name, "smote": r["smote"], **{k: v for k, v in r["test"].items()}}
        if r["cv"]:
            row.update({k: v for k, v in r["cv"].items() if not k.startswith("fold_")})
        row["is_best"] = name == best
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    prefix = cfg.output_dir / mode
    table.to_csv(f"{prefix}_results.csv", index=False)
    plots.plot_model_comparison(table, baseline, cfg.accuracy_target, cfg.output_dir / f"{mode}_model_comparison.png")
    plots.plot_roc_curves(results, y_test, cfg.output_dir / f"{mode}_roc_curves.png")
    plots.plot_confusion_matrix(y_test, results[best]["pred"], f"Confusion matrix - {best}",
                                cfg.output_dir / f"{mode}_confusion_matrix.png")
    if run_cv:
        plots.plot_cv_folds(results[best]["cv"], best, cfg.output_dir / f"{mode}_cv_folds.png")
    joblib.dump(results[best]["pipeline"], cfg.model_dir / f"{mode}_best_model.joblib")

    summary = {
        "mode": mode, "rows": int(len(X)), "features": int(X.shape[1]),
        "attrition_rate": float(y.mean()), "baseline_accuracy": baseline,
        "best_model": best, "selected_by": basis,
        "best_test_metrics": results[best]["test"],
        "best_cv": {k: v for k, v in (results[best]["cv"] or {}).items() if not k.startswith("fold_")},
        "target_met": bool(results[best]["test"]["accuracy"] >= cfg.accuracy_target),
    }
    with open(f"{prefix}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print_summary(summary, table, cfg)
    return {"summary": summary, "table": table, "results": results}


def print_summary(summary: dict, table: pd.DataFrame, cfg: Config) -> None:
    show = ["model", "accuracy", "balanced_accuracy", "recall", "f1", "roc_auc"]
    if "cv_roc_auc_mean" in table.columns:
        show.append("cv_roc_auc_mean")
    print("\n" + "=" * 70)
    print(f"PIPELINE SUMMARY  (mode = {summary['mode']})")
    print("=" * 70)
    print(table[show].round(4).to_string(index=False))
    print("-" * 70)
    m = summary["best_test_metrics"]
    print(f"Rows x features        : {summary['rows']:,} x {summary['features']}")
    print(f"Attrition rate         : {summary['attrition_rate']*100:.1f}%")
    print(f"'Always stays' baseline: {summary['baseline_accuracy']:.4f}  <- any model must beat this")
    print(f"Best model             : {summary['best_model']}  (chosen by {summary['selected_by']})")
    print(f"Test accuracy / AUC    : {m['accuracy']:.4f} / {m['roc_auc']:.4f}")
    print(f"Recall on leavers      : {m['recall']:.4f}")
    if summary["best_cv"]:
        c = summary["best_cv"]
        print(f"CV accuracy            : {c['cv_accuracy_mean']:.4f} +/- {c['cv_accuracy_std']:.4f}")
    gain = m["accuracy"] - summary["baseline_accuracy"]
    print("Accuracy target (%.0f%%)  : %s" % (cfg.accuracy_target * 100,
                                          "reached" if summary["target_met"] else "NOT reached"))
    print(f"Gain over baseline     : {gain*100:+.1f} percentage points")
    if m["accuracy"] > 0.97:
        print("WARNING: accuracy above 97% is unusual for attrition data. Re-check for leakage\n"
              "         and compare with --mode ibm (the IBM data is the harder, more realistic one).")
    print("=" * 70)
