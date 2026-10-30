"""Model definitions and the preprocessing pipeline around them.

Everything that learns from data (imputing, scaling, encoding, SMOTE) lives INSIDE
a scikit-learn Pipeline, so it is fitted on training rows only. That prevents
test-set information from leaking into training.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import Config

logger = logging.getLogger("attrition")

# Optional libraries: the project still runs if some of them are not installed.
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
except Exception:  # ImportError, or a missing libomp on macOS
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    # numeric columns come first in the output, then categorical (used by SMOTENC below)
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def get_model_factories(cfg: Config) -> Dict[str, Callable]:
    """Return {model name: function that creates a fresh, untrained model}."""
    rs = cfg.random_state
    f: Dict[str, Callable] = {}

    f["RandomForest"] = lambda n=None: RandomForestClassifier(
        n_estimators=n or cfg.n_estimators, random_state=rs, n_jobs=1)
    f["HistGB"] = lambda n=None: HistGradientBoostingClassifier(
        max_iter=n or cfg.n_estimators, max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate, random_state=rs)
    if XGBClassifier is not None:
        f["XGBoost"] = lambda n=None: XGBClassifier(
            n_estimators=n or cfg.n_estimators, max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate, tree_method="hist", eval_metric="logloss",
            random_state=rs, verbosity=0, n_jobs=1)
    if LGBMClassifier is not None:
        f["LightGBM"] = lambda n=None: LGBMClassifier(
            n_estimators=n or cfg.n_estimators, max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate, random_state=rs, verbose=-1, n_jobs=1)
    if CatBoostClassifier is not None:
        f["CatBoost"] = lambda n=None: CatBoostClassifier(
            iterations=n or cfg.n_estimators, depth=cfg.max_depth,
            learning_rate=cfg.learning_rate, random_seed=rs, verbose=False,
            allow_writing_files=False, thread_count=1)

    base = [(name, make) for name, make in f.items()]
    f["Stacked"] = lambda n=None: StackingClassifier(
        estimators=[(name, make(100)) for name, make in base],
        final_estimator=LogisticRegression(max_iter=1000, random_state=rs),
        cv=3, n_jobs=1)
    return f


def should_use_smote(name: str, y_train, cfg: Config) -> bool:
    # Stacking runs its own inner cross-validation; SMOTE there would leak
    # synthetic copies between folds, so the stack is trained without it.
    if name == "Stacked" or not cfg.use_smote or not IMBLEARN_AVAILABLE:
        return False
    minority_share = float(np.mean(y_train))
    minority_share = min(minority_share, 1 - minority_share)
    return 0 < minority_share < cfg.smote_threshold


def build_pipeline(estimator, X_train: pd.DataFrame, use_smote: bool, cfg: Config) -> Pipeline:
    pre, num_cols, cat_cols = build_preprocessor(X_train)
    steps = [("preprocess", pre)]
    if use_smote:
        if cat_cols:
            start = len(num_cols)
            sampler = SMOTENC(categorical_features=list(range(start, start + len(cat_cols))),
                              random_state=cfg.random_state, k_neighbors=5)
        else:
            sampler = SMOTE(random_state=cfg.random_state, k_neighbors=5)
        steps += [("smote", sampler), ("model", estimator)]
        return ImbPipeline(steps)
    steps.append(("model", estimator))
    return Pipeline(steps)
