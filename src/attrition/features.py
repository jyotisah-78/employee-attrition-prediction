"""Feature engineering. Every step here works row-by-row (no statistics learned
from the data), so it is safe to run BEFORE the train/test split."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger("attrition")


def assert_no_leakage(X: pd.DataFrame, cfg: Config) -> None:
    """Stop the program if any label-like column reached the features."""
    forbidden = set(cfg.label_cols) | {cfg.target_encoded}
    found = sorted(forbidden & set(X.columns))
    if found:
        raise ValueError(f"Target leakage! These label columns are in the features: {found}")


def engineer_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    y = df[cfg.target_encoded].astype(int)

    drop = [cfg.target_encoded, "data_source", *cfg.label_cols, *cfg.id_cols]
    X = df.drop(columns=[c for c in drop if c in df.columns])

    # booleans -> 0/1 numbers
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)

    # New features
    if "tenure_years" in X.columns:
        X["tenure_log"] = np.log1p(X["tenure_years"])
        X["tenure_squared"] = X["tenure_years"] ** 2
    if "monthly_income" in X.columns:
        X["income_log"] = np.log1p(X["monthly_income"])
    if {"satisfaction_score", "tenure_years"}.issubset(X.columns):
        X["satisfaction_tenure"] = X["satisfaction_score"] * X["tenure_years"]
    if {"OverTime", "DistanceFromHome"}.issubset(X.columns):
        X["commute_stress"] = X["DistanceFromHome"] * (X["OverTime"].astype(str) == "Yes").astype(int)
    if "Age" in X.columns:
        X["age_group"] = pd.cut(
            X["Age"], bins=[0, 25, 35, 45, 120], labels=["Young", "Early", "Mid", "Late"]
        )

    # Columns with a single value teach the model nothing
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant:
        logger.info("Dropping constant columns: %s", constant)
        X = X.drop(columns=constant)

    # Text columns -> plain object columns with real NaN for missing values
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            col_obj = X[col].astype(object)
            X[col] = col_obj.where(col_obj.notna(), np.nan)

    assert_no_leakage(X, cfg)
    logger.info("Feature matrix: %s | attrition rate %.1f%%", X.shape, 100 * y.mean())
    return X, y
