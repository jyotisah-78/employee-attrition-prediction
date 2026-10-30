"""Loading the two datasets and aligning them to one shared schema."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger("attrition")

VALID_MODES = ("merged", "ibm", "employee")


def find_csv(data_dir: Path, filename: str) -> Path:
    """Find a CSV in data_dir (also searches sub-folders)."""
    data_dir = Path(data_dir)
    direct = data_dir / filename
    if direct.exists():
        return direct
    matches = sorted(data_dir.rglob(filename)) if data_dir.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find '{filename}' inside '{data_dir}'.\n"
        "Fix: run  python scripts/download_data.py  (needs Kaggle login),\n"
        "or download the CSV manually from Kaggle and put it in that folder,\n"
        "or create fake demo data with  python scripts/make_synthetic_data.py"
    )


def to_binary(series: pd.Series) -> pd.Series:
    """Turn Yes/No, True/False, 1/0 into integers 1/0. Fails loudly on unknown values."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series) > 0).astype(int)
    mapping = {"yes": 1, "true": 1, "1": 1, "left": 1, "no": 0, "false": 0, "0": 0, "stayed": 0}
    mapped = series.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())[:5]
        raise ValueError(f"Cannot convert label values to 0/1: {bad}")
    return mapped.astype(int)


def rescale_satisfaction(s: pd.Series) -> pd.Series:
    """Bring a satisfaction column onto the IBM 1-4 scale.

    ASSUMPTION: the scale is guessed from the maximum value (<=1, <=4, <=10, <=100).
    Check your real CSV and adjust here if the guess is wrong.
    """
    s = pd.to_numeric(s, errors="coerce")
    mx = s.max()
    if pd.isna(mx) or mx <= 4 and mx > 1:
        return s  # already on a 1-4 style scale
    if mx <= 1:
        top = 1.0
    elif mx <= 10:
        top = 10.0
    elif mx <= 100:
        top = 100.0
    else:
        return s
    return (1 + 3 * s / top).clip(1, 4)


def load_ibm(path: Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Attrition" not in df.columns:
        raise ValueError("IBM file has no 'Attrition' column.")
    df[cfg.target_encoded] = to_binary(df["Attrition"])
    df = df.drop(columns=[c for c in cfg.label_cols if c in df.columns])
    df = df.rename(columns={
        "JobSatisfaction": "satisfaction_score",
        "MonthlyIncome": "monthly_income",
        "YearsAtCompany": "tenure_years",
        "Department": "department",
    })
    df["data_source"] = "ibm"
    logger.info("IBM dataset loaded: %s", df.shape)
    return df


def load_employee(path: Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path)
    label = next((c for c in ("Churn", "left") if c in df.columns), None)
    if label is None:
        raise ValueError("Employee HR file needs a 'Churn' or 'left' column.")
    df[cfg.target_encoded] = to_binary(df[label])
    df = df.drop(columns=[c for c in cfg.label_cols if c in df.columns])

    df = df.rename(columns={
        "Satisfaction": "satisfaction_score",
        "satisfaction_level": "satisfaction_score",
        "time_spent_company": "tenure_years",
        "time_spend_company": "tenure_years",
        "Department": "department",
        "sales": "department",
    })

    if "Salary_INR" in df.columns:
        df["monthly_income"] = (df["Salary_INR"] / 12 / cfg.inr_per_usd).round(2)
        df = df.drop(columns=["Salary_INR"])
    elif "salary" in df.columns:
        df["monthly_income"] = df["salary"].astype(str).map(
            {"low": 3000, "medium": 5000, "high": 7000}).fillna(4500)
        df = df.drop(columns=["salary"])

    if "satisfaction_score" in df.columns:
        df["satisfaction_score"] = rescale_satisfaction(df["satisfaction_score"])

    df["data_source"] = "employee_hr"
    logger.info("Employee HR dataset loaded: %s", df.shape)
    return df


def load_dataset(cfg: Config, mode: str = "merged") -> pd.DataFrame:
    """mode: 'merged' (both), 'ibm' (IBM only) or 'employee' (HR only)."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'")
    frames = []
    if mode in ("merged", "ibm"):
        frames.append(load_ibm(find_csv(cfg.data_dir, cfg.ibm_filename), cfg))
    if mode in ("merged", "employee"):
        frames.append(load_employee(find_csv(cfg.data_dir, cfg.emp_filename), cfg))
    df = pd.concat(frames, axis=0, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
    logger.info("Dataset ready (mode=%s): %s | attrition rate %.1f%%",
                mode, df.shape, 100 * df[cfg.target_encoded].mean())
    return df
