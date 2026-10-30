"""All settings live here, so you only change numbers in one place."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# .../employee-attrition-prediction/  (two levels above this file's folder)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Folders
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    output_dir: Path = PROJECT_ROOT / "results"
    model_dir: Path = PROJECT_ROOT / "models"

    # Kaggle datasets and the CSV file inside each one
    primary_dataset: str = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
    secondary_dataset: str = "prishatank/employee-hr-dataset"
    ibm_filename: str = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    emp_filename: str = "Employee_HR.csv"

    # Target handling
    target_encoded: str = "Target"  # our clean 0/1 column (1 = employee left)
    # Original label columns. These must NEVER be used as features.
    label_cols: tuple = ("Attrition", "Churn", "left")
    # ID / constant columns that carry no real signal
    id_cols: tuple = ("EmployeeNumber", "EmployeeCount", "EmpId", "Over18", "StandardHours")

    # Unit conversion for the second dataset (assumes Salary_INR is per YEAR)
    inr_per_usd: float = 75.0

    # Experiment settings
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = 1  # CPU cores for cross-validation (-1 = all cores)

    # Model settings
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6

    # Class imbalance handling
    use_smote: bool = True
    smote_threshold: float = 0.30  # apply SMOTE only if minority share is below this

    # Claim to test (note: "always predict stays" already scores ~77%)
    accuracy_target: float = 0.80
