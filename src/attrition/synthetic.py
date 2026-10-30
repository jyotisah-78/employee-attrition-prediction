"""FAKE data generator with the same column names as the two Kaggle files.
Use it ONLY to test that the code runs. Results on it mean nothing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def make_ibm_like(n: int = 1470, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    over = r.choice(["Yes", "No"], n, p=[0.28, 0.72])
    sat = r.integers(1, 5, n)
    income = r.integers(1000, 20000, n)
    p = 0.08 + 0.15 * (over == "Yes") + 0.06 * (sat <= 2) - 0.04 * (income > 10000)
    left = r.random(n) < p
    return pd.DataFrame({
        "Age": r.integers(18, 60, n), "Attrition": np.where(left, "Yes", "No"),
        "BusinessTravel": r.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n),
        "DailyRate": r.integers(100, 1500, n),
        "Department": r.choice(["Sales", "Research & Development", "Human Resources"], n),
        "DistanceFromHome": r.integers(1, 30, n), "Education": r.integers(1, 6, n),
        "EducationField": r.choice(["Life Sciences", "Medical", "Marketing", "Other"], n),
        "EmployeeCount": 1, "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": r.integers(1, 5, n), "Gender": r.choice(["Male", "Female"], n),
        "HourlyRate": r.integers(30, 100, n), "JobInvolvement": r.integers(1, 5, n),
        "JobLevel": r.integers(1, 6, n),
        "JobRole": r.choice(["Sales Executive", "Research Scientist", "Laboratory Technician"], n),
        "JobSatisfaction": sat, "MaritalStatus": r.choice(["Single", "Married", "Divorced"], n),
        "MonthlyIncome": income, "MonthlyRate": r.integers(2000, 27000, n),
        "NumCompaniesWorked": r.integers(0, 10, n), "Over18": "Y", "OverTime": over,
        "PercentSalaryHike": r.integers(11, 26, n), "PerformanceRating": r.integers(3, 5, n),
        "RelationshipSatisfaction": r.integers(1, 5, n), "StandardHours": 80,
        "StockOptionLevel": r.integers(0, 4, n), "TotalWorkingYears": r.integers(0, 40, n),
        "TrainingTimesLastYear": r.integers(0, 7, n), "WorkLifeBalance": r.integers(1, 5, n),
        "YearsAtCompany": r.integers(0, 25, n), "YearsInCurrentRole": r.integers(0, 15, n),
        "YearsSinceLastPromotion": r.integers(0, 10, n), "YearsWithCurrManager": r.integers(0, 15, n),
    })


def make_employee_like(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    sat = r.random(n)
    hours = r.integers(100, 300, n)
    p = 0.05 + 0.5 * (sat < 0.3) + 0.15 * (hours > 250)
    return pd.DataFrame({
        "EmpId": np.arange(1, n + 1), "Satisfaction": sat.round(2),
        "Evaluation": r.random(n).round(2), "number_of_projects": r.integers(2, 8, n),
        "average_montly_hours": hours, "time_spent_company": r.integers(2, 10, n),
        "work_accident": r.integers(0, 2, n), "Promotion": r.integers(0, 2, n),
        "Salary_INR": r.integers(250000, 2500000, n),
        "Churn": (r.random(n) < p).astype(int),
    })


def write_synthetic_files(cfg: Config, n_ibm: int = 1470, n_emp: int = 3000) -> tuple:
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    a, b = Path(cfg.data_dir) / cfg.ibm_filename, Path(cfg.data_dir) / cfg.emp_filename
    make_ibm_like(n_ibm).to_csv(a, index=False)
    make_employee_like(n_emp).to_csv(b, index=False)
    return a, b
