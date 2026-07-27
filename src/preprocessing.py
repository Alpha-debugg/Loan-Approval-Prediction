"""
preprocessing.py
-----------------
Data loading and cleaning utilities for the Loan Approval Prediction project.

Responsibilities:
    * Load the raw CSV file(s).
    * Report basic dataset diagnostics (shape, dtypes, missing values).
    * Handle missing values (mode for categorical, median for numerical).
    * Remove duplicate rows.
    * Flag outliers using the IQR method (for reporting; we do not drop
      them, since extreme incomes/loan amounts are real and informative
      for a credit-risk model).
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
NUMERICAL_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def dataset_overview(df: pd.DataFrame) -> None:
    """Print shape, dtypes, head, and descriptive statistics."""
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
    print("\nDescriptive statistics:\n", df.describe(include="all"))


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column, descending."""
    missing = df.isnull().sum().sort_values(ascending=False)
    return missing[missing > 0]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    Categorical columns -> mode (most frequent category). This is the
    standard, low-bias choice for nominal data with a small number of
    categories, and it does not distort the class balance the way a
    constant placeholder would.

    Numerical columns -> median. Income and loan-amount fields are right
    skewed (a handful of very high earners), so the median is a more
    robust central-tendency estimate than the mean, which would be pulled
    upward by outliers.
    """
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate row(s). New shape: {df.shape}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Report the number of outliers per numeric column using the 1.5*IQR rule.
    Returns a small summary DataFrame; does not modify or drop any rows,
    since in a credit-risk context genuinely high incomes/loan amounts are
    legitimate signal rather than data-entry errors.
    """
    columns = columns or ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({"column": col, "lower_bound": lower, "upper_bound": upper,
                      "n_outliers": n_outliers, "pct_outliers": round(100 * n_outliers / len(df), 2)})
    return pd.DataFrame(rows)


def clean_pipeline(path: str) -> pd.DataFrame:
    """Full load -> report -> clean pipeline used by main.py."""
    df = load_data(path)
    print("=" * 70)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 70)
    dataset_overview(df)

    print("\n" + "=" * 70)
    print("STEP 3: DATA CLEANING")
    print("=" * 70)
    print("\nMissing values before cleaning:\n", missing_value_report(df))

    df = remove_duplicates(df)
    df = handle_missing_values(df)

    print("\nMissing values after cleaning:\n",
          "None remaining" if df.isnull().sum().sum() == 0 else missing_value_report(df))

    print("\nOutlier report (IQR method, informational only):\n",
          detect_outliers_iqr(df))

    return df
