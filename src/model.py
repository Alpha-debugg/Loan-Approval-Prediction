"""
model.py
--------
Train/test split, feature scaling, and Logistic Regression training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
    "Total_Income", "Income_to_Loan_Ratio", "Loan_per_Term", "EMI_to_Income",
]
TARGET_COL = "Loan_Status"

# Only continuous / wide-range numeric features need scaling. Binary
# (0/1) columns are left as-is since scaling them adds no value and
# slightly hurts coefficient interpretability.
NUMERIC_TO_SCALE = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Total_Income", "Income_to_Loan_Ratio", "Loan_per_Term", "EMI_to_Income",
]


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """80/20 train-test split, stratified on the target to preserve class balance."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardize numeric features to zero mean / unit variance.

    Logistic Regression is fit by (regularized) gradient-based optimization
    and its coefficients are penalized by an L2 term of equal weight across
    features. Unscaled income (tens of thousands) and binary flags (0/1)
    live on wildly different scales: without scaling, the optimizer
    effectively shrinks the raw-income coefficients far more aggressively
    than binary-feature coefficients, and the resulting coefficient
    magnitudes are not comparable as "importance". Scaling also speeds
    and stabilizes convergence of the solver.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_TO_SCALE] = scaler.fit_transform(X_train[NUMERIC_TO_SCALE])
    X_test_scaled[NUMERIC_TO_SCALE] = scaler.transform(X_test[NUMERIC_TO_SCALE])

    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Hyperparameters (all explicit and explained):
        penalty='l2'        Ridge-style regularization; shrinks coefficients
                             to reduce overfitting without zeroing them out,
                             appropriate given we want every feature's
                             (small) contribution for interpretation.
        C=1.0                Inverse regularization strength; 1.0 is the
                             balanced default -- strong enough to control
                             variance on a dataset this size (~600 rows)
                             without over-penalizing genuine signal.
        solver='liblinear'  Efficient and robust for small/medium binary-
                             classification datasets, supports L2 penalty,
                             deterministic.
        max_iter=1000        Generous ceiling so the solver reliably
                             converges on scaled data.
        class_weight='balanced'
                             The dataset is imbalanced (~69% approved).
                             Balanced weighting re-weights the loss so the
                             minority (rejected) class is not ignored,
                             which matters for recall on rejections.
        random_state=42      Reproducibility.
    """
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
