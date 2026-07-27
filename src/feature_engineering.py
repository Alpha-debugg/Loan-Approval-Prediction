"""
feature_engineering.py
-----------------------
Creates derived features and encodes categorical variables for the
Loan Approval Prediction model.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_MAPS = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    # Property_Area has no natural order, but Semiurban shows the
    # highest historical approval rate, Rural the lowest -> encode
    # ordinally in that observed order so the single numeric column
    # still carries useful monotonic signal for a linear model.
    "Property_Area": {"Rural": 0, "Semiurban": 2, "Urban": 1},
    "Loan_Status": {"Y": 1, "N": 0},
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features:

    Total_Income = ApplicantIncome + CoapplicantIncome
        A single household-income figure is more predictive than the two
        separate income columns, since lenders assess repayment capacity
        against combined household earnings.

    Income_to_Loan_Ratio = Total_Income / (LoanAmount * 1000)
        Captures affordability directly: how many times over could the
        household's income cover the loan. This is closer to what a real
        underwriter evaluates than either raw figure alone.

    Loan_per_Term = LoanAmount / Loan_Amount_Term
        Approximates the monthly repayment burden (loan amount is in
        thousands, term is in months), which relates directly to
        default risk.

    EMI_to_Income = Loan_per_Term / (Total_Income / 1000 + 1)
        A composite affordability index: repayment burden relative to
        income. Small denominators are guarded with +1 to avoid
        division blow-ups for zero-income edge cases.
    """
    df = df.copy()
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_to_Loan_Ratio"] = df["Total_Income"] / (df["LoanAmount"] * 1000 + 1)
    df["Loan_per_Term"] = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["EMI_to_Income"] = df["Loan_per_Term"] / (df["Total_Income"] / 1000 + 1)
    return df


def encode_categoricals(df: pd.DataFrame, fit_encoders: bool = True,
                         encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.

    We use explicit, documented mappings (LABEL_MAPS) rather than a bare
    sklearn LabelEncoder wherever the categories have an obvious binary or
    ordinal meaning (Yes/No, Graduate/Not Graduate, Dependents count,
    Property_Area's observed approval ordering). This keeps the encoding
    interpretable: a coefficient on "Education" unambiguously means
    Graduate=1 vs Not Graduate=0, which matters a great deal when we later
    read Logistic Regression coefficients as business explanations.
    """
    df = df.copy()
    encoders = encoders or {}

    for col, mapping in LABEL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df, encoders
