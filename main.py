"""
main.py
-------
End-to-end pipeline for the Loan Approval Prediction project.

Run with:
    python main.py

This script:
    1. Loads and cleans the raw data (src/preprocessing.py)
    2. Runs the full EDA and saves plots to outputs/ (src/eda.py)
    3. Engineers features and encodes categoricals (src/feature_engineering.py)
    4. Splits, scales, and trains a Logistic Regression model (src/model.py)
    5. Evaluates the model and saves confusion-matrix / ROC plots (src/evaluation.py)
    6. Prints ranked feature importance with odds ratios and business insights
    7. Runs a sample prediction on a custom applicant
"""

import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import clean_pipeline
from eda import run_full_eda, eda_summary_text
from feature_engineering import engineer_features, encode_categoricals
from model import split_data, scale_features, train_logistic_regression, FEATURE_COLS, TARGET_COL
from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    feature_importance_table, plot_feature_importance,
)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # ---------- Steps 1 & 3: Load + Clean ----------
    df = clean_pipeline(DATA_PATH)

    # ---------- Step 2: EDA ----------
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    run_full_eda(df, OUT_DIR)
    print("\nEDA summary:", eda_summary_text(df))

    # ---------- Step 4: Feature Engineering ----------
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)
    df_fe = engineer_features(df)
    print(df_fe[["Total_Income", "Income_to_Loan_Ratio", "Loan_per_Term", "EMI_to_Income"]].describe())

    # ---------- Step 5: Encoding ----------
    print("\n" + "=" * 70)
    print("STEP 5: CATEGORICAL ENCODING")
    print("=" * 70)
    df_enc, _ = encode_categoricals(df_fe)
    print(df_enc[["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]].head())

    # ---------- Step 6: Train/Test Split ----------
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN / TEST SPLIT (80/20, random_state=42)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_data(df_enc)

    # ---------- Step 7: Scaling ----------
    print("\n" + "=" * 70)
    print("STEP 7: FEATURE SCALING")
    print("=" * 70)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("Numeric features standardized (mean=0, std=1) on train, applied to test.")

    # ---------- Step 8: Train Model ----------
    print("\n" + "=" * 70)
    print("STEP 8: TRAIN LOGISTIC REGRESSION")
    print("=" * 70)
    model = train_logistic_regression(X_train_scaled, y_train)
    print("Model trained. Hyperparameters:", model.get_params())

    # ---------- Step 9: Evaluation ----------
    metrics = evaluate_model(model, X_test_scaled, y_test)
    plot_confusion_matrix(model, X_test_scaled, y_test, f"{OUT_DIR}/confusion_matrix.png")
    plot_roc_curve(model, X_test_scaled, y_test, f"{OUT_DIR}/roc_curve.png")
    print(f"\nSaved confusion_matrix.png and roc_curve.png to {OUT_DIR}/")

    # ---------- Step 10: Feature Importance ----------
    print("\n" + "=" * 70)
    print("STEP 10: FEATURE IMPORTANCE (Coefficients & Odds Ratios)")
    print("=" * 70)
    importance_df = feature_importance_table(model, FEATURE_COLS)
    print(importance_df.to_string(index=False))
    plot_feature_importance(importance_df, f"{OUT_DIR}/feature_importance.png")

    top_approval = importance_df[importance_df["direction"] == "Favors Approval"].head(3)
    top_rejection = importance_df[importance_df["direction"] == "Favors Rejection"].head(3)
    print("\nTop factors favoring APPROVAL:\n", top_approval[["feature", "odds_ratio"]].to_string(index=False))
    print("\nTop factors favoring REJECTION:\n", top_rejection[["feature", "odds_ratio"]].to_string(index=False))

    # ---------- Step 11: Sample Prediction ----------
    print("\n" + "=" * 70)
    print("STEP 11: SAMPLE PREDICTION")
    print("=" * 70)
    sample = {
        "Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate",
        "Self_Employed": "No", "ApplicantIncome": 5000, "CoapplicantIncome": 1500,
        "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": 1.0,
        "Property_Area": "Urban",
    }
    result = predict_single(model, scaler, sample)
    print(json.dumps(result, indent=2))

    # ---------- Persist model + scaler ----------
    joblib.dump(model, f"{OUT_DIR}/logistic_regression_model.joblib")
    joblib.dump(scaler, f"{OUT_DIR}/scaler.joblib")
    importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved trained model, scaler, feature_importance.csv, and metrics.json to {OUT_DIR}/")

    print("\nPipeline complete.")


def predict_single(model, scaler, applicant: dict) -> dict:
    """Run the full feature-engineering + encoding + scaling pipeline on a
    single custom applicant dict and return the prediction + probability."""
    from model import NUMERIC_TO_SCALE

    raw = pd.DataFrame([applicant])
    fe = engineer_features(raw)
    enc, _ = encode_categoricals(fe)
    X = enc[FEATURE_COLS].copy()
    X[NUMERIC_TO_SCALE] = scaler.transform(X[NUMERIC_TO_SCALE])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1]

    return {
        "prediction": "Approved" if pred == 1 else "Rejected",
        "approval_probability": f"{proba * 100:.1f}%",
    }


if __name__ == "__main__":
    main()
