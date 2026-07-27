"""
evaluation.py
-------------
Model evaluation, confusion matrix / ROC plotting, and Logistic Regression
coefficient-based feature importance & odds-ratio analysis.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

sns.set_theme(style="whitegrid")


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute and print all core classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<12}: {v:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return metrics


def plot_confusion_matrix(model, X_test, y_test, out_path: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Loan Approval Prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, out_path: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Loan Approval Prediction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def feature_importance_table(model: LogisticRegression, feature_names: list[str]) -> pd.DataFrame:
    """
    Build a ranked feature-importance table from Logistic Regression
    coefficients, including odds ratios (exp(coef)).

    Interpretation:
        coefficient > 0  -> increases the log-odds of approval (Y)
        coefficient < 0  -> decreases the log-odds of approval (favors rejection)
        odds_ratio > 1   -> each 1-unit (scaled) increase multiplies the
                             odds of approval by odds_ratio
        odds_ratio < 1   -> each 1-unit (scaled) increase multiplies the
                             odds of approval by odds_ratio (i.e. shrinks them)
    """
    coefs = model.coef_[0]
    table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table["direction"] = np.where(table["coefficient"] > 0, "Favors Approval", "Favors Rejection")
    return table[["feature", "coefficient", "odds_ratio", "direction", "abs_coefficient"]]


def plot_feature_importance(importance_df: pd.DataFrame, out_path: str, top_n: int = 15):
    df = importance_df.head(top_n).sort_values("coefficient")
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (scaled features)")
    ax.set_title("Feature Importance — Green = Favors Approval, Red = Favors Rejection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
