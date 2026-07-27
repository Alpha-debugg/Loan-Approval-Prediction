"""
eda.py
------
Exploratory Data Analysis plotting functions (Step 2 of the project spec).
Each function saves a figure to the outputs/ directory (or a custom path)
and returns nothing; main.py calls these in sequence.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def run_full_eda(df: pd.DataFrame, out_dir: str) -> None:
    """Generate the full suite of EDA plots requested in the project spec."""

    # 1. Loan approval distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Loan_Status", hue="Loan_Status", palette="Set2", legend=False, ax=ax)
    ax.set_title("Loan Approval Distribution")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_loan_status_distribution.png", dpi=130); plt.close(fig)

    # 2. Income distribution (Applicant)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["ApplicantIncome"], kde=True, color="#2563eb", ax=ax)
    ax.set_title("Applicant Income Distribution")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_applicant_income_distribution.png", dpi=130); plt.close(fig)

    # 3. Loan amount distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["LoanAmount"].dropna(), kde=True, color="#16a34a", ax=ax)
    ax.set_title("Loan Amount Distribution")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_loan_amount_distribution.png", dpi=130); plt.close(fig)

    # 4. Boxplots for outlier inspection
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.boxplot(y=df["ApplicantIncome"], ax=axes[0], color="#93c5fd")
    axes[0].set_title("ApplicantIncome")
    sns.boxplot(y=df["CoapplicantIncome"], ax=axes[1], color="#fca5a5")
    axes[1].set_title("CoapplicantIncome")
    sns.boxplot(y=df["LoanAmount"], ax=axes[2], color="#86efac")
    axes[2].set_title("LoanAmount")
    fig.suptitle("Boxplots — Outlier Inspection")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_boxplots.png", dpi=130); plt.close(fig)

    # 5. Correlation heatmap (numeric columns only)
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_correlation_heatmap.png", dpi=130); plt.close(fig)

    # 6-13. Categorical vs Loan_Status count plots, grouped into one grid
    cat_pairs = [
        ("Education", "Education vs Loan Status"),
        ("Gender", "Gender vs Loan Status"),
        ("Property_Area", "Property Area vs Loan Status"),
        ("Credit_History", "Credit History vs Loan Status"),
        ("Dependents", "Dependents vs Loan Status"),
        ("Married", "Married vs Loan Status"),
        ("Self_Employed", "Self Employed vs Loan Status"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, (col, title) in enumerate(cat_pairs):
        sns.countplot(data=df, x=col, hue="Loan_Status", palette="Set2", ax=axes[i])
        axes[i].set_title(title)
        axes[i].tick_params(axis="x", rotation=20)
    for j in range(len(cat_pairs), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_categorical_vs_status.png", dpi=130); plt.close(fig)

    # 14. Applicant Income vs Loan Status (boxplot)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Loan_Status", y="ApplicantIncome", hue="Loan_Status",
                palette="Set2", legend=False, ax=ax)
    ax.set_title("Applicant Income vs Loan Status")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_income_vs_status.png", dpi=130); plt.close(fig)

    # 15. Coapplicant Income vs Loan Status (boxplot)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Loan_Status", y="CoapplicantIncome", hue="Loan_Status",
                palette="Set2", legend=False, ax=ax)
    ax.set_title("Coapplicant Income vs Loan Status")
    fig.tight_layout(); fig.savefig(f"{out_dir}/eda_coapplicant_income_vs_status.png", dpi=130); plt.close(fig)

    print(f"Saved EDA plots to {out_dir}/")


def eda_summary_text(df: pd.DataFrame) -> str:
    """Return a short, human-readable narrative describing the EDA findings."""
    approval_rate = (df["Loan_Status"] == "Y").mean() * 100
    credit_1_rate = df.loc[df["Credit_History"] == 1, "Loan_Status"].eq("Y").mean() * 100
    credit_0_rate = df.loc[df["Credit_History"] == 0, "Loan_Status"].eq("Y").mean() * 100
    return (
        f"Overall approval rate: {approval_rate:.1f}%. "
        f"Applicants with Credit_History=1 are approved {credit_1_rate:.1f}% of the time, "
        f"versus only {credit_0_rate:.1f}% for Credit_History=0 — by far the strongest "
        f"single relationship in the dataset. Income distributions are right-skewed with a "
        f"long high-earner tail, which motivated median imputation and the engineered "
        f"Total_Income / affordability-ratio features. Categorical splits (Gender, "
        f"Married, Self_Employed, Education) show much smaller differences in approval "
        f"rate than Credit_History, foreshadowing what the model coefficients later confirm."
    )
