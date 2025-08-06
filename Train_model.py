#!/usr/bin/env python3
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = Path("SBAnational.csv")
MODEL_REAL_PATH = Path("loan_model_real.pkl")
MODEL_SYNTH_PATH = Path("loan_model_synth.pkl")

# Feature sets
FEATURES_REAL = [
    "Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob", "FranchiseCode",
    "UrbanRural", "RevLineCr", "LowDoc", "DisbursementGross", "BalanceGross", "GrAppv", "SBA_Appv"
]
FEATURES_SYNTH = FEATURES_REAL + ["CreditScore", "InterestRate", "DSCR", "NetIncomeBeforeDebt"]

# Money columns for cleaning
MONEY_COLS = ["DisbursementGross", "GrAppv", "SBA_Appv", "BalanceGross"]

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── Data Processing Functions ─────────────────────────────────────────────────
def clean_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in MONEY_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace(r"[\$,]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
    return df


def engineer_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    df["CreditScore"] = (
        np.random.normal(670, 50, len(df))
        .astype(int).clip(500, 800)
    )
    df["InterestRate"] = (
        np.random.normal(0.08, 0.015, len(df))
        .clip(0.03, 0.15)
    )
    df["NetIncomeBeforeDebt"] = (
        np.random.normal(100_000, 25_000, len(df))
        .clip(20_000, 200_000)
    )
    return df


def compute_dscr(df: pd.DataFrame) -> pd.DataFrame:
    ir = df["InterestRate"]
    tm = df["Term"]
    df["AnnualDebtService"] = (
        df["DisbursementGross"] * ir /
        (1 - (1 + ir) ** (-tm / 12))
    )
    df["AnnualDebtService"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["AnnualDebtService"].fillna(df["AnnualDebtService"].median(), inplace=True)
    df["DSCR"] = df["NetIncomeBeforeDebt"] / df["AnnualDebtService"]
    df["DSCR"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["DSCR"].fillna(df["DSCR"].median(), inplace=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Filter for PIF/CHGOFF
    df = df[df["MIS_Status"].isin(["P I F", "CHGOFF"])].copy()
    df["Defaulted"] = df["MIS_Status"].map({"P I F": 0, "CHGOFF": 1})

    # Numeric conversions
    df["Term"] = pd.to_numeric(df["Term"], errors="coerce")
    for col in ["NoEmp", "CreateJob", "RetainedJob", "FranchiseCode"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary flags
    df["RevLineCr"] = df["RevLineCr"].fillna("N").map({"Y": 1, "N": 0})
    df["LowDoc"]    = df["LowDoc"].fillna("N").map({"Y": 1, "N": 0})
    if "NewExist" in df.columns:
        df["NewExist"] = df["NewExist"].map({"Existing": 0, "New": 1}).fillna(0).astype(int)

    # Clean money
    df = clean_money_columns(df)

    # Synthetic engineering
    df = engineer_synthetic_features(df)
    df = compute_dscr(df)

    # Fill missing in both feature sets
    for col in FEATURES_SYNTH:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    logging.info("Preprocessing complete.")
    return df

# ─── Model Training & Evaluation ────────────────────────────────────────────────
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, label: str):
    logging.info(f"Training on {label} features: {X.shape[1]} columns, {len(X)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    logging.info(f"{label} Train accuracy: {train_acc:.4f}")
    logging.info(f"{label} Test  accuracy: {test_acc:.4f}")
    return model

# ─── Main Entrypoint ────────────────────────────────────────────────────────────
def main():
    raw_df = pd.read_csv(DATA_PATH)
    df     = preprocess(raw_df)
    y      = df['Defaulted']

    # Prepare X matrices
    X_real  = df[FEATURES_REAL]
    X_synth = df[FEATURES_SYNTH]

    # Train models
    model_real  = train_and_evaluate(X_real,  y, 'Real')
    model_synth = train_and_evaluate(X_synth, y, 'Real+Synthetic')

    # Persist models
    joblib.dump(model_real,  MODEL_REAL_PATH)
    joblib.dump(model_synth, MODEL_SYNTH_PATH)
    logging.info(f"Saved Real-only model to {MODEL_REAL_PATH}")
    logging.info(f"Saved Real+Synth model to {MODEL_SYNTH_PATH}")

if __name__ == "__main__":
    main()
