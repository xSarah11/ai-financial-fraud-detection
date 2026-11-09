"""
Financial Transaction Fraud Detection
-------------------------------------
Trains a Random Forest classifier to flag fraudulent transactions
(PaySim-style dataset: type, amount, balances, isFraud).

Usage:
    python fraud_detection.py --data data/AIML_Dataset.csv            # train + evaluate + save model
    python fraud_detection.py --data data/AIML_Dataset.csv --sample 500000   # faster run on a subset
    python fraud_detection.py --predict                                # check one transaction with the saved model
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TARGET = "isFraud"
TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
NUM_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
MODEL_PATH = Path("models/fraud_model.joblib")
OUTPUT_DIR = Path("outputs")


def load_data(path: str, sample: int | None) -> pd.DataFrame:
    """Read the CSV and check that the required columns exist."""
    df = pd.read_csv(path, usecols=lambda c: c in NUM_COLS + ["type", TARGET])
    missing = [c for c in NUM_COLS + ["type", TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    if sample and sample < len(df):
        df = df.sample(sample, random_state=0)
    return df.dropna()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transactions into model features (same function used for training and prediction)."""
    X = df[NUM_COLS].copy()
    # Balance mismatch: in honest transactions the balances add up; in fraud they often don't.
    X["errorBalanceOrig"] = X["newbalanceOrig"] + X["amount"] - X["oldbalanceOrg"]
    X["errorBalanceDest"] = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
    # One-hot encode the transaction type with a fixed list so columns never change.
    for t in TX_TYPES:
        X[f"type_{t}"] = (df["type"] == t).astype(int)
    return X


def explore(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(df.head(), "\n")
    fraud_rate = df[TARGET].mean() * 100
    print(f"Fraud cases: {int(df[TARGET].sum()):,} ({fraud_rate:.3f}% of all transactions)")
    print("Fraud by transaction type:")
    print(df.groupby("type")[TARGET].sum().to_string(), "\n")


def evaluate(model, X_test, y_test) -> None:
    """Print metrics and save plots. Accuracy alone is misleading for rare fraud, so we use precision/recall."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation on test set ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC : {average_precision_score(y_test, y_prob):.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot.barh(ax=ax, color="#2a6fdb")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"Plots saved in: {OUTPUT_DIR}/")


def train(data_path: str, sample: int | None) -> None:
    df = load_data(data_path, sample)
    explore(df)

    X = build_features(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y  # stratify keeps the same fraud ratio in both sets
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",  # fraud is rare, so give it more weight
        n_jobs=-1,
        random_state=0,
    )
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


def predict_fraud(model, tx: dict) -> tuple[int, float]:
    """Return (label, fraud probability) for one transaction."""
    X = build_features(pd.DataFrame([tx]))
    prob = float(model.predict_proba(X)[0, 1])
    return int(prob >= 0.5), prob


def predict_interactive() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Train first: python fraud_detection.py --data <file.csv>")
    model = joblib.load(MODEL_PATH)

    print(f"Transaction types: {', '.join(TX_TYPES)}")
    t = input("Type: ").strip().upper()
    if t not in TX_TYPES:
        raise SystemExit(f"Unknown type '{t}'.")
    tx = {"type": t}
    for col in NUM_COLS:
        tx[col] = float(input(f"{col}: "))

    label, prob = predict_fraud(model, tx)
    print(f"\nResult: {'FRAUD' if label else 'LEGIT'} (fraud probability: {prob:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial transaction fraud detection")
    parser.add_argument("--data", help="Path to the CSV dataset")
    parser.add_argument("--sample", type=int, help="Use only N random rows (faster)")
    parser.add_argument("--predict", action="store_true", help="Check one transaction with the saved model")
    args = parser.parse_args()

    if args.predict:
        predict_interactive()
    elif args.data:
        train(args.data, args.sample)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
