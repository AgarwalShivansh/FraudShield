
from pathlib import Path
import gc
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)

DATA_FILE = Path("data/processed/train_merged.parquet")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET = "isFraud"
TIME_COLUMN = "TransactionDT"
VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}"
        )

    print("Loading merged dataset...")
    df = pd.read_parquet(DATA_FILE)

    if TARGET not in df.columns or TIME_COLUMN not in df.columns:
        raise ValueError("Target or time column is missing.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET].mean() * 100:.4f}%")

    # Sort chronologically to simulate predicting future transactions.
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    split_index = int(len(df) * (1 - VALIDATION_FRACTION))

    train_df = df.iloc[:split_index].copy()
    valid_df = df.iloc[split_index:].copy()

    y_train = train_df[TARGET].astype("int8")
    y_valid = valid_df[TARGET].astype("int8")

    # The ID is not used as a model feature.
    excluded_columns = [TARGET, "TransactionID"]
    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    X_train = train_df[feature_columns].copy()
    X_valid = valid_df[feature_columns].copy()

    # Convert text columns into categorical features.
    # Use the same category vocabulary for both datasets.
    categorical_columns = (
        X_train.select_dtypes(include=["object", "category"]).columns
    ).tolist()

    for col in categorical_columns:
        train_values = X_train[col].astype("string").fillna("__MISSING__")
        valid_values = X_valid[col].astype("string").fillna("__MISSING__")

        categories = pd.Index(
            pd.concat([train_values, valid_values], ignore_index=True).unique()
        )

        X_train[col] = pd.Categorical(
            train_values, categories=categories
        )
        X_valid[col] = pd.Categorical(
            valid_values, categories=categories
        )

    # Give more weight to the minority fraud class.
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / max(positive_count, 1)

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Positive-class weight: {scale_pos_weight:.2f}")


    model = lgb.LGBMClassifier(
        objective="binary",
        metric="None",              # Disable default binary log-loss metric
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbosity=-1,
    )


    print("\nTraining LightGBM...")
    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_columns,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    print("\nGenerating validation predictions...")
    probabilities = model.predict_proba(
        X_valid,
        num_iteration=model.best_iteration_,
    )[:, 1]

    # Initial benchmark threshold; later we will tune it for business needs.
    threshold = 0.50
    predictions = (probabilities >= threshold).astype("int8")

    metrics = {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "feature_count": int(len(feature_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "train_fraud_rate_percent": float(y_train.mean() * 100),
        "validation_fraud_rate_percent": float(y_valid.mean() * 100),
        "best_iteration": int(model.best_iteration_ or 0),
        "threshold": threshold,
        "average_precision_pr_auc": float(
            average_precision_score(y_valid, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "precision_at_threshold": float(
            precision_score(y_valid, predictions, zero_division=0)
        ),
        "recall_at_threshold": float(
            recall_score(y_valid, predictions, zero_division=0)
        ),
        "f1_at_threshold": float(
            f1_score(y_valid, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_valid, predictions, labels=[0, 1]
        ).tolist(),
    }

    print("\nVALIDATION RESULTS")
    print("=" * 50)
    print(json.dumps(metrics, indent=2))

    print("\nClassification report at threshold 0.50:")
    print(
        classification_report(
            y_valid,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    model_path = MODEL_DIR / "lightgbm_baseline.joblib"
    metrics_path = REPORT_DIR / "baseline_metrics.json"
    features_path = MODEL_DIR / "feature_columns.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(features_path, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    print(f"\nModel saved: {model_path.resolve()}")
    print(f"Metrics saved: {metrics_path.resolve()}")
    print(f"Feature list saved: {features_path.resolve()}")

    del df, train_df, valid_df, X_train, X_valid
    gc.collect()

    print("\nBaseline experiment completed.")


if __name__ == "__main__":
    main()
