
from pathlib import Path
import json

import numpy as np
import shap
import matplotlib.pyplot as plt

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"

MODEL_FILE = MODEL_DIR / "lightgbm_final_baseline.joblib"
FEATURE_FILE = MODEL_DIR / "feature_columns.json"
METRICS_FILE = REPORT_DIR / "final_test_metrics.json"
SHAP_FILE = REPORT_DIR / "shap_global_importance.csv"
PRODUCT_FILE = REPORT_DIR / "dashboard_product_analytics.csv"

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# Load saved artifacts
# --------------------------------------------------


@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(load_model())

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_FILE)


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🛡️ FraudShield AI")
st.caption("Intelligent financial fraud detection and risk analytics")

required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    METRICS_FILE,
    SHAP_FILE,
    PRODUCT_FILE,
    DATA_FILE,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Some project artifacts are missing:")
    for path in missing_files:
        st.write(f"- {path}")
    st.stop()

model = load_model()
df = load_data()
feature_columns = load_json(FEATURE_FILE)
metrics = load_json(METRICS_FILE)
shap_importance = load_csv(SHAP_FILE)
product_analytics = load_csv(PRODUCT_FILE)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Fraud Analytics",
        "Model Performance",
        "Feature Importance",
        "Transaction Scoring",
    ],
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.subheader("Dataset Overview")

    total = len(df)
    fraud_count = int(df["isFraud"].sum())
    legitimate_count = total - fraud_count
    fraud_rate = fraud_count / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraudulent", f"{fraud_count:,}")
    c3.metric("Legitimate", f"{legitimate_count:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.markdown("### Transaction Class Distribution")

    class_df = pd.DataFrame({
        "Class": ["Legitimate", "Fraud"],
        "Count": [legitimate_count, fraud_count],
    })

    fig = px.bar(
        class_df,
        x="Class",
        y="Count",
        color="Class",
        title="Legitimate vs Fraudulent Transactions",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This dataset contains historical labeled transactions. "
        "Its fraud rate is not necessarily representative of live payment traffic."
    )

# --------------------------------------------------
# Fraud Analytics
# --------------------------------------------------
elif page == "Fraud Analytics":
    st.subheader("Fraud Analytics")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Fraud Rate by Product")
        fig = px.bar(
            product_analytics,
            x="ProductCD",
            y="fraud_rate_percent",
            hover_data=["transaction_count", "fraud_count"],
            labels={
                "ProductCD": "Product Category",
                "fraud_rate_percent": "Fraud Rate (%)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Transaction Count by Product")
        fig = px.bar(
            product_analytics,
            x="ProductCD",
            y="transaction_count",
            labels={
                "ProductCD": "Product Category",
                "transaction_count": "Transaction Count",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Product-Level Summary")
    st.dataframe(product_analytics, use_container_width=True)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance — Held-Out Test Set")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Precision",
        f"{metrics['average_precision']:.4f}",
    )
    c2.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.4f}",
    )
    c3.metric(
        "Best Iteration",
        f"{metrics['best_iteration']:,}",
    )

    st.markdown("### Classification Metrics")
    metric_table = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score"],
        "Value": [
            metrics["precision_at_050"],
            metrics["recall_at_050"],
            metrics["f1_at_050"],
        ],
    })

    st.dataframe(metric_table, use_container_width=True)

    st.caption(
        "Metrics are from the saved held-out test evaluation at a "
        "classification threshold of 0.50."
    )

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
elif page == "Feature Importance":
    st.subheader("Global SHAP Feature Importance")

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=30,
        value=15,
    )

    top_features = shap_importance.head(top_n).sort_values(
        "mean_absolute_shap"
    )

    fig = px.bar(
        top_features,
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        title="Mean Absolute SHAP Value",
        labels={
            "mean_absolute_shap": "Mean |SHAP value|",
            "feature": "Feature",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Global SHAP importance measures average contribution magnitude. "
        "It does not imply causation or show the direction of an individual prediction."
    )

# --------------------------------------------------
# Transaction Scoring
# --------------------------------------------------
elif page == "Transaction Scoring":
    st.subheader("Transaction Risk Scoring")

    st.warning(
        "Demo mode: choose a labeled historical transaction. "
        "This is not a live payment gateway and does not authorize or block payments."
    )

    transaction_id = st.selectbox(
        "Choose a historical TransactionID",
        options=df["TransactionID"].sample(
            min(5000, len(df)),
            random_state=42,
        ).tolist(),
    )


    row = df.loc[df["TransactionID"] == transaction_id].iloc[0]

    # Create one-row input with the exact training feature order.
    X_one = pd.DataFrame(
        [row[feature_columns].to_dict()],
        columns=feature_columns,
    )

    # Restore the categorical vocabularies saved by the trained
    # LightGBM model instead of creating new category definitions.
    booster = model.booster_
    pandas_categories = booster.pandas_categorical

    categorical_columns = X_one.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if pandas_categories is not None:
        if len(categorical_columns) != len(pandas_categories):
            st.error(
                "Categorical feature schema mismatch between "
                "the saved model and dashboard input."
            )
            st.stop()

        for col, categories in zip(categorical_columns, pandas_categories):
            X_one[col] = pd.Categorical(
                X_one[col],
                categories=categories,
            )

    # Predict only after the categorical schemas match.
    probability = float(model.predict_proba(X_one)[0, 1])

    st.markdown("---")
    st.subheader("Why did the model give this score?")
    st.caption(
        "SHAP explains how individual features influence the model's "
        "output. These are model explanations, not proof of fraud."
    )

    with st.spinner("Generating transaction explanation..."):
        explainer = load_shap_explainer()
        raw_shap_values = explainer.shap_values(X_one)

        # Handle SHAP output formats across library versions.
        if isinstance(raw_shap_values, list):
            row_shap_values = np.asarray(raw_shap_values[1])[0]
        else:
            raw_shap_values = np.asarray(raw_shap_values)
            if raw_shap_values.ndim == 3:
                row_shap_values = raw_shap_values[0, :, 1]
            else:
                row_shap_values = raw_shap_values[0]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.asarray(expected_value).reshape(-1)
            base_value = float(
                expected_value[1] if len(expected_value) > 1
                else expected_value[0]
            )
        else:
            base_value = float(expected_value)

        explanation = shap.Explanation(
            values=row_shap_values,
            base_values=base_value,
            data=X_one.iloc[0].to_numpy(),
            feature_names=feature_columns,
        )


    # Waterfall chart: largest positive and negative contributions.
    plt.figure(figsize=(10, 7))

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


    # Show the largest contributions in a readable table.
    contribution_table = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP contribution": row_shap_values,
        "Feature value": X_one.iloc[0].astype(str).to_numpy(),
    })

    contribution_table["Absolute contribution"] = (
        contribution_table["SHAP contribution"].abs()
    )

    contribution_table = contribution_table.sort_values(
        "Absolute contribution",
        ascending=False,
    )

    st.markdown("### Most influential features")
    st.dataframe(
        contribution_table[
            [
                "Feature",
                "Feature value",
                "SHAP contribution",
            ]
        ].head(15).round(5),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Positive SHAP contributions generally push the model output "
        "toward the fraud class; negative contributions push it away. "
        "The waterfall uses the model's output scale, which may be "
        "log-odds rather than probability."
    )



    st.metric("Predicted Fraud Probability", f"{probability:.2%}")

    st.progress(min(max(probability, 0.0), 1.0))

    if probability >= 0.50:
        st.error("Flagged by the demonstration threshold.")
    else:
        st.success("Below the demonstration threshold.")

    st.markdown("### Historical Transaction Details")

    st.write({
        "TransactionID": int(row["TransactionID"]),
        "TransactionAmt": float(row["TransactionAmt"]),
        "ProductCD": str(row["ProductCD"]),
        "Actual Label": int(row["isFraud"]),
    })

    st.caption(
        "The actual label is shown for retrospective demonstration only. "
        "Do not use it as an input to the model or as evidence of prospective performance."
    )

st.divider()
st.caption("FraudShield AI | Machine Learning Portfolio Project")
