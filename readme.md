# FraudShield AI — Explainable Financial Fraud Detection

An end-to-end machine learning project for detecting potentially fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset. FraudShield AI combines transaction analytics, LightGBM classification, model evaluation, and SHAP explainability in an interactive Streamlit dashboard.

## Overview

Financial fraud detection is challenging because fraudulent transactions are relatively rare compared with legitimate transactions. A useful model must identify suspicious activity while balancing missed fraud against false alarms.

FraudShield AI explores this problem through chronological validation, gradient-boosted decision trees, threshold analysis, and model explanations.

### Key Features

- **Data profiling and EDA:** Analyze transaction distributions, missing values, fraud prevalence, and identity-data coverage.
- **Feature preparation:** Merge transaction and identity data and prepare features for machine learning.
- **Fraud classification:** Train a LightGBM model to estimate the likelihood of a transaction being fraudulent.
- **Model evaluation:** Evaluate Average Precision, ROC-AUC, precision, recall, and F1-score.
- **Threshold analysis:** Examine how the classification threshold changes fraud detection and alert volume.
- **Global explainability:** Use SHAP to identify features that influence model predictions across a sample.
- **Individual explainability:** Inspect SHAP waterfall plots and feature contributions for selected transactions.
- **Interactive dashboard:** Explore fraud analytics, model performance, feature importance, and transaction scoring through Streamlit.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy, PyArrow |
| Machine learning | LightGBM, scikit-learn |
| Explainable AI | SHAP |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Development | VS Code, Jupyter Notebook, Git |

## Dataset

FraudShield AI uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

The merged dataset contains:

- **590,540** transactions
- **434** columns after merging transaction and identity data
- **20,663** fraudulent transactions
- Approximately **3.50%** fraud prevalence
- Approximately **24.42%** identity-record match rate

The dataset is highly imbalanced, and many columns contain missing values. These characteristics make it useful for exploring fraud detection under realistic modeling challenges.

**Dataset access:** Obtain the data through the official Kaggle competition page: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data).

The dataset is not included in this repository. Download it separately and follow the dataset's applicable terms.

## Project Structure

```text
FraudShield/
├── app.py
├── requirements.txt
├── .gitignore
├── .venv/                       # Local environment; not committed
├── data/
│   ├── raw/                      # Downloaded CSV files; not committed
│   └── processed/
│       └── train_merged.parquet  # Processed dataset; not committed
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_eda_and_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_improvement.ipynb
│   ├── 05_shap_explainability.ipynb
│   └── 06_fraudshield_dashboard.ipynb
├── models/
│   ├── lightgbm_final_baseline.joblib
│   └── feature_columns.json
└── reports/
    ├── final_test_metrics.json
    ├── shap_global_importance.csv
    ├── shap_summary.png
    ├── individual_transaction_shap.png
    ├── dashboard_product_analytics.csv
    └── dashboard_top_features.csv
```

*Adjust the notebook paths or filenames above if your actual repository structure differs.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FraudShield.git
cd FraudShield
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Python 3.12 is the development environment used for this project.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset and model artifacts

Download the required IEEE-CIS CSV files from Kaggle and place them in the expected raw-data directory. Run the notebooks in sequence to reproduce the data preparation, model training, and explainability workflow.

The dashboard also requires the saved model, feature-column metadata, and report artifacts at the paths expected by `app.py`.

### 5. Launch the dashboard

From the project root:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal, typically `http://localhost:8501`.

## Modeling Workflow

1. **Profile the data:** Inspect dimensions, column types, missingness, and target distribution.
2. **Explore and prepare features:** Merge transaction and identity information and examine useful patterns.
3. **Establish a baseline:** Train LightGBM with a chronological train/test split.
4. **Improve the model:** Use chronological training, validation, and test partitions, with early stopping guided by validation performance.
5. **Evaluate generalization:** Measure ranking and classification metrics on held-out test data.
6. **Explain predictions:** Use SHAP to investigate global feature importance and individual transaction predictions.
7. **Present the results:** Load the saved artifacts into the Streamlit dashboard.

Chronological splitting is important because transaction data has a temporal dimension. The validation and test sets should remain separate from model fitting and feature-selection decisions.

## Model Performance

The current improved-model experiment produced the following held-out test results at a classification threshold of 0.50.

| Metric | Result |
|---|---:|
| Average Precision (AP) | 0.5258 |
| ROC-AUC | 0.8894 |
| Precision | 0.7304 |
| Recall | 0.3717 |
| F1-score | 0.4927 |
| Best iteration | 3,118 |

These values describe one experiment and its chosen threshold, not guaranteed production performance.

Average Precision is particularly informative for an imbalanced classification problem because it summarizes precision-recall behavior across thresholds. ROC-AUC measures ranking ability, while precision and recall at a fixed threshold show the practical trade-off between false alarms and missed fraud.

The classification threshold should be selected according to the intended review capacity and the relative costs of false positives and false negatives.

## Explainable AI with SHAP

FraudShield AI uses SHAP (SHapley Additive exPlanations) to investigate how individual features contribute to model outputs.

### Global explanations

Global SHAP importance summarizes the average magnitude of feature contributions across a sample. In the current analysis, influential features include `id_30`, `R_emaildomain`, `id_31`, `id_33`, `P_emaildomain`, and `TransactionAmt`.

### Individual explanations

The transaction-scoring page displays a waterfall plot and a ranked contribution table for a selected transaction. Positive contributions generally move the model output toward the fraud class, while negative contributions move it away.

SHAP explanations describe model behavior; they do not establish that a transaction is fraudulent or prove that a particular feature caused fraud.

## Limitations

- The project is an experimental fraud-detection system, not a validated production banking service.
- Historical dataset performance may not generalize to other institutions, populations, or time periods.
- Missing values and limited identity-data coverage may affect predictions.
- Classification results depend on the chosen threshold and the evaluation dataset.
- Historical transaction scoring is retrospective. It is not the same as validating performance on unseen, live transactions.
- Feature importance and SHAP contributions indicate learned model relationships, not causation.
- Production deployment would require additional monitoring, security controls, latency testing, data-drift checks, and human-review procedures.

## Future Improvements

- Evaluate precision-recall trade-offs under different review-capacity constraints.
- Add time-based performance and drift monitoring.
- Validate calibration of predicted probabilities.
- Build a controlled scoring interface for new transactions with explicit feature validation.
- Add audit logging and role-based access controls for operational use.
- Conduct additional robustness and fairness checks before any real-world deployment.

## Author

**Shivansh Agarwal**  
B.Tech, Computer Science Engineering — Data Science

GitHub: [Add your GitHub profile URL](https://github.com/)

---

*FraudShield AI is an educational and portfolio project focused on machine learning, imbalanced classification, and explainable AI.*
