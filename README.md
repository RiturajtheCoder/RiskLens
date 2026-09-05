# RiskLens

AI-powered financial transaction risk management for the Razorpay AI Builder Internship 2026, Track 2: AI Risk Manager.

RiskLens analyzes transactions, estimates fraud probability, detects anomalous behavior, assigns a 0-100 interpretable risk score, explains top risk factors, and recommends an action for a human risk analyst.

##Access the live website
https://risklensapp.streamlit.app/

## Problem Statement

Fraud operations teams need more than a binary "fraud / not fraud" label. They need to understand how risky a transaction is, why it was flagged, how confident the model is, and what operational action should be taken.

## Solution

RiskLens transforms the original Deep Q-Learning fraud-detection research prototype into a modular demo-ready fintech product:

Transaction -> Validation -> Feature Engineering -> Fraud Prediction -> Anomaly Detection -> Risk Scoring -> Explanation -> Recommended Action -> Analyst Review

## Key Features

- Professional Streamlit dashboard
- CSV upload with schema validation and graceful invalid-row handling
- Fraud probability from a class-weighted supervised model
- Separate anomaly score from Isolation Forest
- Interpretable 0-100 risk score
- LOW / MEDIUM / HIGH risk classification
- APPROVE / REVIEW / HOLD / BLOCK action recommendations
- Transaction detail view with top risk factors
- Model performance page with precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, false positive rate, and false negative rate
- Downloadable analyzed risk report
- Synthetic demo dataset generated locally

## Architecture

```text
app.py
data/
  sample/synthetic_transactions.csv
models/
  risklens_model.joblib
  metrics.json
src/
  preprocessing.py
  feature_engineering.py
  fraud_model.py
  anomaly_detection.py
  risk_scoring.py
  explainability.py
  decision_engine.py
  evaluation.py
  utils.py
scripts/
  create_demo_data.py
  train_models.py
tests/
```

## Technology Stack

- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- Plotly
- pytest
- joblib
- Gymnasium-compatible legacy environment support

## Dataset

The runnable demo uses synthetic, anonymized transaction data created by `scripts/create_demo_data.py`. It includes normal, suspicious, high-risk, and anomalous-looking transactions across amount, merchant category, timing, velocity, and foreign-transaction signals.

The original repository referenced Kaggle's credit card fraud dataset and included a DQN notebook, but no dataset file was committed. RiskLens does not fabricate real financial information.

## ML Methodology

RiskLens separates the product pipeline into independent components:

- Fraud prediction: class-weighted Logistic Regression for transparent, reproducible fraud probability.
- Anomaly detection: Isolation Forest trained primarily on normal transactions.
- Risk scoring: weighted blend of fraud probability, anomaly score, and behavioral signals.
- Decisioning: deterministic threshold-based action recommendation.

The original DQN/Gym code is retained as a legacy research experiment. It is not used as the primary demo classifier because the notebook is Colab-specific, accuracy-only, and not sufficiently explainable for analyst-facing risk management.

## Risk Scoring

Risk score is calculated as:

```text
62% fraud probability + 23% anomaly score + 15% behavioral risk signal
```

Default bands:

- 0-30: LOW
- 31-70: MEDIUM
- 71-100: HIGH

Default actions:

- LOW: APPROVE
- MEDIUM: REVIEW / MONITOR
- HIGH: HOLD FOR MANUAL REVIEW
- 90+: BLOCK

## Explainability

RiskLens generates explanations from structured model and feature signals, not from invented text. Current explanation drivers include elevated fraud probability, high anomaly score, unusual amount relative to customer average, high transaction velocity, foreign transactions, night-time activity, and weekend timing.

## Model Evaluation

The current synthetic demo model reports:

- Precision: 0.340
- Recall: 0.692
- F1-score: 0.456
- ROC-AUC: 0.790
- PR-AUC: 0.608
- False Positive Rate: 0.176
- False Negative Rate: 0.308

These metrics matter because fraud datasets are imbalanced. Accuracy can look high even when fraud is missed. Precision controls analyst review burden, while recall controls missed fraud.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Generate demo data and train saved model artifacts:

```bash
python scripts/create_demo_data.py
python scripts/train_models.py
```

Run the app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## CSV Schema

Uploads should include:

```text
transaction_id, customer_id, amount, merchant_category, hour, day_of_week,
transactions_last_24h, avg_customer_amount, is_foreign
```

`is_fraud` is optional for demo/training data and is not required for inference uploads.

## Security

- No API keys are hardcoded.
- `.env.example` is included for optional future LLM integration.
- Core risk analysis works without an LLM.
- Demo data is synthetic and anonymized.
- This prototype is not intended for direct production financial decisions.

## Existing Repository Audit

Original reusable components:

- Domain framing around credit-card fraud and financial risk.
- Gym environment concept for DQN experimentation.
- DQN notebook as research context.

Issues fixed or avoided:

- Hardcoded Colab paths in the notebook are not used by the product path.
- Legacy Gym environment off-by-one indexing was patched.
- Dataset hardcoding is isolated to legacy code; RiskLens uses configurable scripts and local artifacts.
- Accuracy-only evaluation was replaced with fraud-appropriate metrics.
- Model training no longer happens on every app startup unless artifacts are missing.

## Tests

```bash
python -m pytest -q
```

Current status: 6 tests passing.

## Limitations

- The included model is trained on synthetic demo data, not a bank-grade production dataset.
- Explanations are structured heuristics plus model outputs, not full SHAP yet.
- Thresholds are code-configurable but do not yet have an admin UI.
- The DQN path is retained for experimentation and would need a proper Gymnasium training loop before product use.

## Future Improvements

- Train and tune on a real anonymized fraud dataset with strict leakage controls.
- Add SHAP-based local explanations for the supervised model.
- Add threshold tuning by target recall, precision, or expected investigation cost.
- Add analyst feedback capture and retraining workflow.
- Add API endpoints for batch scoring.
- Add authentication, audit logs, and role-based access controls for production readiness.

