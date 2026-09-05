import numpy as np
import pandas as pd

from src.decision_engine import classify_risk, recommend_action
from src.explainability import attach_explanations
from src.feature_engineering import engineer_features
from src.preprocessing import validate_transactions
from src.risk_scoring import build_risk_report


def sample_df():
    return pd.DataFrame({
        "transaction_id": ["TXN001", "TXN002"],
        "customer_id": ["C1", "C2"],
        "amount": [500, 95000],
        "merchant_category": ["grocery", "jewelry"],
        "hour": [13, 2],
        "day_of_week": [2, 6],
        "transactions_last_24h": [1, 10],
        "avg_customer_amount": [450, 4000],
        "is_foreign": [0, 1],
        "is_fraud": [0, 1],
    })


def test_validation_accepts_required_columns():
    result = validate_transactions(sample_df(), require_target=True)
    assert result.is_valid
    assert len(result.data) == 2


def test_validation_reports_missing_columns():
    result = validate_transactions(sample_df().drop(columns=["amount"]))
    assert not result.is_valid
    assert "amount" in result.errors[0]


def test_feature_engineering_adds_behavioral_features():
    engineered = engineer_features(sample_df())
    assert "amount_to_customer_avg" in engineered
    assert engineered.loc[1, "is_night"] == 1


def test_risk_scoring_and_decisions_are_deterministic():
    engineered = engineer_features(sample_df())
    report = build_risk_report(engineered, np.array([0.05, 0.92]), np.array([0.1, 0.95]))
    assert report.loc[0, "risk_level"] == "LOW"
    assert report.loc[1, "risk_level"] == "HIGH"
    assert report.loc[1, "recommended_action"] in {"HOLD FOR MANUAL REVIEW", "BLOCK"}


def test_classification_boundaries():
    assert classify_risk(30) == "LOW"
    assert classify_risk(31) == "MEDIUM"
    assert classify_risk(71) == "HIGH"
    assert recommend_action(12)[0] == "APPROVE"


def test_explanations_are_attached():
    engineered = engineer_features(sample_df())
    report = build_risk_report(engineered, np.array([0.05, 0.92]), np.array([0.1, 0.95]))
    explained = attach_explanations(report)
    assert explained["top_risk_factors"].str.len().min() > 0

