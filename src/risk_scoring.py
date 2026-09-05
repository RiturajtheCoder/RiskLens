from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_engine import classify_risk, recommend_action


def normalize_anomaly(raw_scores: np.ndarray) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=float)
    if raw_scores.size == 0:
        return raw_scores
    minimum, maximum = raw_scores.min(), raw_scores.max()
    if np.isclose(minimum, maximum):
        return np.zeros_like(raw_scores)
    return (raw_scores - minimum) / (maximum - minimum)


def behavioral_signal(df: pd.DataFrame) -> pd.Series:
    amount_ratio = np.clip((df["amount_to_customer_avg"] - 1.0) / 8.0, 0, 1)
    velocity = np.clip(df["transactions_last_24h"] / 12.0, 0, 1)
    timing = ((df["is_night"] == 1).astype(float) * 0.7) + ((df["is_weekend"] == 1).astype(float) * 0.3)
    foreign = df["is_foreign"].astype(float)
    signal = (0.42 * amount_ratio) + (0.28 * velocity) + (0.18 * timing) + (0.12 * foreign)
    return pd.Series(np.clip(signal, 0, 1), index=df.index)


def build_risk_report(df: pd.DataFrame, fraud_probability: np.ndarray, anomaly_score: np.ndarray) -> pd.DataFrame:
    report = df.copy()
    behavior = behavioral_signal(report)
    fraud_probability = np.asarray(fraud_probability, dtype=float)
    anomaly_score = np.asarray(anomaly_score, dtype=float)
    risk = (0.62 * fraud_probability) + (0.23 * anomaly_score) + (0.15 * behavior.to_numpy())
    report["fraud_probability"] = np.round(fraud_probability, 4)
    report["anomaly_score"] = np.round(anomaly_score, 4)
    report["behavioral_risk_signal"] = np.round(behavior, 4)
    report["risk_score"] = np.round(np.clip(risk * 100, 0, 100), 0).astype(int)
    report["risk_level"] = report["risk_score"].apply(classify_risk)
    actions = report["risk_score"].apply(recommend_action)
    report["recommended_action"] = actions.apply(lambda item: item[0])
    report["action_reason"] = actions.apply(lambda item: item[1])
    report["model_confidence"] = np.round(np.maximum(fraud_probability, 1 - fraud_probability), 4)
    return report

