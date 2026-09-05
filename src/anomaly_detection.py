from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.fraud_model import build_preprocessor
from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, model_matrix


def build_anomaly_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("detector", IsolationForest(n_estimators=150, contamination=0.08, random_state=RANDOM_STATE)),
    ])


def train_anomaly_detector(df: pd.DataFrame) -> Pipeline:
    normal = df[df.get("is_fraud", 0) == 0] if "is_fraud" in df.columns else df
    model = build_anomaly_pipeline()
    model.fit(model_matrix(normal))
    return model


def anomaly_risk_score(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X = model_matrix(df)
    raw = -model.decision_function(X)
    low, high = np.percentile(raw, [5, 95])
    if np.isclose(low, high):
        return np.zeros(len(df))
    return np.clip((raw - low) / (high - low), 0, 1)

