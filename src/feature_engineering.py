from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "amount",
    "hour",
    "day_of_week",
    "transactions_last_24h",
    "avg_customer_amount",
    "is_foreign",
    "amount_to_customer_avg",
    "amount_deviation",
    "is_night",
    "is_weekend",
]

CATEGORICAL_FEATURES = ["merchant_category"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    avg = features["avg_customer_amount"].replace(0, np.nan)
    features["amount_to_customer_avg"] = (features["amount"] / avg).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    features["amount_deviation"] = (features["amount"] - features["avg_customer_amount"]).abs()
    features["is_night"] = features["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
    return features


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_features(df)
    return engineered[MODEL_FEATURES].copy()

