from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACT_PATH, RANDOM_STATE
from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features, model_matrix


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_fraud_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def train_fraud_model(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    engineered = engineer_features(df)
    X = engineered[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = engineered["is_fraud"].astype(int)
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
    )
    model = build_fraud_pipeline()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "pr_auc": float(average_precision_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "test_size": int(len(y_test)),
        "positive_rate": float(y.mean()),
    }
    holdout = X_test.copy()
    holdout["is_fraud"] = y_test.to_numpy()
    holdout["fraud_probability"] = probabilities
    holdout["prediction"] = predictions
    return model, metrics, holdout


def predict_fraud_probability(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X = model_matrix(df)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def save_artifact(artifact: dict, path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    return joblib.load(path)

