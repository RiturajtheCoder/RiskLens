from __future__ import annotations

import argparse
import sys

import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_demo_data import create_demo_data
from src.anomaly_detection import train_anomaly_detector
from src.config import ARTIFACT_PATH, METRICS_PATH, SAMPLE_DATA_PATH
from src.evaluation import confusion_metrics
from src.fraud_model import save_artifact, train_fraud_model
from src.preprocessing import validate_transactions
from src.utils import write_json


def train(data_path=SAMPLE_DATA_PATH) -> dict:
    if not data_path.exists():
        create_demo_data(path=data_path)
    raw = pd.read_csv(data_path)
    validation = validate_transactions(raw, require_target=True)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.errors))

    fraud_model, metrics, holdout = train_fraud_model(validation.data)
    anomaly_model = train_anomaly_detector(validation.data)
    cm = confusion_metrics(holdout["is_fraud"], holdout["prediction"])
    metrics.update(cm)
    artifact = {
        "fraud_model": fraud_model,
        "anomaly_model": anomaly_model,
        "training_columns": list(validation.data.columns),
        "metrics": metrics,
    }
    save_artifact(artifact, ARTIFACT_PATH)
    write_json(METRICS_PATH, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RiskLens fraud and anomaly models.")
    parser.add_argument("--data", default=str(SAMPLE_DATA_PATH), help="Training CSV path.")
    args = parser.parse_args()
    result = train(Path(args.data))
    print(result)
