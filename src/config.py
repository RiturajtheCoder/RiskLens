from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "synthetic_transactions.csv"
MODEL_DIR = ROOT_DIR / "models"
ARTIFACT_PATH = MODEL_DIR / "risklens_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42


@dataclass(frozen=True)
class RiskThresholds:
    low_max: int = 30
    medium_max: int = 70
    review_min: int = 31
    hold_min: int = 71
    block_min: int = 90


DEFAULT_THRESHOLDS = RiskThresholds()

