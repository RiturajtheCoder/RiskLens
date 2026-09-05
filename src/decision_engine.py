from __future__ import annotations

from src.config import DEFAULT_THRESHOLDS, RiskThresholds


def classify_risk(score: float, thresholds: RiskThresholds = DEFAULT_THRESHOLDS) -> str:
    if score <= thresholds.low_max:
        return "LOW"
    if score <= thresholds.medium_max:
        return "MEDIUM"
    return "HIGH"


def recommend_action(score: float, thresholds: RiskThresholds = DEFAULT_THRESHOLDS) -> tuple[str, str]:
    if score >= thresholds.block_min:
        return "BLOCK", "Critical risk score exceeds the block threshold."
    if score >= thresholds.hold_min:
        return "HOLD FOR MANUAL REVIEW", "High risk requires analyst review before approval."
    if score >= thresholds.review_min:
        return "REVIEW / MONITOR", "Moderate risk signals require additional monitoring."
    return "APPROVE", "Risk signals are within the low-risk approval range."

