from __future__ import annotations

import pandas as pd


def explain_transaction(row: pd.Series, feature_contributions: dict[str, float] | None = None, limit: int = 3) -> list[str]:
    reasons: list[tuple[str, float]] = []

    if row.get("amount_to_customer_avg", 1) >= 3:
        reasons.append(("Transaction amount is far above the customer's usual amount.", float(row["amount_to_customer_avg"])))
    if row.get("transactions_last_24h", 0) >= 6:
        reasons.append(("High transaction frequency in the last 24 hours.", float(row["transactions_last_24h"])))
    if row.get("anomaly_score", 0) >= 0.65:
        reasons.append(("Pattern is unusual compared with normal transaction behavior.", float(row["anomaly_score"])))
    if row.get("fraud_probability", 0) >= 0.55:
        reasons.append(("Fraud model assigns an elevated fraud probability.", float(row["fraud_probability"])))
    if row.get("is_foreign", 0) == 1:
        reasons.append(("Foreign transaction adds contextual risk.", 0.25))
    if row.get("is_night", 0) == 1:
        reasons.append(("Transaction occurred during unusual night-time hours.", 0.2))

    if feature_contributions:
        for feature, value in sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]:
            if value > 0:
                reasons.append((f"Model feature contribution from {feature.replace('_', ' ')} increased risk.", abs(value)))

    if not reasons:
        reasons.append(("No major risk driver exceeded configured alert thresholds.", 0.0))

    unique: list[str] = []
    seen: set[str] = set()
    for text, _ in sorted(reasons, key=lambda item: item[1], reverse=True):
        if text not in seen:
            unique.append(text)
            seen.add(text)
        if len(unique) >= limit:
            break
    return unique


def attach_explanations(report: pd.DataFrame) -> pd.DataFrame:
    enriched = report.copy()
    enriched["top_risk_factors"] = enriched.apply(lambda row: " | ".join(explain_transaction(row)), axis=1)
    return enriched

