from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, SAMPLE_DATA_PATH


def create_demo_data(rows: int = 900, path=SAMPLE_DATA_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    categories = np.array(["grocery", "fuel", "travel", "electronics", "gaming", "jewelry", "utility", "restaurant"])
    customer_avg = rng.lognormal(mean=6.4, sigma=0.55, size=rows).round(2)
    amount = rng.lognormal(mean=6.2, sigma=0.75, size=rows).round(2)
    high_risk_idx = rng.choice(rows, size=max(35, rows // 12), replace=False)
    amount[high_risk_idx] *= rng.uniform(3.5, 10.0, size=len(high_risk_idx))
    hours = rng.integers(7, 23, size=rows)
    night_idx = rng.choice(rows, size=rows // 7, replace=False)
    hours[night_idx] = rng.integers(0, 5, size=len(night_idx))
    foreign = rng.binomial(1, 0.12, rows)
    velocity = rng.poisson(2.2, rows)
    velocity[high_risk_idx] += rng.integers(4, 12, size=len(high_risk_idx))
    merchant = rng.choice(categories, size=rows, p=[0.22, 0.13, 0.1, 0.12, 0.1, 0.07, 0.16, 0.1])

    ratio = amount / np.maximum(customer_avg, 1)
    fraud_logit = (
        -4.3
        + 0.48 * np.clip(ratio - 1, 0, 8)
        + 0.28 * velocity
        + 0.8 * foreign
        + 0.75 * np.isin(hours, [0, 1, 2, 3, 4])
        + 0.9 * np.isin(merchant, ["jewelry", "gaming", "electronics"])
    )
    probability = 1 / (1 + np.exp(-fraud_logit))
    fraud = rng.binomial(1, np.clip(probability, 0.01, 0.92))

    df = pd.DataFrame({
        "transaction_id": [f"TXN-{10000 + i}" for i in range(rows)],
        "customer_id": [f"CUST-{rng.integers(1000, 1125)}" for _ in range(rows)],
        "amount": amount.round(2),
        "merchant_category": merchant,
        "hour": hours,
        "day_of_week": rng.integers(0, 7, size=rows),
        "transactions_last_24h": velocity,
        "avg_customer_amount": customer_avg,
        "is_foreign": foreign,
        "is_fraud": fraud,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = create_demo_data()
    print(f"Wrote {len(data)} synthetic transactions to {SAMPLE_DATA_PATH}")
