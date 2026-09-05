from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "amount",
    "merchant_category",
    "hour",
    "day_of_week",
    "transactions_last_24h",
    "avg_customer_amount",
    "is_foreign",
]

OPTIONAL_TARGET = "is_fraud"


@dataclass
class ValidationResult:
    data: pd.DataFrame
    errors: list[str]
    warnings: list[str]
    dropped_rows: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_transactions(df: pd.DataFrame, require_target: bool = False) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    required = REQUIRED_COLUMNS + ([OPTIONAL_TARGET] if require_target else [])
    missing = [column for column in required if column not in df.columns]
    if missing:
        return ValidationResult(df.copy(), [f"Missing required columns: {', '.join(missing)}"], warnings)

    cleaned = df.copy()
    initial_rows = len(cleaned)
    if cleaned.empty:
        return ValidationResult(cleaned, ["CSV contains no transaction rows."], warnings)

    cleaned["transaction_id"] = cleaned["transaction_id"].astype(str).str.strip()
    cleaned["customer_id"] = cleaned["customer_id"].astype(str).str.strip()
    cleaned["merchant_category"] = cleaned["merchant_category"].astype(str).str.strip().str.lower()

    numeric_columns = ["amount", "hour", "day_of_week", "transactions_last_24h", "avg_customer_amount", "is_foreign"]
    if OPTIONAL_TARGET in cleaned.columns:
        numeric_columns.append(OPTIONAL_TARGET)

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    dropped = before_drop - len(cleaned)
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing or invalid required values.")

    invalid_mask = (
        (cleaned["amount"] < 0)
        | (~cleaned["hour"].between(0, 23))
        | (~cleaned["day_of_week"].between(0, 6))
        | (cleaned["transactions_last_24h"] < 0)
        | (cleaned["avg_customer_amount"] < 0)
        | (~cleaned["is_foreign"].isin([0, 1]))
    )
    invalid_rows = int(invalid_mask.sum())
    if invalid_rows:
        cleaned = cleaned.loc[~invalid_mask].copy()
        warnings.append(f"Dropped {invalid_rows} rows with out-of-range transaction values.")

    if OPTIONAL_TARGET in cleaned.columns:
        cleaned = cleaned[cleaned[OPTIONAL_TARGET].isin([0, 1])].copy()

    if cleaned.empty:
        errors.append("No valid transactions remain after validation.")

    return ValidationResult(cleaned.reset_index(drop=True), errors, warnings, initial_rows - len(cleaned))


def expected_columns(include_target: bool = False) -> list[str]:
    return REQUIRED_COLUMNS + ([OPTIONAL_TARGET] if include_target else [])

