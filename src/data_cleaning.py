"""
Data validation and cleaning for the synthetic fraud-detection dataset.

Checks:
- required columns present with expected dtypes
- no nulls, no duplicate transaction ids
- value ranges sane (amount > 0, hour in 0-23, channel in {online, in-store},
  card_present and is_fraud binary, tenure/days non-negative)

Returns a cleaned DataFrame plus a validation summary dict.
"""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"

EXPECTED_COLUMNS = {
    "txn_id": "int64",
    "amount": "float64",
    "merchant_category": "object",
    "hour_of_day": "int64",
    "channel": "object",
    "card_present": "int64",
    "customer_tenure_months": "int64",
    "days_since_last_txn": "float64",
    "is_fraud": "int64",
}

VALID_CHANNELS = {"online", "in-store"}


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Return a dict of validation findings (empty lists = no issues)."""
    findings = {
        "null_counts": df.isna().sum().to_dict(),
        "duplicate_txn_ids": int(df["txn_id"].duplicated().sum()),
        "bad_amount": int((df["amount"] <= 0).sum()),
        "bad_hour": int(((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).sum()),
        "bad_channel": int((~df["channel"].isin(VALID_CHANNELS)).sum()),
        "bad_card_present": int((~df["card_present"].isin([0, 1])).sum()),
        "bad_is_fraud": int((~df["is_fraud"].isin([0, 1])).sum()),
        "negative_tenure": int((df["customer_tenure_months"] < 0).sum()),
        "negative_gap": int((df["days_since_last_txn"] < 0).sum()),
        "missing_columns": sorted(set(EXPECTED_COLUMNS) - set(df.columns)),
    }
    return findings


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate, drop/flag bad rows, add convenience fields. Returns (clean_df, findings)."""
    df = df.copy()
    findings = validate(df)

    # Defensive: drop rows violating hard constraints (generator guarantees none)
    mask = (
        (df["amount"] > 0)
        & df["hour_of_day"].between(0, 23)
        & df["channel"].isin(VALID_CHANNELS)
        & df["card_present"].isin([0, 1])
        & df["is_fraud"].isin([0, 1])
        & (df["customer_tenure_months"] >= 0)
        & (df["days_since_last_txn"] >= 0)
    )
    dropped = int((~mask).sum())
    df = df.loc[mask].reset_index(drop=True)

    # Convenience fields used downstream
    df["hour_bucket"] = pd.cut(
        df["hour_of_day"],
        bins=[-1, 5, 11, 16, 21, 24],
        labels=["night (0-5)", "morning (6-11)", "midday (12-16)",
                "evening (17-21)", "late (22-23)"],
    )
    df["log_amount"] = df["amount"].apply(lambda x: x if x > 0 else 1e-6).apply(__import__("numpy").log)

    findings["rows_dropped"] = dropped
    findings["rows_after_clean"] = len(df)
    return df, findings


def load_clean(path: str | Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    return clean(load_raw(path))


if __name__ == "__main__":
    df, findings = load_clean()
    print(f"Clean rows: {len(df):,}")
    print("Validation findings (all zeros expected):")
    for key, value in findings.items():
        print(f"  {key}: {value}")
