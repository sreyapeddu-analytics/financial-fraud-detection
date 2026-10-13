"""
Exploratory analysis helpers: fraud rate slices and amount distributions.

All functions take the cleaned DataFrame (see data_cleaning.py) and return
small summary tables. `run_all()` prints a compact EDA report; results are
also reused by the notebook.
"""
import pandas as pd

from data_cleaning import load_clean


def fraud_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Transactions, fraud count, and fraud rate grouped by `column`."""
    g = (
        df.groupby(column, observed=True)
        .agg(transactions=("txn_id", "count"), fraud_count=("is_fraud", "sum"))
        .reset_index()
    )
    g["fraud_rate"] = g["fraud_count"] / g["transactions"]
    return g.sort_values("fraud_rate", ascending=False).reset_index(drop=True)


def amount_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats of amount for fraud vs legit."""
    return (
        df.groupby("is_fraud")["amount"]
        .describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        .T.round(2)
    )


def channel_x_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate pivot: hour bucket x channel."""
    return (
        df.pivot_table(
            index="hour_bucket", columns="channel", values="is_fraud",
            aggfunc="mean", observed=True,
        ).round(4)
    )


def top_risk_combinations(df: pd.DataFrame, min_txns: int = 200) -> pd.DataFrame:
    """Fraud rate for (channel, hour_bucket, card_present) slices with enough volume."""
    g = (
        df.groupby(["channel", "hour_bucket", "card_present"], observed=True)
        .agg(transactions=("txn_id", "count"), fraud_count=("is_fraud", "sum"))
        .reset_index()
    )
    g = g[g["transactions"] >= min_txns].copy()
    g["fraud_rate"] = g["fraud_count"] / g["transactions"]
    return g.sort_values("fraud_rate", ascending=False).reset_index(drop=True)


def run_all(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df, _ = load_clean()

    print(f"Transactions: {len(df):,} | Fraud: {int(df['is_fraud'].sum()):,} "
          f"({df['is_fraud'].mean() * 100:.2f}%)\n")

    results = {}
    for label, col in [
        ("merchant_category", "merchant_category"),
        ("channel", "channel"),
        ("hour_bucket", "hour_bucket"),
        ("card_present", "card_present"),
    ]:
        tbl = fraud_rate_by(df, col)
        results[label] = tbl
        print(f"--- Fraud rate by {col} ---")
        print(tbl.to_string(index=False))
        print()

    stats = amount_distribution_stats(df)
    results["amount_stats"] = stats
    print("--- Amount distribution: legit (0) vs fraud (1) ---")
    print(stats.to_string())
    print()

    combos = top_risk_combinations(df)
    results["top_risk_combinations"] = combos
    print("--- Highest fraud-rate slices (channel x hour bucket x card present) ---")
    print(combos.head(8).to_string(index=False))
    return results


if __name__ == "__main__":
    run_all()
