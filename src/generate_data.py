"""
Synthetic credit-card transaction generator for the financial fraud detection
portfolio project.

- 20,000 transactions, ~1.5% labeled fraud
- Seeded (random_state=42) so results are fully reproducible
- Fraud is engineered to correlate with: high amounts, odd hours (0-5),
  online channel, and card-not-present transactions
- NO real data of any kind is used; everything is simulated for portfolio use
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_TRANSACTIONS = 20_000
TARGET_FRAUD_RATE = 0.015

MERCHANT_CATEGORIES = [
    "grocery", "gas", "restaurant", "retail", "travel",
    "online_marketplace", "electronics", "healthcare",
]

# Baseline mix of merchant categories for legitimate traffic
BASE_CATEGORY_PROBS = np.array([0.24, 0.14, 0.18, 0.16, 0.06, 0.08, 0.06, 0.08])
# Fraud skews toward categories that are easy to monetize quickly
FRAUD_CATEGORY_PROBS = np.array([0.03, 0.02, 0.05, 0.22, 0.16, 0.26, 0.20, 0.06])


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    txn_id = np.arange(1, N_TRANSACTIONS + 1)

    is_fraud = rng.random(N_TRANSACTIONS) < TARGET_FRAUD_RATE

    # ---- amount: legit ~ lognormal centered low, fraud shifted much higher ----
    legit_amount = rng.lognormal(mean=3.6, sigma=0.9, size=N_TRANSACTIONS)
    fraud_amount = rng.lognormal(mean=5.6, sigma=0.8, size=N_TRANSACTIONS)
    amount = np.where(is_fraud, fraud_amount, legit_amount).round(2)
    amount = np.clip(amount, 1.0, 25_000.0)

    # ---- hour of day: legit clusters in daytime/evening, fraud clusters 0-5 ----
    legit_hour_probs = np.array(
        [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.06,
         0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.06, 0.05, 0.03,
         0.02, 0.01]
    )
    legit_hour_probs /= legit_hour_probs.sum()
    fraud_hour_probs = np.array(
        [0.16, 0.16, 0.14, 0.12, 0.10, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02,
         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
         0.02, 0.02]
    )
    fraud_hour_probs /= fraud_hour_probs.sum()
    hour_of_day = np.where(
        is_fraud,
        rng.choice(24, size=N_TRANSACTIONS, p=fraud_hour_probs),
        rng.choice(24, size=N_TRANSACTIONS, p=legit_hour_probs),
    )

    # ---- channel: fraud is overwhelmingly online ----
    channel = np.where(
        is_fraud,
        rng.choice(["online", "in-store"], size=N_TRANSACTIONS, p=[0.82, 0.18]),
        rng.choice(["online", "in-store"], size=N_TRANSACTIONS, p=[0.35, 0.65]),
    )

    # ---- card_present: fraud is overwhelmingly card-not-present ----
    card_present = np.where(
        is_fraud,
        rng.choice([0, 1], size=N_TRANSACTIONS, p=[0.88, 0.12]),
        rng.choice([0, 1], size=N_TRANSACTIONS, p=[0.25, 0.75]),
    )

    # ---- customer tenure: fraud skews toward newer accounts ----
    legit_tenure = rng.integers(1, 121, size=N_TRANSACTIONS)
    fraud_tenure = rng.integers(1, 25, size=N_TRANSACTIONS)
    customer_tenure_months = np.where(is_fraud, fraud_tenure, legit_tenure)

    # ---- days since last transaction: fraud more often rapid-fire ----
    legit_gap = rng.exponential(scale=3.0, size=N_TRANSACTIONS)
    fraud_gap = rng.exponential(scale=0.4, size=N_TRANSACTIONS)
    days_since_last_txn = np.where(is_fraud, fraud_gap, legit_gap).round(2)
    days_since_last_txn = np.clip(days_since_last_txn, 0.0, 60.0)

    # ---- merchant category ----
    merchant_category = np.where(
        is_fraud,
        rng.choice(MERCHANT_CATEGORIES, size=N_TRANSACTIONS, p=FRAUD_CATEGORY_PROBS),
        rng.choice(MERCHANT_CATEGORIES, size=N_TRANSACTIONS, p=BASE_CATEGORY_PROBS),
    )

    df = pd.DataFrame(
        {
            "txn_id": txn_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "hour_of_day": hour_of_day,
            "channel": channel,
            "card_present": card_present.astype(int),
            "customer_tenure_months": customer_tenure_months.astype(int),
            "days_since_last_txn": days_since_last_txn,
            "is_fraud": is_fraud.astype(int),
        }
    )

    # Shuffle so fraud labels are not ordered, then sort back by txn_id for readability
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).sort_values("txn_id").reset_index(drop=True)

    out = "data/transactions.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} transactions to {out}")
    print(f"Fraud count: {int(df['is_fraud'].sum()):,} "
          f"({df['is_fraud'].mean() * 100:.2f}%)")
    print(f"Fraud by channel:\n{df.groupby('channel')['is_fraud'].mean().round(4)}")


if __name__ == "__main__":
    main()
