"""
Fraud detection: rule-based flags + unsupervised anomaly detection.

Detection itself is label-free (rules + IsolationForest), mirroring how fraud
teams operate before labels exist. Labels are used ONLY for evaluation, since
this synthetic dataset ships with ground truth for validation.

Risk score:  risk_score = 0.6 * (rule_flags_hit / 4) + 0.4 * isolation_anomaly
where isolation_anomaly is the IsolationForest anomaly score min-max scaled
to [0, 1]. Transactions scoring above a threshold join the review queue.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from data_cleaning import load_clean

RULE_WEIGHT = 0.6
IFOREST_WEIGHT = 0.4
IFOREST_CONTAMINATION = 0.02  # expected anomaly share, slightly above fraud rate
THRESHOLDS = [0.3, 0.5, 0.7]


# ---------------------------------------------------------------- features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_amount"] = np.log(df["amount"].clip(lower=1e-6))
    X["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    X["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    X["channel_online"] = (df["channel"] == "online").astype(int)
    X["card_not_present"] = 1 - df["card_present"]
    X["log_tenure"] = np.log1p(df["customer_tenure_months"])
    X["log_gap"] = np.log1p(df["days_since_last_txn"])
    dummies = pd.get_dummies(df["merchant_category"], prefix="mcc", dtype=int)
    return pd.concat([X, dummies], axis=1)


# ------------------------------------------------------------------- rules
def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Return one binary column per business rule."""
    rules = pd.DataFrame(index=df.index)
    rules["R1_high_value_card_not_present"] = (
        (df["amount"] >= 2000) & (df["card_present"] == 0)
    ).astype(int)
    rules["R2_odd_hour_high_amount"] = (
        df["hour_of_day"].between(0, 5) & (df["amount"] >= 500)
    ).astype(int)
    rules["R3_new_account_online"] = (
        (df["customer_tenure_months"] <= 3)
        & (df["channel"] == "online")
        & (df["amount"] >= 800)
    ).astype(int)
    rules["R4_rapid_fire_high_amount"] = (
        (df["days_since_last_txn"] < 0.1) & (df["amount"] >= 1500)
    ).astype(int)
    return rules


# ------------------------------------------------------------------- model
def fit_isolation_forest(X: pd.DataFrame) -> tuple[IsolationForest, np.ndarray]:
    """Fit IsolationForest (unsupervised) and return model + [0,1] anomaly scores."""
    model = IsolationForest(
        n_estimators=200,
        contamination=IFOREST_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    raw = -model.decision_function(X)  # higher = more anomalous
    scaled = (raw - raw.min()) / (raw.max() - raw.min())
    return model, scaled


def risk_score(rules: pd.DataFrame, iforest_anomaly: np.ndarray) -> pd.Series:
    rule_component = rules.sum(axis=1) / rules.shape[1]
    score = RULE_WEIGHT * rule_component + IFOREST_WEIGHT * pd.Series(
        iforest_anomaly, index=rules.index
    )
    return score.clip(0, 1)


def evaluate_at_thresholds(
    y_true: pd.Series, score: pd.Series, thresholds: list[float] = THRESHOLDS
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = (score >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "flagged": int(pred.sum()),
                "flagged_pct": round(100 * pred.mean(), 2),
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
            }
        )
    return pd.DataFrame(rows)


def top_risk_factors(
    X: pd.DataFrame, y: pd.Series, top_n: int = 8
) -> pd.DataFrame:
    """Supervised logistic regression used ONLY to rank/interpret risk factors."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xs, y)
    coefs = pd.DataFrame(
        {"feature": X.columns, "coefficient": lr.coef_[0]}
    ).sort_values("coefficient", ascending=False)
    return coefs.head(top_n).reset_index(drop=True)


def run() -> dict:
    df, _ = load_clean()
    y = df["is_fraud"]

    X = engineer_features(df)
    rules = apply_rules(df)
    _, iforest_anomaly = fit_isolation_forest(X)
    score = risk_score(rules, iforest_anomaly)

    metrics = evaluate_at_thresholds(y, score)
    factors = top_risk_factors(X, y)

    print(f"Transactions: {len(df):,} | Fraud: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    print(f"\nRule coverage (share of txns hitting each rule):")
    print((rules.mean() * 100).round(2).to_string())
    print("\n--- Precision / Recall / F1 by risk-score threshold ---")
    print(metrics.to_string(index=False))
    print("\n--- Top risk factors (logistic regression, interpretation only) ---")
    print(factors.to_string(index=False))

    return {"df": df, "X": X, "rules": rules, "score": score,
            "metrics": metrics, "factors": factors}


if __name__ == "__main__":
    run()
