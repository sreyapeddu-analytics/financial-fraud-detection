# Fraud Operations Dashboard — KPI definitions

A fraud-operations dashboard for this project would track these KPIs. The
charts in `../images/` and the notebook cover the same measures, and the SQL
in `../sql/analysis_queries.sql` powers the underlying slices.

## KPIs

| KPI | Definition | Why it matters |
|---|---|---|
| **Fraud rate** | Confirmed fraud transactions ÷ total transactions | Baseline prevalence; expected ~1.5% in this synthetic data |
| **Detection rate (recall)** | Fraud caught by the review queue ÷ total fraud | Share of fraud the system actually surfaces |
| **Precision at review threshold** | Fraud in the review queue ÷ queue size | Investigator efficiency — higher precision = fewer wasted reviews |
| **Investigation queue size** | Transactions with risk_score ≥ threshold | Workload planning for the fraud-ops team |
| **Rule hit rate** | Transactions triggering ≥1 business rule ÷ total | Coverage of the transparent rule layer |
| **Median time-to-flag** | (Operational metric — not in this dataset) | How quickly a fraud pattern is caught after it starts |

## Suggested dashboard layout

1. **Overview row:** fraud rate, detection rate, precision, queue size (scorecards)
2. **Fraud by hour:** bar chart of fraud rate across 24 hours
3. **Amount distribution:** fraud vs legit amount histograms
4. **Threshold trade-off:** precision/recall/F1 across candidate thresholds
5. **Risk factors:** top contributing features from the risk model
6. **Queue table:** highest-risk transactions with triggered rules and score
