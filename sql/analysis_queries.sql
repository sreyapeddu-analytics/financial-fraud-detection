-- Financial Fraud Detection — exploratory SQL (SQLite-compatible)
-- Run against data/transactions.csv loaded into a `transactions` table,
-- e.g.:  sqlite3 fraud.db ".import --csv --skip 1 data/transactions.csv transactions"
--
-- Table schema:
--   txn_id INT, amount REAL, merchant_category TEXT, hour_of_day INT,
--   channel TEXT ('online' | 'in-store'), card_present INT (0/1),
--   customer_tenure_months INT, days_since_last_txn REAL, is_fraud INT (0/1)

-- 1. Fraud rate by merchant category ------------------------------------------
SELECT merchant_category,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS fraud_count,
       ROUND(1.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate DESC;

-- 2. Fraud rate by hour bucket -------------------------------------------------
SELECT CASE
         WHEN hour_of_day BETWEEN 0 AND 5  THEN 'night (0-5)'
         WHEN hour_of_day BETWEEN 6 AND 11 THEN 'morning (6-11)'
         WHEN hour_of_day BETWEEN 12 AND 16 THEN 'midday (12-16)'
         WHEN hour_of_day BETWEEN 17 AND 21 THEN 'evening (17-21)'
         ELSE 'late (22-23)'
       END AS hour_bucket,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS fraud_count,
       ROUND(1.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate
FROM transactions
GROUP BY hour_bucket
ORDER BY fraud_rate DESC;

-- 3. Fraud rate by channel ------------------------------------------------------
SELECT channel,
       COUNT(*)                        AS transactions,
       SUM(is_fraud)                   AS fraud_count,
       ROUND(1.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate
FROM transactions
GROUP BY channel
ORDER BY fraud_rate DESC;

-- 4. Average / median-ish amount: fraud vs legit ---------------------------------
SELECT is_fraud,
       COUNT(*)                        AS transactions,
       ROUND(AVG(amount), 2)           AS avg_amount,
       ROUND(MIN(amount), 2)           AS min_amount,
       ROUND(MAX(amount), 2)           AS max_amount
FROM transactions
GROUP BY is_fraud;

-- 5. Rule-flagged transaction counts (mirrors src/model.py business rules) ------
SELECT
  SUM(CASE WHEN amount >= 2000 AND card_present = 0 THEN 1 ELSE 0 END)
      AS r1_high_value_card_not_present,
  SUM(CASE WHEN hour_of_day BETWEEN 0 AND 5 AND amount >= 500 THEN 1 ELSE 0 END)
      AS r2_odd_hour_high_amount,
  SUM(CASE WHEN customer_tenure_months <= 3 AND channel = 'online'
            AND amount >= 800 THEN 1 ELSE 0 END)
      AS r3_new_account_online,
  SUM(CASE WHEN days_since_last_txn < 0.1 AND amount >= 1500 THEN 1 ELSE 0 END)
      AS r4_rapid_fire_high_amount,
  SUM(CASE WHEN (amount >= 2000 AND card_present = 0)
              OR (hour_of_day BETWEEN 0 AND 5 AND amount >= 500)
              OR (customer_tenure_months <= 3 AND channel = 'online' AND amount >= 800)
              OR (days_since_last_txn < 0.1 AND amount >= 1500)
           THEN 1 ELSE 0 END)
      AS any_rule_flagged,
  COUNT(*) AS total_transactions
FROM transactions;
