-- ============================================================
-- 1. Fraud rate by hour-of-day, with a window function comparing each
--    hour's fraud rate to the overall average
--
--    Recreates EDA.ipynb's hour derivation exactly (cell: "Time analysis" -
--    df["Hour"] = (df["Time"]/3600) % 24), independently, in SQL. Oracle's
--    FLOOR(MOD(x, 24)) on time_seconds/3600 gives the identical integer
--    hour pandas' int((Time/3600) % 24) does - verified against several
--    boundary values (0, 3599, 3600, 86399, values that wrap past 24h)
--    before trusting this query. The notebook only plots hour vs count
--    separately for fraud and legitimate transactions; this adds the
--    actual rate and the vs-overall-average comparison on top.
-- ============================================================
WITH hourly AS (
    SELECT
        FLOOR(MOD(time_seconds / 3600, 24))    AS hour_of_day,
        class
    FROM transactions_raw
),
hourly_rates AS (
    SELECT
        hour_of_day,
        COUNT(*)                                       AS transactions,
        SUM(class)                                     AS fraud_count,
        AVG(class)                                      AS fraud_rate
    FROM hourly
    GROUP BY hour_of_day
)
SELECT
    hour_of_day,
    transactions,
    fraud_count,
    ROUND(fraud_rate * 100, 4)                                              AS fraud_rate_pct,
    ROUND(AVG(fraud_rate) OVER () * 100, 4)                                 AS overall_fraud_rate_pct,
    ROUND((fraud_rate - AVG(fraud_rate) OVER ()) * 100, 4)                  AS pct_pts_vs_overall
FROM hourly_rates
ORDER BY hour_of_day;


-- ============================================================
-- 2. Fraud rate by transaction amount bucket
--
--    EDA.ipynb only plots the Amount distribution as a histogram/boxplot
--    (no discrete buckets or rates) - these bucket boundaries are new
--    analysis, not a recreation of an existing notebook cell. Boundaries
--    chosen to separate micro-transactions (often used to test a stolen
--    card before a larger charge) from normal and high-value amounts.
-- ============================================================
WITH bucketed AS (
    SELECT
        CASE
            WHEN amount < 1    THEN '01: under $1'
            WHEN amount < 10   THEN '02: $1-10'
            WHEN amount < 50   THEN '03: $10-50'
            WHEN amount < 100  THEN '04: $50-100'
            WHEN amount < 500  THEN '05: $100-500'
            ELSE                    '06: $500+'
        END AS amount_bucket,
        class
    FROM transactions_raw
),
bucket_rates AS (
    SELECT
        amount_bucket,
        COUNT(*)                AS transactions,
        SUM(class)              AS fraud_count,
        AVG(class)               AS fraud_rate
    FROM bucketed
    GROUP BY amount_bucket
)
SELECT
    amount_bucket,
    transactions,
    fraud_count,
    ROUND(fraud_rate * 100, 4)                                             AS fraud_rate_pct,
    ROUND(AVG(fraud_rate) OVER () * 100, 4)                                AS overall_fraud_rate_pct,
    ROUND((fraud_rate - AVG(fraud_rate) OVER ()) * 100, 4)                 AS pct_pts_vs_overall
FROM bucket_rates
ORDER BY amount_bucket;


-- ============================================================
-- 3. The 3 highest-risk hours, cross-referenced against the 3
--    highest-risk amount buckets, using a single combined query -
--    is fraud concentrated at the intersection of both, or are the two
--    risk factors independent of each other?
-- ============================================================
WITH combined AS (
    SELECT
        FLOOR(MOD(time_seconds / 3600, 24))    AS hour_of_day,
        CASE
            WHEN amount < 1    THEN '01: under $1'
            WHEN amount < 10   THEN '02: $1-10'
            WHEN amount < 50   THEN '03: $10-50'
            WHEN amount < 100  THEN '04: $50-100'
            WHEN amount < 500  THEN '05: $100-500'
            ELSE                    '06: $500+'
        END AS amount_bucket,
        class
    FROM transactions_raw
)
SELECT
    hour_of_day,
    amount_bucket,
    COUNT(*)                                       AS transactions,
    SUM(class)                                     AS fraud_count,
    ROUND(AVG(class) * 100, 4)                     AS fraud_rate_pct
FROM combined
GROUP BY hour_of_day, amount_bucket
HAVING COUNT(*) >= 30   -- drop near-empty cells; a 100% fraud rate on 2 transactions isn't a signal
ORDER BY fraud_rate_pct DESC
FETCH FIRST 10 ROWS ONLY;
