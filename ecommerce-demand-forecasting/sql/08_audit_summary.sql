-- ----------------------------------------------------------------------------
-- Step 7: Data-quality audit summary (paste this table into your README)
-- ----------------------------------------------------------------------------

SELECT
    (SELECT COUNT(*) FROM raw_online_retail)                                  AS raw_rows,
    (SELECT COUNT(*) FROM stg_online_retail)                                  AS after_cleaning,
    (SELECT COUNT(*) FROM stg_online_retail WHERE is_cancellation)            AS cancellation_rows,
    (SELECT COUNT(*) FROM stg_online_retail WHERE customer_id IS NULL)        AS null_customer_rows,
    (SELECT COUNT(DISTINCT stock_code) FROM fact_demand_daily_filled)         AS distinct_skus,
    (SELECT MIN(sale_date) FROM fact_demand_daily_filled)                     AS date_range_start,
    (SELECT MAX(sale_date) FROM fact_demand_daily_filled)                     AS date_range_end,
    (SELECT COUNT(*) FROM fact_demand_daily_filled)                           AS filled_rows_per_sku_lifetime;

-- Sanity check: what would the OLD (buggy, global-range) fill have produced,
-- for comparison - run this once to see how many ghost rows the fix avoided.
-- Not part of the pipeline, just a one-time diagnostic.
WITH date_bounds AS (
    SELECT MIN(sale_date) AS min_date, MAX(sale_date) AS max_date FROM fact_demand_daily
),
old_style_row_count AS (
    SELECT COUNT(DISTINCT stock_code) * (
        SELECT (max_date - min_date) + 1 FROM date_bounds
    ) AS would_be_rows
    FROM fact_demand_daily
)
SELECT
    (SELECT would_be_rows FROM old_style_row_count) AS old_buggy_fill_rowcount,
    (SELECT COUNT(*) FROM fact_demand_daily_filled) AS fixed_per_sku_fill_rowcount,
    (SELECT would_be_rows FROM old_style_row_count) - (SELECT COUNT(*) FROM fact_demand_daily_filled) AS ghost_rows_avoided;

-- Net-demand clipping check: how many (stock_code, date) rows had negative
-- net_quantity before the GREATEST(..., 0) clip in 05_daily_demand.sql -
-- i.e. how often a cancellation landed on a different day than its original
-- sale. Worth reporting alongside the ghost-rows number above, since both
-- are "how much did a documented cleaning decision actually change."
SELECT
    COUNT(*) FILTER (WHERE net_quantity < 0) AS negative_net_quantity_rows,
    COUNT(*)                                 AS total_stg_net_transaction_rows,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE net_quantity < 0) / NULLIF(COUNT(*), 0), 2
    ) AS pct_negative
FROM stg_net_transactions;


-- ----------------------------------------------------------------------------
-- Export for Phase 2 (Python feature engineering / modeling):
--   \copy fact_demand_features TO 'data/processed/demand_features.csv' WITH CSV HEADER;
-- ----------------------------------------------------------------------------
