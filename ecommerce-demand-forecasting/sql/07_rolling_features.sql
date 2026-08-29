-- ----------------------------------------------------------------------------
-- Step 6: Window functions - rolling demand signals + category ranking
-- ----------------------------------------------------------------------------

DROP TABLE IF EXISTS fact_demand_features;

CREATE TABLE fact_demand_features AS
SELECT
    stock_code,
    description,
    category_code,
    sale_date,
    units_sold,
    revenue,
    distinct_customers,
    avg_unit_price,

    -- Rolling averages computed causally: window EXCLUDES the current row
    -- (1 PRECEDING as the end bound, not CURRENT ROW). A window that includes
    -- the current row's own units_sold would mean the "feature" contains the
    -- target itself - direct leakage, not a subtle one, and it would silently
    -- inflate every backtest that used these columns as model inputs.
    AVG(units_sold) OVER (
        PARTITION BY stock_code ORDER BY sale_date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_7d_avg_units,

    AVG(units_sold) OVER (
        PARTITION BY stock_code ORDER BY sale_date
        ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS rolling_28d_avg_units,

    STDDEV(units_sold) OVER (
        PARTITION BY stock_code ORDER BY sale_date
        ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS rolling_28d_std_units,

    -- Week-over-week growth vs. same weekday last week. Reporting/EDA column
    -- only - uses the CURRENT day's units_sold by design, so this is NOT
    -- causal and must never be added to FEATURE_COLUMNS in src/features.py.
    -- Using it as a model input would be direct leakage (same-day target
    -- value feeding the model that predicts that same value).
    units_sold - LAG(units_sold, 7) OVER (
        PARTITION BY stock_code ORDER BY sale_date
    ) AS wow_change_units,

    -- Revenue rank within category, per day (identifies top movers).
    -- Same caveat as wow_change_units: computed from CURRENT day's revenue,
    -- reporting/EDA use only, never a model feature.
    RANK() OVER (
        PARTITION BY category_code, sale_date ORDER BY revenue DESC
    ) AS category_revenue_rank_today

FROM fact_demand_daily_filled;
