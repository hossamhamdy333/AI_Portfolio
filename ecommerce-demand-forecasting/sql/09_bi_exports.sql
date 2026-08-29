-- ----------------------------------------------------------------------------
-- Step 8: BI exports - date dimension + flat exports for Power BI
-- ----------------------------------------------------------------------------
-- Power BI's time-intelligence DAX functions (SAMEPERIODLASTYEAR, TOTALYTD,
-- DATEADD, etc.) need a proper contiguous date dimension table marked as a
-- Date Table in the model - they don't work reliably against a date column
-- that only has the dates present in the fact table (gaps break them).
--
-- Requires: fact_demand_daily_filled (06_fill_calendar.sql)

DROP TABLE IF EXISTS dim_date;

CREATE TABLE dim_date AS
SELECT
    d::date                                AS date,
    EXTRACT(YEAR FROM d)::int              AS year,
    EXTRACT(MONTH FROM d)::int             AS month,
    TRIM(TO_CHAR(d, 'Month'))              AS month_name,
    EXTRACT(QUARTER FROM d)::int           AS quarter,
    EXTRACT(WEEK FROM d)::int              AS iso_week,
    EXTRACT(ISODOW FROM d)::int            AS day_of_week,      -- 1=Mon .. 7=Sun
    TRIM(TO_CHAR(d, 'Day'))                AS day_name,
    (EXTRACT(ISODOW FROM d) IN (6, 7))     AS is_weekend,
    TO_CHAR(d, 'YYYY-"W"IW')               AS year_week,
    TO_CHAR(d, 'YYYY-MM')                  AS year_month
FROM generate_series(
    (SELECT MIN(sale_date) FROM fact_demand_daily_filled),
    (SELECT MAX(sale_date) FROM fact_demand_daily_filled),
    interval '1 day'
) AS d;

-- ----------------------------------------------------------------------------
-- BI-ready fact export: full history at (stock_code, date) grain, actuals
-- only (no model columns - those come from the Python-side export, see
-- bi_forecast_export.csv in 03_forecasting_models.ipynb). Kept separate
-- from fact_demand_features so Power BI isn't loading the 25 model feature
-- columns it doesn't need.
-- ----------------------------------------------------------------------------

-- pgAdmin users: this pgAdmin build doesn't show "Import/Export Data" on
-- views (only tables), and the grid download button hangs on 2M+ rows.
-- So this is a TABLE, not a view - right-click it in the tree same as
-- dim_date/dim_product_category above -> Import/Export Data -> Export ->
-- CSV. That path is fast at any row count since it's server-side.
DROP TABLE IF EXISTS bi_actuals_export_tbl;
CREATE TABLE bi_actuals_export_tbl AS
SELECT stock_code, description, category_code, sale_date, units_sold, revenue, avg_unit_price
FROM fact_demand_daily_filled;

-- psql users: uncomment and run this line directly (\copy is psql-only,
-- it will NOT run inside pgAdmin's Query Tool)
-- \copy (SELECT * FROM bi_actuals_export_tbl) TO 'data/processed/bi_actuals_export.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- BI-ready dimension exports
-- ----------------------------------------------------------------------------
-- \copy dim_product_category TO 'data/processed/bi_dim_category.csv' WITH CSV HEADER;
-- \copy dim_date TO 'data/processed/bi_dim_date.csv' WITH CSV HEADER;

-- Sanity check: date dim should have zero gaps
SELECT COUNT(*) AS dim_date_rows,
       MAX(date) - MIN(date) + 1 AS expected_rows
FROM dim_date;
