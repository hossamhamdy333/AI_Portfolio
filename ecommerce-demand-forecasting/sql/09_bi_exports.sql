-- ----------------------------------------------------------------------------
-- Step 8: BI exports - date dimension + flat exports for Power BI
-- ----------------------------------------------------------------------------
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

DROP TABLE IF EXISTS bi_actuals_export_tbl;
CREATE TABLE bi_actuals_export_tbl AS
SELECT stock_code, description, category_code, sale_date, units_sold, revenue, avg_unit_price
FROM fact_demand_daily_filled;

SELECT COUNT(*) AS dim_date_rows,
       MAX(date) - MIN(date) + 1 AS expected_rows
FROM dim_date;
