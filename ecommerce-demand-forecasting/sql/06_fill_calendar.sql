-- ----------------------------------------------------------------------------
-- Step 5: Fill calendar gaps (a SKU with zero sales on a given day is a real
-- demand signal - a "0" - not a missing row. 
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS fact_demand_daily_filled;

CREATE TABLE fact_demand_daily_filled AS
WITH sku_bounds AS (
    -- Each SKU's own observed lifetime, not the dataset-wide range.
    SELECT stock_code, description, category_code,
           MIN(sale_date) AS first_sale_date,
           MAX(sale_date) AS last_sale_date
    FROM fact_demand_daily
    GROUP BY stock_code, description, category_code
),
sku_calendar AS (
    -- One calendar row per SKU per day, but only within THAT SKU's active window.
    SELECT
        b.stock_code, b.description, b.category_code,
        generate_series(b.first_sale_date, b.last_sale_date, interval '1 day')::date AS cal_date
    FROM sku_bounds b
)
SELECT
    s.stock_code,
    s.description,
    s.category_code,
    s.cal_date               AS sale_date,
    COALESCE(f.units_sold, 0)      AS units_sold,
    COALESCE(f.revenue, 0)         AS revenue,
    COALESCE(f.distinct_customers, 0) AS distinct_customers,
    f.avg_unit_price          -- left NULL on zero-demand days deliberately;
                               -- imputing a fake price would be worse than a null
FROM sku_calendar s
LEFT JOIN fact_demand_daily f
    ON f.stock_code = s.stock_code AND f.sale_date = s.cal_date;
