-- ----------------------------------------------------------------------------
-- Step 4: Daily demand per SKU (core modeling table)
-- ----------------------------------------------------------------------------

DROP TABLE IF EXISTS fact_demand_daily;

CREATE TABLE fact_demand_daily AS
SELECT
    t.stock_code,
    MAX(t.description)         AS description,
    c.category_code,
    t.sale_date,
    GREATEST(SUM(t.net_quantity), 0)                                       AS units_sold,
    SUM(t.net_revenue)         AS revenue,
    COUNT(DISTINCT t.customer_id) FILTER (WHERE t.customer_id IS NOT NULL) AS distinct_customers,
    AVG(t.avg_price)           AS avg_unit_price
FROM stg_net_transactions t
JOIN dim_product_category c ON c.stock_code = t.stock_code
GROUP BY t.stock_code, c.category_code, t.sale_date;

CREATE INDEX idx_fact_stockcode_date ON fact_demand_daily (stock_code, sale_date);
