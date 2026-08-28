-- ----------------------------------------------------------------------------
-- Step 2: Net demand per line item - offset cancellations against original sales
-- ----------------------------------------------------------------------------

DROP TABLE IF EXISTS stg_net_transactions;

CREATE TABLE stg_net_transactions AS
SELECT
    stock_code,
    MAX(description)                       AS description,  
    DATE(invoice_date)                      AS sale_date,
    customer_id,
    country,
    SUM(quantity)                           AS net_quantity,
    SUM(quantity * price)                   AS net_revenue,
    AVG(price)                              AS avg_price
FROM stg_online_retail
GROUP BY stock_code, DATE(invoice_date), customer_id, country;
