-- ----------------------------------------------------------------------------
-- Step 3: Category assignment
-- ----------------------------------------------------------------------------

DROP TABLE IF EXISTS dim_product_category;

CREATE TABLE dim_product_category AS
SELECT DISTINCT
    stock_code,
    LEFT(stock_code, 2) AS category_code
FROM stg_net_transactions;
