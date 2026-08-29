-- ----------------------------------------------------------------------------
-- Step 1: Staging table - apply explicit, documented cleaning decisions
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS stg_online_retail;

CREATE TABLE stg_online_retail AS
WITH source_renamed AS (
    -- Alias the mixed-case, quoted CSV column names to lowercase once, here,
    -- so every downstream step in this pipeline can use plain lowercase names.
    SELECT
        "Invoice"      AS invoice,
        "StockCode"    AS stock_code,
        "Description"  AS description,
        "Quantity"     AS quantity,
        "InvoiceDate"  AS invoice_date,
        "Price"        AS price,
        "Customer ID"  AS customer_id,
        "Country"      AS country
    FROM raw_online_retail
),
dedup AS (
    SELECT DISTINCT ON (invoice, stock_code, quantity, invoice_date, customer_id)
        invoice,
        UPPER(TRIM(stock_code))            AS stock_code,
        TRIM(description)                  AS description,
        quantity,
        invoice_date,
        price,
        customer_id,
        TRIM(country)                      AS country,
        (invoice LIKE 'C%')                AS is_cancellation
    FROM source_renamed
    WHERE quantity <> 0
      AND price > 0
    ORDER BY invoice, stock_code, quantity, invoice_date, customer_id, price DESC
)
SELECT *
FROM dedup
WHERE stock_code NOT IN (
    'POST', 'D', 'DOT', 'M', 'S', 'B', 'AMAZONFEE', 'BANK CHARGES', 'CRUK',
    'C2', 'TEST001', 'TEST002', 'ADJUST', 'ADJUST2'
);

CREATE INDEX idx_stg_stockcode_date ON stg_online_retail (stock_code, invoice_date);
