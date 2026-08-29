-- ----------------------------------------------------------------------------
-- Step -1: Create the raw table before loading data/raw/online_retail_combined.csv
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS raw_online_retail;

CREATE TABLE raw_online_retail (
    "Invoice"      TEXT,
    "StockCode"    TEXT,
    "Description"  TEXT,
    "Quantity"     INTEGER,
    "InvoiceDate"  TIMESTAMP,
    "Price"        NUMERIC,
    "Customer ID"  NUMERIC,
    "Country"      TEXT
);


