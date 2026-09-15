-- SQL*Loader control file - Oracle's native bulk CSV import tool, the
-- direct equivalent of the \copy command the Postgres projects in this
-- portfolio use via pgAdmin's import wizard. Run with:
--   sqlldr userid=<user>/<password>@<connect_string> control=transactions_raw.ctl log=load.log
--
-- Expects data/creditcard.csv with its original header row
-- (Time,V1,V2,...,V28,Amount,Class) sitting next to this file, or edit
-- the INFILE path below to point at it directly.

LOAD DATA
INFILE 'data/creditcard.csv'
APPEND
INTO TABLE transactions_raw
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
SKIP 1                                  -- the CSV's own header row (Time,V1,V2,...,Amount,Class)
(
    time_seconds,
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26, v27, v28,
    amount,
    class
)
