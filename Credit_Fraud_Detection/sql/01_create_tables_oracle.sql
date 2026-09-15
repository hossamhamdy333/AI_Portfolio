-- Schema matches data/creditcard.csv exactly (see notebooks/preprocessing.ipynb,
-- cell 2: pd.read_csv("../data/creditcard.csv")) - V1-V28 are the PCA-anonymized
-- features Kaggle publishes this dataset with; TIME_SECONDS is seconds elapsed
-- since the first transaction in the dataset, not a wall-clock timestamp.

DROP TABLE transactions_raw;

CREATE TABLE transactions_raw (
    transaction_id      NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    time_seconds         NUMBER,             -- seconds since the first transaction, not a real timestamp
    v1  NUMBER, v2  NUMBER, v3  NUMBER, v4  NUMBER, v5  NUMBER,
    v6  NUMBER, v7  NUMBER, v8  NUMBER, v9  NUMBER, v10 NUMBER,
    v11 NUMBER, v12 NUMBER, v13 NUMBER, v14 NUMBER, v15 NUMBER,
    v16 NUMBER, v17 NUMBER, v18 NUMBER, v19 NUMBER, v20 NUMBER,
    v21 NUMBER, v22 NUMBER, v23 NUMBER, v24 NUMBER, v25 NUMBER,
    v26 NUMBER, v27 NUMBER, v28 NUMBER,      -- PCA components, anonymized, no real-world meaning individually
    amount               NUMBER(12, 2),
    class                NUMBER(1, 0)        -- 0 = legitimate, 1 = fraud
);

-- One index on the column every query below actually filters/groups by -
-- CLASS is 0.17% ones and 99.83% zeros, so an index here is what makes the
-- fraud-only aggregates in 02_fraud_analysis_queries_oracle.sql cheap
-- instead of a full table scan every time.
CREATE INDEX idx_transactions_class ON transactions_raw (class);
