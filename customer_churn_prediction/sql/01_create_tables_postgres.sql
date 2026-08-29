DROP TABLE IF EXISTS scored_customers;
DROP TABLE IF EXISTS customers_raw;

CREATE TABLE customers_raw (
    customerID          VARCHAR PRIMARY KEY,
    gender               VARCHAR,
    SeniorCitizen         INTEGER,          -- 0 / 1
    Partner              VARCHAR,          -- Yes / No
    Dependents           VARCHAR,          -- Yes / No
    tenure                INTEGER,          -- months
    PhoneService         VARCHAR,          -- Yes / No
    MultipleLines        VARCHAR,          -- Yes / No / No phone service
    InternetService      VARCHAR,          -- DSL / Fiber optic / No
    OnlineSecurity       VARCHAR,          -- Yes / No / No internet service
    OnlineBackup         VARCHAR,
    DeviceProtection     VARCHAR,
    TechSupport          VARCHAR,
    StreamingTV          VARCHAR,
    StreamingMovies      VARCHAR,
    Contract             VARCHAR,          -- Month-to-month / One year / Two year
    PaperlessBilling     VARCHAR,          -- Yes / No
    PaymentMethod        VARCHAR,
    MonthlyCharges         DOUBLE PRECISION,
    TotalCharges          VARCHAR,          -- text on purpose, see comment above
    Churn                VARCHAR           -- Yes / No
);

CREATE TABLE scored_customers (
    customerID          VARCHAR PRIMARY KEY,
    churn_probability      DOUBLE PRECISION,  -- from the calibrated model, test set only
    actual_churn           INTEGER            -- 0 / 1, ground truth for the test set
);

