DROP TABLE IF EXISTS staging_employees;
CREATE TABLE staging_employees (
    Age                         SMALLINT,
    Attrition                   TEXT,
    BusinessTravel              TEXT,
    DailyRate                   INTEGER,
    Department                  TEXT,
    DistanceFromHome            SMALLINT,
    Education                   SMALLINT,
    EducationField               TEXT,
    EmployeeCount                SMALLINT,
    EmployeeNumber              INTEGER PRIMARY KEY,
    EnvironmentSatisfaction     SMALLINT,
    Gender                      TEXT,
    HourlyRate                  INTEGER,
    JobInvolvement              SMALLINT,
    JobLevel                    SMALLINT,
    JobRole                     TEXT,
    JobSatisfaction             SMALLINT,
    MaritalStatus               TEXT,
    MonthlyIncome                INTEGER,
    MonthlyRate                  INTEGER,
    NumCompaniesWorked          SMALLINT,
    Over18                      TEXT,
    OverTime                    TEXT,
    PercentSalaryHike           SMALLINT,
    PerformanceRating           SMALLINT,
    RelationshipSatisfaction    SMALLINT,
    StandardHours                SMALLINT,
    StockOptionLevel             SMALLINT,
    TotalWorkingYears            SMALLINT,
    TrainingTimesLastYear       SMALLINT,
    WorkLifeBalance              SMALLINT,
    YearsAtCompany                SMALLINT,
    YearsInCurrentRole          SMALLINT,
    YearsSinceLastPromotion    SMALLINT,
    YearsWithCurrManager        SMALLINT
);

DROP TABLE IF EXISTS dim_department;
CREATE TABLE dim_department (
    department_id  SERIAL PRIMARY KEY,
    department     TEXT UNIQUE NOT NULL
);

DROP TABLE IF EXISTS dim_job_role;
CREATE TABLE dim_job_role (
    job_role_id    SERIAL PRIMARY KEY,
    job_role       TEXT UNIQUE NOT NULL
);

DROP TABLE IF EXISTS dim_education_field;
CREATE TABLE dim_education_field (
    education_field_id SERIAL PRIMARY KEY,
    education_field     TEXT UNIQUE NOT NULL
);