INSERT INTO dim_department (department)
SELECT DISTINCT Department FROM staging_employees ORDER BY 1
ON CONFLICT (department) DO NOTHING;

INSERT INTO dim_job_role (job_role)
SELECT DISTINCT JobRole FROM staging_employees ORDER BY 1
ON CONFLICT (job_role) DO NOTHING;

INSERT INTO dim_education_field (education_field)
SELECT DISTINCT EducationField FROM staging_employees ORDER BY 1
ON CONFLICT (education_field) DO NOTHING;

DROP TABLE IF EXISTS fact_employee;
CREATE TABLE fact_employee AS
SELECT
    s.EmployeeNumber           AS employee_id,
    s.Age                      AS age,
    (s.Attrition = 'Yes')      AS attrition,
    s.BusinessTravel           AS business_travel,
    d.department_id,
    r.job_role_id,
    ef.education_field_id,
    s.DistanceFromHome         AS distance_from_home,
    s.Education                AS education_level,
    s.Gender                   AS gender,
    s.MaritalStatus            AS marital_status,
    (s.OverTime = 'Yes')       AS overtime,
    s.MonthlyIncome            AS monthly_income,
    s.JobLevel                 AS job_level,
    s.JobSatisfaction          AS job_satisfaction,
    s.EnvironmentSatisfaction  AS environment_satisfaction,
    s.RelationshipSatisfaction AS relationship_satisfaction,
    s.WorkLifeBalance          AS work_life_balance,
    s.NumCompaniesWorked       AS num_companies_worked,
    s.TotalWorkingYears        AS total_working_years,
    s.YearsAtCompany           AS years_at_company,
    s.YearsInCurrentRole       AS years_in_current_role,
    s.YearsSinceLastPromotion  AS years_since_last_promotion,
    s.YearsWithCurrManager     AS years_with_curr_manager,
    s.StockOptionLevel         AS stock_option_level,
    s.PercentSalaryHike        AS percent_salary_hike,
    s.PerformanceRating        AS performance_rating
FROM staging_employees s
JOIN dim_department d       ON s.Department = d.department
JOIN dim_job_role r          ON s.JobRole = r.job_role
JOIN dim_education_field ef  ON s.EducationField = ef.education_field;

DROP INDEX IF EXISTS idx_fact_employee_dept;
CREATE INDEX idx_fact_employee_dept ON fact_employee (department_id);
DROP INDEX IF EXISTS idx_fact_employee_role;
CREATE INDEX idx_fact_employee_role ON fact_employee (job_role_id);