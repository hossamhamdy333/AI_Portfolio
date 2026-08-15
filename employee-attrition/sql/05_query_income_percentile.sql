SELECT
    f.employee_id,
    r.job_role,
    f.monthly_income,
    PERCENT_RANK() OVER (PARTITION BY r.job_role_id ORDER BY f.monthly_income) AS income_percentile_in_role,
    f.attrition
FROM fact_employee f
JOIN dim_job_role r ON f.job_role_id = r.job_role_id
ORDER BY r.job_role, income_percentile_in_role
LIMIT 20;