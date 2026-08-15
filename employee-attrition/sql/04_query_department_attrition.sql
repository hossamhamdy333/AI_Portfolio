SELECT
    d.department,
    COUNT(*) AS headcount,
    SUM(CASE WHEN f.attrition THEN 1 ELSE 0 END) AS attritions,
    ROUND(100.0 * SUM(CASE WHEN f.attrition THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM fact_employee f
JOIN dim_department d ON f.department_id = d.department_id
GROUP BY d.department
ORDER BY attrition_rate_pct DESC;