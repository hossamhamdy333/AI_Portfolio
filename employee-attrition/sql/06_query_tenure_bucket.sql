WITH tenure_buckets AS (
    SELECT
        *,
        CASE
            WHEN years_at_company < 2 THEN '0-1 yrs'
            WHEN years_at_company < 5 THEN '2-4 yrs'
            WHEN years_at_company < 10 THEN '5-9 yrs'
            ELSE '10+ yrs'
        END AS tenure_bucket
    FROM fact_employee
)
SELECT
    tenure_bucket,
    COUNT(*) AS headcount,
    ROUND(100.0 * SUM(CASE WHEN attrition THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM tenure_buckets
GROUP BY tenure_bucket
ORDER BY MIN(years_at_company);