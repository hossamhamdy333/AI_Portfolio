-- ============================================================
-- 1. Churn rate by contract type
-- ============================================================
SELECT
    Contract,
    COUNT(*)                                    AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers_raw
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ============================================================
-- 2. Churn rate by tenure band, with a window function comparing
--    each band's churn rate to the overall average
-- ============================================================
WITH tenure_bands AS (
    SELECT
        *,
        CASE
            WHEN tenure <= 12 THEN '0-1y'
            WHEN tenure <= 24 THEN '1-2yr'
            WHEN tenure <= 48 THEN '2-4yr'
            WHEN tenure <= 60 THEN '4-5yr'
            ELSE                    '5-6yr'
        END AS tenure_group
    FROM customers_raw
),
band_rates AS (
    SELECT
        tenure_group,
        COUNT(*) AS customers,
        AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) AS churn_rate
    FROM tenure_bands
    GROUP BY tenure_group
)
SELECT
    tenure_group,
    customers,
    ROUND(churn_rate * 100, 2)                                        AS churn_rate_pct,
    ROUND(AVG(churn_rate) OVER () * 100, 2)                           AS overall_churn_rate_pct,
    ROUND((churn_rate - AVG(churn_rate) OVER ()) * 100, 2)            AS pct_pts_vs_overall
FROM band_rates
ORDER BY
    CASE tenure_group
        WHEN '0-1y' THEN 1 WHEN '1-2yr' THEN 2 WHEN '2-4yr' THEN 3
        WHEN '4-5yr' THEN 4 ELSE 5
    END;


-- ============================================================
-- 3. Rule-based high_risk / high_value flags, recreated in SQL exactly
--    as EDA_and_Preprocessing.ipynb cell 39 defines them:
--      high_risk  = Month-to-month contract AND MonthlyCharges > median
--      high_value = MonthlyCharges > 75th percentile AND tenure > 24
-- ============================================================
WITH thresholds AS (
    SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY MonthlyCharges)   AS median_charge,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY MonthlyCharges)  AS p75_charge
    FROM customers_raw
),
flagged AS (
    SELECT
        c.*,
        CASE WHEN c.Contract = 'Month-to-month' AND c.MonthlyCharges > t.median_charge
             THEN 1 ELSE 0 END AS high_risk,
        CASE WHEN c.MonthlyCharges > t.p75_charge AND c.tenure > 24
             THEN 1 ELSE 0 END AS high_value
    FROM customers_raw c
    CROSS JOIN thresholds t
)
SELECT
    high_risk,
    COUNT(*)                                                          AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM flagged
GROUP BY high_risk
ORDER BY high_risk DESC;
-- Cross-check against the notebook's own printout in EDA cell 42
-- ("High risk churn rate: ...%", "Low risk churn rate: ...%").


-- ============================================================
-- 4. The priority matrix itself: customer count by value segment x
--    risk segment, rebuilt from the model's exported scores.
--    Bucket edges match Customer_Segmentation.ipynb exactly:
--      risk_segment  = pd.cut(prob, [0, .2, .4, .6, .8, 1.0])
--      value_segment = pd.qcut(MonthlyCharges, q=3)  -- tertiles
-- ============================================================
WITH value_tertiles AS (

    SELECT
        c.customerID,
        c.MonthlyCharges,
        NTILE(3) OVER (ORDER BY c.MonthlyCharges) AS value_tertile
    FROM customers_raw c
    JOIN scored_customers s USING (customerID)
),
bucketed AS (
    SELECT
        s.customerID,
        s.actual_churn,
        CASE
            WHEN s.churn_probability <= 0.2 THEN 'Very Low'
            WHEN s.churn_probability <= 0.4 THEN 'Low'
            WHEN s.churn_probability <= 0.6 THEN 'Medium'
            WHEN s.churn_probability <= 0.8 THEN 'High'
            ELSE                                  'Very High'
        END AS risk_segment,
        CASE v.value_tertile
            WHEN 1 THEN 'Low Value'
            WHEN 2 THEN 'Mid Value'
            ELSE        'High Value'
        END AS value_segment
    FROM scored_customers s
    JOIN value_tertiles v USING (customerID)
)
SELECT
    value_segment,
    risk_segment,
    COUNT(*) AS customers
FROM bucketed
GROUP BY value_segment, risk_segment
ORDER BY
    CASE value_segment WHEN 'Low Value' THEN 1 WHEN 'Mid Value' THEN 2 ELSE 3 END,
    CASE risk_segment WHEN 'Very Low' THEN 1 WHEN 'Low' THEN 2 WHEN 'Medium' THEN 3
                       WHEN 'High' THEN 4 ELSE 5 END;


-- ============================================================
-- 5. The actual retention target: high value, high/very high risk
-- ============================================================
WITH value_tertiles AS (
    SELECT
        c.customerID,
        NTILE(3) OVER (ORDER BY c.MonthlyCharges) AS value_tertile
    FROM customers_raw c
    JOIN scored_customers s USING (customerID)
)
SELECT COUNT(*) AS priority_customers
FROM scored_customers s
JOIN value_tertiles v USING (customerID)
WHERE v.value_tertile = 3            -- High Value
  AND s.churn_probability > 0.6;     -- High or Very High risk
