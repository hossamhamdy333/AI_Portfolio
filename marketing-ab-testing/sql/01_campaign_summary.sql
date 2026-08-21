-- 01_campaign_summary.sql
--
-- Basic descriptive summary of the A/B test: conversion rate by group,
-- by day of week, and by hour of day. This is descriptive only -- the
-- actual significance testing happens in notebooks/02_ab_test_analysis.ipynb.
--
-- Written against DuckDB, querying the raw CSV directly. Column names
-- have spaces in the source data, so they're quoted throughout.

-- 1. Overall conversion rate per test group, with sample size so the
--    group imbalance is visible right away.
SELECT
    "test group",
    COUNT(*)                                   AS users,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS conversions,
    ROUND(AVG(CASE WHEN converted THEN 1.0 ELSE 0.0 END), 4) AS conversion_rate
FROM read_csv_auto('../data/raw/marketing_AB_fake.csv')
GROUP BY "test group"
ORDER BY "test group";


-- 2. Conversion rate by group and day of week.
SELECT
    "test group",
    "most ads day"                             AS day_of_week,
    COUNT(*)                                   AS users,
    ROUND(AVG(CASE WHEN converted THEN 1.0 ELSE 0.0 END), 4) AS conversion_rate
FROM read_csv_auto('../data/raw/marketing_AB_fake.csv')
GROUP BY "test group", "most ads day"
ORDER BY "test group", day_of_week;


-- 3. Conversion rate by group and hour of day, plus a running share of
--    total conversions within each group (window function), so we can
--    see which hours drive the bulk of conversions.
SELECT
    "test group",
    "most ads hour"                            AS hour_of_day,
    COUNT(*)                                   AS users,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) AS conversions,
    ROUND(AVG(CASE WHEN converted THEN 1.0 ELSE 0.0 END), 4) AS conversion_rate,
    ROUND(
        SUM(SUM(CASE WHEN converted THEN 1 ELSE 0 END))
            OVER (PARTITION BY "test group" ORDER BY "most ads hour")
        / SUM(SUM(CASE WHEN converted THEN 1 ELSE 0 END))
            OVER (PARTITION BY "test group"),
        4
    ) AS cumulative_share_of_conversions
FROM read_csv_auto('../data/raw/marketing_AB_fake.csv')
GROUP BY "test group", "most ads hour"
ORDER BY "test group", hour_of_day;
