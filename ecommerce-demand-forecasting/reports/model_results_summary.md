# Model Results Summary

*(This file is regenerated automatically by `notebooks/03_forecasting_models.ipynb`
— run that notebook on the real data and it overwrites this placeholder with
real numbers. Do not hand-edit numbers into this file; if a number needs to
change, it should come from re-running the notebook.)*

## What changed since the previous version of this project

The original SQL pipeline (`sql/01_clean_and_aggregate.sql`) had a real bug:
it filled every SKU's calendar against the *global* dataset date range
instead of that SKU's own observed active window. This created large
numbers of "ghost" zero-demand rows — days when a product hadn't launched
yet or was already discontinued, misrepresented as "in stock, zero sales."
Confirmed via a direct row-count comparison (included as a diagnostic query
in the SQL file itself) that this inflated row counts substantially, and is
the most likely cause of the previous version's best model scoring **worse
than a trivial "always predict zero" baseline** (WAPE > 100%).

A second, smaller leakage bug was also found and fixed: the SQL's rolling
average columns (`rolling_7d_avg_units`, `rolling_28d_avg_units`,
`rolling_28d_std_units`) included the current row in their own window —
meaning a day's "rolling average" partly consisted of that same day's
actual sales. Fixed to exclude the current row (`1 PRECEDING` as the end
bound, not `CURRENT ROW`).

**Once you've re-run the corrected pipeline on the real Online Retail II
data, the real WAPE numbers will replace this section.** The evaluation in
`03_forecasting_models.ipynb` always reports the "always-zero" floor (100%
WAPE) alongside every model, specifically so this comparison is impossible
to accidentally omit again.
