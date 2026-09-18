<div align="center">

# E-Commerce Demand Forecasting

`PostgreSQL` `pandas` `LightGBM` `scikit-learn` `SHAP` `holidays` `matplotlib` `Power BI`

</div>

---

### Contents

- [Summary](#summary)
- [Problem & motivation](#problem--motivation)
- [Approach](#approach)
- [Data](#data)
- [Results](#results)
- [What I'd do differently / limitations](#what-id-do-differently--limitations)
- [Stack](#stack)

---

## Summary

This project forecasts daily, SKU-level product demand for a real UK online
retailer and turns that forecast into inventory decisions. Raw transaction
data goes through a documented SQL cleaning pipeline, then Python feature
engineering, then a two-stage LightGBM "hurdle" model (a classifier for
whether a SKU sells at all on a given day, times a regressor for how much,
gated rather than blended), and the model's own forecast errors are used to
size safety stock and reorder points per SKU. Final result: **86.8% WAPE**,
beating both a naive always-predict-zero baseline (100% WAPE by definition)
and a seasonal-naive baseline (126.8%). That headline number undersells the
project a little — two earlier, more "obvious" versions of this same model
scored *worse* than predicting zero, and finding out why is most of what's
actually in here.

## Problem & motivation

Retail demand at the SKU level is sparse and bursty: most (SKU, day) pairs
have zero sales, and the nonzero ones range from a handful of units to
genuine spikes. A model optimized the usual way — minimize squared error, or
even minimize MAPE — gets dominated by the zero-demand majority or blows up
on the days with zero actual demand (MAPE divides by an actual value that's
often zero). That's why this project scores everything in WAPE (weighted
absolute percentage error) instead, and why it treats "beat the trivial
always-zero baseline" as the real floor to clear, not just a formality.

The naive approach to intermittent demand is a standard two-stage hurdle
model: a classifier for P(demand > 0) multiplied by a regressor for demand
magnitude. That's exactly what this project started with, and it scored
108.9% WAPE — worse than predicting zero for every row. The interesting part
of this project isn't the final architecture, it's diagnosing why the
"textbook" version of that architecture failed and what specifically fixed
it, documented in `reports/PROJECT_REPORT.md`.

## Approach

### SQL layer (PostgreSQL, `sql/00`–`sql/09`, run in order):

- Staging (`02_staging_clean.sql`) deduplicates, drops zero/negative-price
  rows, and excludes a fixed list of non-product stock codes (`POST`, `D`,
  `DOT`, `M`, `S`, `B`, `AMAZONFEE`, `BANK CHARGES`, `CRUK`, `C2`,
  `TEST001`, `TEST002`, `ADJUST`, `ADJUST2`) identified by hand in EDA.
  Cancellations (invoice numbers starting with `C`) are kept, not dropped,
  and netted against same-day sales in `03_net_transactions.sql` — dropping
  them outright would overstate net demand.
- Categories don't exist natively in this dataset. `04_category_assignment.sql`
  derives one from the first two characters of the stock code
  (`LEFT(stock_code, 2)`). This is stated as a heuristic, not a real product
  taxonomy, and every category-level number downstream inherits that
  limitation.
- Daily demand per SKU (`05_daily_demand.sql`) clips negative net demand
  (a cancellation landing on a different calendar day than its original
  sale) at zero, since the model can't take negative targets.
- Calendar gaps are filled **per SKU, within that SKU's own observed
  lifetime** (`06_fill_calendar.sql`), not the dataset's global date range.
  This mattered more than it sounds like it should — see Results.
- `07_rolling_features.sql` computes reporting-only window functions
  (week-over-week change, category revenue rank). These deliberately
  include the current day's own value and are explicitly commented as
  never to be used as model inputs, since that would be same-day-target
  leakage.
- `08_audit_summary.sql` quantifies what the cleaning decisions actually
  changed — see Results.
- `09_bi_exports.sql` builds a proper contiguous `dim_date` table (Power
  BI's time-intelligence DAX functions need one with no gaps) plus flat
  exports for the dashboard.

The SQL pipeline is run manually against Postgres (via `psql` or pgAdmin,
per the comments in each file) — there's no Python DB connector wiring it
into the notebooks; the notebooks pick up from CSV exports.

**Feature engineering (`src/features.py`, used by
`02_feature_engineering.ipynb`):** every feature is built causally by
construction — a feature at (SKU, day *t*) only ever uses information
through *t-1*, except calendar features, which are known in advance.
Concretely: lags at 1/7/14/28 days; rolling 7/14/28-day mean and std,
computed by shifting the series one day before rolling so the window never
touches the current row; a 7-vs-28-day momentum feature; `days_since_last_sale`
(a streak-based intermittency signal, implemented without
`groupby().apply()` to sidestep a pandas quirk that breaks on single-SKU
slices); calendar features (day of week, weekend, month, ISO week, UK
holiday flag, and distance to the nearest Christmas — computed with an
explicit wraparound so early-January dates measure distance to the
Christmas that just passed, not the one 11 months away); and category-level
7/28-day rolling trend features, added because a SKU's own rolling average
is too slow to catch a demand ramp before it's already inside the averaging
window. SKU-level average demand, SKU zero-rate, and category frequency
encoding are all fit on training rows only and then applied to the full
frame, so nothing from the test period leaks backward into them.

### Model:

a gated two-stage hurdle model, both stages `LightGBM`
(`n_estimators=300, learning_rate=0.05, num_leaves=63`): an `LGBMClassifier`
for P(demand > 0), and an `LGBMRegressor` with a **quantile objective at the
median** (`alpha=0.5`), trained only on rows with nonzero demand, for
magnitude. The two stages are combined with a **hard threshold**, not a
probability-weighted blend (`p × magnitude`): the magnitude prediction only
fires if the classifier's probability clears a cutoff, otherwise the
prediction is zero. The cutoff itself is chosen by 4-fold cross-validation
on the training set only, sweeping thresholds from 0.05 to 0.95 in steps of
0.05 and picking whichever minimizes out-of-fold WAPE — never by checking
against test-set performance, since that would just be a subtler version of
the same leakage the project is otherwise careful about.

### Inventory sizing (`04_inventory_risk.ipynb`):

safety stock and reorder
point are derived directly from the model's own test-set residuals, not an
assumed error rate — `safety_stock = z × residual_std × sqrt(lead_time_days)`,
`reorder_point = avg_daily_forecast × lead_time_days + safety_stock`, computed
at 90/95/99% service levels (z = 1.28 / 1.65 / 2.33). Lead time is a flat,
stated assumption of 7 days, not measured against real supplier data.

## Data

Source: Online Retail II (a real UK-based online retailer, invoices from
December 2009 to December 2011), loaded from the dataset's two Excel sheets
and concatenated. Raw size is roughly **1.07 million transaction line
items**. EDA (`01_eda.ipynb`) found: ~23% of rows missing a customer ID
(no monthly spike, read as guest checkouts rather than a data bug, and kept
since SKU-level demand doesn't need it); ~6% exact duplicate rows, a known
re-export artifact in this dataset, dropped in staging; ~1.8% cancellations,
kept and netted against sales rather than dropped; ~6,200 rows with
price ≤ 0, mostly samples/damages, dropped; roughly 92% of rows from the UK
(all countries are kept, since the model operates at SKU/date grain rather
than by country).

After cleaning, aggregation, feature engineering, and dropping warm-up rows
(rows without enough history for the longer lags/rolling windows to be
defined), the final `model_ready.csv` splits by **date, not randomly** —
train/test cutoff at roughly the 85th percentile of the date range — giving
**1,791,719 training rows** and **332,891 test rows**.

## Results

| Model | WAPE |
|---|---|
| Always-zero baseline | 100.0% |
| Seasonal-naive baseline (same weekday, prior week) | 126.8% |
| Hurdle model, original (probability × magnitude blend) | 108.9% |
| Hurdle model, threshold gate + Tweedie (mean) regressor | 95.7% |
| **Hurdle model, threshold gate + quantile (median) regressor — final** | **86.8%** |

Numbers come straight from `reports/model_results_summary.json`, rewritten
on every run of `03_forecasting_models.ipynb`.

The path to that final number is the actual story:

- **The original hurdle model was worse than predicting zero.** Breaking
  test error down by row showed roughly half of total error came from rows
  where actual demand was zero — not because the classifier was bad
  (AUC ≈ 0.71), but because multiplying a probability by a magnitude puts a
  small nonzero prediction on *every* row, and with about two-thirds of all
  rows genuinely zero-demand, those small overpredictions add up fast.
- **Switching to a hard threshold gate fixed most of it**, taking WAPE from
  108.9% to 95.7%.
- **Switching the magnitude regressor from a Tweedie (mean-targeting)
  objective to a quantile (median-targeting) one closed most of the rest**,
  down to 86.8%. WAPE/MAE is minimized by the conditional median, and mean
  regression runs high on demand this skewed. A quick synthetic test had
  suggested this would be worth roughly 0.6 WAPE points; the real effect on
  this data was much larger, most likely because the real distribution is
  more skewed than the synthetic approximation used to sanity-check the
  idea beforehand.
- **A separate, earlier bug is worth calling out on its own**: an earlier
  version of the calendar-fill step filled every SKU's zero-demand days
  against the dataset's *global* date range instead of that SKU's own
  active lifetime. That manufactured a large number of fake zero-demand
  rows for periods before a product existed or after it was discontinued,
  and is the most likely reason an even earlier version of this pipeline
  produced badly inflated error metrics. `08_audit_summary.sql` measures
  the fix directly by comparing the old (buggy) row count formula against
  the actual per-SKU-lifetime row count.
- **A threshold that varies by SKU (lower for spike-prone SKUs) was tested
  and didn't help** — 95.2% vs. 95.1% WAPE on the same setup, within noise.
  Chasing this further pointed at something more useful: the demand spikes
  the model still misses have a median lag-1 value of zero (no sale the day
  before), versus a lag-1 of 10 for spikes it catches, and they don't
  concentrate in November or any particular SKU. Since every feature here
  is derived from a SKU's own sales history, a SKU that's been quiet and
  then suddenly sells a lot has nothing in its recent history to signal
  that — a real ceiling on what's predictable from this data, not a bug to
  keep chasing with more feature engineering.

The Power BI dashboard (`dashboard/ecommerce_dashboard.pbix`,
4 pages — Executive Overview, Forecast Accuracy, Inventory Risk, SKU
drill-down) reports, over the full history: 11M total units sold, £19.25M
total revenue, ~3K distinct SKUs. On the test window specifically: WAPE
0.87 and forecast bias -3.86 (the model slightly underpredicts on average),
against the always-zero baseline of 1.00; total actual test-period demand
of 2,271,125 units against 985,368 forecast. On the inventory side: ~3K
SKUs flagged as at some stockout risk under a zero-safety-stock policy,
199.90K total units of safety stock at the 95% service level, and an
average 95% reorder point of 82.16 units per SKU.

## What I'd do differently / limitations

- **Category assignment is a heuristic, not a real taxonomy.** It's stated
  as such everywhere it's used, but every category-level number in this
  project — including the category-trend features that feed the model —
  inherits whatever error that heuristic introduces, and there's no
  measurement of how often it misfiles a SKU.
- **Cancellation netting only matches same-day cancellations.** A
  cancellation landing a week after its original sale isn't matched to it;
  it shows up as a standalone negative-quantity row, which gets clipped to
  zero rather than silently kept as-is. `08_audit_summary.sql` measures how
  often this happens rather than assuming it's rare.
- **The unpredictable, no-lead-in demand spikes are the real ceiling on
  this model**, and closing that gap needs data this project doesn't have
  — a promotions calendar or supply/restock events — not more feature
  engineering on the SKU's own sales history.
- **Lead time (7 days) is an assumption, not measured against real
  supplier data**, and there's no real on-hand inventory in this dataset,
  so the "stockout risk" flag is a proxy based on forecast uncertainty
  (any SKU with nonzero residual std is flagged), not a check against
  actual stock levels.
- **The classification threshold minimizes WAPE, not cost.** If a missed
  spike costs more in stockouts than a false positive costs in holding
  inventory, the threshold should reflect that tradeoff instead of treating
  both error types as equally expensive. That needs an actual cost ratio
  from whoever owns the inventory decision, which wasn't available here.
- **One high-volume/long-tail split was never tried.** Very high-volume
  SKUs and the long tail of intermittent ones may need different models
  rather than one model serving both regimes.
- **A weekly-aggregated feature set (`model_ready_weekly.csv`) is built in
  `02_feature_engineering.ipynb` but never used by the modeling notebook**
  — it's a leftover branch from an earlier exploration, not part of the
  current pipeline.

## Stack

- `PostgreSQL` (14+) for the cleaning/aggregation pipeline, run directly via
  `psql`/pgAdmin — no Python DB layer sits between the SQL and the CSV
  exports the notebooks read
- `pandas`, `numpy` for feature engineering and evaluation
- `LightGBM` (`LGBMClassifier`, `LGBMRegressor` with a quantile objective)
  for the hurdle model
- `scikit-learn` (`KFold`) for the train-only cross-validated threshold search
- `SHAP` (`TreeExplainer`) for feature importance on the magnitude regressor
- `holidays` for UK public holiday flags
- `matplotlib` for EDA and diagnostic plots
- `Power BI` for the 4-page dashboard, fed by `sql/09_bi_exports.sql`'s
  flat exports plus a BI-specific forecast export from
  `03_forecasting_models.ipynb`
