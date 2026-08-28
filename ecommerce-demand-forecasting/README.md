# E-Commerce Demand Forecasting

SKU-level daily demand forecasting on the Online Retail II dataset (UK online retailer, 2009–2011), from raw transaction data through a trained model to inventory reorder recommendations.

**Result:** WAPE 86.6%, beating both an always-predict-zero baseline (100%) and a seasonal-naive baseline (126.2%). Full writeup of how that number was reached — including two earlier model versions that scored *worse* than predicting zero, and why — is in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Structure

```
sql/
  00_create_raw_table.sql       raw landing table
  01_raw_audit.sql              raw row count sanity check
  02_staging_clean.sql          documented cleaning decisions
  03_net_transactions.sql       net demand, cancellations offset against sales
  04_category_assignment.sql    category heuristic (first 2 chars of stock_code)
  05_daily_demand.sql           daily demand per SKU, negative-net clipped at zero
  06_fill_calendar.sql          per-SKU calendar fill (own active window, not global range)
  07_rolling_features.sql       reporting-only window functions (not model features)
  08_audit_summary.sql          data-quality numbers for this README
  09_bi_exports.sql             date dimension + flat exports for Power BI

notebooks/
  01_eda.ipynb                  exploratory analysis
  02_feature_engineering.ipynb  builds model_ready.csv from fact_demand_features
  03_forecasting_models.ipynb   trains the hurdle model, evaluates WAPE
  04_inventory_risk.ipynb       safety stock + reorder point from model residuals

src/
  features.py                   shared feature logic (notebooks + API both import this)

dashboard/
  powerbi_build_spec.md         data model, DAX measures, page-by-page build spec
```

## Setup

```
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires PostgreSQL 14+ for the SQL pipeline.

## Running the pipeline

1. Download the Online Retail II dataset, combine both sheets, save to `data/raw/online_retail_combined.csv` (see `notebooks/01_eda.ipynb`).
2. Run `sql/00` through `sql/09` in order against your Postgres database (each file's header states its `Requires:`).
3. Run `notebooks/02_feature_engineering.ipynb` → `03_forecasting_models.ipynb` → `04_inventory_risk.ipynb`, in that order. `04` depends on the model artifact `03` saves, so `03` must be re-run first after any change to it.

## Key design decisions

- **Every feature is strictly causal**: a feature at row (stock_code, date=t) only uses information available up to t-1 (calendar features are the one exception, since they're known in advance). This is enforced consistently in both `src/features.py` and the SQL rolling-window step.
- **Calendar days are filled per-SKU, within each SKU's own observed lifetime** — not the dataset-wide date range. Filling globally silently manufactures ghost zero-demand rows for periods before a SKU existed or after it was discontinued, which inflates every downstream error metric. See `08_audit_summary.sql` for the measured impact of this fix.
- **Train-only fitting for every leakage-prone step**: SKU-level statistics, category frequency encoding, and the model's classification threshold are all selected using train data only, cross-validated where relevant, never by looking at test-set performance.
- **Hurdle model uses a hard threshold gate**, not a continuous probability-weighted blend — see `PROJECT_REPORT.md` for why this mattered more than any single feature.

## Dashboard

`dashboard/powerbi_build_spec.md` has the full data model, DAX measures, and a page-by-page build spec (Executive Overview, Forecast Accuracy, Inventory Risk, SKU Drill-down) for Power BI Desktop. Data sources are the flat exports from `sql/09_bi_exports.sql` and a small BI-specific export added to `notebooks/03_forecasting_models.ipynb`.

## Known limitations

See `PROJECT_REPORT.md` for the full list — category assignment is a heuristic, same-day-only cancellation netting, and a genuine structural ceiling on predicting demand spikes with no lead-in signal (would need external data like a promotions calendar to close further).
