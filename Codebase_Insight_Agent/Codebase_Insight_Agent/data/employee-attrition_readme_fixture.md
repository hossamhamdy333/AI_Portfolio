<div align="center">

# Employee Attrition & Retention Risk Analytics

An end-to-end HR analytics project: SQL data modeling, EDA-driven feature engineering, a classification model comparison chosen on cross-validated PR-AUC (not accuracy, given the dataset's imbalance), Kaplan-Meier and Cox survival analysis to quantify *when* people leave and why, and a final cost-of-attrition model that ties the risk scores to real dollar figures and a targeted-retention ROI estimate.

`Python` `SQL (PostgreSQL)` `scikit-learn` `LightGBM` `lifelines` `SHAP` `Streamlit` `Power BI`

Every claim in this README traces back to a number produced in one of the five notebooks — nothing here is asserted without a cell that generated it.

</div>

---

### Contents

- [Dashboard](#dashboard)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [How to run this](#how-to-run-this)
- [Notes on scope decisions](#notes-on-scope-decisions)
- [Key findings](#key-findings)
- [Techniques used](#techniques-used)

## Dashboard

![Executive Summary](dashboard_screenshots/page1_executive_summary.png)
![Attrition Drivers](dashboard_screenshots/page2_attrition_drivers.png)
![Tenure & Survival](dashboard_screenshots/page3_tenure_survival.png)
![Cost & Retention ROI](dashboard_screenshots/page4_cost_roi.png)

Full interactive Power BI file: `dashboard/Dashboard-employee-attrition.pbix` (includes live What-If sliders for the retention ROI scenario on page 4). A static PDF export is also included: `dashboard/Dashboard-employee-attrition.pdf`. A Streamlit version with the same 4 pages is at `dashboard/streamlit_app.py` (`streamlit run dashboard/streamlit_app.py`).

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employees, 35 columns, single flat snapshot.

**Getting the data:**

1. Go to the Kaggle link above and sign in (or create a free account)
2. Click **Download** — this is a plain dataset, not a competition, so no rules-acceptance step is needed
3. You'll get one file: `WA_Fn-UseC_-HR-Employee-Attrition.csv` (227 KB)
4. Place it in `data/raw/` in this project, keeping that exact filename — the notebooks reference it by name

## Project structure

```
employee-attrition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                      # gitignored — put the Kaggle CSV here
│   └── processed/                # gitignored — notebook outputs land here
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_load_staging.sql
│   ├── 03_build_dimensions_and_fact.sql
│   ├── 04_query_department_attrition.sql
│   ├── 05_query_income_percentile.sql
│   └── 06_query_tenure_bucket.sql
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification_models.ipynb
│   ├── 04_survival_analysis.ipynb
│   └── 05_cost_of_attrition.ipynb
├── models/                        # gitignored
├── reports/
│   └── model_results_summary.md
├── dashboard/
│   ├── Dashboard-employee-attrition.pbix
│   ├── Dashboard-employee-attrition.pdf
│   └── streamlit_app.py
└── dashboard_screenshots/
    ├── page1_executive_summary.png
    ├── page2_attrition_drivers.png
    ├── page3_tenure_survival.png
    └── page4_cost_roi.png
```

## How to run this

1. Activate your Python environment, `pip install -r requirements.txt`
2. Download the CSV into `data/raw/` — see [Dataset](#dataset) above
3. Run notebooks in order: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_classification_models.ipynb` → `04_survival_analysis.ipynb` → `05_cost_of_attrition.ipynb` (the last one exports `data/processed/fact_risk_scores.csv`, needed by the dashboards)
4. (Optional) load into Postgres via `sql/01`–`06`, in order, for the SQL-driven analytical queries (department attrition rate, income percentile within role, tenure-bucket attrition rate)
5. `streamlit run dashboard/streamlit_app.py` for the interactive web dashboard, or open `dashboard/Dashboard-employee-attrition.pbix` in Power BI Desktop for the full report with What-If sliders

## Notes on scope decisions

- **No Prophet.** Dropped after hitting a Windows long-path install failure (Prophet's bundled Stan/TBB library has deeply nested paths past Windows' 260-char limit). Baseline is seasonal-naive/logistic-regression instead; the real comparison is Logistic Regression vs. LightGBM, decided on 5-fold cross-validated PR-AUC rather than a single train/test split.

## Key findings

- **Class imbalance:** 84% No / 16% Yes (1,233 vs 237 employees) — drove the choice of PR-AUC over accuracy, and stratified splits, throughout.

- **Overtime is the single strongest driver of attrition**, and it compounds with role: overtime workers leave at 30.5% vs 10.4% for those without it, and Sales Representatives working overtime hit 66.7% attrition — the highest of any group in the data. The Cox model confirms this isn't just correlation: holding income, satisfaction, and commute distance constant, overtime workers leave at **~3.2x the rate** (hazard ratio 3.19, p < 0.005).

- **Department cost vs. department risk are different questions.** R&D carries the largest total dollar cost ($6.89M of the $10.15M workforce-wide expected attrition cost) simply because it's the largest department — but Sales has the highest per-employee attrition *rate* (20.6% vs R&D's 13.8%), with Sales Representative the single highest-risk role (39.8%).

- **Model:** Logistic Regression and LightGBM were compared on 5-fold cross-validation (not a single train/test split, which proved unreliable with only 47 positive test cases) — LightGBM won clearly (PR-AUC 0.578 vs 0.478) and was used for the final risk scores and SHAP explanations.

- **Survival analysis:** median tenure for leavers is 3 years vs 6 for those who stayed; the Cox model reaches a concordance index of 0.80, meaning it correctly ranks "who leaves sooner" about 80% of the time. Independently confirmed in SQL: attrition rate drops from 34.9% (0-1 yrs) to 10.4% (10+ yrs), monotonically, across every tenure bucket.

- **Cost of attrition:** total expected annual attrition cost across the workforce is an estimated **$10.15M**, using a standard 50%-of-salary replacement cost assumption. A targeted intervention aimed at the 81 highest-risk employees who also work overtime — combining the classification model's risk scores with the survival analysis's OverTime finding — is estimated at $162K in cost against $967K in avoided attrition cost (**ROI ~497%**, under the notebook's stated assumptions about intervention cost and effectiveness — not a measured result; see `reports/model_results_summary.md` for the full breakdown, including how quickly this ROI turns negative if the targeting is too broad).

## Techniques used

SQL data modeling (star schema, window functions, CTEs) · EDA-driven feature engineering · binary classification with class-imbalance handling (Logistic Regression, LightGBM, 5-fold cross-validation, SHAP) · survival analysis (Kaplan-Meier, Cox Proportional Hazards) · cost-of-attrition and retention-ROI modeling · interactive dashboarding (Power BI with DAX measures and What-If parameters, Streamlit).
