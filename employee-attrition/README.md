# Employee Attrition & Retention Risk Analytics

An end-to-end HR analytics project: SQL data modeling, EDA-driven feature
engineering, a classification model comparison chosen on cross-validated
PR-AUC (not accuracy, given the dataset's imbalance), Kaplan-Meier and Cox
survival analysis to quantify *when* people leave and why, and a final
cost-of-attrition model that ties the risk scores to real dollar figures and
a targeted-retention ROI estimate. Every claim in this README traces back to
a number produced in one of the five notebooks — nothing here is asserted
without a cell that generated it.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
— 1,470 employees, 35 columns, single flat snapshot. See `docs/kaggle_setup.md`.

## Project structure

```
employee-attrition/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── kaggle_setup.md
├── data/
│   ├── raw/                      # gitignored
│   └── processed/                # gitignored
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_load_staging.sql
│   └── 03_build_fact_employee.sql
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification_models.ipynb
│   ├── 04_survival_analysis.ipynb
│   └── 05_cost_of_attrition.ipynb
├── models/                        # gitignored
├── reports/
│   └── model_results_summary.md
└── dashboard/
    ├── streamlit_app.py
    └── powerbi_spec.md
```


## Key findings

- **Class imbalance:** 84% No / 16% Yes (1,233 vs 237 employees) — drove the
  choice of PR-AUC over accuracy, and stratified splits, throughout.

- **Overtime is the single strongest driver of attrition**, and it compounds
  with role: overtime workers leave at 30.5% vs 10.4% for those without it,
  and Sales Representatives working overtime hit 66.7% attrition — the
  highest of any group in the data. The Cox model confirms this isn't just
  correlation: holding income, satisfaction, and commute distance constant,
  overtime workers leave at **~3.2x the rate** (hazard ratio 3.19, p < 0.005).

- **Department cost vs. department risk are different questions.**
  R&D carries the largest total dollar cost ($6.89M of the $10.15M
  workforce-wide expected attrition cost) simply because it's the largest
  department — but Sales has the highest per-employee attrition *rate*
  (20.6% vs R&D's 13.8%), with Sales Representative the single highest-risk
  role (39.8%).

- **Model:** Logistic Regression and LightGBM were compared on 5-fold
  cross-validation (not a single train/test split, which proved unreliable
  with only 47 positive test cases) — LightGBM won clearly (PR-AUC 0.578 vs
  0.478) and was used for the final risk scores and SHAP explanations.

- **Survival analysis:** median tenure for leavers is 3 years vs 6 for
  those who stayed; the Cox model reaches a concordance index of 0.80,
  meaning it correctly ranks "who leaves sooner" about 80% of the time.

- **Cost of attrition:** total expected annual attrition cost across the
  workforce is an estimated **$10.15M**, using a standard 50%-of-salary
  replacement cost assumption. A targeted intervention aimed at the 81
  highest-risk employees who also work overtime — combining the
  classification model's risk scores with the survival analysis's OverTime
  finding — is estimated at $162K in cost against $967K in avoided
  attrition cost (**ROI ~497%**, under the notebook's stated assumptions
  about intervention cost and effectiveness — not a measured result).

## What's different from Project 1

Different domain (HR, not retail) and different core techniques: binary
classification with class-imbalance handling, survival analysis
(Kaplan-Meier / Cox), and cost-of-attrition ROI modeling — none of which
appear in the retail-forecasting project. Same underlying discipline
(SQL → EDA-driven features → model comparison → SHAP → dashboard) carried
across both, which is itself worth calling out as a consistent skill set.
