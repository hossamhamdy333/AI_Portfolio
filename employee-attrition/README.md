<div align="center">

# Employee Attrition & Retention Risk Analytics

`pandas` `scikit-learn` `LightGBM` `SHAP` `lifelines` `PostgreSQL` `Streamlit` `Power BI` `matplotlib`

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

An HR analytics project that goes past "predict who leaves" and into "what would it actually cost, and is intervening worth it." A star-schema SQL layer and a classification model (LightGBM, chosen on 5-fold cross-validated PR-AUC) both quantify attrition risk on IBM's 1,470-employee HR dataset; a Cox proportional-hazards model adds *when* people leave and confirms overtime as the dominant driver (hazard ratio 3.19, ~3.2x faster attrition, holding other factors constant); and a final cost model translates the model's risk scores into dollars — $10.15M in estimated workforce-wide annual attrition cost, and a targeted retention intervention on the 81 highest-risk overtime workers projected at 497% ROI under stated cost assumptions. The two dashboards (Power BI with live What-If sliders, and a matching Streamlit app) exist specifically so someone other than the analyst can stress-test those assumptions instead of trusting one baked-in number.

## Problem & motivation

The naive version of this project stops at "train a classifier, report accuracy." Two things make that the wrong approach here. First, the data is imbalanced (84% stayed / 16% left), so accuracy is close to meaningless — a model that predicts "stays" for everyone is 84% accurate and catches zero leavers, which is useless for a retention team. Second, and less obvious: even after switching to PR-AUC, a single train/test split is still not a reliable way to compare two models when the test set only has 47 positive cases. This project hit that directly — on the single 80/20 split, Logistic Regression's PR-AUC (0.552) actually beat LightGBM's (0.478), which would pick the wrong model. Five-fold cross-validation on the training set reversed that ranking (LightGBM 0.578 vs. Logistic Regression 0.478), and that's the number the project actually trusts, specifically because 47 positive cases is small enough for one split to favor either model by chance.

The second half of the problem — turning a risk score into a business decision — has its own naive failure mode: "classify everyone as high or low risk" throws away the fact that a $2,000/employee intervention is only worth it if the targeting is narrow and confident. The project treats this as its own thing to model, not an afterthought, and shows explicitly (via the dashboard's What-If sliders) that widening the target group and weakening the intervention effect turns the same ROI calculation negative.

## Approach

### SQL layer

(`sql/01`–`06`, run against PostgreSQL): a star schema (dimension tables plus a fact table) built from the raw CSV, then three independent analytical queries — department-level attrition rate, income percentile within role (window functions), and attrition rate by tenure bucket. This layer exists to cross-check the Python-side findings with plain SQL, not to replace them — the tenure-bucket query in particular is used later to independently confirm the survival-analysis result.

### Feature engineering

(`02_feature_engineering.ipynb`): three constant columns (`EmployeeCount`, `Over18`, `StandardHours` — zero variance across all 1,470 rows in EDA) and the `EmployeeNumber` ID column are dropped. Two engineered features are added: `role_tenure_ratio` (`YearsInCurrentRole` / `YearsAtCompany`, divide-by-zero guarded by replacing 0 with 1, then clipped to [0, 1]) and `overtime_role` (a concatenation of `OverTime` and `JobRole`, since the two interact — see Results). Categorical columns are one-hot encoded with `drop_first=True` to avoid redundant collinear columns. Split 80/20, stratified on `Attrition` (`random_state=42`), giving 1,176 train / 294 test rows, both preserving the ~16% attrition rate (16.2% train, 16.0% test).

### Classification

(`03_classification_models.ipynb`): Logistic Regression (`class_weight="balanced"`) and LightGBM (`scale_pos_weight` set to the actual train-set class ratio), both against class imbalance rather than left at library defaults. Compared two ways deliberately: a single 80/20 split (shown for context, explicitly *not* used to pick the winner) and 5-fold stratified cross-validation on the training set (the metric actually used to decide). LightGBM wins on CV PR-AUC and is carried forward for SHAP explanation and the downstream risk scores.

### Survival analysis

(`04_survival_analysis.ipynb`): Kaplan-Meier curves and a Cox proportional-hazards model, duration = `YearsAtCompany`, event = `Attrition`, via `lifelines`. This answers a different question than the classifier — not just who's at risk, but how much faster a given factor moves someone toward leaving, expressed as a hazard ratio, holding the other covariates constant.

### Cost of attrition

(`05_cost_of_attrition.ipynb`): each employee's classifier-derived risk score is combined with a standard 50%-of-annual-salary replacement-cost assumption to produce a per-employee and department-level expected attrition cost. A retention-intervention ROI scenario then targets a specific, narrow group (top 20% by risk score, AND working overtime) against a stated intervention cost and assumed risk-reduction effectiveness, exporting `data/processed/fact_risk_scores.csv` for both dashboards to consume.

### Dashboards

: a 4-page Power BI report (`dashboard/Dashboard-employee-attrition.pbix`, with live What-If DAX parameters for the ROI scenario on the last page, plus a static PDF export) and a matching 4-page Streamlit app (`dashboard/streamlit_app.py`) built off the same exported CSV, so the cost/effectiveness assumptions behind the ROI number are adjustable rather than fixed in a notebook.

**Scope note — Prophet was dropped.** An earlier version considered Prophet for a time-series angle on attrition trends; it was dropped after a Windows long-path install failure (Prophet's bundled Stan/TBB library paths exceed Windows' 260-character limit), and the project moved forward with the Logistic Regression vs. LightGBM comparison instead rather than working around the install issue.

## Data

- Source: IBM HR Analytics Employee Attrition & Performance (Kaggle), 1,470 employees, 35 raw columns, a single flat snapshot (no time series).
- Class balance: 84% No / 16% Yes (1,233 stayed / 237 left) — drove the choice of PR-AUC over accuracy and stratified splits throughout.
- After dropping 3 constant columns and the ID column, one-hot encoding 8 categorical columns (including the engineered `overtime_role`) with `drop_first=True`: 62 features.
- Split: 80/20 stratified on `Attrition` (`random_state=42`) → 1,176 train rows (16.2% attrition) / 294 test rows (16.0% attrition, 47 positive cases).

## Results

Classification — winner picked on 5-fold cross-validated PR-AUC, not the single split, because the single split actually reverses the ranking:

| Model | Single-split PR-AUC | 5-fold CV PR-AUC (winner metric) | Recall, leavers* | Precision, leavers* |
|---|---|---|---|---|
| Logistic Regression | 0.552 | 0.478 (±0.052) | 0.64 | 0.33 |
| **LightGBM (winner)** | 0.478 | **0.578 (±0.053)** | 0.26 | 0.52 |

\* From the single 80/20 split's classification report at the default 0.5 threshold — shown for context on what each model's predictions look like, not used to pick the winner. LightGBM's CV PR-AUC (0.578) clearly beats Logistic Regression's (0.478), reversing what the single split alone would suggest.

Survival analysis: Cox model concordance index **0.797** (correctly ranks who leaves sooner ~80% of the time). Hazard ratios:

| Factor | Hazard ratio | Interpretation |
|---|---|---|
| **OverTime** | **3.19** | ~3.2x faster attrition, holding other factors constant |
| JobSatisfaction | 0.79 | Each point (1–4 scale) cuts risk ~21% |
| DistanceFromHome | 1.02 | ~2% higher risk per mile |
| MonthlyIncome | 1.00 | Statistically significant, but a tiny per-dollar effect |

Log-rank test on the OverTime Kaplan-Meier split: p < 0.0001. Independently confirmed by SQL (`sql/06_query_tenure_bucket.sql`): attrition rate by tenure bucket falls monotonically — 0–1 yrs 34.9%, 2–4 yrs 18.1%, 5–9 yrs 11.1%, 10+ yrs 10.4%.

Overtime and role compound: overtime workers leave at 30.5% vs. 10.4% for those without it; Sales Representatives working overtime hit 66.7% attrition, the single highest-risk group in the data.

Cost of attrition: **$10.15M** total expected annual attrition cost workforce-wide. R&D carries the largest total dollar cost ($6.89M, 68%) purely on headcount size, while Sales has the higher per-employee rate (20.6% vs. R&D's 13.8%); HR is smallest on both counts ($0.63M, 6%).

Retention intervention ROI (target: top 20% by risk score AND working overtime; $2,000/employee intervention cost; assumed 30% risk reduction):

| Metric | Value |
|---|---|
| Employees targeted | 81 |
| Intervention cost | $162,000 |
| Expected savings | $966,663 |
| **ROI** | **497%** |

This number is entirely a function of the two stated assumptions (cost per employee, risk-reduction effectiveness), not a measured outcome. The dashboard's What-If sliders show it's genuinely sensitive to targeting: widening the target group to 45% of the workforce with a weaker 10% risk-reduction assumption turns the same calculation negative.

## What I'd do differently / limitations

- **The single-split-vs-CV reversal is the most important methodological finding in this project, and it's worth stating plainly: with only 47 positive test cases, a single train/test split is not a reliable way to rank two models here.** Anyone re-running just the single-split cell would pick Logistic Regression and be wrong by the project's own more trustworthy metric.
- **The $2,000 intervention cost and 30% risk-reduction figure are stated assumptions, not measured numbers.** Neither comes from a real HR program's historical data; the ROI figure should be read as "what this would be worth if these hold," and the dashboard's sliders exist specifically so a reader doesn't have to take the default numbers on faith.
- **The 50%-of-salary replacement-cost assumption behind the $10.15M figure is a common industry rule of thumb, not this company's actual measured replacement cost** — a real HR team would have (or could measure) a better number.
- **SHAP was only run on the winning LightGBM model**, and only as a summary bar chart in the notebook — there's no committed table of exact mean-|SHAP| values or per-feature ranking beyond the plot itself, so a reader can see which features matter visually but can't pull an exact ranked list from this README alone. [ADD: top-10 SHAP feature ranking with values, from `03_classification_models.ipynb`'s SHAP cell, if a precise list is needed.]
- **No time dimension in the underlying data.** This is a single flat snapshot (1,470 employees at one point in time), not a longitudinal dataset, so the model can't distinguish a genuine trend from a one-time snapshot artifact, and Prophet (which would have needed a time series) was dropped for a Windows install issue rather than a data-availability one — worth revisiting if a longitudinal version of this dataset is ever available.
- **`role_tenure_ratio`'s divide-by-zero guard (replacing `YearsAtCompany=0` with 1) is a reasonable but arbitrary choice** — it silently treats a brand-new employee's ratio as if they'd been there one year, rather than flagging tenure-zero rows separately.
- **The Cox model's hazard ratios are reported without confidence intervals in this README** — the underlying `lifelines` output has them, but they're not surfaced here. [ADD: 95% CI for each hazard ratio from `04_survival_analysis.ipynb`.]

## Stack

- `pandas`, `numpy` for data handling and feature engineering
- `scikit-learn` (`LogisticRegression`, `train_test_split`, `StratifiedKFold`, `cross_val_score`, `precision_recall_curve`, `average_precision_score`) for the classification comparison and threshold analysis
- `LightGBM` (`LGBMClassifier`) for the winning classification model
- `SHAP` (`TreeExplainer`) for feature importance on the LightGBM model
- `lifelines` for Kaplan-Meier curves and the Cox proportional-hazards model
- `PostgreSQL`, run via `psql`/pgAdmin, for the independent star-schema SQL layer
- `Streamlit` for the 4-page interactive dashboard
- `Power BI` for the 4-page stakeholder dashboard with live What-If DAX parameters
- `matplotlib` for EDA and SHAP plots
