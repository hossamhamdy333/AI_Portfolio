<div align="center">

# Customer Churn Prediction — Model, SQL, and Two Dashboards

`pandas` `scikit-learn` `XGBoost` `SHAP` `MLflow` `joblib` `PostgreSQL` `Streamlit` `Power BI`

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

Predicts churn for a 7,043-customer telecom dataset (26.54% churn rate), then turns that model into something a retention team could actually use: a SQL layer that reproduces the segmentation independently of Python, a Streamlit app for day-to-day use, and a Power BI dashboard for stakeholders who live in Excel, not notebooks. Four models are compared (Logistic Regression, Random Forest, XGBoost, LightGBM); Random Forest wins on ROC-AUC (0.8418) and gets isotonic-calibrated so its probabilities are trustworthy, not just well-ranked (Brier score 0.1587 → 0.1363, a 14.1% improvement). The classification threshold isn't the default 0.5 — it's the one that maximizes expected profit given a retention-offer cost model, which comes out to 0.01 under this project's cost assumptions: essentially "flag almost everyone." A SQL/pandas cross-check on the resulting priority matrix caught a real discrepancy — `pandas.cut()` silently drops customers with a predicted probability of exactly 0.0 — that's documented rather than fixed by rounding it away.

## Problem & motivation

Churn prediction is a fairly standard tabular classification problem, but the part that actually matters for a retention team isn't the model, it's the decision the model feeds: who gets called, and is calling them worth it. Two things separate a demo from something usable here. First, a model's raw predicted probabilities aren't automatically trustworthy just because its ranking (ROC-AUC, AUC-PR) is good — a model can rank customers correctly by risk while still being badly overconfident or underconfident about the actual probability, which matters a lot once that probability gets plugged into a cost calculation. Second, "which customers to prioritize" is really a two-dimensional question — risk of leaving times value if they stay — and collapsing it to one number throws away the fact that a low-value high-risk customer and a high-value high-risk customer deserve different responses.

The naive approach is to pick the model with the best accuracy, apply a 0.5 cutoff, and call it done. That's close to the worst threshold available here: at 0.5, the profit-maximizing calculation below actually loses money ($-2,602 in expected profit on the test set), because a missed churner (an average customer worth $2,279.73) costs far more than a wasted $55 outreach attempt on someone who wasn't going to leave anyway. A model that's individually accurate can still recommend the wrong action if the threshold ignores the asymmetry in that cost.

## Approach

### Preprocessing

(`EDA_and_Preprocessing.ipynb`): `TotalCharges` arrives as text because 11 rows are blank (all customers with `tenure = 0`, i.e. brand new); those get parsed to numeric and filled with 0. All categorical columns get plain `LabelEncoder` (not one-hot — chosen to keep the feature count manageable for the tree models, at the cost of imposing an arbitrary ordinal relationship the linear model doesn't actually need). Five engineered features: `tenure_group` (binned into 0-1/1-2/2-4/4-5/5-6 year buckets), `charges_per_tenure` (falls back to `MonthlyCharges` for zero-tenure customers to avoid a divide-by-zero), `num_services` (count of active add-on services), and two rule-based flags, `high_value` (`MonthlyCharges` above the 75th percentile AND tenure > 24 months) and `high_risk` (month-to-month contract AND `MonthlyCharges` above the median). Split 80/20, stratified on churn (`random_state=42`), giving 5,634 train / 1,409 test rows, both preserving the 26.54% churn rate.

### Modeling

(`Modeling_and_Evaluation.ipynb`): four models, all with `class_weight='balanced'` or `scale_pos_weight` set to the actual class ratio rather than left at defaults, since a naive classifier on 73/27 data can hit reasonable accuracy by mostly predicting "no churn." Compared on ROC-AUC, AUC-PR, recall, precision, F1, and Brier score (calibration quality) on the held-out test set. Random Forest wins on ROC-AUC and AUC-PR and gets carried forward.

### Calibration

`CalibratedClassifierCV` with 5-fold internal cross-validation and isotonic regression, fit on the training set only (the calibrator needs data the base model hasn't already seen, so it can't be fit on the same rows the base model trained on). This matters specifically because the profit calculation downstream needs realistic probabilities, not just a good ranking — ROC-AUC is invariant to any monotonic transformation of the scores, so calibration doesn't change it, but Brier score (a direct measure of probability accuracy) does.

### Threshold selection

Rather than the default 0.5, the project sweeps thresholds from 0.01 to 0.99 and picks the one that maximizes `TP × (avg_customer_value − offer_cost − outreach_cost) + FP × (−offer_cost − outreach_cost) + FN × (−avg_customer_value)` on the calibrated test-set probabilities, using $50 as the assumed retention-offer cost and $5 as the assumed cost to reach a customer.

### SQL layer

Two files, run against Postgres. `01_create_tables_postgres.sql` stages the raw CSV schema plus a `scored_customers` table for the model's exported probabilities. `02_segment_queries_postgres.sql` does two things: pure rule-based analysis that needs no model at all (churn by contract type; churn by tenure band, with a window function comparing each band to the overall average; the `high_risk`/`high_value` flags recreated exactly as SQL, using `PERCENTILE_CONT` for the same thresholds the notebook computes with pandas), and an independent rebuild of the priority matrix from the model's exported scores, using `NTILE(3)` for value tertiles and the same bin edges as the notebook's `pd.cut()` — but written with `<=` throughout instead of pandas' default half-open intervals.

### Two dashboards on the same two exported CSVs

A 4-page Streamlit app (`streamlit_app.py`: churn overview, calibration curve, priority matrix as a sortable table with a call-list view, and a what-if page with offer cost / outreach cost / threshold as live sliders) and a 2-page Power BI file (`churn_dashboard.pbix`: an executive overview and a retention-priority page with the same three sliders feeding an Expected Profit measure in DAX). Built to make the threshold and cost assumptions adjustable by whoever owns the retention budget, rather than locking the notebook's specific dollar figures into a dashboard nobody can change.

## Data

- Source: Kaggle's Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), 7,043 customers, 21 raw columns, 26.54% churn rate (1,869 churned / 5,174 retained).
- 3 numeric columns as loaded (`SeniorCitizen`, `tenure`, `MonthlyCharges`); `TotalCharges` is technically numeric but arrives as text due to 11 blank values, all zero-tenure customers.
- After preprocessing: 24 features (dropping `customerID`), split 80/20 stratified — 5,634 train rows / 1,409 test rows, churn rate preserved to four decimal places in both (0.2654).
- Mann-Whitney U tests confirm all three original numeric features differ significantly between churned and retained customers (`tenure`: p ≈ 2.42e-208; `MonthlyCharges`: p ≈ 3.31e-54; `TotalCharges`: p ≈ 5.69e-83). Correlation with churn: `tenure` −0.352, `TotalCharges` −0.198, `MonthlyCharges` +0.193, `SeniorCitizen` +0.151 — tenure is the strongest single linear signal, in the direction you'd expect (longer-tenured customers churn less).
- The rule-based `high_risk` flag (month-to-month + above-median charges) separates the base cleanly on its own, no model required: 52.8% churn rate for flagged customers vs. 15.8% for everyone else.

## Results

Four-model comparison on the held-out test set (1,409 customers), pre-calibration:

| Model | ROC-AUC | AUC-PR | Recall | Precision | F1 | Brier |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.8418** | **0.6544** | 0.7834 | 0.5416 | 0.6404 | 0.1587 |
| Logistic Regression | 0.8387 | 0.6238 | 0.8048 | 0.5050 | 0.6206 | 0.1699 |
| LightGBM | 0.8333 | 0.6418 | 0.7380 | 0.5359 | 0.6209 | 0.1602 |
| XGBoost | 0.8310 | 0.6336 | 0.6925 | 0.5243 | 0.5968 | 0.1585 |

Random Forest wins on ROC-AUC and AUC-PR despite XGBoost having a marginally better raw Brier score — calibration is what actually fixes RF's probability quality, taking its Brier score from 0.1587 to 0.1363 (14.1% improvement) via isotonic regression.

Threshold optimization on the calibrated model's probabilities, using $50 offer cost / $5 outreach cost / $2,279.73 average customer value:

| Threshold | Expected profit |
|---|---|
| Default (0.50) | −$2,602 |
| **Optimal (0.01)** | **$775,241** |

The optimal threshold of 0.01 means treating nearly the entire customer base as a churn risk — not a targeted list. That's not a bug in the search, it's what the math says when a missed churner ($2,279.73) costs roughly 41x more than a wasted $55 outreach attempt: false negatives are so expensive relative to false positives that the model recommends acting on almost everyone. Whether that's the right call operationally depends on real campaign capacity and offer acceptance rates the notebook doesn't have, which is exactly why both dashboards expose offer cost, outreach cost, and threshold as adjustable sliders rather than hardcoding 0.01.

SHAP (`TreeExplainer` on the uncalibrated Random Forest — the model actually deployed and calibrated) ranks feature importance as: `Contract`, `OnlineSecurity`, `tenure`, `high_risk`, `TechSupport`, `tenure_group`, `TotalCharges`, `InternetService`, `charges_per_tenure`, `MonthlyCharges` — the engineered `high_risk` flag lands in the global top 4, and `charges_per_tenure` outranks the raw `MonthlyCharges` it's derived from. A single example: the customer the model is most confident will churn (predicted probability 0.9678) did in fact churn.

The priority matrix (test set, 1,409 customers, risk bands from calibrated probability × value tertiles from `MonthlyCharges`):

| Value \ Risk | Very Low | Low | Medium | High | Very High |
|---|---|---|---|---|---|
| Low Value | 319 | 73 | 50 | 12 | 0 |
| Mid Value | 220 | 92 | 77 | 61 | 20 |
| High Value | 186 | 109 | 92 | 55 | 25 |

These cells sum to 1,391, not 1,409. The missing 18 are customers whose predicted probability came out to exactly 0.0 — `pandas.cut()`'s default bins are left-exclusive, so a value of exactly 0 falls outside every bin and gets silently dropped rather than landing in "Very Low." The SQL rebuild of this same matrix uses `<=` throughout and keeps all 1,409 rows, which is a genuine, documented discrepancy between the two implementations, not a rounding difference. The retention call list — High Value + High/Very High risk — is 55 + 25 = **80 customers**.

## What I'd do differently / limitations

- **`pandas.cut()`'s dropped-zero-probability rows is a real gap, not just a footnote.** 18 of 1,409 test customers (1.3%) are silently excluded from the notebook's own risk segmentation and every plot built on it, while the SQL version keeps them. Anyone comparing the Python-side numbers to the SQL-side numbers side by side will see a mismatch in the "Very Low" bucket and needs to know why.
- **The profit-optimal threshold of 0.01 is entirely a function of the assumed $50/$5 cost inputs**, which aren't measured — they're stated assumptions. A 2x change to either number would move the optimal threshold meaningfully, and the project doesn't show that sensitivity beyond letting the dashboards' sliders explore it interactively.
- **Two different "high value" definitions exist in this repo and mean different things.** The EDA notebook's `high_value` flag is `MonthlyCharges` above the 75th percentile AND tenure > 24 months (a rule-based flag, unused after EDA). The segmentation notebook's `value_segment` is `MonthlyCharges` tertiles with no tenure condition at all. They're never reconciled or even cross-referenced, so a reader skimming both notebooks could reasonably think there's one "high value" concept when there are actually two, with different customers qualifying under each.
- **The profit model assumes every retention offer succeeds and every non-churner contacted is a pure loss** — a genuine simplification the notebook itself flags rather than hides, but it means "act on almost everyone" is a ceiling case that would look very different against a realistic offer-acceptance rate below 100%.
- **Calibration is checked on the same test set it's evaluated on.** There's no separate calibration-holdout distinct from the final test set, so the reported 14.1% Brier improvement is optimistic to whatever degree the calibrator is (mildly) fit to this particular test split via the 5-fold internal CV during `fit()`.
- **Label encoding, not one-hot, for nominal categorical features** (e.g. `PaymentMethod`, `InternetService`) imposes an arbitrary ordinal relationship a tree model can exploit as spurious splits and a linear model shouldn't be given at all. It's a reasonable tradeoff against feature-count blowup on a 15-categorical-column dataset, but it's a tradeoff, not a free choice.
- **No model card or drift-monitoring plan.** The Telco dataset is a static snapshot; there's nothing here that checks whether a production customer base still resembles this training distribution over time.

## Stack

- `pandas` 2.3.3, `numpy` 2.4.3, `scipy` 1.17.1 (Mann-Whitney U tests) for data handling and EDA
- `scikit-learn` 1.8.0 (`LogisticRegression`, `RandomForestClassifier`, `CalibratedClassifierCV`, `StratifiedKFold`, calibration/ROC/PR metrics)
- `XGBoost` 3.2.0, `LightGBM` 4.6.0 for the gradient-boosted comparison models
- `SHAP` 0.51.0 for feature importance and individual-prediction explanations
- `MLflow` 3.10.1, local database-backed tracking store, for run logging
- `joblib` for model persistence (`models/calibrated_model.pkl`, gitignored — retrain to regenerate)
- `PostgreSQL`, run via `psql`/pgAdmin, for the independent SQL segmentation layer
- `Streamlit` 1.61.1 for the 4-page interactive dashboard
- `Power BI` for the 2-page stakeholder dashboard (`churn_dashboard.pbix`), DAX for the live Expected Profit measure
-e 

---

# Supplementary document: `reports/segment_summary.md`

<div align="center">

# Customer Retention — Who to Call This Week

</div>

> **Real data.** Numbers below are pulled directly from the executed notebooks (`notebooks/EDA_and_Preprocessing.ipynb`, `Modeling_and_Evaluation.ipynb`, `Customer_Segmentation.ipynb`) run on the actual Telco dataset, not the fixture. Two numbers — average monthly charge and average churn probability *specifically for the call list* — aren't printed anywhere in the notebooks as run, so they're left as "needs `data/scored_test.csv`" below rather than guessed. See the note at the bottom for how to fill them in.

---

### Contents

- [The headline](#the-headline)
- [Where the 80 customers come from](#where-the-80-customers-come-from)
- [What the other segments mean](#what-the-other-segments-mean)
- [The bigger picture](#the-bigger-picture-why-this-matters-at-all)
- [The profit-threshold picture](#the-profit-threshold-picture)
- [Caveats, in plain terms](#caveats-in-plain-terms)

## The headline

**80 customers** are High Value and High or Very High risk of churning. That's the group `Customer_Segmentation.ipynb` calls out for "immediate personal outreach, best offer."

If even half of them are saved with a retention offer, that's real revenue protected for the cost of 80 phone calls. That's the trade the priority matrix exists to make visible.

## Where the 80 customers come from

The model scores every customer's churn probability, then two independent cuts are laid on top of each other:

- **Value**: monthly charges, split into thirds (Low / Mid / High Value).
- **Risk**: predicted churn probability, split into five bands (Very Low → Very High).

The full grid, customer counts per cell (test set, from `results/priority_matrix.png`):

| Value \ Risk | Very Low | Low | Medium | High | Very High |
|---|---|---|---|---|---|
| Low Value  | 319 | 73  | 50 | 12 | 0  |
| Mid Value  | 220 | 92  | 77 | 61 | 20 |
| High Value | 186 | 109 | 92 | 55 | 25 |

The call list is the bottom-right corner: **High Value** rows, **High** and **Very High** risk columns (55 + 25 = **80**).

Note: these cells sum to 1,391, not the full 1,409-customer test set. The missing 18 are customers whose predicted churn probability came out to exactly 0.0 — pandas' `pd.cut()` excludes the lowest bin edge by default, so they get silently dropped from the notebook's risk_segment column rather than landing in "Very Low." This isn't a bug introduced by this report; it's how the notebook's own `pd.cut()` call behaves. The SQL version in `sql/02_segment_queries_postgres.sql` uses `<=` throughout and keeps all 1,409, so running that query against the real data will show a slightly higher "Very Low" count than the plot does. Worth knowing if the two are compared side by side.

## What the other segments mean

| Segment | Strategy | Why |
|---|---|---|
| High Value + High/Very High Risk | Immediate personal outreach, best offer | Highest revenue at stake, most likely to leave |
| High Value + Medium Risk | Proactive engagement, loyalty rewards | Worth protecting before risk climbs further |
| Low Value + High Risk | Automated, low-cost retention campaign | Not worth a phone call, but an email or discount code is cheap insurance |
| Low/Mid Value + Low/Very Low Risk | No action needed | Not at risk, don't spend budget here |

## The bigger picture (why this matters at all)

- Overall churn rate: **26.54%** of customers.
- Average revenue per user (ARPU): **$64.76/month**.
- Average tenure: **32.4 months**.
- Average customer lifetime value (mean `TotalCharges`, the value figure used in the profit-threshold calculation): **$2,279.73**.
- Rule-based high-risk customers (month-to-month contract, above-median monthly charge) churn at **52.8%**, vs. **15.8%** for everyone else — see `EDA_and_Preprocessing.ipynb`, cell 42.
- Contract-type breakdown (month-to-month vs. one-year vs. two-year churn rate) isn't printed in the notebooks as run — run query 1 in `sql/02_segment_queries_postgres.sql` against the real CSV in pgAdmin to get exact figures. On the earlier fixture that query showed month-to-month churning roughly 8x more than two-year contracts; expect something in that range on real data, but don't quote a number until the query's been run.

## The profit-threshold picture

`Modeling_and_Evaluation.ipynb`'s threshold optimization (cell 28), using $50 as the retention offer cost and $5 as the cost to reach a customer:

- **Optimal threshold: 0.01** — i.e., the model recommends treating almost every customer as a churn risk under these cost assumptions.
- **Expected profit at optimal threshold: $775,241**
- **Expected profit at the default 0.50 threshold: –$2,602**
- **Improvement: $777,843**

Worth flagging to whoever owns the retention budget: a threshold of 0.01 means acting on nearly the entire customer base, not a targeted list. That math only works because the model's average predicted probability across low-risk customers is still low enough that reaching out rarely wastes the $55 in campaign + discount cost, against a $2,280 average customer value. If the real cost of a retention offer is higher than $50, or the offer acceptance rate is much less than 100%, the optimal threshold moves — that's exactly what the dashboard's What-If page is for.

## Caveats, in plain terms

- This is a snapshot of the **test set only** (1,409 of 7,043 customers), the same set the notebook evaluates the model on. It's a representative sample, not the full customer base — the SQL queries can be pointed at the full base for a company-wide count once every customer has been scored, not just the held-out test set.
- The risk bands come from a model, and models are wrong sometimes. Treat "Very High risk" as "worth a closer look," not as certainty.
- Average monthly charge and average predicted probability *for the 80 customers on the call list specifically* aren't in this report because they require reading `data/scored_test.csv` directly rather than a notebook printout. Quick way to get them:
  ```python
  import pandas as pd
  scored = pd.read_csv("data/scored_test.csv")
  call_list = scored[(scored.value_segment == "High Value") &
                      (scored.risk_segment.isin(["High", "Very High"]))]
  print(len(call_list), call_list.MonthlyCharges.mean(), call_list.churn_probability.mean())
  ```
- The profit-threshold numbers assume every retention offer succeeds and every non-churner contacted is a pure loss — a simplification the notebook itself flags. Treat the optimal threshold as a starting point for discussion with whoever owns the retention budget, not a rule to follow automatically.
