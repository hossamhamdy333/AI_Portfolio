<div align="center">

# Credit Card Fraud Detection

End-to-end fraud detection on a highly imbalanced (0.17% fraud) credit card transaction dataset — sampling strategy, threshold tuning, and metrics chosen for imbalance, not accuracy.

`Python 3.12` `scikit-learn 1.8` `XGBoost 3.2` `LightGBM 4.6` `imbalanced-learn` `SHAP` `MLflow 3.10` `SQL (Oracle)`

</div>

---

### Contents

- [Overview](#overview)
- [Results](#results)
- [Key findings](#key-findings)
- [Project structure](#project-structure)
- [SQL layer (Oracle)](#sql-layer-oracle)
- [Tech stack](#tech-stack)
- [How to run](#how-to-run)
- [Key visualizations](#key-visualizations)
- [What I learned](#what-i-learned)

## Overview

The dataset is extremely imbalanced — about 599 legitimate transactions for every 1 fraud case. The project focuses on sampling strategies, decision-threshold tuning, and metrics that actually matter for imbalanced data, rather than accuracy.

## Results

| Model | Recall | Precision | F1 | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost (no sampling, class weights)** | **0.7895** | **0.9036** | **0.8427** | **0.8183** | **0.9748** |
| LightGBM (SMOTE) | 0.8211 | 0.5200 | 0.6367 | 0.8040 | 0.9670 |
| XGBoost (SMOTE) | 0.8000 | 0.5507 | 0.6524 | 0.8034 | 0.9698 |

**Best model: XGBoost, no sampling, class weights — AUC-PR 0.8183**

At the default 0.5 threshold this model catches 79% of fraud with 90% precision. After tuning the decision threshold against an estimated business cost (missed fraud vs. false alarms), the optimal threshold of 0.05 catches 83% of fraud and lowers total estimated cost from $2,484 to $2,275 on the test set.

## Key findings

- The dataset is extremely imbalanced — about 599 legitimate transactions for every 1 fraud case
- Class weights on the original data beat every resampling strategy (SMOTE, undersampling, SMOTETomek) on AUC-PR
- Resampling raises recall a little but costs a lot of precision, meaning far more false alarms
- V17, V14, and V3 are the PCA features that separate fraud from legitimate transactions the most
- Tuning the decision threshold against business cost catches more fraud than the default 0.5 threshold, at the cost of more false alarms — worth it here since missed fraud costs far more than a false alarm

## Project structure

```
Credit_Fraud_Detection/
├── notebooks/
│   ├── EDA.ipynb              # Exploratory data analysis
│   ├── preprocessing.ipynb    # Cleaning, feature engineering, sampling
│   ├── modeling.ipynb         # Model training, comparison, threshold tuning
│   └── explainability.ipynb   # SHAP and MLflow tracking
├── sql/
│   ├── 01_create_tables_oracle.sql          # staging table matching creditcard.csv's schema
│   ├── transactions_raw.ctl                 # SQL*Loader control file to bulk-load the CSV
│   └── 02_fraud_analysis_queries_oracle.sql # fraud rate by hour / by amount bucket / both combined
├── results/
│   ├── shap_summary.png              # SHAP feature importance
│   ├── model_comparison.png          # Model comparison chart
│   ├── pr_curve.png                  # Precision-recall curve
│   ├── threshold_analysis.png        # Cost vs threshold analysis
│   ├── correlation_heatmap.png       # Feature correlations
│   └── ...                           # All other plots
├── .gitignore
├── requirements.txt
└── README.md
```

## SQL layer (Oracle)

Independent, SQL-only fraud-rate analysis straight from the raw CSV — no model needed, same idea as the SQL layers in `customer_churn_prediction`/`ecommerce-demand-forecasting`/`employee-attrition`, Oracle instead of Postgres this time.

**Why Oracle specifically, not Postgres again:** fraud detection is one of the most common real-world Oracle domains — transaction systems of record in banking/fintech very often run on it. Repeating Postgres a fourth time would've said less about range than picking the engine that's actually the industry default for exactly this kind of data.

Three things live here:
1. **Fraud rate by hour-of-day** — independently recreates `EDA.ipynb`'s own hour derivation (`Hour = (Time/3600) % 24`) in SQL, then goes further than the notebook's plot with an actual rate and a window function comparing each hour to the overall average.
2. **Fraud rate by amount bucket** — new analysis, not in the notebook (which only plots amount as a histogram/boxplot, no discrete buckets).
3. **Both combined** — the 10 riskiest hour × amount-bucket combinations together, to see whether the two risk factors compound or are independent.

**Setup:**
```bash
# Oracle XE (free, official Docker image) if you don't have an Oracle
# instance already:
docker run -d -p 1521:1521 -e ORACLE_PASSWORD=<a-real-password> gvenzl/oracle-xe:21-slim

sqlplus <user>/<password>@localhost:1521/XEPDB1 @sql/01_create_tables_oracle.sql
sqlldr userid=<user>/<password>@localhost:1521/XEPDB1 control=sql/transactions_raw.ctl log=load.log
sqlplus <user>/<password>@localhost:1521/XEPDB1 @sql/02_fraud_analysis_queries_oracle.sql
```

**Honest limitation, stated plainly:** unlike the Postgres SQL layers elsewhere in this portfolio, these queries haven't been run against a live Oracle instance — there's no in-memory Oracle equivalent to SQLite to test against without standing up a real server first. What *was* verified: the hour-bucketing arithmetic (`FLOOR(MOD(time_seconds/3600, 24))`) against several boundary values (0, 3599, 3600, 86399) to confirm it matches pandas' `int((Time/3600) % 24)` exactly, and the overall grouping/bucketing approach against a synthetic dataset with an injected fraud signal at night hours and small amounts — both correctly surfaced by the same logic these queries use. Standard Oracle SQL syntax throughout (no exotic features), so the risk of an actual syntax error is low, but "verified logic, unexecuted against real Oracle" is a real gap from "tested," not the same claim.

## Tech stack

| Category | Tools |
|---|---|
| Data | `pandas` `numpy` `scipy` |
| Visualization | `matplotlib` `seaborn` |
| ML | `scikit-learn` `XGBoost` `LightGBM` |
| Imbalanced data | `imbalanced-learn` (SMOTE, undersampling, SMOTETomek) |
| Explainability | `SHAP` |
| Tracking | `MLflow` |

## How to run

```bash
# Clone the repo
git clone https://github.com/hossamhamdy333/AI_Portfolio
cd AI_Portfolio/Credit_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Download the data from Kaggle
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# place creditcard.csv in a data/ folder in this project

# Run notebooks in order
# 1. EDA.ipynb
# 2. preprocessing.ipynb
# 3. modeling.ipynb
# 4. explainability.ipynb
```

## Key visualizations

**SHAP feature importance**
![SHAP Summary](results/shap_summary.png)

**Model comparison**
![Model Comparison](results/model_comparison.png)

**Threshold analysis**
![Threshold Analysis](results/threshold_analysis.png)

## What I learned

- AUC-PR is a much more useful metric than accuracy or AUC-ROC at this level of class imbalance
- Resampling is not always the answer — class weights on the original data outperformed every SMOTE variant here
- The right decision threshold depends on the actual cost of a missed fraud vs. a false alarm, not just on the model's metrics
- V14 stands out as important in both the EDA analysis and the SHAP importance, though the two rankings differ beyond that — EDA and SHAP are measuring different things (raw separation vs. actual contribution to the trained model's predictions)
