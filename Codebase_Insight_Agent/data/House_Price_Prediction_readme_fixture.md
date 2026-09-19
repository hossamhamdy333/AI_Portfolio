<div align="center">

# House Price Prediction — Advanced Regression

`pandas` `matplotlib` `scikit-learn` `XGBoost` `SHAP` `Optuna` `MLflow` `jupyter`

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

Predicts Ames, Iowa house sale prices from 79 structural and quality features. The pipeline runs outlier removal, missing-value imputation, skew correction, feature engineering, and categorical encoding, then compares six models (Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM) plus a stacking ensemble under 5-fold cross-validation on log-transformed price. The best result is a Ridge/Lasso/XGBoost/LightGBM stacking ensemble with a Ridge meta-learner: **CV RMSE 0.1104** on log(price), against a Ridge baseline of 0.1119. Optuna tuning improves XGBoost by 4.9% and LightGBM by 3.6% over their untuned baselines. SHAP confirms `OverallQual` and `GrLivArea` as the two strongest predictors, with two engineered features (`TotalBathrooms`, `OverallScore`) both ranking in the top 4.

## Problem & motivation

This is the Kaggle "House Prices: Advanced Regression Techniques" dataset — 79 features (structural, quality, and neighborhood attributes) predicting `SalePrice`, evaluated on RMSE of log-transformed price so a $50k error on a $600k house and a $50k error on a $100k house aren't treated as equally bad.

The naive version of this problem is "throw XGBoost at the raw CSV." Two things make that undersell the dataset: first, `SalePrice` is strongly right-skewed (skewness 1.8829 on the raw scale) and several individual features are skewed even harder (`MiscVal` at 24.5, `PoolArea` at 14.8), so a model trained without correcting for that spends its capacity fighting the skew rather than learning the relationship. Second, several features that look numeric are actually near-categorical (garage/basement quality ratings encoded as text with a genuine order — `Po` < `Fa` < `TA` < `Gd` < `Ex`), and treating them as arbitrary categories throws away information a model could use directly. Getting both of those right, and engineering a few interaction features on top, is most of what separates a mid-pack score from a good one on this specific dataset.

## Approach

- **Outlier removal**: `remove_outliers()` in `src/preprocessing.py` drops any training row with `GrLivArea > 4000`, regardless of price. EDA specifically flagged two houses (`Id` 524 and 1299) as suspicious — large living area, unusually low sale price — but the actual filter is a blunter square-footage cutoff, which removed 4 rows, not 2. It happens to catch the two flagged outliers plus two others.
- **Missing values**: three different strategies depending on what "missing" means for a given column, all inside `handle_missing_values()`:
  - Categorical columns where `NaN` means "feature doesn't exist" (`PoolQC`, `Alley`, `FireplaceQu`, `GarageType`, `BsmtQual`, etc.) → filled with the string `'None'`.
  - Numerical columns with the same meaning (`GarageArea`, `TotalBsmtSF`, `MasVnrArea`, etc.) → filled with `0`.
  - `LotFrontage` → filled with the **neighborhood median**, since lot frontage is a local-context property, not a global average.
  - `MSZoning` → filled with the mode within the same `MSSubClass`.
  - A handful of low-missing categorical columns (`Electrical`, `KitchenQual`, `Exterior1st/2nd`, `SaleType`, `Functional`) → global mode.
  - `Utilities` is dropped outright — near-zero variance, no predictive signal.
- **Skew correction**: any numeric feature with skew > 0.75 gets `log1p`-transformed. 20 features crossed that threshold on this dataset.
- **Feature engineering** (`engineer_features()`): `TotalSF` (basement + 1st + 2nd floor area), `TotalBathrooms` (full baths + 0.5×half baths, basement included), `HouseAge` and `YearsSinceRemodel` (both relative to sale year), a `Remodeled` flag, `TotalPorchSF`, four binary flags (`HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace`), and two interaction terms, `OverallScore` (quality × condition) and `GarageScore` (cars × area).
- **Encoding**: quality-rated columns (`ExterQual`, `KitchenQual`, `HeatingQC`, etc.) use an explicit ordinal map (`None=0 … Ex=5`) rather than arbitrary label encoding, since the order is real. Other ordinal-ish columns (`BsmtExposure`, `LotShape`, `Fence`, etc.) get plain `LabelEncoder`. Everything else nominal is one-hot encoded via `pd.get_dummies`.
- **Modeling**: linear models (Ridge, Lasso, ElasticNet) run inside a `Pipeline` with `RobustScaler`, chosen over `StandardScaler` specifically because it scales on median/IQR and is less distorted by the outliers that survive the `GrLivArea` filter. Tree models (Random Forest, XGBoost, LightGBM) run unscaled, since scaling doesn't affect split-based models.
- **Stacking ensemble**: `StackingRegressor` combining Ridge (α=10), Lasso (α=0.0005), a 300-estimator XGBoost, and a 300-estimator LightGBM as base learners, with a **Ridge (α=10) meta-learner** and internal 5-fold CV — Ridge was picked as the meta-learner specifically to avoid the meta-learner itself overfitting on the base models' out-of-fold predictions.
- **Hyperparameter tuning**: Optuna, TPE sampler (default, not explicitly seeded), minimizing 5-fold CV RMSE — 50 trials for Ridge (just `alpha`), 100 trials each for LightGBM and XGBoost (`n_estimators`, `learning_rate`, `max_depth`, plus regularization and sampling parameters, all on log or uniform scales as appropriate).
- **Final submissions**: two separate weighted blends exist, from two different notebooks, and they don't agree on which models to use. `modeling.ipynb` produces `submission.csv` from `0.1×Ridge + 0.1×Lasso + 0.3×XGBoost + 0.3×LightGBM + 0.2×Stacking` (untuned base models). `Hyperparameters_Tuning.ipynb` produces `submission_tuned.csv` from `0.2×Ridge + 0.4×LightGBM(tuned) + 0.4×XGBoost(tuned)`, dropping Lasso and the stacking ensemble entirely. Neither notebook reconciles these into one canonical "final" model — see limitations.

## Data

- Source: Kaggle's Ames Housing dataset (`house-prices-advanced-regression-techniques`). Raw train: 1,460 rows × 81 columns. Raw test: 1,459 rows × 80 columns (no `SalePrice`).
- `SalePrice` is log1p-transformed before modeling (`np.log1p`), which brings skewness from 1.8829 down to 0.1213.
- After removing 4 outliers, the training set used for everything downstream is 1,456 rows.
- Missing data is real and uneven: `PoolQC` is missing in 99.5% of training rows, `MiscFeature` 96.3%, `Alley` 93.8%, `Fence` 80.8%, `MasVnrType` 59.7%, `FireplaceQu` 47.3%, `LotFrontage` 17.7%; garage-related columns are each missing in 5.5% of rows (the same 81 rows — houses with no garage).
- Train and test are concatenated before cleaning/encoding so both get identical transformations, then split back apart by row count before modeling. Final processed shape after feature engineering and encoding: 2,915 combined rows × 238 columns → `X_train` (1,456 × 238), `X_test` (1,459 × 238).
- No separate validation set: model comparison uses 5-fold `KFold(shuffle=True, random_state=42)` cross-validation on the training set, scored on RMSE of log-transformed price. The 1,459-row test set has no public labels (it's the Kaggle submission set), so every number in Results is a CV number, not a test-set number.

## Results

5-fold CV RMSE on log-transformed `SalePrice` (lower is better):

| Model | CV RMSE | Std |
|---|---|---|
| ElasticNet (α=0.0005, l1_ratio=0.9) | 0.1118 | ±0.0071 |
| Lasso (α=0.0005) | 0.1119 | ±0.0072 |
| Ridge (α=10, baseline) | 0.1119 | ±0.0081 |
| XGBoost (untuned) | 0.1198 | ±0.0105 |
| LightGBM (untuned) | 0.1233 | ±0.0132 |
| Random Forest (500 trees) | 0.1354 | ±0.0129 |
| **Stacking ensemble** | **0.1104** | **±0.0104** |

The three linear models are essentially tied with each other and already competitive with untuned tree models on this dataset — the feature engineering and skew correction do a lot of the work a tree model would otherwise have to learn on its own.

Optuna tuning results (against baselines re-declared inside `Hyperparameters_Tuning.ipynb`, which lists the XGBoost baseline as 0.1207 — inconsistent with the 0.1198 XGBoost actually scored in `modeling.ipynb`; the LightGBM and Ridge baselines match across both notebooks):

| Model | Tuned CV RMSE | Improvement (vs. that notebook's stated baseline) |
|---|---|---|
| Ridge | 0.1119 | 0.0% |
| LightGBM | 0.1189 | 3.6% |
| XGBoost | 0.1148 | 4.9% |

Best tuned XGBoost params found: `n_estimators=830, learning_rate=0.0233, max_depth=4, min_child_weight=1, subsample=0.594, colsample_bytree=0.501, reg_alpha=0.00225, reg_lambda=0.179`. Best tuned LightGBM params: `n_estimators=748, learning_rate=0.0431, max_depth=3, num_leaves=39, min_child_samples=27, subsample=0.501, colsample_bytree=0.522, reg_alpha=0.0106, reg_lambda=4.367`.

**These do not match the "best params from the tuning notebook" hardcoded in `explainability.ipynb`** — that notebook uses `n_estimators=954` for XGBoost (vs. 830) and `n_estimators=484` for LightGBM (vs. 748), with every other parameter also different. The Optuna study isn't seeded, so re-running the tuning cells produces a different "best" each time; whoever copied the params into `explainability.ipynb` did so from a different run than the one whose output is saved in `Hyperparameters_Tuning.ipynb`. MLflow logged from `explainability.ipynb`'s versions: `XGBoost_tuned` RMSE 0.1159 ± 0.0111, `LightGBM_tuned` RMSE 0.1184 ± 0.0129 — close to but not identical to the tuning notebook's numbers, consistent with sampler variance rather than a bug in either notebook individually.

SHAP (`TreeExplainer` on the `explainability.ipynb` XGBoost model, a 1,456 × 238 SHAP value matrix) ranks feature importance as: `OverallQual`, `GrLivArea`, `TotalBathrooms`, `OverallScore`, `TotalBsmtSF`, `LotArea`, `KitchenQual`, `ExterQual`, `GarageScore`, `BsmtFinSF1` — the top 10 by mean absolute SHAP value. Two of the top four (`TotalBathrooms`, `OverallScore`) are engineered, not raw. A single waterfall example: the most expensive house in the training set actually sold for $625,000; the model predicted $579,430 (a $45,570 miss).

Raw-data correlation with `SalePrice` (before any modeling, on untransformed features) ranks `OverallQual` highest at 0.791, followed by `GrLivArea` at 0.709, `GarageCars` at 0.640, and `GarageArea` at 0.623 — consistent with what SHAP finds on the trained model, though the raw ranking has no equivalent for the engineered features since they don't exist yet at that stage.

## What I'd do differently / limitations

- **Optuna isn't seeded, and it shows.** The XGBoost and LightGBM "best" hyperparameters differ meaningfully between `Hyperparameters_Tuning.ipynb`'s own saved output and the copy hardcoded into `explainability.ipynb`, because `optuna.create_study()` doesn't fix a sampler seed. The resulting CV RMSE only moves by about 0.001–0.003 between the two parameter sets, so it isn't costing much here, but it means the specific hyperparameter values in this repo aren't reproducible from the code alone — running the tuning notebook again would produce a third, different set.
- **Two final submissions, not one.** `modeling.ipynb` and `Hyperparameters_Tuning.ipynb` each ship a different weighted blend of different model sets (one includes Lasso and the stacking ensemble, the other drops both and reweights toward tuned LightGBM/XGBoost). Anyone reading only one of the two notebooks would come away with a different idea of what the "final" model actually is. Picking one blend, deleting the other's submission file, and stating why would remove the ambiguity.
- **The outlier filter is a size cutoff, not a price-relative one.** `GrLivArea > 4000` happens to catch the two houses EDA actually flagged as suspicious (large + underpriced), but it would also drop a legitimately large, legitimately expensive house if one existed in a future data refresh, since the rule never looks at price at all.
- **No held-out test-set score to validate the CV estimate against.** Every number in this README is 5-fold CV RMSE on the training set; there's no leaderboard or held-out score confirming the CV estimate generalizes, since the competition test set has no public labels.
- **MLflow tracking is local only** (a local database-backed store, not a hosted server) — runs aren't shared or comparable across machines, and there's no experiment UI screenshot or exported run table in the repo, just the code that produces the runs.
- **Trial counts (50 for Ridge, 100 for LightGBM/XGBoost) look chosen for round numbers, not for a demonstrated convergence point** — there's no plot showing the search had actually plateaued by the last trial versus still improving.
- **A previous version of this README's write-up cited a 0.673 correlation for `TotalBathrooms` that isn't computed anywhere in the current notebooks.** I couldn't verify it against any cell output, so it's left out here rather than repeated. **Open item:** add the notebook cell that computes it if the figure should be reported.

## Stack

- `pandas` 2.3.3, `numpy` 2.4.3, `scipy` 1.17.1 for data handling
- `matplotlib` 3.10.8, `seaborn` 0.13.2 for EDA and result plots
- `scikit-learn` 1.8.0 (`Ridge`, `Lasso`, `ElasticNet`, `RandomForestRegressor`, `StackingRegressor`, `RobustScaler`, `KFold`)
- `XGBoost` 3.2.0, `LightGBM` 4.6.0 for gradient-boosted models
- `SHAP` 0.51.0 for model explainability (summary, waterfall, dependence plots)
- `Optuna` for hyperparameter tuning (TPE sampler, not seeded)
- `MLflow` 3.10.1, local database-backed tracking store
- `jupyter`, five sequential notebooks (`EDA` → `feature_engineering` → `modeling` → `Hyperparameters_Tuning` → `explainability`)
