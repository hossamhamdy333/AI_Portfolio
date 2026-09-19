<div align="center">

# Credit Card Fraud Detection

`pandas` `scikit-learn` `imbalanced-learn` `SHAP` `MLflow` `SQL*Loader`

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

End-to-end fraud detection on a highly imbalanced credit card transaction
dataset (473 fraud cases out of 283,726 after deduplication, a 599:1
imbalance ratio). The best model is XGBoost trained on the original,
unresampled data with class weighting (`scale_pos_weight`), not any
SMOTE variant, an AUC-PR of 0.8183, recall 0.7895, precision 0.9036 at
the default threshold. Tuning the decision threshold against an
estimated dollar cost (missed fraud vs. false alarm) instead of using
the default 0.5 cutoff raises recall to 0.8316 and lowers total
estimated cost on the test set from $2,484.20 to $2,275.36. A separate
SQL-only analysis layer, run against a real Oracle XE instance with all
284,807 raw rows loaded, finds fraud is 10x more likely in certain
overnight hours than the daily average.

## Problem & motivation

At 0.17% fraud, a model that predicts "legitimate" for every transaction
is 99.83% accurate and catches zero fraud, so accuracy is actively
misleading here, and the entire project deliberately avoids it as a
headline metric. The two things that actually determine whether a fraud
model is useful are (1) whether it can find the rare positive class at
all without drowning in false alarms, which is what AUC-PR (not AUC-ROC)
measures, and (2) where to set the decision threshold, since a model
that ranks fraud correctly is still only as good as the cutoff applied
to it. The tempting shortcut, resampling the training data (SMOTE,
undersampling) to "fix" the imbalance before training, turns out not to
help here: it's tested directly against the alternative (class weights
on the untouched data) rather than assumed to be the right move.

## Approach

- **Feature engineering**: drops three PCA features (`V13`, `V15`,
  `V22`) identified as weak separators in EDA, and scales `Amount` with
  `RobustScaler` (the `V1`-`V28` PCA components arrive pre-scaled from
  the source dataset, so only `Amount` needs it).
- **Split**: 80/20 stratified train/test split (`random_state=42`),
  preserving the fraud ratio in both sets so the test set's class
  balance still reflects reality.
- **Four resampling strategies compared head to head on the training
  set**: no resampling (class weights only), SMOTE (`k_neighbors=5`,
  brings the training set to a 1:1 ratio), random undersampling
  (`sampling_strategy=0.5`, a 2:1 ratio), and SMOTETomek (SMOTE followed
  by Tomek-link cleaning). Each trains a Logistic Regression baseline,
  XGBoost, and LightGBM, six models times four sampling strategies.
- **Winning model**: `XGBClassifier(n_estimators=300, learning_rate=0.05,
  max_depth=6, scale_pos_weight=<legit/fraud ratio>, eval_metric='aucpr')`,
  trained on the original (unresampled) data. `scale_pos_weight` handles
  the imbalance inside the loss function itself, rather than changing
  what the model sees at the data level the way resampling does.
- **Threshold tuning against a real cost model, not just F1**: sweeps
  thresholds from 0.01 to 0.99 and computes
  `total_cost = (false_negatives * avg_fraud_amount) + (false_positives * false_alarm_cost)`,
  using `avg_fraud_amount = $122.21` (the actual mean fraud transaction
  amount from EDA) and a flat `false_alarm_cost = $5` (an assumed
  investigation cost per false alarm). The threshold that minimizes this
  cost, not the one that maximizes F1 or accuracy, is the one the
  project recommends.
- **SQL layer, genuinely independent of the notebooks**: three Oracle
  queries against the raw, unmodeled data, fraud rate by hour-of-day
  (with a window function comparing each hour to the overall average),
  fraud rate by amount bucket, and the combined riskiest
  hour × amount-bucket pairs (filtered to buckets with at least 30
  transactions, so a 100%-fraud-rate cell built on 2 transactions can't
  masquerade as a real signal). The hour derivation
  (`FLOOR(MOD(time_seconds/3600, 24))`) was checked against pandas'
  equivalent on boundary values (0, 3599, 3600, 86399) before trusting
  it, and the whole pipeline was actually run against a live Oracle XE
  Docker instance, not just written and logic-checked.

## Data

Source: the standard Kaggle "Credit Card Fraud Detection" dataset
(`mlg-ulb/creditcardfraud`), 284,807 transactions, 492 fraud (0.17%),
`V1`-`V28` as anonymized PCA components plus `Time` and `Amount`. EDA
found and removed 1,081 duplicate rows, leaving 283,726 transactions
(473 fraud, 0.1667%, 599:1 imbalance ratio) for everything downstream.
Fraud transactions have a higher mean amount ($122.21) than legitimate
ones ($88.29), though a lower median ($9.25 vs. $22.00), consistent with
fraud including both small testing charges and occasional large ones.

## Results

| Model | Recall | Precision | F1 | AUC-PR | AUC-ROC | MCC |
|---|---|---|---|---|---|---|
| **XGBoost (no sampling, class weights)** | **0.7895** | **0.9036** | **0.8427** | **0.8183** | **0.9748** | **0.8444** |
| LightGBM (SMOTE) | 0.8211 | 0.5200 | 0.6367 | 0.8040 | 0.9670 | 0.6527 |
| LightGBM (SMOTETomek) | 0.8211 | 0.5200 | 0.6367 | 0.8040 | 0.9670 | 0.6527 |
| XGBoost (SMOTE) | 0.8000 | 0.5507 | 0.6524 | 0.8034 | 0.9698 | 0.6631 |
| XGBoost (SMOTETomek) | 0.8000 | 0.5507 | 0.6524 | 0.8034 | 0.9698 | 0.6631 |

Full comparison across six model types and four sampling strategies is
in `modeling.ipynb`; this table is the top five by AUC-PR. The SMOTE and
SMOTETomek rows for both XGBoost and LightGBM are identical to four
decimal places, not a coincidence: SMOTETomek's Tomek-link cleaning step
found and removed exactly zero pairs on this training set (both end up
at 226,602/226,602, a clean 1:1 split, identical to plain SMOTE's
output), so the two models were trained on byte-identical data. On this
dataset, at least, "combined" resampling bought nothing over SMOTE
alone.

Threshold tuning on the winning model, against the dollar cost model
described above:

| | Default (0.5) | Optimal (0.05) |
|---|---|---|
| Recall | 0.7895 | 0.8316 |
| Precision | 0.9036 | 0.5524 |
| False negatives | 20 | 16 |
| False positives | 8 | 64 |
| Total cost | $2,484.20 | $2,275.36 |

Lowering the threshold to 0.05 trades 56 more false alarms for 4 fewer
missed frauds, an 8.4% reduction in total estimated cost, worth it under
this cost model since a missed fraud ($122.21 average) costs roughly
24x more than a false alarm ($5 assumed).

SQL layer: run end to end against a live Oracle XE instance, all
284,807 rows loaded with zero rejects. Highest fraud-rate hours were
2 AM (1.71%) and 4 AM (1.04%), both well above the overall average.

## What I'd do differently / limitations

- **The false-alarm cost ($5) is an assumption, not a measured
  number.** It's a reasonable placeholder for "cost of a human
  reviewing one flagged transaction," but the threshold-tuning result is
  only as good as this number. A 2x change to it would shift the optimal
  threshold meaningfully; the project doesn't show that sensitivity.
- **SMOTETomek removing zero points here is worth double-checking, not
  just reporting.** It could be a genuine property of this dataset
  (SMOTE-generated synthetic points landing far enough from the decision
  boundary that no majority/minority pair ends up mutually nearest), or
  it could mean Tomek-link cleaning was configured in a way that made it
  ineffective on this specific feature space. Either way, the two
  resampling strategies should not be assumed interchangeable on a
  different dataset just because they matched here.
- **SHAP's top features and EDA's top features genuinely disagree, and
  the project surfaces this rather than picking one.** EDA's raw
  separation metrics rank `V3`, `V14`, `V17` highest by mean difference
  and `V17`, `V14`, `V12` highest by correlation with the fraud label.
  SHAP on the actual trained XGBoost model ranks `V14`, `V4`, `V12`,
  `V10`, `V11` highest. `V4` in particular shows up as SHAP's second most
  important feature despite not appearing in EDA's top-10 separation
  list at all, a concrete example of why "what separates the classes in
  raw data" and "what the trained model actually relies on" are
  different questions with different answers.
- **The winning model's hyperparameters (`n_estimators=300,
  learning_rate=0.05, max_depth=6`) don't appear to come from a
  documented search.** They work well, but there's no grid/random search
  log showing they were chosen over nearby alternatives rather than
  picked once and kept.
- **No holdout time split.** The 80/20 split is stratified by class but
  random in time; a fraud model in production faces genuinely
  time-shifted data (fraud patterns evolve), which a random split
  doesn't test for at all.

## Stack

- `pandas`, `numpy`, `scipy` for data handling and stats (Mann-Whitney U
  tests per feature in EDA)
- `scikit-learn` (`RobustScaler`, `train_test_split`, metrics),
  `XGBoost`, `LightGBM` for modeling
- `imbalanced-learn` (`SMOTE`, `RandomUnderSampler`, `SMOTETomek`) for
  the resampling comparison
- `SHAP` for model explainability (summary, dependence, and waterfall
  plots)
- `MLflow` for experiment tracking across the six-model × four-sampling
  comparison
- Oracle XE (via Docker) + `SQL*Loader` for the independent SQL fraud
  analysis layer
