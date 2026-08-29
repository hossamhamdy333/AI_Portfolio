<div align="center">

# Model Results Summary

</div>

---

### Contents

- [Classification — attrition prediction](#classification--attrition-prediction)
- [Survival analysis — Cox proportional hazards](#survival-analysis--cox-proportional-hazards)
- [Cost of attrition](#cost-of-attrition)
- [Retention intervention ROI](#retention-intervention-roi-default-assumptions)
- [Headline takeaway](#headline-takeaway)

## Classification — attrition prediction

Compared on **5-fold cross-validation** for the headline metric (PR-AUC) — not a single train/test split, which was shown to give an unreliable, misleading ranking with only 47 positive cases in the test set (see `03_classification_models.ipynb` for the comparison that caught this). Recall/precision below are from the single 80/20 split's classification report at the default 0.5 threshold — a different evaluation than the CV column, shown for context on what each model's predictions actually look like, not as the basis for picking the winner.

| Model | 5-fold CV PR-AUC (winner metric) | Recall*, leavers | Precision*, leavers |
|---|---|---|---|
| Logistic Regression | 0.478 | 0.64 | 0.33 |
| **LightGBM (winner)** | **0.578** | 0.26 | 0.52 |

\* From the single train/test split, 0.5 threshold — shown for context, not used to pick the winner. LightGBM won on 5-fold CV PR-AUC.

LightGBM won on the trustworthy metric (cross-validated PR-AUC) and was used for all downstream risk scores and SHAP explanations.

## Survival analysis — Cox proportional hazards

Duration = `YearsAtCompany`, event = `Attrition`. Concordance index: **0.797** (correctly ranks who leaves sooner vs. later ~80% of the time).

| Factor | Hazard Ratio | Interpretation |
|---|---|---|
| **OverTime** | **3.19** | ~3.2x faster attrition, holding other factors constant |
| JobSatisfaction | 0.79 | Each point (1–4 scale) cuts risk ~21% |
| DistanceFromHome | 1.02 | ~2% higher risk per mile |
| MonthlyIncome | 1.00 | Statistically significant, but tiny effect per dollar |

Log-rank test on the OverTime Kaplan-Meier split: **p < 0.0001** — the gap is real, not chance.

Independently confirmed by SQL: attrition rate by tenure bucket (`sql/06_query_tenure_bucket.sql`) shows the same pattern — 0-1 yrs: 34.9%, 2-4 yrs: 18.1%, 5-9 yrs: 11.1%, 10+ yrs: 10.4%.

## Cost of attrition

- Total expected annual attrition cost (workforce-wide): **$10.15M** (50%-of-salary replacement-cost assumption × each employee's model risk score)
- Cost concentration: R&D $6.89M (68%), Sales $2.63M (26%), HR $0.63M (6%) — R&D leads on total dollars due to headcount size, not attrition rate; Sales has the higher per-employee rate (20.6% vs R&D's 13.8%)

## Retention intervention ROI (default assumptions)

Target: top 20% by risk score, AND working overtime. Intervention cost $2,000/employee, assumed 30% risk reduction.

| Metric | Value |
|---|---|
| Employees targeted | 81 |
| Intervention cost | $162,000 |
| Expected savings | $966,663 |
| **ROI** | **497%** |

This figure depends entirely on the two stated assumptions (cost per employee, risk-reduction effectiveness) — it is a "what this would be worth if the assumptions hold" estimate, not a measured result. The Power BI dashboard's What-If sliders let you test other assumptions directly (e.g. widening the target group to 45% of the workforce with a weaker 10% risk reduction turns this negative — the ROI is genuinely sensitive to targeting a narrow, high-confidence group, not just "throw money at everyone").

## Headline takeaway

LightGBM beat Logistic Regression on the metric that matters (cross-validated PR-AUC), overtime is by far the strongest and most confident driver of attrition (both in the classifier's feature importance and independently in the Cox model's hazard ratio), and targeting a narrow, correctly-identified high-risk group produces a strong ROI — but that ROI collapses quickly if the targeting is too broad or the intervention too weak, which is the real business lesson: precision in *who* you target matters more than the size of the intervention budget.
