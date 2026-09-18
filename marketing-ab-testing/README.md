<div align="center">

# Marketing Campaign & A/B Test Analytics

`pandas` `scipy` `DuckDB` `Streamlit` `matplotlib` `plotly` `jupyter` `pytest`

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

Answers two separate questions about a real ad campaign, and keeps them separate on purpose: did the ads change conversion rate (yes, measurably), and did the campaign pay for itself (no, not under the assumptions available). A two-proportion z-test on Kaggle's 588,101-user "Marketing A/B Testing" dataset finds a statistically significant lift (2.55% ad-group conversion vs. 1.79% control, p = 1.7×10⁻¹³, full statistical power) but a small effect size (Cohen's h = 0.053). Translating that lift into dollars under stated cost/revenue assumptions ($0.02/impression, $25/conversion) gives an ROI of 0.39x — the campaign lost money. The core finding of the project is really the gap between those two answers: statistical significance and business profitability are different questions, and this dataset is large enough that even a genuinely small, commercially weak effect comes back significant.

## Problem & motivation

The naive version of an A/B test writeup treats "p < 0.05" as the whole answer — significant means it worked, ship it. Two things make that the wrong read here. First, this dataset has 588,101 users, which means the test has enough statistical power to detect a real but tiny effect; a small, commercially unimportant lift and a large, meaningful one can both come back "significant" at this sample size, so effect size (Cohen's h) and power have to be reported alongside the p-value, not left out, or a marketer reading only "significant" would draw an unwarranted conclusion about how much the campaign actually moved the needle.

Second, "significant" and "profitable" are genuinely different questions that this project deliberately doesn't conflate. A statistically real 0.77-percentage-point lift still has to be compared against what showing that many ads actually cost — and cost/revenue-per-conversion aren't in the dataset at all, so any ROI figure necessarily rests on stated assumptions. The naive approach picks one assumption and reports one ROI number as if it were fact; this project reports it as conditional and includes a sensitivity table so the reader can see exactly how much the conclusion depends on numbers that were guessed, not measured.

## Approach

### Statistical test

(`notebooks/02_ab_test_analysis.ipynb`): conversion is binary, compared between two independent groups — a two-proportion z-test (`statsmodels.stats.proportion.proportions_ztest`, two-tailed, alpha = 0.05), not a t-test, since t-tests compare means of continuous data, not proportions. Reports p-value, a 95% confidence interval on the difference (`proportion_confint`), Cohen's h as effect size (`proportion_effectsize`), and achieved power (`statsmodels.stats.power.NormalIndPower`). Power matters specifically because it's what distinguishes "no effect" from "an effect too small to reliably detect at this sample size" — a distinction a bare p-value can't make on its own.

### Validated against synthetic data before touching the real dataset

(`tests/make_fake_data.py`, `tests/test_ab_analysis.py`): three synthetic scenarios, each with a known ground-truth effect, exercise the same `run_ab_test()` logic that later runs on the real data — `real_effect` (true rates 5% vs. 2%, expected significant), `no_effect` (both groups at the true 2.5% rate, expected not significant), and `borderline` (2.6% vs. 2.4% on a small 300/300 sample, informational only — this is where power matters, since a "not significant" result here should read as "underpowered to detect a small gap," not "no gap exists"). This exists because a bug that looks fine on real data can hide in a subtle mislabeling — a one-tailed test called two-tailed, a t-test applied to binary data, an underpowered sample misread as a null result — and synthetic data with a known answer is what actually catches that.

### SQL layer

(`sql/01_campaign_summary.sql`, DuckDB, run via `sql/run_summary.py`): conversion rate by group, by day of week, and by hour of day, with a window function computing each hour's cumulative share of conversions within its group. Purely descriptive — the actual significance test lives in the notebook, not here. The SQL file itself is written against a small fixture (`marketing_AB_fake.csv`), the same one `tests/test_sql.py` runs against; `run_summary.py` string-substitutes the real filename in before running against the actual 588K-row CSV, so the same query is exercised against a known-shape fixture before it ever touches real data, and isn't duplicated as two separate files.

### ROI

(`notebooks/03_campaign_roi.ipynb`): campaign cost = total ad impressions shown to the `ad` group (the `psa` control group is unpaid) × an assumed cost per impression. Incremental revenue = actual `ad`-group conversions minus the conversions that group would have produced at the `psa` group's (lower) conversion rate — i.e., revenue attributable to the campaign, not just raw conversions — × an assumed revenue per conversion. Both dollar inputs are explicitly labeled as placeholder assumptions, not dataset values, and the notebook ends with a 4×4 sensitivity table sweeping both assumptions rather than reporting one number as settled fact.

### Dashboard

(`dashboard/streamlit_app.py`): interactive version of the same funnel, significance test, and ROI calculation, with the cost/revenue assumptions exposed as sliders so a reader can test other numbers directly instead of trusting the notebook's defaults. Checked with Streamlit's `AppTest` (`tests/test_dashboard.py`) against the same small fixture the SQL tests use, so the dashboard is smoke-tested without needing the real dataset either.

## Data

- Source: Kaggle "Marketing A/B Testing" (by faviovaz), 588,101 users total.
- Two groups: `ad` (shown the campaign, 564,577 users, 96%) and `psa` (control, shown a public service announcement instead, 23,524 users, 4%) — an intentionally unbalanced split, flagged directly in the EDA notebook. It doesn't invalidate the test, but it does make the smaller `psa` group's conversion-rate estimate noisier, visible in the dashboard's confidence-interval chart as a noticeably wider interval for `psa` than for `ad`.
- Columns: `user id`, `test group`, `converted` (bool), `total ads` (impressions shown to that user), `most ads day`, `most ads hour`.
- No train/test split — this is a hypothesis test on observational-experiment data, not a trained model.

## Results

### Significance test

, real data:

| Group | n | Conversion rate |
|---|---:|---:|
| `ad` | 564,577 | 2.5547% |
| `psa` | 23,524 | 1.7854% |

| Metric | Value |
|---|---|
| Difference (ad − psa) | 0.7692 pp (95% CI: 0.5951–0.9434 pp) |
| z-statistic | 7.370 |
| p-value | 1.7 × 10⁻¹³ |
| Effect size (Cohen's h) | 0.0530 (small) |
| Achieved power | 100% |

The gap is real, not noise — full power at this sample size means the test could reliably detect an effect this size if it exists, and it found one. But the effect itself is small: ads move conversion up by about 0.77 percentage points, not a dramatic shift.

### When the lift happens

(`sql/01_campaign_summary.sql` against the real data): conversion peaks Monday (3.32%, ad group) and is lowest Saturday (2.13%, ad group); both groups follow the same weekday shape, pointing to a general user-behavior pattern rather than something specific to the campaign. By hour, traffic and conversion are low overnight, ramp through the morning, and stay elevated midday into evening, peaking around hour 16 (3.09%, ad group) — nothing in the hourly breakdown looks like a data artifact.

### ROI

, at the stated $0.02/impression, $25/conversion assumptions:

| Metric | Value |
|---|---:|
| Total ad impressions | 14,014,701 |
| Campaign cost | $280,294.02 |
| Actual conversions | 14,423 → $360,575.00 |
| Counterfactual conversions (at `psa` rate) | 10,080 → $252,000.45 |
| Incremental revenue | $108,574.55 |
| **ROI** | **0.39x** |

Under these assumptions, the campaign lost money — $0.39 returned per $1 spent. The full sensitivity sweep (cost ∈ {$0.01, $0.02, $0.05, $0.10}, revenue ∈ {$10, $25, $50, $100} per conversion) shows this isn't a knife-edge result: ROI stays below 1.0x (loses money) across most of the grid, and only clears breakeven when cost is at its lowest tested value ($0.01) and revenue is $50 or higher (1.55x), or revenue is $100 even at $0.02 cost (1.55x). At the stated $0.02/$25 assumptions specifically, ROI is 0.39x; at the cheapest-cost / cheapest-revenue corner ($0.01/$10) it's 0.31x; at the most expensive corner ($0.10/$100) it's still only 0.31x.

### Spend concentration

The top 1% of `ad`-group users (200+ ads shown each) account for 13.2% of total ad impressions — cost is concentrated in a small segment of heavy-exposure accounts, not spread evenly across everyone who saw an ad.

## What I'd do differently / limitations

- **Both ROI inputs are stated placeholders, not measured values** — neither cost-per-impression nor revenue-per-conversion exists in the source dataset. The sensitivity table shows the conclusion is fairly robust (ROI stays under 1.0x across most of the grid), but "under most of the grid" is still a statement about assumed numbers, not this campaign's real unit economics.
- **The spend-concentration finding (top 1% of users = 13.2% of impressions) is reported but not connected back to ROI.** A natural next step — does cutting the heaviest-impression segment change the ROI calculation meaningfully, since those users are costing the most but not shown to convert proportionally more — isn't computed here.
- **Single hypothesis test, so no multiple-comparison correction needed** — but that also means no sub-group significance testing (e.g., is the lift itself significant within just the Monday cohort, or within heavy- vs. light-impression users) was run. The day/hour breakdown is descriptive only.
- **The `borderline` synthetic test scenario is informational, with no assertion** — by design, since a true 2.6%-vs-2.4% gap on 300/300 samples could reasonably go either way. That's the right call for a scenario meant to demonstrate underpowered-test behavior, but it does mean this specific test doesn't fail CI if the borderline behavior regresses; only a human reading the printed p-value and power would notice.
- **Ad exposure (`total ads`) isn't randomized-controlled within the `ad` group** — the A/B split itself (ad vs. psa) is the randomized comparison; how many ads a given `ad`-group user happened to see is an observed quantity, not something this design controls for, so any relationship between impression count and conversion within the `ad` group is correlational, not causal.
- **Group imbalance (96/4) is flagged but not addressed with a rebalancing or weighting approach** — it doesn't invalidate the z-test, but a report could go further by explicitly quantifying how much narrower the CI would be at a more balanced split, rather than only noting qualitatively that `psa`'s interval is wider.

## Stack

- `pandas`, `numpy` for data handling
- `scipy`, `statsmodels` (`proportions_ztest`, `proportion_confint`, `proportion_effectsize`, `NormalIndPower`) for the significance test, confidence interval, effect size, and power analysis
- `DuckDB`, querying the CSV directly via `read_csv_auto`, for the descriptive SQL layer
- `Streamlit` (`streamlit.testing.v1.AppTest` for headless dashboard testing) for the interactive dashboard with adjustable ROI sliders
- `matplotlib` for the EDA and results charts
- `plotly` (dashboard-side charting, per `requirements.txt`)
- `jupyter`/`nbformat`/`nbconvert`/`ipykernel` for the notebook pipeline
- Plain `pytest`-free scripts (`test_ab_analysis.py`, `test_dashboard.py`, `test_sql.py` run directly, not via a `pytest` collector) validating stats logic against three synthetic scenarios and the dashboard/SQL against a shared fixture, before any of it touches the real 588,101-row dataset
