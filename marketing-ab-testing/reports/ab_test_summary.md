# Marketing A/B Test — Summary Report

Full result on the real Kaggle dataset: 588,101 users (564,577 `ad` / 23,524 `psa`).

**Bottom line:** the ads measurably worked, but they didn't pay for themselves.

---

## Setup

Two groups were compared:

- **`ad`** — shown the marketing campaign
- **`psa`** — control group, shown a public service announcement instead

Question: does showing ads change conversion rate, by how much, and does it pay for itself?

---

## Method

Two-proportion z-test comparing conversion rate between `ad` and `psa` (two-tailed, alpha = 0.05), with a 95% confidence interval on the difference and Cohen's h as effect size.

Code: [`notebooks/02_ab_test_analysis.ipynb`](../notebooks/02_ab_test_analysis.ipynb)

### Group balance

| Group | Share of users |
|---|---|
| `ad` | 96% |
| `psa` | 4% |

The split is unbalanced — flagged directly in [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). This doesn't invalidate the test, but it does make the `psa` group's estimate noisier, visible directly in the dashboard's confidence-interval chart, where the `psa` interval is much wider than `ad`'s.

---

## Result

| Group | n | Conversion rate |
|---|---:|---:|
| `ad` | 564,577 | 2.55% |
| `psa` | 23,524 | 1.79% |

| Metric | Value |
|---|---|
| Difference (ad − psa) | 0.7692 pp (95% CI: 0.5951 pp – 0.9434 pp) |
| z-statistic | 7.370 |
| p-value | 1.7 × 10⁻¹³ (significant at alpha = 0.05) |
| Effect size (Cohen's h) | 0.0530 — small |
| Achieved power | 100% |

The gap is real, not noise. At this sample size the test has full power to detect it. But the effect size is small — ads move conversion up by about 0.77 percentage points, not a dramatic shift.

---

## When the lift happens

Generated from [`sql/01_campaign_summary.sql`](../sql/01_campaign_summary.sql), run against the real data.

**Data note:** at the time of this writeup, `sql/01_campaign_summary.sql` referenced a fixture filename that didn't exist yet. Re-run `sql/run_summary.py` against the real data to confirm the specific numbers below still hold before relying on them.

**By day of week**

| Pattern | Detail |
|---|---|
| Peak | Monday — 3.32% (ad group) |
| Low | Saturday — 2.13% (ad group) |

Both groups follow the same shape (highest Monday, lowest Saturday), which points to a general weekday effect on user behavior rather than something specific to the ad campaign.

**By hour of day**

Traffic and conversion are low overnight (roughly midnight–6am), ramp up through the morning, and stay elevated from midday into the evening. Ad conversion peaks around hour 16 (3.09%). Nothing in the hourly breakdown looks like a data error or an artificial pattern — it tracks normal daily activity.

---

## ROI

Assumptions: $0.02 per ad impression, $25 revenue per conversion — both are estimates, not values present in the dataset. See [`notebooks/03_campaign_roi.ipynb`](../notebooks/03_campaign_roi.ipynb) for the full sensitivity table across other assumptions.

| Metric | Value |
|---|---:|
| Total ad impressions | 14,014,701 |
| Campaign cost | $280,294.02 |
| Actual conversions | 14,423 → $360,575.00 |
| Counterfactual conversions (at `psa` rate) | 10,080 → $252,000.45 |
| Incremental revenue | $108,574.55 |
| ROI | 0.39x |

Under these assumptions, the campaign lost money — $0.39 returned for every $1 spent. The lift is statistically real but not large enough to cover the cost of showing that many ads.

**Spend concentration:** the top 1% of `ad`-group users (200+ ads each) account for 13.2% of total ad impressions. Spend is noticeably concentrated in a small segment of heavy-exposure users, not spread evenly across everyone who saw an ad.

---

## Caveats

- ROI depends entirely on the $0.02 / $25 assumptions above, not on the dataset itself — neither figure exists in the original data. The dashboard's ROI sliders let you test other assumptions directly.
- Single hypothesis test (`ad` vs `psa`), so no multiple-comparison correction is needed.
- Statistical significance confirms the conversion gap is real. It does not by itself mean the campaign was profitable — that's what the ROI section answers, and here the answer is no, under these assumptions.

---

## Bottom line

The ads worked in the sense that they measurably increased conversion rate (p < 0.001, full power, effect confirmed). They did not pay for themselves under the assumed cost and revenue figures.

Before running this campaign again, one of the following needs to change:

1. The true cost-per-impression needs to be lower, or
2. The true value-per-conversion needs to be higher, or
3. The campaign needs to cut the heavy-impression segment that's driving a disproportionate share of the cost without a matching share of the return.

---

## Where to look for more

| Resource | Contents |
|---|---|
| [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) | Group balance, funnel, ad exposure and timing |
| [`notebooks/02_ab_test_analysis.ipynb`](../notebooks/02_ab_test_analysis.ipynb) | The significance test, run against three synthetic scenarios first, then the real data |
| [`notebooks/03_campaign_roi.ipynb`](../notebooks/03_campaign_roi.ipynb) | ROI calculation and sensitivity table |
| [`sql/01_campaign_summary.sql`](../sql/01_campaign_summary.sql) | Conversion rate by group, day, and hour |
| [`dashboard/`](../dashboard) | Interactive version of all of the above, with adjustable ROI assumptions |
