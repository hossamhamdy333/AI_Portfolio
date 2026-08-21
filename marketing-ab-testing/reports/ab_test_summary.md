# Marketing A/B Test - Summary

Full result on the real Kaggle dataset: 588,101 users (564,577 ad / 23,524 psa).

## Setup

Two groups: `ad` (shown the campaign) and `psa` (control, shown a public service
announcement instead). Question: does showing ads change conversion rate, by
how much, and does it pay for itself.

## Method

Two-proportion z-test comparing conversion rate between `ad` and `psa`
(two-tailed, alpha = 0.05), with a 95% CI on the difference and Cohen's h as
effect size. Code in `notebooks/02_ab_test_analysis.ipynb`.

## Group balance

Split is unbalanced: 96% ad, 4% psa. Flagged in `notebooks/01_eda.ipynb`.
Doesn't invalidate the test, but the psa group's estimate is noisier - visible
directly in the dashboard's confidence-interval chart, where psa's interval
is much wider than ad's.

## Result

| Group | n | Conversion rate |
|---|---|---|
| ad | 564,577 | 2.55% |
| psa | 23,524 | 1.79% |

- Difference (ad - psa): 0.7692 percentage points, 95% CI [0.5951pp, 0.9434pp]
- z = 7.370, p = 1.7e-13, significant at alpha = 0.05
- Effect size (Cohen's h) = 0.0530, small
- Achieved power: 100%

The gap is real, not noise. At this sample size the test has full power to
detect it. But the effect size is small: ads move conversion up by about
0.77 percentage points, not a dramatic shift.

## When the lift happens

From `sql/01_campaign_summary.sql`, run against the real data:

- By day of week: ad conversion peaks Monday (3.32%) and is lowest Saturday
  (2.13%). Same general pattern holds for psa (highest Monday, lowest
  Saturday), so this looks like a weekday effect on user behavior, not
  something specific to the ad campaign.
- By hour: both groups show very low traffic and conversion overnight
  (roughly midnight to 6am), ramping up through the morning and staying
  elevated from midday into the evening. Ad conversion rate peaks around
  hour 16 (3.09%). Nothing in the hourly breakdown looks like a data error
  or an artificial pattern - it tracks normal daily activity.

## ROI

Assumptions: $0.02 per ad impression, $25 revenue per conversion (both are
guesses, not in the dataset - see `notebooks/03_campaign_roi.ipynb` for the
full sensitivity table across other assumptions).

- Total ad impressions: 14,014,701
- Campaign cost: $280,294.02
- Actual conversions: 14,423, worth $360,575.00
- Counterfactual conversions (at psa's rate): 10,080, worth $252,000.45
- Incremental revenue: $108,574.55
- ROI: 0.39x

Under these assumptions, the campaign lost money: $0.39 back for every $1
spent. The lift is statistically real but not large enough to cover the
cost of showing that many ads.

Top 1% of ad-group users (200+ ads each) account for 13.2% of total ad
impressions. Spend is noticeably concentrated in a small segment of
heavy-exposure users, not spread evenly across everyone who saw an ad.

## Caveats

- ROI depends entirely on the $0.02 / $25 assumptions above, not on the
  dataset. Neither figure exists in the original data - the dashboard's
  ROI sliders let you test other assumptions directly.
- Single hypothesis test (ad vs psa), so no multiple-comparison correction
  needed.
- Statistical significance confirms the conversion gap is real. It doesn't
  by itself mean the campaign was profitable - that's what the ROI section
  answers, and here the answer is no, under these assumptions.

## Bottom line

The ads worked in the sense that they measurably increased conversion rate
(p < 0.001, full power, effect confirmed). They did not pay for themselves
under the assumed cost and revenue figures. Before running this campaign
again, either the true cost-per-impression needs to be lower, the true
value-per-conversion needs to be higher, or the campaign needs to cut the
heavy-impression segment that's driving a disproportionate share of the
cost without a matching share of the return.

## Where to look for more

- `notebooks/01_eda.ipynb` - group balance, funnel, ad exposure and timing
- `notebooks/02_ab_test_analysis.ipynb` - the significance test, run against
  three synthetic scenarios first, then the real data
- `notebooks/03_campaign_roi.ipynb` - ROI calculation and sensitivity table
- `sql/01_campaign_summary.sql` - conversion rate by group, day, and hour
- `dashboard/` - interactive version of all of the above, with adjustable
  ROI assumptions
