# Marketing Campaign & A/B Test Analytics

Does showing ads actually change conversion rate, and does the campaign
pay for itself? Full analysis of the Kaggle "Marketing A/B Testing"
dataset: 588,101 users, split into an `ad` group (shown the campaign) and
a `psa` control group (shown a public service announcement instead).

Result, in short: the ads work. Conversion rate goes up by about 0.77
percentage points (2.55% vs 1.79%), and it's statistically real (p =
1.7e-13, full power). But the campaign doesn't pay for itself under
reasonable cost/revenue assumptions - ROI comes out to 0.39x. Full
writeup in `reports/ab_test_summary.md`.

## What's here

```
sql/01_campaign_summary.sql       conversion rate by group / day / hour, in DuckDB SQL
sql/run_summary.py                runs the SQL file above and prints the results
notebooks/01_eda.ipynb            group balance, conversion funnel, ad exposure and timing
notebooks/02_ab_test_analysis.ipynb   two-proportion z-test, confidence interval,
                                       effect size (Cohen's h), power analysis
notebooks/03_campaign_roi.ipynb   cost/revenue assumptions -> ROI, with a sensitivity table
dashboard/streamlit_app.py        interactive version: funnel, significance test, ROI sliders
reports/ab_test_summary.md        the final writeup, with real numbers
tests/                            checks the stats logic against synthetic data before
                                   trusting it on the real dataset
```

## Method

Conversion is binary, compared between two independent groups - that's a
two-proportion z-test, not a t-test (t-tests compare means of continuous
data, not proportions). The notebook reports p-value, a 95% confidence
interval on the difference, Cohen's h as effect size, and a power
analysis, since at this sample size even a small, practically
unimportant difference can come back statistically significant. Effect
size and power are what tell you whether that's actually happened here.

The group split is imbalanced (96% ad, 4% psa) - called out directly in
the EDA notebook, since it means the smaller group's conversion rate
estimate is noisier. The dashboard's confidence-interval chart shows this
visually: psa's interval is much wider than ad's.

## Tested before trusting it on real data

Since a project like this can look right and still have a subtle bug (a
one-tailed test mislabeled as two-tailed, a t-test used on binary data,
an underpowered sample mistaken for "no effect"), the core stats logic is
checked against three synthetic scenarios before being run on anything
real:

- **real_effect** - a clear, real gap between groups -> expect significant
- **no_effect** - both groups convert at the same true rate -> expect not significant
- **borderline** - a tiny true gap -> could go either way, and this is
  where the power analysis matters: a "not significant" result here
  should come with a low power reading, not get read as "no effect
  exists"

The dashboard is checked with Streamlit's `AppTest`, and the SQL runs
against a fixture through DuckDB before ever touching the real CSV.

## Running it

```bash
pip install -r requirements.txt
```

Download the real dataset from Kaggle ("Marketing A/B Testing" by
faviovaz), and place it at `data/raw/marketing_AB.csv`.

Run the notebooks in order: `01_eda.ipynb` -> `02_ab_test_analysis.ipynb`
-> `03_campaign_roi.ipynb`.

Run the SQL summary:
```bash
cd sql
python run_summary.py
```

Run the dashboard:
```bash
cd dashboard
streamlit run streamlit_app.py
```

Run the tests (optional, but they're there):
```bash
cd tests
python test_ab_analysis.py
python test_dashboard.py
```

## Key findings

- Ad conversion rate: 2.55% (n=564,577). Psa (control): 1.79% (n=23,524).
- Difference is statistically significant (p = 1.7e-13) with full power,
  but the effect size is small (Cohen's h = 0.053).
- Under $0.02/impression and $25/conversion, the campaign generated
  $108,574.55 in incremental revenue against $280,294.02 in cost - an ROI
  of 0.39x. It lost money at these assumptions.
- The top 1% of ad-exposed users account for 13.2% of total ad spend -
  cost is concentrated in a small segment of heavy-exposure accounts, not
  spread evenly.
- Conversion rate follows a weekday pattern (higher Monday, lower
  Saturday) in both groups, so this is a general user-behavior pattern,
  not something specific to the ad campaign.

Full detail, caveats, and the sensitivity table across other cost/revenue
assumptions are in `reports/ab_test_summary.md`.
