# Customer Retention — Who to Call This Week

> **Real data.** Numbers below are pulled directly from the executed
> notebooks (`notebooks/EDA_and_Preprocessing.ipynb`,
> `Modeling_and_Evaluation.ipynb`, `Customer_Segmentation.ipynb`) run on the
> actual Telco dataset, not the fixture. Two numbers — average monthly
> charge and average churn probability *specifically for the call list* —
> aren't printed anywhere in the notebooks as run, so they're left as
> "needs `data/scored_test.csv`" below rather than guessed. See the note
> at the bottom for how to fill them in.

## The headline

**80 customers** are High Value and High or Very High risk of churning.
That's the group `Customer_Segmentation.ipynb` calls out for "immediate
personal outreach, best offer."

If even half of them are saved with a retention offer, that's real revenue
protected for the cost of 80 phone calls. That's the trade the priority
matrix exists to make visible.

## Where the 80 customers come from

The model scores every customer's churn probability, then two independent
cuts are laid on top of each other:

- **Value**: monthly charges, split into thirds (Low / Mid / High Value).
- **Risk**: predicted churn probability, split into five bands (Very Low →
  Very High).

The full grid, customer counts per cell (test set, from
`results/priority_matrix.png`):

| Value \ Risk | Very Low | Low | Medium | High | Very High |
|---|---|---|---|---|---|
| Low Value  | 319 | 73  | 50 | 12 | 0  |
| Mid Value  | 220 | 92  | 77 | 61 | 20 |
| High Value | 186 | 109 | 92 | 55 | 25 |

The call list is the bottom-right corner: **High Value** rows, **High** and
**Very High** risk columns (55 + 25 = **80**).

Note: these cells sum to 1,391, not the full 1,409-customer test set. The
missing 18 are customers whose predicted churn probability came out to
exactly 0.0 — pandas' `pd.cut()` excludes the lowest bin edge by default,
so they get silently dropped from the notebook's risk_segment column
rather than landing in "Very Low." This isn't a bug introduced by this
report; it's how the notebook's own `pd.cut()` call behaves. The SQL
version in `sql/02_segment_queries_postgres.sql` uses `<=` throughout and
keeps all 1,409, so running that query against the real data will show a
slightly higher "Very Low" count than the plot does. Worth knowing if the
two are compared side by side.

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
- Average customer lifetime value (mean `TotalCharges`, the value figure
  used in the profit-threshold calculation): **$2,279.73**.
- Rule-based high-risk customers (month-to-month contract, above-median
  monthly charge) churn at **52.8%**, vs. **15.8%** for everyone else —
  see `EDA_and_Preprocessing.ipynb`, cell 42.
- Contract-type breakdown (month-to-month vs. one-year vs. two-year churn
  rate) isn't printed in the notebooks as run — run query 1 in
  `sql/02_segment_queries_postgres.sql` against the real CSV in pgAdmin to
  get exact figures. On the earlier fixture that query showed
  month-to-month churning roughly 8x more than two-year contracts; expect
  something in that range on real data, but don't quote a number until
  the query's been run.

## The profit-threshold picture

`Modeling_and_Evaluation.ipynb`'s threshold optimization (cell 28), using
$50 as the retention offer cost and $5 as the cost to reach a customer:

- **Optimal threshold: 0.01** — i.e., the model recommends treating almost
  every customer as a churn risk under these cost assumptions.
- **Expected profit at optimal threshold: $775,241**
- **Expected profit at the default 0.50 threshold: –$2,602**
- **Improvement: $777,843**

Worth flagging to whoever owns the retention budget: a threshold of 0.01
means acting on nearly the entire customer base, not a targeted list. That
math only works because the model's average predicted probability across
low-risk customers is still low enough that reaching out rarely wastes the
$55 in campaign + discount cost, against a $2,280 average customer value.
If the real cost of a retention offer is higher than $50, or the offer
acceptance rate is much less than 100%, the optimal threshold moves — that's
exactly what the dashboard's What-If page is for.

## Caveats, in plain terms

- This is a snapshot of the **test set only** (1,409 of 7,043 customers),
  the same set the notebook evaluates the model on. It's a representative
  sample, not the full customer base — the SQL queries can be pointed at
  the full base for a company-wide count once every customer has been
  scored, not just the held-out test set.
- The risk bands come from a model, and models are wrong sometimes. Treat
  "Very High risk" as "worth a closer look," not as certainty.
- Average monthly charge and average predicted probability *for the 80
  customers on the call list specifically* aren't in this report because
  they require reading `data/scored_test.csv` directly rather than a
  notebook printout. Quick way to get them:
  ```python
  import pandas as pd
  scored = pd.read_csv("data/scored_test.csv")
  call_list = scored[(scored.value_segment == "High Value") &
                      (scored.risk_segment.isin(["High", "Very High"]))]
  print(len(call_list), call_list.MonthlyCharges.mean(), call_list.churn_probability.mean())
  ```
- The profit-threshold numbers assume every retention offer succeeds and
  every non-churner contacted is a pure loss — a simplification the
  notebook itself flags. Treat the optimal threshold as a starting point
  for discussion with whoever owns the retention budget, not a rule to
  follow automatically.
