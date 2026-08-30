<div align="center">

# Model Results Summary

</div>

The numbers below are a human-readable copy of `model_results_summary.json`, which is written automatically by `notebooks/03_forecasting_models.ipynb` every time it runs. This file is not regenerated automatically — if you re-run the notebook and the json changes, update the numbers here to match by hand. If the two ever disagree, trust the json.

---

### Contents

- [Result](#result)
- [Background, briefly](#background-briefly)

## Result

| Model | WAPE |
|---|---|
| Always-zero baseline | 100.0% |
| Seasonal-naive baseline | 126.8% |
| Hurdle model (final) | 86.8% |

The final model beats the always-predict-zero floor. Earlier versions of it didn't — see `PROJECT_REPORT.md` for the full story of why, since it turned out to be more interesting than the final number itself.

## Background, briefly

An earlier version of the SQL pipeline filled every SKU's calendar against the full dataset date range instead of that SKU's own active window, which manufactured a large number of fake zero-demand rows for periods before a product existed or after it stopped selling. That inflated the row count and was the most likely reason that version's model scored worse than a trivial always-zero baseline. It's fixed now — SKUs are filled only within their own observed lifetime, and the row-count difference is checked directly in `08_audit_summary.sql`.

A second, smaller bug was in the SQL's rolling-average columns, which originally included the current row in their own window — meaning a day's "average" partly contained that day's own actual sales. Fixed to exclude the current row.

Even with both of those fixed, the model still came in worse than the zero floor (108.9% WAPE). That one turned out to be about how the hurdle model combines its two stages: multiplying a probability by a magnitude puts a small nonzero prediction on every row, including the roughly two-thirds of rows that are genuinely zero-demand, and that adds up to about half the total error. Switching to a hard threshold — predict the magnitude only above a cross-validated probability cutoff, otherwise predict zero — along with switching the magnitude model from a mean-targeting objective to a median-targeting one, brought it down to the final 86.8%.
