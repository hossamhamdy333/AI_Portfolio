<div align="center">

# E-Commerce Demand Forecasting — Project Report

</div>

---

### Contents

- [Overview](#overview)
- [Data and pipeline](#data-and-pipeline)
- [Why the first two versions were worse than doing nothing](#why-the-first-two-versions-were-worse-than-doing-nothing)
- [A dead end worth keeping](#a-dead-end-worth-keeping)
- [Results](#results)
- [What this doesn't do](#what-this-doesnt-do)
- [If this continued](#if-this-continued)

## Overview

This project builds a daily, SKU-level demand forecast on the Online Retail II dataset (a real UK online retailer, Dec 2009 to Dec 2011), and uses the forecast to size safety stock and reorder points. The pipeline runs from raw transaction data through cleaning, feature engineering, model training, and inventory sizing.

Final model: WAPE 86.8%, which beats both a trivial "predict zero every day" baseline (100%) and a seasonal-naive baseline (126.8%). That sounds like a modest number until you know that two earlier versions of this same model scored *worse* than predicting zero — getting past that turned out to be most of the actual work, and most of the interesting part of the project.

## Data and pipeline

The raw dataset is about 1.07 million transaction line items. It's messy in the ways real retail data usually is: cancellations mixed in with sales, non-product line items (postage, bank fees, manual adjustments), a meaningful chunk of null customer IDs, and duplicate rows from what looks like a re-export artifact.

The SQL pipeline (`sql/00` through `sql/09`) handles cleaning and aggregation:

- Staging removes obvious non-products and zero/null-price rows, and deduplicates. Cancellations are kept rather than dropped, since dropping them would overstate net demand — they get netted against same-day sales instead.
- Categories don't exist natively in this dataset, so they're derived from the first two characters of the stock code. That's stated plainly as a heuristic, not presented as if it were real taxonomy.
- Daily demand is aggregated per SKU, with negative net demand (a cancellation landing on a different day than its original sale) clipped at zero, since the downstream model can't take negative targets.
- Calendar gaps are filled per SKU, within that SKU's own observed lifetime — not the dataset's full date range. This one mattered more than it sounds like it should. Filling zero-demand rows against the global date range manufactures a huge number of fake rows for periods before a product existed or after it was discontinued, and an earlier version of this pipeline did exactly that. It was the reason that version's error metrics looked broken.

Feature engineering happens in Python (`src/features.py`), used by the training notebook. Every feature is built to only use information available at or before the day it's predicting for — lags, rolling means and standard deviations (computed with a one-day shift before rolling, so the window never includes the current day), days-since-last-sale, and calendar features like day of week and distance to the nearest Christmas.

## Why the first two versions were worse than doing nothing

The first real model trained here was a hurdle model: a classifier estimating P(demand > 0), multiplied by a regressor estimating the demand magnitude given that a sale happens. Standard approach for intermittent demand. It scored 108.9% WAPE — worse than just predicting zero for every row, every day.

The obvious guess was that the model was missing something about demand spikes. A quick check of the biggest test-period days showed actual demand running 3–4x higher than what the model predicted, so the first fix attempt was a category-level demand trend feature, on the theory that a SKU's own rolling average is too slow to catch a ramp before it's already inside the averaging window. That helped, but only by about a point. Not the answer.

The real problem showed up once the test-set error was broken down by row: roughly half of the total error was coming from rows where actual demand was zero. Not because the classifier was bad at telling sale days from non-sale days — its AUC was a reasonable 0.71 — but because multiplying a probability by a magnitude produces a small nonzero prediction on *every* row, including the majority of rows that are genuinely zero-demand. With about two-thirds of all rows being zero-demand, thousands of small overpredictions add up fast.

Switching from that probability-weighted blend to a hard threshold — predict the magnitude only if P(demand > 0) clears a cutoff, otherwise predict zero — fixed most of it. The cutoff itself is chosen by cross-validating on the training data only, never by checking it against the test set, since tuning a threshold against test performance is just a subtler form of the same leakage problem everyone is careful to avoid in the features. That change alone took WAPE from 108.9% to 95.7%.

One more change closed most of the remaining gap. WAPE (and MAE generally) is minimized by predicting the conditional median, not the mean. The regressor had been using a Tweedie objective, which targets the mean — and on demand data this skewed, with plenty of small values and a handful of very large ones, the mean runs noticeably higher than the median. Swapping to a quantile regression objective at the median brought WAPE down to 86.8%, a bigger improvement in practice than a quick synthetic test had suggested it would be, likely because the real distribution here is more skewed than the synthetic approximation used to sanity-check the idea beforehand.

## A dead end worth keeping

One idea that didn't pan out: giving spike-prone SKUs a lower classification threshold than stable ones, on the assumption that missing a big SKU's spike is more costly than a false positive on a quiet one. Tested it properly, and it made essentially no difference — 95.2% versus 95.1% on the same setup, within noise.

Looking into why led somewhere more useful. The demand spikes the model still misses aren't tied to particular SKUs or to the holiday season specifically — they're spikes with no lead-in at all. The median lag-1 value for a missed spike is zero (no sale the day before), versus 10 for spikes the model catches, and the miss rate in November is about the same as everywhere else in the year. Every feature this model has access to is derived from a SKU's own sales history, so a SKU that's been quiet and then suddenly sells a lot has nothing in its recent history to signal that. That's a real ceiling on what's predictable from this data, not a bug to chase further — closing it would need something external, like a promotions calendar or stock-availability data, neither of which exists in this dataset.

## Results

| Model | WAPE |
|---|---|
| Always-zero baseline | 100.0% |
| Seasonal-naive baseline | 126.8% |
| Hurdle model, original (probability × magnitude) | 108.9% |
| Hurdle model, threshold gate + Tweedie regressor | 95.7% |
| Hurdle model, threshold gate + quantile regressor (final) | 86.8% |

All model selection — the classification threshold, hyperparameters that were touched, feature choices — was validated on training data only, with test-set numbers checked at the end and not used to make any decisions along the way.

## What this doesn't do

Category assignment is a heuristic derived from stock code prefixes, not a real product taxonomy, and every category-level number in this project inherits that limitation. Cancellation netting only matches a cancellation to a sale on the same day; a cancellation landing a week after its original purchase isn't matched and instead shows up as a standalone negative row, which gets clipped to zero rather than passed through unexamined — the rate of this is measured in `08_audit_summary.sql` rather than ignored. Lead time in the inventory-risk calculation is a stated assumption (7 days), not measured against actual supplier data, and there's no real on-hand inventory in this dataset, so the "stockout risk" flag is a proxy based on forecast uncertainty rather than a measurement against real stock levels.

The unpredictable, no-warning demand spikes described above are the main thing standing between this model and a meaningfully lower error rate, and closing that gap would take data this project doesn't have — a promotions calendar or supply-chain events, most likely — rather than more feature engineering on what's already here.

## If this continued

The most direct next step would be turning the classification threshold into a cost-weighted decision rather than a pure WAPE-minimizing one — if a missed spike costs more in stockouts than a false positive costs in holding inventory, the threshold should reflect that tradeoff instead of a metric that treats both kinds of error as equally expensive. That needs an actual cost ratio from whoever owns the inventory decision, so it wasn't something to guess at here. Beyond that, the most useful additions would be external signals for the promotions/restock problem described above, and possibly splitting the very high-volume SKUs from the long tail of intermittent ones into separate models rather than asking one model to serve both regimes well.
