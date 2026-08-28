# E-Commerce Demand Forecasting — Project Report

## Summary

Built a SKU-level daily demand forecasting pipeline on the Online Retail II dataset (UK online retailer, 2009–2011), covering data cleaning, feature engineering, model training, and downstream inventory-risk sizing. The headline result: **WAPE 86.6%**, beating the always-predict-zero baseline (100%) and the seasonal-naive baseline (126.2%) — the first version of this model to actually beat doing nothing.

Getting there required diagnosing why two earlier model versions scored *worse* than predicting zero everywhere, which turned out to be more interesting than the final result itself.

## Data and pipeline

Source: Online Retail II (UCI/Kaggle), ~1.07M raw transaction line items, Dec 2009–Dec 2011, real UK retailer selling mostly gift/homeware items.

Pipeline, in order (`sql/00` through `sql/09`, `notebooks/01` through `04`):

1. **Raw load + audit** — load CSV into Postgres, verify row count.
2. **Staging clean** — explicit, documented exclusions: non-product stock codes (postage, bank charges, manual adjustments), zero/null price rows, zero-quantity rows, deduplication of known re-export artifacts. Cancellations (`C`-prefixed invoices) are kept, not dropped, and netted against sales downstream.
3. **Net transactions** — net demand at (stock_code, date, customer) grain, offsetting same-day cancellations against sales.
4. **Category assignment** — categories derived from the first 2 characters of stock_code (a heuristic, documented as such — this dataset has no native category field).
5. **Daily demand** — aggregate to (stock_code, date), clip negative net demand at zero (a cancellation landing on a later day than its original sale can leave a standalone negative row; documented and measured in the audit summary, not silently absorbed).
6. **Calendar fill** — every SKU gets a row for every day in *its own* observed active window (first sale to last sale), not the global dataset range. This was the single most consequential correctness decision in the whole pipeline — filling against the global range instead was confirmed as the root cause of an earlier version's WAPE exceeding 100% purely from ghost rows.
7. **Rolling features (SQL)** — reporting-only window functions (week-over-week change, category revenue rank), explicitly excluded from the model's feature set since they use same-day values.
8. **Audit summary** — data-quality numbers for the README: raw vs cleaned row counts, cancellation rate, ghost-rows-avoided from the calendar-fill fix, negative-net-quantity rate.
9. **BI exports** — flat, denormalized exports and a date dimension table for Power BI.

Python side (`notebooks/02` onward, feature logic shared with the API in `src/features.py`):

- Lag features (1/7/14/28 day), rolling mean/std (7/14/28 day, shifted before rolling to stay strictly causal), days-since-last-sale, calendar features (day of week, holiday, distance to Christmas — including a wraparound fix so early-January dates measure distance to the *just-passed* Christmas, not the next one 11 months away), SKU-level and category-level statistics fit on train data only, and a category-level demand-trend feature (added during debugging, see below).

## The debugging journey

### First version: WAPE 108.9% — worse than predicting zero

The original hurdle model (a classifier for P(demand>0) multiplied by a Tweedie regressor for magnitude, `p × mag`) scored *worse* than the trivial baseline of predicting zero for every row. That's not "weak," it's actively harmful, and it took real diagnosis to understand why, since the obvious guesses were wrong.

**First hypothesis (wrong):** the model must be missing a demand-ramp signal, since a spike-underprediction check showed actual demand 3–4x higher than predicted on the biggest test days. Added a category-level trend feature to address this. It helped by less than a point.

**Second finding (right):** decomposing total test error by row showed **48.5% of it came from zero-actual-demand rows.** Not because the classifier couldn't tell sale-days from non-sale-days — its AUC was a workable 0.71 — but because `p × mag` is a continuous expected-value blend: it puts a *small* nonzero prediction on every single row, including the ~65% of rows that are genuinely zero-demand. With that much zero-demand data, thousands of small false-positive errors add up to roughly half the total error budget. The magnitude regressor itself was actually good in isolation (33% WAPE on days it should fire, 0.91 correlation with actual).

**The fix:** replace the continuous `p × mag` blend with a hard threshold gate (`prediction = magnitude if p > threshold else 0`), with the threshold chosen by cross-validating on train data only — never by looking at test WAPE, which would just be a different leakage bug. This alone took WAPE from 108.9% to 95.7%, the first version to beat the zero floor.

**Refinement:** WAPE/MAE is minimized by the conditional median, not the mean. Swapping the magnitude regressor's objective from Tweedie (a mean estimator) to quantile loss at the median took WAPE to 86.6% — a much larger real-world gain than a synthetic test had predicted, likely because the real demand distribution is more heavily skewed than the synthetic approximation used to validate the idea before shipping it.

### One honest dead end, kept in the record rather than hidden

Tested whether per-SKU "spike-proneness" tiers (a lower classifier threshold for historically spikier SKUs) would recover more of the missed big-demand days. It didn't help (95.2% vs 95.1% on the same test setup — no meaningful difference). Digging into *why* revealed the real pattern: missed spikes aren't a property of certain SKUs or the holiday season specifically — they're "surprise" spikes with no lead-in signal (median `lag_1d` of 0 for missed spikes vs 10 for caught ones), spread roughly evenly across months rather than concentrated around the holidays. Every feature available is built from a SKU's own sales history; when a SKU goes quiet and then spikes with zero warning, nothing in that history could have predicted it. This is a real ceiling on what history-based features can do, not a bug — closing it further would need external signal (a promotions calendar, wholesale order backlog, restock events) that isn't present in this dataset.

## Final results

| Model | WAPE |
|---|---|
| Always-zero baseline | 100.0% |
| Seasonal-naive baseline | 126.2% |
| Hurdle, continuous blend (original) | 108.9% |
| Hurdle, gated + Tweedie regressor | 95.7% |
| Hurdle, gated + quantile regressor (final) | **86.6%** |

Threshold for the gate (0.70) and all model artifacts are selected/validated using train-only cross-validation throughout — no test-set leakage at any stage of this project, including during threshold tuning, which is where it would have been easiest to introduce silently.

## Known limitations

- **Category assignment is a heuristic** (first 2 characters of stock_code), not ground-truth product taxonomy — documented in the SQL and carried through to every downstream category-level feature and BI view.
- **Cancellation netting is same-day only.** A cancellation booked on a later day than its original sale isn't matched to it; this is measured (see `08_audit_summary.sql`'s negative-net-quantity check) and handled by clipping at zero rather than passed through silently, but it's a real gap in how precisely net demand is computed.
- **"Surprise" spikes with no lead-in signal are a structural blind spot**, not a fixable bug — see above. Downstream, the inventory-risk safety-stock calculation partially compensates for this automatically, since a SKU with more missed spikes shows higher residual variance and gets a wider safety-stock buffer as a result — but the point forecast itself can't see these coming.
- **Lead time (7 days) in the inventory-risk model is an assumption, not measured** — real deployment would use actual supplier lead times per SKU/vendor.
- **No real on-hand inventory data exists in this dataset**, so the "stockout risk" flag in `04_inventory_risk.ipynb` is a proxy (any nonzero forecast uncertainty), not a measurement against actual stock levels.

## What's next, if extended

- **Cost-weighted threshold tuning**: the current gate threshold minimizes plain WAPE. If the actual business cost of a missed spike (stockout) vs. a false positive (excess holding cost) were known, the threshold could be chosen to minimize that cost directly rather than a proxy metric — deliberately trading some WAPE for fewer catastrophic misses. Not implemented since it requires a real cost ratio from the business, not a modeling assumption.
- **External signal integration** (promotions calendar, restock events) would be the highest-leverage next step for closing the surprise-spike gap specifically.
- **Per-segment models** for very different demand regimes (e.g., a small number of extremely high-volume SKUs vs. the long tail of intermittent ones) rather than one global model serving both.
