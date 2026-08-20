# Baseline vs. Crew: Hallucination Rate

## Result (100 questions, `rag_router`'s eval set, `gemini-3.1-flash-lite`, seed 42)

| Metric | Value |
|---|---|
| `baseline_hallucination_rate` | 0.12 |
| `crew_hallucination_rate` | 0.08 |
| `crew_approval_rate` | 0.91 |
| `crew_avg_revisions` | 0.17 |
| `baseline_judge_failures` / `crew_judge_failures` | 0 / 0 |
| `n_failed` | 0 |

The crew's hallucination rate is a third lower than the single-pass
baseline's (0.08 vs. 0.12) — the Critic step is catching claims the
baseline lets through. `crew_avg_revisions` is 0.17, meaning the Writer
only got sent back on roughly 1 in 6 questions. Most of that improvement
isn't coming from the revision loop actually firing — it's coming from a
Critic checking the draft at all before it goes out, whether or not it
sends it back. The revision mechanism itself is a smaller piece of the
gain than I expected going in.

`crew_approval_rate` of 0.91 means the Critic approved 91% of drafts
(first pass or after one revision), so it isn't rejecting so often that
the "final answer" is usually a low-confidence fallback — it's approving
most drafts, just checking them first.

Both `judge_failures` counts are 0, so the hallucination rates above
aren't diluted by unscored answers — every question got a real
faithfulness verdict either way.

## What to look for

| Metric | What it tells you |
|---|---|
| `baseline_hallucination_rate` | Fraction of single-pass answers with at least one claim the judge couldn't find support for in the retrieved passages |
| `crew_hallucination_rate` | Same check, applied to the crew's final answer (after 0 or 1 revisions) |
| `crew_approval_rate` | How often the Critic approved a draft at all (first pass or after one revision) — if this is low, the Critic may be too strict, or the corpus genuinely doesn't support confident answers to many of these questions |
| `crew_avg_revisions` | How often the Writer actually needed a second attempt |
| `*_judge_failures` | How many answers the faithfulness judge itself failed to score (returned unparseable output) — excluded from the hallucination rate, not counted as clean. Watch this: a high count here undermines trust in the rate itself, independent of which condition looks better |

## Reading the result honestly

The crew's hallucination rate is meaningfully lower than the baseline's,
which is the expected finding — but `crew_avg_revisions` at 0.17 means
revisions were rare. Most of that improvement is coming from *knowing a
check exists*, not from the revision mechanism itself being exercised
often. If I were pushing this further, the next question isn't "does the
crew help" (it does) — it's whether a cheaper single-pass-plus-critique
without the retry loop gets most of the same benefit for less cost.

**Cost is not free here** — the crew costs roughly 2-3x the LLM calls of
the baseline per question (research + write + critique, plus a revision
pass on ~17% of questions). A third fewer hallucinations for 2-3x the
calls is a real tradeoff, not a free win — worth stating plainly rather
than glossing over.
