# Baseline vs. Crew: Hallucination Rate

## Status: not yet run

Run `notebooks/02_evaluation.ipynb` and replace this with real numbers.

## What to look for

| Metric | What it tells you |
|---|---|
| `baseline_hallucination_rate` | Fraction of single-pass answers with at least one claim the judge couldn't find support for in the retrieved passages |
| `crew_hallucination_rate` | Same check, applied to the crew's final answer (after 0 or 1 revisions) |
| `crew_approval_rate` | How often the Critic approved a draft at all (first pass or after one revision) — if this is low, the Critic may be too strict, or the corpus genuinely doesn't support confident answers to many of these questions |
| `crew_avg_revisions` | How often the Writer actually needed a second attempt |
| `*_judge_failures` | How many answers the faithfulness judge itself failed to score (returned unparseable output) — excluded from the hallucination rate, not counted as clean. Watch this: a high count here undermines trust in the rate itself, independent of which condition looks better |

## Reading the result honestly

**If `crew_hallucination_rate` is meaningfully lower** than the baseline's,
that's the expected finding — a critique step catching what a single pass
missed — and it's worth checking `crew_avg_revisions` alongside it: if
revisions were rare, most of that improvement is coming from *knowing a
check exists*, not from the revision mechanism itself actually being
exercised often.

**If the two rates are close, or the crew isn't clearly better**, that's
not a failed project — it's a real finding: it means either the Critic
isn't strict enough, or a single well-prompted call was already about as
faithful as the sources allow. Both are legitimate, useful things to write
up, not a result to fish for a different framing of.

**Cost is not free here** — the crew costs roughly 2-3x the LLM calls of
the baseline per question (research + write + critique, plus a revision
pass sometimes). If the hallucination-rate improvement is small, that's a
genuine cost/benefit tradeoff worth stating plainly, not glossing over.
