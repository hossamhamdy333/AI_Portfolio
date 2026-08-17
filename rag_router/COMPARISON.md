# LLM Selector vs. Embedding Selector

## Result: LLM selector wins on every metric

Both selectors were run start-to-finish on the same 400-question eval set
via `notebooks/04_evaluation.ipynb`, with the per-selector checkpoint fix
in place (see below) so neither run's numbers are contaminated by the
other's progress.

| Metric | Embedding selector | LLM selector |
|---|---|---|
| `routing_accuracy` | 0.6450 | **0.6975** |
| `citation_accuracy` | 0.6275 | **0.6650** |
| MRR | 0.6010 | **0.6319** |
| NDCG@10 | 0.6074 | **0.6398** |
| Selector-parsing-bug failures | 4 / 400 (1.0%) | 39 / 400 (9.75%) |

Unlike `rag-vanilla-vs-langchain`'s `COMPARISON.md`, this file doesn't carry forward any
pre-fix historical numbers, even flagged as stale — the two numbers
originally reported here (0.725 and 0.66) weren't real measurements at
all; they were two Colab sessions resuming from the same un-versioned
checkpoint file at two different points. `eval_routing.py`'s `run_eval()`
now keys checkpoints per `selector_name`, so that failure mode can't
recur. Separately, the original version's eval notebooks both called
`build_embedding_router()` regardless of filename, so there was never a
real historical LLM-selector number either — the 0.6975 above is the
first one actually measured.

## The failure-rate gap is the interesting part

The LLM selector hits the known LlamaIndex `choice=None` parsing bug (see
README) almost 10x more often than the embedding selector — 39 failed
questions vs. 4, out of 400. Every failed question counts as a wrong
routing (there's no domain to compare against `None`), which makes the
LLM selector's 0.6975 routing accuracy an understatement of how good it
is when the bug doesn't hit:

- Embedding, non-failed only: 258 correct / 396 = **65.2%**
- LLM, non-failed only: 279 correct / 361 = **77.3%**

So the real gap between the two selectors is wider than the headline
routing_accuracy numbers suggest — the LLM selector is meaningfully
better at picking the right domain, it just also eats the parsing bug far
more often. Both of those are genuine, separate findings about this
pipeline, not one canceling the other out.

## What this answers

Is spending an extra LLM call on routing worth it, on this corpus? Yes —
the LLM selector wins on routing accuracy, citation accuracy, and
retrieval quality (MRR/NDCG), even after accounting for its higher
parsing-bug failure rate. The embedding selector's only advantage is cost
and latency (no extra Gemini call per question); if a use case can't
tolerate that extra call, the embedding selector is the fallback, not the
default. The 4 domain descriptions in `router.py` are similar enough in
embedding space that pure similarity matching alone is a measurably worse
tool for telling them apart than asking the LLM directly.
