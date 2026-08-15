# LLM Selector vs. Embedding Selector

## Status: not yet run under the current pipeline

Unlike `rag_chatbot`'s `COMPARISON.md`, this file doesn't carry forward any
historical numbers, even flagged as stale. The two numbers that used to be
reported here (routing accuracy 0.725 and 0.66) turned out not to be two
real, complete measurements of anything — they were two different Colab
sessions resuming from the *same* un-versioned checkpoint file at two
different points, each reporting a "final" score for a run that was never
actually run start-to-finish as a single, coherent attempt. Reporting
either of those numbers here, even with a caveat, would imply they mean
something they don't. See `eval_routing.py`'s docstring and
`../rag_chatbot`-style fix pattern for the actual fix (checkpoints are now
keyed per selector, never shared).

Separately, **the true LLM-selector accuracy was never measured at all** in
the original version of this project — both of the old eval notebooks
called `build_embedding_router()` regardless of what their filenames
implied, so there's no historical LLM-selector number to report either,
stale or otherwise.

Run `notebooks/04_evaluation.ipynb` and replace this section with real
results from both selectors.

## What to look for once it's run

| Metric | What it tells you |
|---|---|
| `routing_accuracy` | Did the router pick the correct domain, independent of whether the answer was any good |
| `citation_accuracy` | Domain routing *and* whether the right article got cited in the final answer |
| MRR / NDCG@10 | Retrieval quality specifically, within whichever domain got routed to |
| Failure breakdown | How many of a selector's "wrong" routings are actually the known LlamaIndex parsing bug (see README) vs. genuine misrouting — a low routing_accuracy that's mostly parsing-bug failures is a very different finding than one that's mostly real confusion between domains |

## The real question this comparison answers

Is spending an extra LLM call on routing worth it, on this corpus? If the
embedding selector matches the LLM selector's routing accuracy (once
parsing-bug failures are separated out from genuine misrouting), that's a
legitimate, useful finding: routing by embedding similarity is strictly
better here (same accuracy, no extra API cost, no extra latency). If the
LLM selector clearly wins, that's the opposite finding, and also useful —
it means the 4 domain descriptions aren't linearly separable enough in
embedding space for similarity matching alone to reliably tell them apart,
and paying for an LLM call is buying something real.

Either outcome is a good result to report. The point of running both for
real was to find out which one it actually is, not to assume.
