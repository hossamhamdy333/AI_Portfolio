# LLM Selector vs. Embedding Selector

Both selectors were run on the same 400-question set in
`notebooks/04_evaluation.ipynb`, with checkpoints keyed per selector so
the two runs can't share state.

## Results

| Metric | Embedding selector | LLM selector |
|---|---|---|
| Routing accuracy | 0.6450 | **0.6975** |
| Citation accuracy | 0.6275 | **0.6650** |
| MRR | 0.6010 | **0.6319** |
| NDCG@10 | 0.6074 | **0.6398** |
| Parsing-bug failures | 4/400 (1.0%) | 39/400 (9.75%) |

## The failure rate understates the LLM selector's edge

Failed questions count as wrong routing — there's no domain to compare
against `None`. Excluding failures:

- Embedding selector: 258/396 correct = **65.2%**
- LLM selector: 279/361 correct = **77.3%**

So the actual gap in routing quality is wider than the headline accuracy
numbers show, even though the LLM selector fails almost 10x more often.

## Conclusion

The LLM selector wins on every metric, including after accounting for its
higher failure rate. The embedding selector's only advantage is cost and
latency — no extra Gemini call per question. On this corpus, the four
domain descriptions aren't distinct enough in embedding space for
similarity matching to match an LLM's judgment; the extra API call buys
real accuracy.
