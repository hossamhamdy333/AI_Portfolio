# RAG Chatbot — Three Implementations Compared

Same core idea, built three ways, scored against a shared eval methodology
(`shared/eval_set.py`, `shared/metrics.py`) so the numbers below are directly
comparable wherever the architecture allows it.

| | impl_vanilla | impl_langchain | impl_llamaindex |
|---|---|---|---|
| Framework | none (hand-rolled) | LangChain (LCEL) | LlamaIndex |
| Data | XLSum Arabic (37.5K articles) | XLSum Arabic (same) | new, topic-split corpus |
| Retrieval pattern | flat fixed-chunk + cross-encoder rerank | ParentDocumentRetriever + rerank | RouterQueryEngine over per-domain indices |
| MRR / NDCG@10 | 0.810 / 0.840 | _pending_ | _pending_ |
| RAGAS faithfulness / relevancy / recall | 0.978 / 0.854 / 1.000 | _pending_ | _pending_ |
| Citation accuracy | 96.67% | _pending_ | _pending_ |
| Routing accuracy | n/a | n/a | _pending_ |
| Avg latency/query | _pending_ | _pending_ | _pending_ |
| Avg cost/query | $0.000454 | _pending_ | _pending_ |

## Reading this table

- **vanilla vs langchain** is the controlled experiment: same data, same eval
  set, same metrics — the only variable is the framework and retrieval
  pattern. This is where "did the abstraction earn its complexity" gets a
  real answer.
- **llamaindex** uses different data and a different retrieval architecture
  (routing across domains instead of one flat index), so its MRR/NDCG numbers
  aren't directly comparable to the other two — it's included for range, and
  brings its own metric (routing accuracy) that the others don't have.

## Status

Only `impl_vanilla` numbers exist so far. Fill in `impl_langchain` and
`impl_llamaindex` rows once those are built.
