# impl_vanilla vs. impl_langchain

## Setup

Both implementations use the same corpus (XLSum Arabic), the same 100-question
eval sample, the same embedding model, the same cross-encoder reranker, and
the same LLM for generation. The variable under test is retrieval
architecture:

- **impl_vanilla**: flat chunking (fixed-size), direct vector search.
- **impl_langchain**: `ParentDocumentRetriever`. Small chunks are embedded
  and searched, but the larger parent chunk is passed to the LLM.

## Results

| Metric | impl_vanilla | impl_langchain |
|---|---|---|
| Eval sample size | 100 | 100 |
| MRR | 0.802 | 0.925 |
| NDCG@10 | 0.836 | 0.926 |
| Citation accuracy | 94.00% | 92.00% |
| Faithfulness (RAGAS) | 0.991 | 0.978 |
| Answer relevancy (RAGAS) | 0.824 | 0.906 |
| Context recall (RAGAS) | 0.928 | 0.885 |

## Metric definitions

| Metric | Measures |
|---|---|
| MRR / NDCG@10 | Retrieval quality: rank of the correct source among retrieved results. |
| Citation accuracy | End-to-end: retrieval quality combined with whether the LLM cited the correct source in its answer. |
| Faithfulness | Whether the answer's claims are supported by the retrieved context. |
| Answer relevancy | Whether the answer addresses the question asked. |
| Context recall | Whether the retrieved context contains what the reference answer required. |

## Findings

`impl_langchain` outperforms `impl_vanilla` on retrieval quality (MRR,
NDCG@10) and answer relevancy. `impl_vanilla` outperforms on citation
accuracy, faithfulness, and context recall. Neither implementation
dominates across all metrics.

`ParentDocumentRetriever`'s higher MRR/NDCG indicates that indexing on
small chunks while passing larger parent chunks to the LLM improves what
gets retrieved. Its lower citation accuracy and faithfulness indicate that
the larger context passed to the LLM does not translate into more
consistently correct citations or fully grounded answers.

## Chunking strategy (impl_vanilla)

`fixed` and `sentence` chunking are statistically tied on retrieval
quality (MRR 0.789 both; NDCG@10 0.822 vs. 0.819). `semantic` chunking
underperforms both (MRR 0.768, NDCG@10 0.809). `impl_vanilla` uses `fixed`
chunking, per `configs/config.yaml`.

## Not yet measured

Cost and latency per query. Both implementations log per-query cost
(`impl_vanilla` via `shared/llm_client.py`, `impl_langchain` via
LangSmith). `ParentDocumentRetriever`'s larger context windows are
expected to increase input tokens per generation call relative to
`impl_vanilla`'s flat chunks; not yet quantified.
