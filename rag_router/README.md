# rag_router — Multi-Domain RAG with Query Routing

Routes incoming questions to one of four topic-specific indexes (sports, tech,
history, English literature — built from a Wikipedia sample) and answers from
whichever one gets picked. Two routing methods are compared: an LLM call that
selects the domain, and an embedding-similarity match against each domain's
description.

Standalone project — no shared code, corpus, or config with
[`../rag-vanilla-vs-langchain`](../rag-vanilla-vs-langchain). See that
project's README for the vanilla-vs-LangChain retrieval comparison.

## Results

Evaluated on 400 questions (`notebooks/04_evaluation.ipynb`):

| Metric | LLM selector | Embedding selector |
|---|---|---|
| Routing accuracy | **0.6975** | 0.6450 |
| Citation accuracy | **0.6650** | 0.6275 |
| MRR | **0.6319** | 0.6010 |
| NDCG@10 | **0.6398** | 0.6074 |
| Parsing-bug failures | 39/400 (9.75%) | 4/400 (1.0%) |
| Extra API call for routing | Yes | No |

The LLM selector wins on every accuracy metric, despite hitting the known
LlamaIndex bug below far more often. Excluding failed questions, its routing
accuracy is 77.3% vs. 65.2% for the embedding selector — full breakdown in
[COMPARISON.md](./COMPARISON.md).

## Architecture

- 4 domains, each its own Qdrant Cloud collection
- Routing via LlamaIndex's `RouterQueryEngine`, either `LLMSingleSelector`
  or `EmbeddingSingleSelector`
- Corpus: `wikimedia/wikipedia` (20231101.en), filtered into domains by
  keyword match, 300 articles per domain
- Eval set: 400 synthetic Q&A pairs (50 articles × 2 questions × 4 domains),
  generated with Gemini
- Tracking: MLflow via DagsHub
- Data and model artifacts: DVC

## Known issue: LlamaIndex selector parsing bug

LlamaIndex's selector-result parsing occasionally returns `choice=None`,
which crashes with `TypeError: unsupported operand type(s) for -: 'NoneType'
and 'int'`. It's deterministic per question — retrying doesn't help — and it
affects both selectors, though the LLM selector hits it about 10x more often.
`eval_routing.py` detects this specific error and skips retries for it, while
still retrying transient failures like rate limits normally.

## Notebooks (run in order)

1. `01_build_corpus.ipynb` — streams and filters the Wikipedia corpus into
   4 domains, saves to DVC
2. `02_ingest_and_router.ipynb` — embeds articles into Qdrant, builds the
   router, sanity-checks routing on a few questions
3. `03_synthetic_qa.ipynb` — generates the 400-question eval set
4. `04_evaluation.ipynb` — runs both selectors on the eval set, reports
   routing accuracy, citation accuracy, MRR, and NDCG

Each notebook pulls DVC artifacts the previous one produced.

## Tests

```bash
cd rag_router
PYTHONPATH=src:shared pytest tests/ shared/tests/ -v
```

30 tests — pure logic plus in-memory Qdrant and real LlamaIndex machinery,
no API keys required.
