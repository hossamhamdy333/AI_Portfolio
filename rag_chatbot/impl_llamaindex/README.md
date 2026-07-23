# RAG Chatbot — LlamaIndex Implementation (Router Pattern)

Different dataset from vanilla/langchain, and a different retrieval pattern —
not just vector-search-then-generate again.

## What's different

- **Data**: 4 topic domains — sports, tech, history, english_literature —
  instead of one flat corpus. Each domain gets its own `VectorStoreIndex`.
- **Data source**: `src/build_corpus.py` streams the `wikimedia/wikipedia`
  dataset (properly licensed, CC BY-SA, built for downstream ML use) and
  buckets ~300 articles per domain by keyword-matching title/text — the same
  "real dataset + real filtering logic" discipline as `impl_vanilla`'s XLSum
  pipeline, not a hand-picked page list.
- **Pattern**: a `RouterQueryEngine` picks the right domain index for a given
  question before retrieving, instead of searching one big index every time.
- **New metric**: routing accuracy (did it pick the right domain?), on top of
  the same MRR/NDCG/RAGAS numbers used elsewhere, via `shared/metrics.py`.
- **New eval set**: the 400-question synthetic Q&A set from vanilla doesn't
  apply here (wrong corpus) — a new one gets generated for this data, tagged
  with which domain each question belongs to.

## Structure

```
impl_llamaindex/
├── data/
│   └── processed/          # <domain>.parquet, one per domain, built by build_corpus.py
├── src/
│   ├── build_corpus.py     # streams wikimedia/wikipedia, filters into the 4 domains
│   ├── ingest.py            # builds one VectorStoreIndex per domain
│   ├── router.py            # RouterQueryEngine wiring the domain indices
│   ├── qa_generation.py     # new synthetic eval set for this corpus (reuse vanilla's approach)
│   ├── eval_routing.py      # checks router's domain pick against ground truth
│   └── generation.py        # citation-grounded answer generation
├── notebooks/
└── README.md
```

## Status

All code written (`build_corpus.py`, `ingest.py`, `router.py`, `qa_generation.py`,
`generation.py`, `eval_routing.py`). **None of it has been run** — no real
corpus, no real eval numbers yet. Next steps, in order:
1. `pip install datasets llama-index llama-index-embeddings-huggingface google-genai mlflow`
2. Run `build_corpus.py` in Colab → produces the 4 domain parquet files. Eyeball
   a sample per domain for keyword-match false positives before trusting it.
3. Run `ingest.py`'s `build_all_indexes()` → builds + persists the 4 VectorStoreIndexes
4. Run `qa_generation.py`'s `generate_qa_dataset()` over the 4 domain dataframes → synthetic eval set
5. Run `eval_routing.py`'s `run_eval()` → routing accuracy, citation accuracy, MRR/NDCG
6. Fill in `COMPARISON.md`'s `impl_llamaindex` column
