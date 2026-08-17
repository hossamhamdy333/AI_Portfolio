# RAG Chatbot: Vanilla vs. LangChain

Two implementations of the same RAG chatbot -- an Arabic news Q&A system
over XLSum, with citation checking -- built to answer one question: does
`ParentDocumentRetriever`'s extra complexity actually buy better retrieval
than flat chunking, on this corpus, holding everything else constant?

**Full comparison, methodology, and results: [COMPARISON.md](./COMPARISON.md)**

## Layout

```
rag-vanilla-vs-langchain/
├── configs/config.yaml       # one config, both implementations read it
├── shared/                   # code genuinely shared by both implementations
│   ├── metrics.py            # MRR/NDCG
│   ├── eval_set.py           # load/sample the eval question set
│   ├── llm_client.py         # Gemini retry/cost/token tracking
│   └── tests/
├── impl_vanilla/             # flat chunking, hand-rolled retrieval loop, Qdrant
│   ├── notebooks/
│   ├── src/
│   └── tests/
├── impl_langchain/           # ParentDocumentRetriever, LCEL chain, Chroma
│   ├── notebooks/
│   ├── src/
│   └── tests/
└── COMPARISON.md
```

## Why two implementations share one repo

Same corpus, same eval question set (`shared/eval_set.py` guarantees this
-- both implementations sample the same `n` with the same seed), same
embedding model, same reranker, same LLM. The only variable that's
actually different is the retrieval architecture. That's what makes
`COMPARISON.md`'s numbers meaningful instead of two unrelated projects
with a table stapled between them.

## Running either implementation

Both are built as Colab notebooks (Drive mount, DVC for data/artifacts,
MLflow for experiment tracking). Open a notebook via its Colab badge and
run top to bottom -- see each implementation's own README for specifics.

Run order matters within `impl_vanilla` (`01_eda` → `02_synthetic_qa` →
`03_chunking` → `04_evaluation`) and within `impl_langchain`
(`01_build_retriever` → `02_rag_pipeline` → `03_evaluation`), since each
notebook pulls DVC artifacts the previous one produced.

## What's tracked where

- **DVC**: corpus, chunks, synthetic Q&A set, `impl_langchain`'s Chroma
  vector store, eval reports.
- **Qdrant Cloud** (`impl_vanilla` only, not DVC): the vector store behind
  `src/api.py`/`src/mcp_server.py`. Used to be a local embedded instance
  written to Colab's Drive mount, meaning the serving layer only ever
  worked from inside the exact session that built it. A free cluster at
  cloud.qdrant.io gives it a real URL instead.
- **MLflow, hosted on DagsHub** (not a local file): both implementations
  log to the same DagsHub-hosted tracking server and experiment
  (`shared/tracking.py`, configured via `configs/config.yaml`'s
  `dagshub:` section), so runs from either one show up side-by-side at
  `https://dagshub.com/<owner>/<repo>/experiments` -- no `mlflow ui`,
  no localhost, viewable by anyone with the link.
- **LangSmith** (optional, `impl_langchain` only): per-step chain traces
  at [smith.langchain.com](https://smith.langchain.com) -- see
  `02_rag_pipeline.ipynb`.
- **MCP** (`impl_vanilla` only, optional): `src/mcp_server.py` exposes the
  pipeline as a tool for Claude Desktop or any other MCP client, alongside
  the REST API in `src/api.py`.
