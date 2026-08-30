<div align="center">

# impl_vanilla

Hand-rolled RAG pipeline: flat chunking (fixed-size, sentence-based, or semantic — see `03_chunking.ipynb` for which one wins and why), Qdrant for vector search, a cross-encoder reranker, Gemini for generation with enforced `[N]` citation tags, RAGAS for answer-quality scoring.

`Python` `Qdrant` `cross-encoder` `Gemini API` `RAGAS` `FastAPI` `MCP`

See [../COMPARISON.md](../COMPARISON.md) for how this stacks up against `impl_langchain`.

</div>

---

### Contents

- [Notebooks](#notebooks-run-in-order)
- [Serving](#serving)
- [MCP](#mcp)
- [Experiment tracking](#experiment-tracking)
- [Tests](#tests)

## Notebooks (run in order)

1. **`01_eda.ipynb`** — pulls XLSum Arabic, explores it, validates the corpus with Pydantic (drops empty/too-short/non-Arabic rows, logs why), writes `src/data_utils.py`.
2. **`02_synthetic_qa.ipynb`** — generates the synthetic Q&A eval set with Gemini, writes `src/qa_generation.py`.
3. **`03_chunking.ipynb`** — compares fixed/sentence/semantic chunking by MRR/NDCG against the eval set, writes `src/chunking.py`. Asserts that the winning strategy actually matches `configs/config.yaml`, so this can't silently drift out of sync again.
4. **`04_evaluation.ipynb`** — builds the full pipeline (embed → Qdrant → rerank → generate), then scores citation accuracy, MRR/NDCG, and RAGAS in one pass over the shared eval sample. Writes `src/ingest.py`, `src/retrieval.py`, `src/generation.py`, `src/evaluate.py`, and the test files under `tests/`.

## Serving

`src/api.py` wraps the pipeline in a FastAPI `/chat` endpoint, backed by Qdrant Cloud rather than a local embedded vector store — so it works from anywhere, not just the Colab session that built the collection.

Local run:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Needs `GEMINI_API_KEY` and `QDRANT_API_KEY` in the environment.

## MCP

`src/mcp_server.py` exposes the same pipeline as an MCP tool (`ask_arabic_news`) instead of/alongside the REST endpoint, so it's usable directly from Claude Desktop or any other MCP client. Add it to your client's MCP config pointing at `python -m src.mcp_server` (stdio transport), from the same runtime with Drive mounted and `04_evaluation.ipynb` already run.

## Experiment tracking

Both implementations log to a DagsHub-hosted MLflow server, not a local file — see [shared/tracking.py](../shared/tracking.py) for one-time setup. Runs are viewable at `https://dagshub.com/<owner>/<repo>/experiments`, no local `mlflow ui` needed.

## Tests

```bash
cd rag-vanilla-vs-langchain
PYTHONPATH=impl_vanilla:. pytest impl_vanilla/tests/ -v
```
