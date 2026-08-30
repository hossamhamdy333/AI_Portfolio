<div align="center">

# impl_langchain

`ParentDocumentRetriever` over the same XLSum Arabic corpus `impl_vanilla` uses: small child chunks get embedded and searched, but the larger parent chunk is what actually reaches the LLM. Same cross-encoder reranker and same Gemini model as `impl_vanilla`, wired together as an LCEL chain.

`Python` `LangChain` `ChromaDB` `Gemini API` `MLflow` `LangSmith`

See [../COMPARISON.md](../COMPARISON.md) for how this stacks up against `impl_vanilla`, including whether the extra architectural complexity here actually pays for itself.

</div>

---

### Contents

- [Notebooks](#notebooks-run-in-order)
- [Observability](#observability)
- [Tests](#tests)

## Notebooks (run in order)

1. **`01_build_retriever.ipynb`** — builds the `ParentDocumentRetriever` against the same 300-article sample `impl_vanilla`'s RAG eval uses, persists both the Chroma vectorstore and the parent-document store to disk, pushes to DVC.
2. **`02_rag_pipeline.ipynb`** — loads the persisted retriever (no re-embedding), builds the LCEL chain, sanity-checks it on one question. Optional LangSmith tracing setup lives here.
3. **`03_evaluation.ipynb`** — runs the same eval sample `impl_vanilla` scores against (`shared/eval_set.py` guarantees this), computes citation accuracy / MRR / NDCG, then RAGAS.

## Observability

- **MLflow**: logs to the same tracking store/experiment `impl_vanilla` uses (`configs/config.yaml`'s `mlflow:` section) — runs from both implementations are comparable in one MLflow UI.
- **LangSmith** (optional): set up in `02_rag_pipeline.ipynb`. Once enabled, every `chain.invoke()` call afterwards — including the whole eval loop in `03_evaluation.ipynb` — gets per-step tracing (retrieve → rerank → generate) for free at [smith.langchain.com](https://smith.langchain.com), since `langchain-core` reads the tracing env vars automatically. Leave the key blank when prompted to skip it.

## Tests

```bash
cd rag-vanilla-vs-langchain
PYTHONPATH=impl_langchain:. pytest impl_langchain/tests/ -v
```
