# impl_vanilla vs. impl_langchain

Same corpus (XLSum Arabic), same eval question set, same embedding model,
same reranker, same LLM. The one thing that's different between these two
implementations is the retrieval architecture: flat fixed/sentence-sized
chunks in `impl_vanilla`, vs. `langchain`'s `ParentDocumentRetriever` --
small chunks get embedded and searched, but the larger *parent* chunk
(more surrounding context) is what actually gets passed to the LLM.
Reranker, prompt shape, and citation-correctness definition are held
constant across both so a metric delta traces back to that one variable,
not a second confounding one.

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

## What changed (bugs fixed before these numbers were measured)

- **`configs/config.yaml`'s `chunking.strategy`** used to say `"fixed"`
  with a comment claiming fixed had won the chunking ablation, but the
  config's actual value had drifted out of sync with what
  `03_chunking.ipynb` measured on an earlier run. Re-running the ablation
  for real showed `fixed` and `sentence` are essentially tied (MRR 0.789
  both, NDCG@10 0.822 vs 0.819) with `semantic` clearly behind (0.768/0.809)
  -- `fixed` is what's actually configured and what the numbers above were
  measured on. `04_evaluation.ipynb` also now pulls whichever strategy's
  chunk file the config actually names instead of a hardcoded
  `fixed_chunks.parquet`, so this can't silently drift again.
- **Eval sample size** was inconsistent -- vanilla scored itself on 30 out
  of ~400 available questions, langchain used 100, through two different
  sampling code paths. Both now go through `shared/eval_set.py` with one
  `n` read from `configs/config.yaml`.
- **MLflow tracking store** was split -- langchain logged to its own
  `impl_langchain/mlflow.db` under experiment `"rag_chatbot_langchain"`,
  vanilla to `mlflow.db` under `"rag_chatbot_comparison"` -- both local
  sqlite files, neither viewable without running `mlflow ui` on the exact
  machine that produced them. Both now log to a shared, DagsHub-hosted
  MLflow server (`shared/tracking.py`), so runs from either implementation
  are viewable side-by-side at a real URL, by anyone, without you running
  anything locally.
- **`impl_langchain`'s retriever used to get rebuilt from scratch in every
  notebook** (`01_build_retriever.ipynb`, `02_rag_pipeline.ipynb`, and
  `03_evaluation.ipynb` each re-embedded all 300 documents), because
  `ParentDocumentRetriever`'s parent-document store (`InMemoryStore`) was
  never actually persisted to disk -- only the child-chunk Chroma vectors
  were. Worse, re-running `build_parent_document_retriever()` against an
  already-populated `persist_directory` silently duplicated the child
  chunks in Chroma each time. `retriever.py` now persists both stores
  (Chroma + a `LocalFileStore`-backed docstore), and only `01_build_retriever.ipynb`
  builds; `02` and `03` load. This shouldn't change the *quality* numbers,
  but it does change what a fresh run of `02`/`03` actually does, so it's
  worth confirming nothing regressed.

## Reading the numbers

- **MRR/NDCG** isolate retrieval quality specifically -- did the right
  article end up near the top of what got retrieved, independent of
  whether the LLM used it well.
- **Citation accuracy** is end-to-end: retrieval *and* whether the LLM
  actually cited the right source in its answer. A model can retrieve the
  right chunk and still fail to cite it (or cite something else instead),
  so this can be lower than MRR even with good retrieval.
- **RAGAS faithfulness** asks whether the answer's claims are actually
  supported by the retrieved context (hallucination check, independent of
  whether the *right* context was retrieved).
- **RAGAS context recall** asks whether the retrieved context contains
  what the reference answer needed -- closely related to MRR/NDCG but
  scored by an LLM judge rather than exact article-ID matching, so it can
  disagree with MRR/NDCG on individual questions even when they agree on
  average.

None of these four are redundant with each other -- a real regression in
just one of them (e.g. good MRR but poor citation accuracy) is itself a
finding worth writing up, not noise to explain away.

## Architecture note: is the added complexity worth it?

`ParentDocumentRetriever` is a genuinely more complex retrieval design
than vanilla's flat chunking -- two splitters, two stores, and (until the
fix above) a persistence story that didn't actually work. Whether that
complexity earns its keep is exactly what the MRR/NDCG/citation numbers
above are supposed to answer: if `impl_langchain` doesn't clearly beat
`impl_vanilla` once both are measured on the same sample size and the same
chunking-strategy correctness, the honest conclusion is that
`ParentDocumentRetriever` didn't pay for its complexity on this corpus --
which is itself a useful, legitimate result for this comparison to
produce, not a disappointing one.

## Cost and latency

Not yet tracked side-by-side in this table. Both implementations log
`cost_usd` per question (vanilla via `shared/llm_client.py`'s cost
tracking, langchain via LangSmith once tracing is enabled in
`02_rag_pipeline.ipynb`) -- worth adding a cost/latency row here once both
are re-run, since `ParentDocumentRetriever`'s larger parent chunks mean
more input tokens per generation call than vanilla's flat chunks, which a
pure quality-metric table doesn't capture.
