# RAG Chatbot with Citations (Arabic)

A retrieval-augmented QA chatbot over ~37K Arabic news articles, where every answer comes with `[N]` citation tags pointing back to the exact source chunk. The chunking strategy, retrieval, and generation aren't just picked and shipped — chunking is A/B/C-tested against a synthetic eval set before one strategy is selected, and the final pipeline is scored end-to-end with RAGAS rather than eyeballed.

```
XL-Sum Arabic (HuggingFace, 37.5K articles)
    → EDA (length, vocab, Arabic-ratio, duplicates)
    → Pydantic row validation → clean parquet (versioned with DVC)
    → synthetic Q&A generation (Gemini, 200 articles → 400 questions) = eval set
    → 3 chunking strategies scored on the same eval set → winner selected
    → winning chunks embedded (multilingual SBERT) → Qdrant
    → retrieve top-10 → cross-encoder rerank → top-5 → Gemini generation w/ citations
    → RAGAS end-to-end scoring (faithfulness, answer relevancy, context recall)
    → Prefect flow for nightly re-ingestion
    → pytest + GitHub Actions CI
```

## Why build it this way

The easy version of this project is "chunk it somehow, embed it, ask an LLM." The actual engineering questions are earlier than that: how do you even know which chunking strategy is right for this corpus, and once you have an answer, how do you know the RAG system isn't just hallucinating a fluent-sounding response on top of it? So this repo builds a synthetic Q&A set first, uses it to pick the chunking strategy with real numbers instead of a guess, and then closes the loop with RAGAS instead of stopping at "the demo looks right."

## The dataset

XL-Sum Arabic (`csebuetnlp/xlsum`, BBC Arabic news), train split only.

| | |
|---|---|
| Articles loaded | 37,519 |
| Duplicate rows found | 94 |
| Dropped on validation (< 50% Arabic characters) | 3 |
| Clean corpus size | 37,516 |
| Avg article length | 429 words / 2,536 characters |
| Vocabulary | 459K unique tokens (215K singletons) out of 15.9M total |
| Mean Arabic-character ratio | 0.81 |

Cleaning is enforced with a Pydantic `ArticleRow` model (`src/data_utils.py`) rather than ad-hoc pandas filtering — empty titles, articles under 20 characters, and articles under 50% Arabic text are all rejected row-by-row, and every dropped row is logged with its reason to `outputs/reports/dropped_rows.csv` instead of silently vanishing. Full EDA — length distributions, title-vs-article correlation, top tokens — lives in `notebooks/01_eda.ipynb`.

## Building the eval set first

There's no labeled "is this the right answer" set for this corpus, so one is generated synthetically before any retrieval work starts: 200 articles are sampled, and Gemini (`gemini-3.1-flash-lite`) is prompted to write 2 answerable questions per article with a short factual answer, giving 400 question–answer pairs each tied back to its source article ID (`notebooks/02_synthetic_q&a.ipynb`, `src/qa_generation.py`). This is the same set both the chunking comparison and the final RAG evaluation are scored against, so every downstream number is measured against the same ground truth.

## Picking a chunking strategy with numbers, not intuition

Three strategies (`src/chunking.py`) are applied to the same corpus and scored by embedding each strategy's chunks, retrieving against all 400 synthetic questions, and computing MRR / NDCG@10 (`notebooks/03_chunking.ipynb`):

| Strategy | Chunks produced | MRR | NDCG@10 |
|---|---|---|---|
| **Fixed (256 words, 32 overlap)** | **747** | **0.810** | **0.840** |
| Sentence-boundary | 692 | 0.809 | 0.838 |
| Semantic (embedding similarity drop) | 640 | 0.805 | 0.839 |

All three land within a point of each other — on single-topic news articles there just isn't much structural ambiguity for sentence- or semantic-boundary chunking to exploit over a fixed window. Fixed-size wins narrowly and is also the simplest and cheapest to compute, so it's what gets carried forward into the actual pipeline (`configs/config.yaml: chunking.strategy: fixed`). Semantic chunking has the most variable chunk sizes (word-count std of 235 vs. ~70–80 for the other two), which is expected — it splits wherever the topic actually shifts rather than at a fixed length, but that variability isn't paying for itself here.

## The pipeline

**Retrieval** — chunks embedded with `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim) and upserted into Qdrant (cosine distance). Top 10 candidates retrieved per query (`src/retrieval.py`, `src/ingest.py`).

**Reranking** — `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` rescores the 10 candidates against the query and cuts down to the top 5 that actually go to the LLM.

**Generation with citations** — the 5 reranked chunks are numbered and passed to Gemini with a prompt that requires every claim to be tagged `[N]`; `src/generation.py` then parses which source numbers actually appear in the response and maps them back to article/chunk IDs, so "citations" are checked against the model's actual output rather than assumed.

**Example** (`notebooks/04_rag_pipeline.ipynb`):
> **Q:** ما هي اللغات الرسمية الأربع في سويسرا؟
> **A:** اللغات الرسمية الأربع في سويسرا هي الألمانية، والفرنسية، والإيطالية، والرومانشية `[2][3]`
> Cost: $0.000419 · both citations resolved to the correct source article

Run across a 30-question eval sample: **96.67% citation accuracy** (the article actually cited matches the article the question was generated from) at an average cost of **$0.000454/query**.

## RAGAS evaluation

The same 30-question sample scored end-to-end with RAGAS (`notebooks/05_ragas_evaluation.ipynb`):

| Metric | Score |
|---|---|
| Faithfulness | 0.978 |
| Answer relevancy | 0.854 |
| Context recall | 1.000 |

Context recall of 1.0 says retrieval is essentially never the bottleneck on this sample — the right chunk is always in the top 5. Faithfulness at 0.978 means answers are staying tightly grounded in the retrieved sources rather than drifting into the model's own knowledge. Answer relevancy is the softest of the three (0.854), which tracks with RAGAS's own definition — it penalizes answers that are correct but slightly under- or over-specific relative to the question, not factual errors.

## What's in the repo

```
rag_chatbot/
├── notebooks/
│   ├── 01_eda.ipynb              # corpus stats, Arabic-ratio, vocab, duplicates
│   ├── 02_synthetic_q&a.ipynb    # Gemini-generated eval set (400 Q&A pairs)
│   ├── 03_chunking.ipynb         # fixed vs sentence vs semantic, MRR/NDCG comparison
│   ├── 04_rag_pipeline.ipynb     # Qdrant ingest, rerank, citation generation
│   └── 05_ragas_evaluation.ipynb # RAGAS scoring, Prefect flow, CI setup
├── src/
│   ├── data_utils.py    # Pydantic row validation, Arabic-ratio filtering
│   ├── qa_generation.py # synthetic Q&A generation + cost tracking
│   ├── chunking.py       # fixed / sentence / semantic chunkers
│   ├── ingest.py         # Qdrant collection build + upsert
│   ├── retrieval.py      # vector search + cross-encoder rerank
│   ├── generation.py     # citation-grounded answer generation
│   ├── evaluate.py       # MRR/NDCG scoring, RAGAS wrapper
│   └── pipeline.py       # Prefect nightly ETL flow
├── tests/                # 14 unit tests across data_utils, evaluate, qa_generation
├── .github/workflows/    # CI: pytest on every push
├── configs/config.yaml   # every hyperparameter, in one place
├── outputs/
│   ├── figures/           # chunking comparison, length distributions
│   ├── reports/           # dropped_rows.csv, RAGAS results
│   ├── indexes/           # Qdrant-related artifacts
│   └── synthetic_qa/      # the 400-question eval set (DVC-tracked)
└── data/                  # cleaned corpus + per-strategy chunks (DVC-tracked)
```

## Try it

```bash
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY

docker run -p 6333:6333 qdrant/qdrant   # Qdrant needs to be running

# reproduce a notebook end to end, e.g. the RAG pipeline itself:
jupyter notebook notebooks/04_rag_pipeline.ipynb
```

```bash
pytest tests/ -v
```

There's no FastAPI serving layer yet (see below) — right now the pipeline is driven from the notebooks and `src/`, not from a live endpoint.

## The MLOps pieces (not just the pipeline)

- **DVC (with a Google Drive remote)** — cleaned corpus, per-strategy chunks, and the synthetic Q&A set are all version-controlled without living in git directly
- **MLflow** — every Gemini call (synthetic Q&A generation and answer generation) logs prompt/response tokens and real cost math per run, not printed-and-forgotten
- **Prefect** — a nightly ETL flow (`nightly_etl_flow`, cron `0 2 * * *`) re-loads the corpus, re-embeds, and re-upserts into Qdrant, with automatic retries on the load step
- **Config-driven** — chunk size/overlap, retrieval/rerank model names, top_k values, Gemini generation settings, and cost rates all live in one `config.yaml`
- **pytest + GitHub Actions** — 14 unit tests (Pydantic validation, MRR/NDCG math, synthetic Q&A parsing), CI runs on every push to this folder

## A few things I'd improve with more time

- Build the FastAPI `/chat` endpoint the config is already set up for (`serving.host`/`serving.port`) — right now the pipeline only runs from notebooks
- Run the chunking comparison and RAGAS evaluation on a larger sample than 30–400 questions to get tighter confidence on the numbers, especially citation accuracy
- Try a hybrid BM25 + dense retrieval step before reranking, since Arabic news text has a lot of exact proper-noun and place-name matching that keyword search is well-suited to
- Add a RAGAS "answer correctness" or human-labeled sample to catch faithfulness failures that don't show up in a 30-question sample
