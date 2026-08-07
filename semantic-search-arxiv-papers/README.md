# Semantic Search Engine over ArXiv ML Papers

A search engine over ~50K arXiv machine learning papers, built up in stages — keyword search, then dense embeddings, then a proper vector database, then a cross-encoder reranker on top — and benchmarked at each stage so I could actually see how much each layer was buying me instead of just assuming "bigger model = better."

```
ArXiv ML papers (HuggingFace, 50K abstracts)
    → EDA (length, vocab, duplicates)
    → clean + dedupe → parquet (versioned with DVC)
    → Stage 1: BM25 sparse retrieval (keyword baseline)
    → Stage 2: SBERT dense embeddings + FAISS (semantic retrieval)
    → Stage 3: same embeddings upserted into Qdrant (production-style vector DB)
    → Stage 4: cross-encoder reranking on top of retrieval (two-stage retrieval)
    → same 200-query eval set, same metrics, at every stage
    → FastAPI /search endpoint serving the full retrieve-then-rerank pipeline
```

## Why build it in stages instead of just using the best approach

Because "just use embeddings" isn't obviously the right call until you've actually measured it against a keyword baseline, and "just add a reranker" isn't obviously worth the extra latency until you've measured that too. Each notebook in this repo adds exactly one thing on top of the previous one and re-runs the same evaluation, so the results table below shows the real, isolated effect of each piece — not a single end-to-end number with no way to tell what mattered.

## The dataset

50,000 ML papers (`CShorten/ML-ArXiv-Papers` on HuggingFace) — title + abstract per paper.

| | |
|---|---|
| Papers loaded | 50,000 |
| Duplicate abstracts found | 31 |
| Clean corpus size | 49,969 |
| Avg abstract length | 160 words |
| Abstracts longer than chunk size (256 words) | 2,239 (4.5%) |
| Total tokens across corpus | ~8.0M |
| Unique tokens | ~202K |

EDA (`notebooks/01_eda.ipynb`) also pulled the top topic words in paper titles — unsurprisingly dominated by *learning*, *deep*, *networks*, *neural* — which is really just a sanity check that the corpus is what it claims to be before spending compute embedding all of it. The 4.5% of abstracts that exceed the configured chunk size (256 words) is why chunking exists in the ingest pipeline at all, rather than embedding each abstract as one unbroken block.

## Evaluation methodology

There's no labeled query set for "does this paper match this search," so I built one synthetically: **200 papers are sampled at random, and each paper's own title is used as a stand-in query**, with that paper's ID as the one correct answer. It's not a perfect proxy for how a person actually searches, but it's a repeatable, label-free way to compare retrieval methods against each other on identical footing — and the same 200 queries (same random seed) are reused across all four notebooks, so the comparison is fair.

Metrics: **MRR**, **Recall@k**, and **NDCG@k** for k ∈ {1, 5, 10} (`src/evaluate.py`).

## The four stages

**1. BM25 baseline** (`notebooks/02_sparse_retriever.ipynb`)
Classic keyword-matching sparse retrieval (`rank-bm25`) over all 49,969 abstracts, tokenized with a simple regex tokenizer. This is the "no ML" baseline everything else has to beat.

**2. Dense retrieval — SBERT + FAISS** (`notebooks/03_dense_retriever.ipynb`)
Abstracts encoded with `BAAI/bge-base-en-v1.5` (768-dim, normalized embeddings) on a T4 GPU, indexed with `faiss.IndexFlatIP` (exact inner-product search = cosine similarity on normalized vectors, no approximation).

**3. Vector database — Qdrant** (`notebooks/04_vector_db.ipynb`)
Same embeddings, same model, this time upserted into a Qdrant collection (cosine distance, batched upserts) instead of an in-memory FAISS index — the point being to prove out the retrieval quality on infrastructure that actually looks like what you'd run in production, with a real client/server vector store rather than a flat index loaded into RAM.

**4. Two-stage retrieval with reranking** (`notebooks/05_reranking.ipynb`)
The bi-encoder (BGE) retrieves the top 50 candidates, then `cross-encoder/ms-marco-MiniLM-L-6-v2` rescoring each (query, abstract) pair directly to re-rank down to the final top 5. Cross-encoders are slower (they can't be pre-computed like bi-encoder embeddings) but see the query and document together, which lets them catch relevance signals a bi-encoder's separately-encoded vectors miss.

## Results

Same 200-query eval set at every stage:

| Model | MRR | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| BM25 (keyword baseline) | 0.712 | 0.635 | 0.785 | 0.825 | 0.736 |
| SBERT + FAISS | 0.753 | 0.670 | 0.850 | 0.910 | 0.789 |
| SBERT + Qdrant | 0.753 | 0.670 | 0.850 | 0.910 | 0.789 |
| **SBERT + FAISS + Cross-Encoder rerank** | **0.818** | **0.760** | **0.880** | **0.920** | **0.841** |

A few things worth calling out:

- **Dense retrieval beats keyword search everywhere**, but the gap is real, not massive — BM25 is a genuinely strong baseline on this kind of technical text, where the exact terminology in a paper title (e.g. "convolutional", "reinforcement") tends to also appear in its own abstract, which is exactly the case BM25 is built for.
- **FAISS and Qdrant post identical numbers**, which is the expected — and reassuring — result: same model, same vectors, same similarity metric, just different infrastructure underneath. That's the whole point of that stage; it's evidence the swap to production-style infra didn't cost any retrieval quality, not a new finding about search quality itself.
- **Reranking gives the single biggest jump** in the whole pipeline — MRR goes from 0.753 to 0.818 and Recall@1 from 0.670 to 0.760, a bigger lift than swapping BM25 for embeddings in the first place. The cross-encoder's advantage is exactly that it can weigh query and document together instead of comparing two vectors computed in isolation, which matters most for query/paper pairs where the top candidates are all topically close and the deciding signal is subtle.
- The trade-off is latency: reranking means running a full transformer forward pass on 50 (query, document) pairs per search instead of one cheap vector lookup, so it's the right call when result quality matters more than shaving milliseconds — which is why it's what's wired into the actual serving endpoint.

## What's in the repo

```
semantic_search/
├── notebooks/
│   ├── 01_eda.ipynb              # corpus stats, length distribution, top keywords, dedup
│   ├── 02_sparse_retriever.ipynb # BM25 baseline + eval-set construction
│   ├── 03_dense_retriever.ipynb  # SBERT embeddings + FAISS index + eval
│   ├── 04_vector_db.ipynb        # same embeddings into Qdrant + eval
│   └── 05_reranking.ipynb        # cross-encoder reranking, final 4-way comparison
├── src/
│   ├── ingest.py     # chunking + embedding + Qdrant upsert
│   ├── retrieval.py  # FAISS / Qdrant search, query encoding
│   ├── evaluate.py   # MRR, Recall@k, NDCG@k
│   └── serve.py      # FastAPI two-stage search endpoint (retrieve + rerank)
├── scripts/
│   └── build_index.py    # populates the persistent Qdrant Cloud collection
├── streamlit_app.py       # free-hosted demo UI (Streamlit Community Cloud)
├── configs/config.yaml    # every hyperparameter — chunk size, top_k, model names
├── .env.example            # required env vars (Qdrant URL/key), no secrets committed
├── outputs/
│   ├── figures/    # length distributions, keyword frequency plots
│   ├── indexes/    # FAISS index (DVC-tracked, not committed directly)
│   └── reports/    # JSON results for all four stages
└── data/           # arxiv_subset.parquet (DVC-tracked)
```

## Setup

Nothing here runs against localhost — the vector DB is a hosted Qdrant Cloud cluster, and the data/index binaries live in a DVC remote, not in git.

**1. Get the two things you need (both free, no card required):**
- A Qdrant Cloud cluster: [cloud.qdrant.io](https://cloud.qdrant.io) → free tier → copy the cluster URL and create an API key
- Your own Google Drive folder to use as the DVC remote (see below)

**2. Configure secrets:**
```bash
cp .env.example .env
# fill in QDRANT_URL and QDRANT_API_KEY
```

**3. DVC remote (Google Drive):**
`.dvc/config` in this repo points at a placeholder folder ID — swap it for your own:
```bash
pip install -r requirements.txt
dvc remote modify gdrive_storage url gdrive://<your-google-drive-folder-id>
dvc pull   # first run opens a browser to authenticate with your Google account
```
(The folder ID is the string after `/folders/` in the Drive folder's URL.)

**4. Build the index** (one-time, or whenever the data changes):
```bash
python scripts/build_index.py
```
This is the step that was actually missing before — it's what creates the Qdrant collection and embeds+upserts every abstract into it. Notebook `04_vector_db.ipynb` still exists for the *evaluation* comparison (it intentionally uses a throwaway in-memory Qdrant collection so the notebook is self-contained and reproducible), but it was never wired up to populate a real, persistent collection — `scripts/build_index.py` is that missing wiring.

**5. Serve locally:**
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer models for time series forecasting"}'
```
```json
{
  "query": "transformer models for time series forecasting",
  "results": [
    {"title": "...", "score": 8.42, "abstract": "..."},
    {"title": "...", "score": 7.91, "abstract": "..."}
  ]
}
```
Each request runs the full pipeline server-side: encode the query → retrieve top 50 from Qdrant → cross-encoder reranks → top 5 returned.

## Deploy for free (no credit card)

`streamlit_app.py` is a thin UI over the exact same retrieval code (`src/retrieval.py`) and the exact same Qdrant Cloud collection that `serve.py` uses — same models, same config, same index. Deployed via **Streamlit Community Cloud**, which authenticates with GitHub only and never asks for payment details:

1. Push this repo to GitHub (public, or Streamlit Cloud's one free private-app slot)
2. Run `python scripts/build_index.py` once, locally, so the Qdrant Cloud collection is populated
3. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → "New app" → point it at this repo, main file `streamlit_app.py`
4. Under **Settings → Secrets**, add:
   ```toml
   QDRANT_URL = "https://your-cluster.cloud.qdrant.io"
   QDRANT_API_KEY = "your-key"
   ```
5. Deploy — you get a public `*.streamlit.app` URL

Free tier is ~1GB RAM; `bge-base-en-v1.5` + the MiniLM cross-encoder comfortably fit. Apps sleep after ~12 hours idle and wake on the next visit (~30s cold start) — fine for a portfolio demo, not for something needing guaranteed uptime.

The FastAPI app (`src/serve.py`) is still in the repo and works the same way locally or on any host you do have access to — Streamlit Community Cloud is just the specific path that's genuinely free with no card, so it's what the demo is built around.

## The infra pieces (not just the retrieval models)

- **DVC** — the 50K-paper parquet and the FAISS index are version-controlled without living in git directly; `.dvc` pointer files are what's committed, the actual binaries live in the DVC remote (Google Drive)
- **Qdrant Cloud** — a real, persistent, hosted client/server vector database (cosine distance, batched upserts), not an in-memory index or a localhost container, so the retrieval-quality numbers hold up against something you'd actually deploy
- **FastAPI two-stage serving** — `/search` runs bi-encoder retrieval then cross-encoder reranking server-side per request, reading model names and top_k straight from `config.yaml` so it can't silently drift from what was benchmarked
- **Streamlit demo** — same retrieval code, deployable free with zero payment info required
- **Config-driven** — chunk size/overlap, model names, top_k vs. rerank_top_k, collection name, all live in one `config.yaml`, not hardcoded across notebooks or duplicated in `serve.py`

## A few things I'd improve with more time

- Build a real labeled eval set (or at least a human-annotated relevance sample) instead of relying solely on the title-as-query proxy, since it can't capture queries phrased very differently from how the paper describes itself
- Try approximate FAISS indexes (HNSW/IVF) and measure the recall/latency trade-off at this corpus size, rather than only using exact search
- Hybrid retrieval — combine BM25 and dense scores (e.g. reciprocal rank fusion) before reranking, since the two methods clearly catch different things
- Chunk-level retrieval instead of whole-abstract embeddings, since ~4.5% of abstracts already exceed the configured chunk size
