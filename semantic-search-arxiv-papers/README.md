# Semantic Search over ArXiv ML Papers — Four Stages, One Eval Set

**Live demo:** [semantic-search-arxiv-papers.streamlit.app](https://semantic-search-arxiv-papers.streamlit.app)

## Summary

A search engine over 49,969 arXiv machine-learning abstracts, built in four
stages and re-benchmarked at every stage on the same 200-query set: BM25
keyword search, then dense retrieval with `BAAI/bge-base-en-v1.5` over a
FAISS exact index, then the same embeddings in a Qdrant collection, then a
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker on top of the dense
candidates. MRR goes 0.7122 → 0.7526 → 0.7526 → 0.8178, and Recall@1 goes
0.635 → 0.670 → 0.670 → 0.760. The result worth knowing is the shape of
that progression: **reranking bought more than switching from keyword
search to embeddings did** (+0.065 MRR vs. +0.040), and it bought it almost
entirely by reordering documents the bi-encoder had already retrieved —
Recall@10 barely moves (0.910 → 0.920) while Recall@1 jumps 9 points. The
FAISS → Qdrant step produces byte-identical metrics, which is the expected
and desired outcome for an infrastructure swap rather than a finding about
search quality.

One caveat belongs in the summary rather than buried below: **the index
these numbers were measured on is not the index the deployed app queries.**
The evaluation indexes one vector per whole abstract; the production
`build_index.py` chunks abstracts first and indexes 55,752 vectors with a
different payload shape, on Qdrant Cloud's approximate HNSW index rather
than exact search. See Limitations.

## Problem & motivation

The default way to build this is to reach straight for embeddings and a
vector database, because that's what a "semantic search" project is
supposed to look like. Two things make that worth resisting.

First, BM25 is a genuinely strong baseline on academic text, and the
project's own numbers show it: 0.635 Recall@1 with no model, no GPU, no
embedding step, and no index to keep in sync. Technical abstracts repeat
the exact terminology of their own titles ("convolutional",
"reinforcement", "variational"), which is precisely the case lexical
matching is built for. Any claim that dense retrieval is worth its
operational cost has to be made against that number, not against nothing.

Second, "add a reranker" and "use a vector DB" are usually asserted rather
than measured, and they're very different kinds of change. One is a
quality/latency trade — a full transformer forward pass over 50
(query, document) pairs per search instead of one vector lookup. The other
should, if done correctly, change nothing about quality at all. Building
the stages separately and re-running identical evaluation after each one is
what makes it possible to say which is which, instead of reporting one
end-to-end number with no way to attribute it.

The harder, quieter problem is that there is no labeled relevance data for
"does this paper answer this search," and inventing one changes what you're
measuring. The proxy used here is defensible but has a specific bias, laid
out below.

## Approach

Everything — chunk size, model names, `top_k`, collection name, k-values —
lives in `configs/config.yaml`, read by all six notebooks, by
`scripts/build_index.py`, by `src/serve.py`, and by `streamlit_app.py`, so
no component can quietly diverge from the benchmarked settings.

**Evaluation method** (`notebooks/02_sparse_retriever.ipynb` builds it,
`src/evaluate.py` scores it). 200 papers are sampled from the cleaned
corpus with `random_state=42`; each paper's own **title** becomes a query,
and that paper's row id is the single correct answer. The same seed and
the same sample are rebuilt identically in notebooks 02–05, so all four
stages are scored on the same 200 (query, answer) pairs. Metrics are MRR,
Recall@k and NDCG@k for k ∈ {1, 5, 10}. Because there is exactly one
relevant document per query, two things follow that are worth stating
plainly rather than leaving implicit: Recall@k here is a hit rate (did the
one correct paper land in the top k), and NDCG@k reduces to
`1/log2(rank+1)` with an ideal DCG of 1, so it's a rank-discounted version
of the same signal rather than independent information.

**Stage 1 — BM25** (`rank-bm25`'s `BM25Okapi`, default k1/b). Abstracts are
tokenized with `re.findall(r"[a-z]+", text.lower())` — no stemming, no
stopword removal, since BM25's IDF term already discounts terms that appear
everywhere. Each query is scored against all 49,969 documents and the full
corpus is ranked.

**Stage 2 — dense retrieval** (`BAAI/bge-base-en-v1.5`, 768-dim). All
49,969 abstracts encoded on a Colab T4 in 781 batches of 64, with
`normalize_embeddings=True`, then indexed with `faiss.IndexFlatIP`. Inner
product over L2-normalized vectors is cosine similarity, and `IndexFlat` is
exhaustive — so this stage measures the embedding model's ceiling with no
approximation error mixed in. At 49,969 vectors, exact search is cheap
enough that an ANN index would only add a recall/latency knob with nothing
to tune it against yet.

**Stage 3 — Qdrant** (`notebooks/04_vector_db.ipynb`). The same encoder,
the same vectors, upserted in batches of 500 into a Qdrant collection
(768-dim, `Distance.COSINE`) with `paper_id`, `title`, `abstract` and
`categories` in the payload, then re-evaluated. This runs against
`QdrantClient(":memory:")` — the client's local in-process mode, not a
server — so it validates the client API, the payload round-trip and the
distance configuration, not the network behavior of a hosted cluster.

**Stage 4 — cross-encoder reranking** (`notebooks/05_reranking.ipynb`). The
bi-encoder retrieves `top_k = 50` candidates from FAISS, then
`cross-encoder/ms-marco-MiniLM-L-6-v2` scores each `[query, abstract]` pair
and the 50 candidates are re-sorted by that score. A cross-encoder can't be
precomputed the way bi-encoder vectors can — it needs the query and the
document in the same forward pass — which is exactly why it sees
interactions two independently-encoded vectors can't, and exactly why it
costs 50 forward passes per search.

**Serving** (`src/serve.py`, `streamlit_app.py`, both on the same code in
`src/retrieval.py`). A `POST /search` FastAPI endpoint encodes the query,
pulls `top_k = 50` from a hosted Qdrant Cloud collection, reranks with the
cross-encoder, and returns `rerank_top_k = 5` results. Connection details
come from `QDRANT_URL` / `QDRANT_API_KEY` in the environment, never from
code, and the app raises a specific error at startup if `QDRANT_URL` is
missing rather than failing on the first request. `GET /health` exists for
free-tier hosts that ping to keep an app awake. The Streamlit demo hits the
same collection through the same functions.

**Index provisioning** (`scripts/build_index.py`, run by
`notebooks/06_build_index.ipynb`). Recreates the Qdrant Cloud collection
from scratch (delete-then-create, so a re-run can't leave stale vectors
behind), chunks each abstract at 256 words with 32-word overlap, embeds
every chunk, and upserts in batches of 256. This is the piece that makes
the hosted demo possible at all — notebook 04 only ever built an in-memory
collection, which has no path to a queryable production index.

## Data

Source: `CShorten/ML-ArXiv-Papers` on Hugging Face (117,592 rows total),
title + abstract per paper.

| | |
|---|---:|
| Papers loaded | 50,000 |
| Duplicate abstracts dropped | 31 |
| Corpus used everywhere downstream | 49,969 |
| Mean abstract length | 160.4 words |
| Median / min / max | 157 / 1 / 439 words |
| Abstracts over the 256-word chunk size | 2,239 (4.5%) |
| Total tokens | 8,020,392 |
| Unique tokens | 202,077 |
| Missing titles or abstracts | 0 |

Two details about how the 50,000 are selected matter for reading every
number below. They're taken with `dataset.select(range(50000))` — the
**first** 50,000 rows, not a random sample, despite `random_seed: 42` being
present in `config.yaml` (it's used for the query sample, not the corpus
sample). Since the source is ordered, that means the corpus is the oldest
~43% of the dataset; the first three rows are papers on compressed
observations, sensor networks, and on-line shortest paths. A query about a
recent architecture has correspondingly little to find. And the `id` field
is the row index cast to a string, not a real arXiv identifier, so results
can't be linked back to arxiv.org without rejoining the source.

The `categories` column is the literal string `"cs.LG"` assigned to every
row in the ingest loop, not read from the dataset — hence the EDA line
"Unique categories: 1". It carries no information and nothing filters on
it.

Top title words are `learning` (15,226), `deep` (6,527), `networks`
(6,045), `neural` (6,031), `based` (3,989) — a sanity check that the corpus
is what it claims to be before spending GPU time embedding all of it.

## Results

Same 200 queries, same correct-answer ids, at every stage. Numbers come
straight from the four committed JSON files in `outputs/reports/`:

| Stage | MRR | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (keyword baseline) | 0.7122 | 0.635 | 0.785 | 0.825 | 0.7231 | 0.7360 |
| SBERT + FAISS | 0.7526 | 0.670 | 0.850 | 0.910 | 0.7695 | 0.7888 |
| SBERT + Qdrant | 0.7526 | 0.670 | 0.850 | 0.910 | 0.7695 | 0.7888 |
| **+ cross-encoder rerank** | **0.8178** | **0.760** | **0.880** | **0.920** | **0.8281** | **0.8411** |

**Dense beats keyword, but not by much, and the gap is smaller than it
looks on MRR.** BM25 ranks the entire 49,969-document corpus, so a correct
answer sitting at rank 400 still contributes 1/400 to its MRR. The three
dense stages only ever produce a 50-document ranking, so anything past rank
50 scores 0. The comparison is therefore mildly generous to BM25 on MRR and
exactly fair on every Recall@k and NDCG@k in the table, since those only
look at the top 10.

**FAISS and Qdrant are identical to 16 decimal places** (the two JSON files
have the same `mrr` value, 0.7525725798588703). Same model, same vectors,
same cosine metric, exhaustive search on both sides — this is the result
that says the infrastructure swap was lossless, which is the point of the
stage. It is not evidence about how a hosted, HNSW-indexed Qdrant cluster
behaves; see Limitations.

**Reranking is the biggest single win, and it's a reordering win.**
Recall@10 moves 0.910 → 0.920 while Recall@1 moves 0.670 → 0.760. The
cross-encoder can only re-sort the 50 candidates the bi-encoder already
found — its Recall@k is hard-capped by the bi-encoder's Recall@50 — so
nearly all of the gain is promoting a correct document that was already in
the candidate set from somewhere in the top 50 to position 1. That's the
practical argument for two-stage retrieval: the bi-encoder's job is to not
lose the answer, the cross-encoder's job is to put it first, and they're
good at different things.

**What's still missing at the top.** Even after reranking, Recall@1 is
0.760 — on a quarter of queries, the paper whose own title was used as the
query does not come back first. Some of that is genuine near-duplicate
competition in a corpus of 50K ML abstracts, where several papers can
legitimately match a title-shaped query; none of it has been inspected
qualitatively, so that's a hypothesis, not a finding.

[ADD: the bi-encoder's Recall@50, which is the hard ceiling on every
reranked metric above — computable from `05_reranking.ipynb` by calling
`recall_at_k(all_rankings_reranked, correct_ids, 50)`, but not currently
run.]

[ADD: measured per-query latency for each of the four stages. The
latency/quality trade-off is the stated reason for choosing reranking, and
nothing in this repo times a search.]

**Production index**, from `06_build_index.ipynb`'s run against the live
cluster: 55,752 vectors indexed (49,969 abstracts expanded into chunks),
collection status green, `indexed_vectors_count = 55752`, HNSW at `m=16`,
`ef_construct=100`.

## What I'd do differently / limitations

- **The benchmarked index and the served index are different systems, and
  nothing measures the served one.** Notebooks 03–05 index one vector per
  whole abstract (49,969 points) and search exhaustively.
  `scripts/build_index.py` chunks each abstract at 256 words with 32-word
  overlap, producing 55,752 points on Qdrant Cloud's approximate HNSW
  index, with a payload of `{title, abstract-chunk}` and **no `paper_id`**
  — so the eval code, which scores by `hit.payload["paper_id"]`, cannot
  even be pointed at the production collection as written. Every number in
  the Results table describes the offline configuration. Re-running the
  200-query eval against the deployed collection, with `paper_id` added to
  the payload, is the single highest-value change here.
- **Nothing deduplicates by paper at serve time.** Since long abstracts
  become multiple chunks, `POST /search`'s top 5 can be five chunks of
  two papers. Both `serve.py` and `streamlit_app.py` sort by cross-encoder
  score and slice — neither groups by title or paper id first.
- **Title-as-query is a biased proxy, and it's biased toward the
  baseline.** A title shares vocabulary with its own abstract far more
  often than a real user's phrasing does, which is the case BM25 is
  strongest on; the dense models' margin would likely widen on natural
  queries ("papers about making transformers cheaper to fine-tune") and
  the whole table would shift. It also means every query has exactly one
  correct answer, so nothing here measures how well a system returns
  *several* relevant papers, which is what a search engine is actually
  for. A few dozen human-judged queries with graded relevance would test
  that; the synthetic set can't.
- **200 queries is a small sample.** The 0.670 → 0.760 Recall@1 gain is 18
  queries changing their top result. No confidence intervals or
  significance tests are computed, and a single fixed seed means there's no
  variance estimate for any number in the table.
- **The corpus is the oldest 50K papers, not a sample of the field.**
  `dataset.select(range(50000))` takes the head of a 117,592-row ordered
  dataset. The demo will be visibly weak on anything recent, and the config
  key `random_seed` next to `subset_size` makes it easy to assume otherwise.
  Switching to `dataset.shuffle(seed=42).select(range(50000))` is a
  one-line fix that would make the corpus representative.
- **No tests and no CI in this project.** Other projects in this portfolio
  run pytest under GitHub Actions; this one has neither, despite
  `src/evaluate.py` containing exactly the kind of off-by-one-prone code
  (1-indexed ranks, `[:k]` slicing, NDCG's `log2(rank+1)`) that unit tests
  catch cheaply. A handful of hand-built ranking fixtures with known MRR
  and NDCG values would be maybe 30 lines.
- **`categories` is a hardcoded constant**, written as `"cs.LG"` for every
  row rather than read from the source. It's harmless because nothing
  filters on it, but it's a column that looks like metadata and isn't, and
  it would silently break any future faceted-search feature built on top.
- **The FAISS/Qdrant equivalence result doesn't transfer to the cloud
  cluster.** Both benchmarked paths are exact; the deployed one is HNSW
  with `ef_construct=100`, which trades recall for speed by design. Whether
  that costs anything at this corpus size is untested — and it's the same
  experiment as the "try approximate indexes and measure the trade-off"
  item, which the offline FAISS setup could answer directly with
  `IndexHNSWFlat` or `IndexIVFFlat`.
- **No hybrid retrieval.** BM25 and the bi-encoder clearly fail on
  different queries — that's implied by the metric gaps but never measured
  per-query — so reciprocal rank fusion before reranking is the obvious
  next arm, and the per-query overlap analysis that would justify it is
  cheap to run from the artifacts already committed.
- **The DVC remote is incomplete.** `01_eda.ipynb`'s own `dvc push` output
  ends with `Some of the cache files do not exist neither locally nor on
  remote`, listing 14 missing hashes. The parquet corpus does pull
  successfully in notebooks 02–06, so the project is reproducible in
  practice, but the remote isn't in the state the README's `dvc pull`
  instructions assume.
- **The Qdrant Cloud cluster URL is visible in `06_build_index.ipynb`'s
  saved output.** The API key was entered through `getpass` and isn't
  exposed, so this isn't a credential leak, but the endpoint of a live
  cluster is in the repo permanently; clearing that cell's output before
  committing would cost nothing.
- **Chunking exists but is never evaluated.** 4.5% of abstracts exceed 256
  words, which is the stated reason chunking is in the ingest path — but no
  stage of the evaluation uses chunked documents, so whether chunk-level
  retrieval helps or hurts on this corpus is an open question the repo
  doesn't answer despite shipping the chunking code to production.

## Stack

- `rank-bm25` 0.2.2 (`BM25Okapi`) for the sparse baseline
- `sentence-transformers` 2.7.0 — `SentenceTransformer` with
  `BAAI/bge-base-en-v1.5` (768-dim) for the bi-encoder, `CrossEncoder` with
  `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- `faiss-cpu` 1.8.0 (`IndexFlatIP`, exact inner-product search over
  normalized vectors)
- `qdrant-client` 1.9.1 — `":memory:"` mode for the offline evaluation,
  a hosted Qdrant Cloud cluster (cosine, HNSW `m=16`/`ef_construct=100`)
  for the deployed app
- `FastAPI` 0.111.0 + `uvicorn` 0.29.0 + `pydantic` 2.7.1 for the
  `/search` and `/health` endpoints
- `Streamlit` 1.38.0 for the public demo, deployed on Streamlit Community
  Cloud with `QDRANT_URL` / `QDRANT_API_KEY` as app secrets
- `datasets` 2.19.0 for loading `CShorten/ML-ArXiv-Papers`
- `torch` 2.2.0 (embedding runs done on a Colab Tesla T4), `numpy` <2.0
  pinned for FAISS compatibility
- `DVC` 3.49.0 + `dvc-gdrive` 3.0.1 for the corpus parquet and the FAISS
  index binary; only `.dvc` pointers are committed
- `pandas` 2.2.2, `matplotlib` 3.8.4, `seaborn` 0.13.2 for EDA and figures
- `PyYAML` 6.0.1 (`configs/config.yaml` as the single source of
  hyperparameters), `python-dotenv` 1.0.1 for local env loading
