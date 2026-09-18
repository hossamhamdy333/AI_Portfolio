<div align="center">

# RAG Chatbot: Vanilla vs. LangChain

`sentence-transformers` `qdrant-client` `langchain` `google-genai` `ragas` `pydantic` `FastAPI` `MLflow` `DVC` `datasets` `pytest`

</div>

---

### Contents

- [Summary](#summary)
- [Problem & motivation](#problem--motivation)
- [Approach](#approach)
- [Data](#data)
- [Results](#results)
- [What I'd do differently / limitations](#what-id-do-differently--limitations)
- [Stack](#stack)

---

## Summary

Two implementations of the same Arabic news Q&A system over BBC XLSum,
built in one repo to answer one question: does LangChain's
`ParentDocumentRetriever` actually retrieve better than flat chunking with
a hand-rolled retrieval loop? Both index the same 300-article corpus,
answer the same 100 synthetic questions, rerank with the same
cross-encoder, generate with the same Gemini model, and get scored by the
same MRR/NDCG code. Neither wins outright: `impl_langchain` takes
retrieval quality (MRR 0.925 vs. 0.802, NDCG@10 0.926 vs. 0.836) and
answer relevancy (0.906 vs. 0.824); `impl_vanilla` takes citation accuracy
(94% vs. 92%), faithfulness (0.991 vs. 0.978), and context recall (0.928
vs. 0.885).

The comparison is less controlled than the repo claims, and that belongs in
the summary rather than a footnote. `impl_langchain`'s child chunks are
sized in **characters** (256 of them) while `impl_vanilla`'s chunks are
sized in **words** (256 of those), because `RecursiveCharacterTextSplitter`
and `chunk_fixed()` count different units from the same config value. The
two sides also rank different numbers of candidates. So the MRR gap is a
gap between two systems that differ in more than their retriever
architecture. Everything downstream of that — the corpus, the questions,
the reranker, the LLM, the scoring code — genuinely is shared, verifiably,
through `shared/`.

## Problem & motivation

"Should I use a framework or write it myself" is one of the least
empirically answered questions in RAG work. Most comparisons are two
separate projects with a results table stapled between them, where the
corpora, the question sets, or the evaluation code differ enough that the
numbers aren't comparable at all. The design choice here is to force the
shared parts to be literally shared code: `shared/eval_set.py` is the only
place the eval set gets loaded and sampled, `shared/metrics.py` is the only
place MRR and NDCG are computed, `shared/llm_client.py` is the only place
Gemini gets called with retry and cost tracking, and both implementations
read one `configs/config.yaml`. If a difference shows up in the results, it
should be traceable to the one thing that's different.

That framing matters because the repo's own history shows how easily it
breaks. `shared/eval_set.py`'s docstring records that `impl_vanilla` used
to sample 30 questions inline in two separate notebooks while
`impl_langchain` sampled 100 through a different code path — two numbers
that looked like a comparison and weren't. `shared/metrics.py` records that
the same MRR/NDCG math existed in three copies. Both were consolidated.
The chunk-unit mismatch described below is the same class of problem,
still present.

The second question underneath is whether retrieval quality and answer
quality even move together. They don't, here, and that's the most useful
thing in the results: the implementation that retrieves better produces
answers that are *less* faithful and cite the right source *less* often.

## Approach

### Corpus and eval set

(built once, in `impl_vanilla`, reused by both).
`01_eda.ipynb` pulls XLSum Arabic, validates every row through a Pydantic
model (`ArticleRow`: non-empty title, article over 20 characters, at least
50% Arabic characters by `arabic_ratio`), and logs the id and reason for
every dropped row to `outputs/reports/dropped_rows.csv` rather than
discarding silently. `02_synthetic_qa.ipynb` samples 200 articles and asks
Gemini for 2 question/answer pairs each in JSON mode, giving 400 questions
where the source article's id is the ground-truth answer.
`shared/eval_set.py` then samples `eval.n_questions = 100` of those with
`random_state=42` for both implementations.

### `impl_vanilla`

— flat chunking, explicit retrieval loop.
`03_chunking.ipynb` implements three strategies (`fixed`, `sentence`,
`semantic`) and picks between them by MRR/NDCG rather than by preference;
the notebook asserts that the winner matches `configs/config.yaml`'s
`chunking.strategy`, so the config can't silently drift from the
experiment that chose it. `04_evaluation.ipynb` embeds the winning chunks
with `paraphrase-multilingual-mpnet-base-v2`, upserts them into a Qdrant
Cloud collection, retrieves `top_k_retrieve = 10`, reranks to
`top_k_rerank = 5` with `mmarco-mMiniLMv2-L12-H384-v1`, and generates with
Gemini under a prompt that forces `[N]` citation tags. Citations are then
re-checked against what was actually retrieved rather than taken at face
value — a `[3]` in the answer maps back to the third reranked chunk's real
`article_id`.

### `impl_langchain`

— `ParentDocumentRetriever` and an LCEL chain. Child
chunks are embedded and searched; the larger parent chunk is what reaches
the LLM, which is the thing flat chunking can't do without restructuring.
`chain.py` is a two-stage LCEL graph: a `RunnableParallel` that
retrieves-and-reranks on one branch while passing the question through
untouched on the other, then a second `RunnableParallel` that generates the
answer while carrying the document list forward, so `run(question)` returns
`(answer, docs)` — the same shape `impl_vanilla`'s `generate_answer()`
returns. The reranker object is passed *into* `build_chain()` rather than
constructed there, so both implementations provably hold the same model.
`citation.py` re-verifies `[N]` tags the same way `impl_vanilla` does.

### Persistence, and a bug fixed along the way

`retriever.py`'s docstring
records that `ParentDocumentRetriever` was originally built with
LangChain's `InMemoryStore` for parent documents — which never touches
disk, despite `dvc add data/chroma_db` implying the whole retriever was
saved. Every downstream notebook therefore had to rebuild from scratch, and
since `build_parent_document_retriever()` called `add_documents()`
unconditionally, each rebuild silently duplicated child chunks already in
Chroma. Now both stores persist (Chroma for vectors, `LocalFileStore` via
`create_kv_docstore` for parents), building is a no-op when the vectorstore
is non-empty unless `force_rebuild=True`, and
`load_parent_document_retriever()` exists for the normal reconnect case.

### Serving

(`impl_vanilla` only). `src/api.py` exposes `POST /chat`
returning an answer plus structured citations, and `src/mcp_server.py`
exposes the identical pipeline as an MCP tool (`ask_arabic_news`) over
stdio for Claude Desktop or any MCP client. Both load models once and read
everything from `config.yaml`. Both point at Qdrant Cloud rather than an
embedded instance — the config records why: `QdrantClient(path=...)`
ignored the old `host`/`port` keys entirely, and an embedded collection
only existed inside the Colab session that built it, so anything serving
from it worked nowhere else.

### Tracking

`shared/tracking.py` points MLflow at a DagsHub-hosted server
so both implementations' runs land in one experiment
(`rag_chatbot_comparison`) at a real URL instead of a local sqlite file.
`shared/llm_client.py` logs prompt/response tokens and computed cost per
call as nested MLflow runs, wrapped in try/except so a tracking-server
hiccup can't kill a long eval loop. Both eval loops checkpoint to disk and
resume, which is not theoretical: the committed RAGAS output is full of
Gemini free-tier 429s (15 requests/minute), and both runs were completed
across multiple sessions.

## Data

- Source: `csebuetnlp/xlsum`, Arabic train split — 37,519 articles (BBC
  Arabic news), fields `id`, `url`, `title`, `summary`, `text`.
- Validation drops 3 rows, all for falling under the 50%-Arabic threshold
  (ratios 0.221, 0.405, 0.465) → **37,516 articles** in
  `xlsum_arabic_clean.parquet`. No missing values anywhere; no
  empty articles.
- 94 articles are exact duplicates of another article. The EDA notebook
  counts them and does not drop them (see Limitations).
- `arabic_ratio` is tight across the corpus: mean 0.810, std 0.013, min
  0.221, max 0.856. Titles average 9.4 words and correlate with article
  length at 0.032 — i.e. not at all.
- Vocabulary (Arabic-script tokens only): 15,937,111 total, 459,148
  unique, 215,324 singletons (47%).
- **The RAG corpus is not the full 37,516.** `03_chunking.ipynb` samples
  300 articles (`random_state=42`), and everything downstream — the Qdrant
  collection, the Chroma store, both eval runs, and the served API — uses
  only those 300.
- Eval set: 200 articles sampled for question generation × 2 questions =
  400 Q&A pairs, of which `shared/eval_set.py` samples 100 for the
  comparison. All 400 fall inside the 300-article RAG corpus (the chunking
  notebook's own filter reports 400 of 400 matched).
- Chunk counts on the 300-article corpus: `fixed` 747 chunks (avg 199.8
  words, std 78.5), `sentence` 692 (avg 196.0, std 67.7), `semantic` 640
  (avg 212.0, std **234.8**). `impl_langchain`'s Chroma store holds
  **4,045 child chunks** over the same 300 articles.

## Results

### Chunking strategy

(`03_chunking.ipynb`, all 400 questions, cosine
similarity over each strategy's own chunk set):

| Strategy | MRR | NDCG@10 |
|---|---:|---:|
| **fixed** (selected) | **0.789** | **0.822** |
| sentence | 0.789 | 0.819 |
| semantic | 0.768 | 0.809 |

`fixed` and `sentence` are tied on MRR to three decimals and separated by
0.003 on NDCG@10 — a tie-break, not a win, and `config.yaml` says so in a
comment rather than presenting `fixed` as a clear winner. `semantic`
trails both while producing the most erratic chunks (word-count std 234.8
vs. 78.5 and 67.7), which is what a similarity-threshold splitter does when
consecutive sentences in news prose rarely drop below 0.2 cosine: it mostly
declines to split, then occasionally splits hard.

### Head-to-head

(100 questions, same sample, same seed):

| Metric | impl_vanilla | impl_langchain |
|---|---:|---:|
| MRR | 0.802 | **0.925** |
| NDCG@10 | 0.836 | **0.926** |
| Citation accuracy | **94.00%** | 92.00% |
| Faithfulness (RAGAS) | **0.991** | 0.978 |
| Answer relevancy (RAGAS) | 0.824 | **0.906** |
| Context recall (RAGAS) | **0.928** | 0.885 |

`impl_vanilla`'s run cost **$0.0655 for 100 questions** ($0.000655/query at
the config's stated `gemini-3.1-flash-lite` rates of $0.25/M input and
$1.50/M output). One of the 100 questions failed generation and was
excluded from RAGAS scoring, so its RAGAS column is over 99 rows;
`impl_langchain`'s is over 100.

**What the split actually says.** `ParentDocumentRetriever` puts the right
article higher in the ranking (MRR 0.925 vs. 0.802) and its answers address
the question better (relevancy 0.906 vs. 0.824) — both consistent with the
LLM receiving a larger, more coherent parent chunk instead of a 200-word
window that may cut mid-argument. But the same larger context makes the
answers slightly less grounded (faithfulness 0.978 vs. 0.991) and the
citations slightly less often correct (92% vs. 94%): more text in the
prompt is more surface for a claim that isn't quite supported, and more
sources to mis-number. Better retrieval did not produce better-cited
answers. On a system whose whole point is citation-checked answers, that's
the tradeoff worth knowing.

The gaps are small in absolute terms. At 100 questions, 2 percentage points
of citation accuracy is 2 questions.

**Tests.** 29 test functions: `shared/tests/test_metrics.py` (8),
`shared/tests/test_eval_set.py` (4), `impl_vanilla/tests/test_data_utils.py`
(5), `test_evaluate.py` (5), `test_qa_generation.py` (2),
`impl_langchain/tests/test_citation.py` (5). I ran them directly — the 12
shared and 10 of the 12 vanilla tests pass locally, and the 5
`impl_langchain` tests pass; `test_qa_generation.py`'s 2 need the
`google-genai` SDK installed, which CI does install.

**MCP dependency check.** `mcp_server.py` imports `MCPServer` from
`mcp.server`, with a comment claiming this replaced the older
`mcp.server.fastmcp.FastMCP`. That checks out against the currently
published package: `mcp` 2.2.0's `mcp/server/__init__.py` exports
`MCPServer` in its `__all__`.

## What I'd do differently / limitations

- **The child chunk size is passed in the wrong unit, which undermines the
  central comparison.** `build_retriever_from_config()` and
  `03_evaluation.ipynb` both pass `config["chunking"]["chunk_size"]` (256)
  as `child_chunk_size` to `RecursiveCharacterTextSplitter`, which measures
  **characters**. `impl_vanilla`'s `chunk_fixed()` measures **words**. So
  vanilla searches over ~200-word chunks while LangChain searches over
  ~256-character (~40-word) children — roughly a 5x difference in the unit
  that actually gets embedded. `retriever.py`'s docstring explicitly claims
  the opposite ("Mirrors impl_vanilla's chunk_size=256/overlap=32 choice
  ... so the retrieval-quality comparison isn't confounded by a different
  chunk size"). Smaller child chunks are a well-known way to improve
  embedding precision, so some unknown share of the MRR gap is chunk
  granularity, not the parent-document architecture. Fixing this means
  either converting units explicitly or giving the two splitters separate,
  clearly-named config keys.
- **The two sides rank different numbers of candidates.**
  `impl_vanilla` retrieves 10 and reranks to 5, so its `ranked_ids` list is
  5 long. `chain.py` calls `retriever.invoke(question)` with no
  `search_kwargs`, so `ParentDocumentRetriever` uses its default k and
  returns at most ~4 parent documents, which the reranker then "reranks to
  5" — a no-op slice. `rag.top_k_retrieve = 10` never reaches the LangChain
  retriever at all. Both sides' MRR is therefore truncated, at different
  lengths, and "NDCG@10" is computed over lists shorter than 10 on both
  sides.
- **`impl_vanilla`'s RAGAS cell references `ragas_llm` and
  `ragas_embeddings`, which the committed notebook never defines.** They're
  passed into `evaluate()` and assigned nowhere in the file. The scores are
  real (the checkpoint resumed 40/99 rows from a prior session), but the
  judge model and embedding model that produced them aren't recoverable
  from the repo, so "both sides used the same RAGAS setup" isn't verifiable
  — `impl_langchain`'s notebook does define its judge explicitly
  (`ChatGoogleGenerativeAI` on `gemini-3.1-flash-lite` plus
  `models/gemini-embedding-001` embeddings). Re-running the vanilla RAGAS
  cell as committed would raise `NameError`.
- **`shared/tracking.py` reads the wrong environment variable.**
  `os.environ.get("DAGSHUB_TOKEN:")` has a trailing colon in the key name,
  so the `DAGSHUB_TOKEN` a user actually exports is never found and
  `getpass` prompts on every run. Harmless interactively in Colab —
  which is why it survived — but it makes any unattended or CI-driven run
  hang on a prompt. One-character fix.
- **The RAGAS judge is the same model family being evaluated.** Gemini
  scoring Gemini's own Arabic answers for faithfulness, with no
  human-labeled subset to calibrate against, means a systematic blind spot
  in the judge would move both columns by the same unmeasured amount. For
  Arabic specifically, there's no evidence here about how well RAGAS's
  English-designed prompts transfer.
- **Cost and latency per query are measured on only one side.**
  `impl_vanilla` reports $0.000655/query; `impl_langchain` routes through
  LangSmith, and no number is committed.
  `ParentDocumentRetriever` sends larger contexts, so its input-token cost
  per generation should be higher — the comparison's most practical
  dimension, and it's an open number. **Open item:** impl_langchain's mean input
  tokens and cost per query, from the LangSmith run for `langchain_eval`.
- **94 duplicate articles are counted in EDA and never dropped.** They
  survive into the cleaned parquet and therefore into the 300-article
  sample and both indexes. For a retrieval eval with exactly one correct
  article per question, a duplicate article competing with the true source
  is a guaranteed miss that isn't the retriever's fault.
- **Synthetic questions only ever see the first 2,000 characters of an
  article.** `qa_generation.py` truncates with `article_text[:2000]` while
  retrieval indexes the whole article. Questions are therefore
  systematically about article openings, which flatters any chunking scheme
  that keeps the opening intact and means late-article content is indexed
  but never queried.
- **The served API answers from 300 of 37,516 articles.** The Qdrant Cloud
  collection holds 747 chunks — the experimental sample, not the corpus.
  `src/api.py` and `src/mcp_server.py` are wired correctly; they just point
  at an index built for measurement. Indexing the full corpus is an
  embedding run, not a code change, but nothing in the repo says the
  deployed surface is this narrow.
- **CI doesn't run `shared/tests/`.** The workflow runs
  `impl_vanilla/tests/` and `impl_langchain/tests/` with two different
  `PYTHONPATH` prefixes, and the 12 shared tests — covering
  `shared/metrics.py`, the code every number in COMPARISON.md depends on —
  are never executed. They also need a third invocation form
  (`PYTHONPATH=shared`), since they import `from metrics import ...` rather
  than `from shared.metrics import ...`. Three path conventions for one
  repo is itself worth collapsing.
- **100 questions, one seed, one run per implementation.** No confidence
  intervals, no repeated runs, no significance test on any gap. The
  faithfulness difference (0.991 vs. 0.978) is well inside what a
  single-run LLM-judged sample of this size could produce by chance.
- **The Qdrant Cloud cluster URL is committed in `config.yaml`.** The API
  key correctly is not, so this isn't a credential leak, but a live
  cluster's endpoint sits in the repo permanently.
- **`impl_vanilla`'s `ingest.py` uses `recreate_collection`**, which the
  qdrant-client already emits a `DeprecationWarning` for in the committed
  notebook output; `collection_exists` plus `create_collection` is the
  current form.

## Stack

- `sentence-transformers` 2.7.0 — `paraphrase-multilingual-mpnet-base-v2`
  (768-dim) as the shared bi-encoder, `CrossEncoder` with
  `mmarco-mMiniLMv2-L12-H384-v1` as the shared reranker
- `qdrant-client` against Qdrant Cloud (cosine) — `impl_vanilla`'s vector
  store, and what the API and MCP server both read
- `langchain` <1.0 with `langchain-chroma`, `langchain-huggingface`,
  `langchain-text-splitters`, `langchain-google-genai` — LCEL chain,
  `ParentDocumentRetriever`, Chroma for child vectors, `LocalFileStore` +
  `create_kv_docstore` for parent documents
- `google-genai` (`gemini-3.1-flash-lite`, temperature 0.1) for generation
  and synthetic Q&A, behind `shared/llm_client.py`'s retry wrapper — which
  parses Google's own "retry in Xs" hint out of the error rather than
  guessing a backoff
- `ragas` 0.2.15 (faithfulness, answer relevancy, context recall) for
  answer-quality scoring on both sides
- `pydantic` for corpus validation (`ArticleRow`) and FastAPI request and
  response models
- `FastAPI` + `uvicorn` for `POST /chat`; `mcp` (stdio transport) for the
  `ask_arabic_news` MCP tool
- `MLflow` on a DagsHub-hosted tracking server (experiment
  `rag_chatbot_comparison`), plus optional `langsmith` tracing on the
  LangChain side
- `DVC` + `dvc-gdrive` for the corpus, the three chunk sets, the synthetic
  Q&A parquet, the Chroma store (830 files, 30.9 MB), and the eval reports
- `datasets` 2.19.0 for `csebuetnlp/xlsum`, `pandas` 2.2.2, `numpy` <2
- `pytest` 8.2.0, 29 tests, plus GitHub Actions on pushes touching
  `rag-vanilla-vs-langchain/**`
