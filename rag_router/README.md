# rag_router: Multi-Domain RAG with Query Routing

A router sits in front of 4 separate topic-specific indexes (sports, tech,
history, English literature — built from a real Wikipedia sample, not
hand-picked pages) and decides which one to query for a given question,
before that domain's retriever and the LLM take over. Two routing
strategies are compared head-to-head: asking the LLM to pick a domain vs.
picking by embedding similarity to each domain's description.

Standalone project — doesn't share code, corpus, or config with
[`../rag_chatbot`](../rag_chatbot); see that project's own README for the
single-domain retrieval-architecture comparison instead.

## The comparison

| | LLM selector | Embedding selector |
|---|---|---|
| Routing call | 1 extra Gemini call per question | No LLM call — pure embedding similarity |
| Cost/latency | Higher | Lower |
| Known failure mode | Both hit the same LlamaIndex selector-parsing bug (see below) — this isn't a difference between them |

Both are evaluated on the same 400-question set in
`notebooks/04_evaluation.ipynb`; real, current numbers live in
[`COMPARISON.md`](./COMPARISON.md) once that notebook's been run — see that
file for why the ⚠️-flagged historical numbers there shouldn't be trusted
as-is.

## A known LlamaIndex bug, and how this project handles it

LlamaIndex's selector-result parsing occasionally decodes a response with
`choice=None`, which crashes downstream with `TypeError: unsupported
operand type(s) for -: 'NoneType' and 'int'`. It's deterministic for a
given question — retrying the identical call doesn't help — and it
reproduces under **both** selectors, not just the LLM-based one (routing
by embedding similarity means no LLM call happens for routing, but doesn't
avoid this particular parsing path). `src/eval_routing.py` detects this
specific error and short-circuits immediately instead of burning 5
identical retries that can't succeed; every other failure still gets
real, transient-error-appropriate retries. See `router.py`'s and
`eval_routing.py`'s docstrings for the full story.

## Notebooks (run in order)

1. **`01_build_corpus.ipynb`** — streams `wikimedia/wikipedia`, filters
   into 4 domains by keyword match (not hand-picked titles), saves one
   parquet per domain via DVC.
2. **`02_ingest_and_router.ipynb`** — embeds each domain's articles into
   its own Qdrant Cloud collection, builds the embedding-based router,
   sanity-checks it on a few questions.
3. **`03_synthetic_qa.ipynb`** — generates the 400-question eval set
   (50 articles × 2 questions × 4 domains) with Gemini, tagged by domain
   for scoring routing accuracy later.
4. **`04_evaluation.ipynb`** — evaluates **both** selectors on the same
   question set, with separate checkpoint files per selector (see below
   for why that matters), reports routing accuracy / citation accuracy /
   MRR / NDCG for each.

## Nothing local

- **Qdrant Cloud**, not local disk-persisted indexes — a free cluster at
  [cloud.qdrant.io](https://cloud.qdrant.io). The original version of this
  project persisted each domain's index as JSON directly into the repo's
  working directory, which then got swept into git (not DVC) by a blind
  `git add .`, permanently bloating the repo's history with binary index
  data every run.
- **MLflow, hosted on DagsHub** — see `shared/tracking.py`, same one-time
  setup as `rag_chatbot`.
- **DVC** for the corpus and eval-set parquet files — properly this time.
  An earlier version of this project's `03_synthetic_qa.ipynb` committed
  the eval-set parquet straight to git, then tried to `dvc add` the
  already-git-tracked file in the next cell, leaving DVC's own cache in a
  warned, inconsistent state. Data goes through `dvc add` first now; the
  GitHub Push cells only ever add code, config, and `.dvc` pointer files.

## A real, fixed bug worth knowing about: checkpoint collisions

An earlier version of `04_evaluation.ipynb`'s ancestor notebooks used one
shared, un-versioned checkpoint file across separate Colab sessions. Two
different sessions resumed from that same file at two different points and
each produced a different "final" routing accuracy for what was supposed
to be one run — the two numbers that used to be reported (0.725 and 0.66)
weren't a real discrepancy to explain, they were two incomplete runs of the
same thing. `eval_routing.py`'s `run_eval()` now takes a `selector_name`
and builds a checkpoint filename from it, so two different runs can never
silently share state.

## Tests

```bash
cd rag_router
PYTHONPATH=src:shared pytest tests/ shared/tests/ -v
```
30 tests, all pure logic or against a real (in-memory) Qdrant client and
real LlamaIndex machinery — no live API keys needed.
