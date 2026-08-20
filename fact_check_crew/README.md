# fact_check_crew: Does a Verify-and-Revise Loop Catch Hallucinations?

Three CrewAI agents — Researcher, Writer, Critic — instead of one model
answering directly. The Researcher searches, the Writer drafts from what
was found, the Critic checks every claim in the draft against the actual
sources and can send it back to the Writer once. The real question this
answers: **does that extra structure catch unsupported claims a single
LLM call, answering from the same sources, would have let through?**

Standalone in the sense of having its own agents and eval logic, but
deliberately *not* standalone in its data: it reads
[`../rag_router`](../rag_router)'s existing Qdrant collections (read-only)
and reuses its 400-question eval set as-is. Building a second Wikipedia
corpus and a second question set for this would have tested nothing new —
the thing worth testing here is the agent architecture, not the retrieval.

## Why 3 agents and not more

Researcher → Writer → Critic is the minimum that makes "verify and revise"
a real behavior instead of a label. An earlier draft of this project had a
4th "Editor" role that just rubber-stamped whatever the Critic approved —
folding "finalize" into the Critic's own approval removes a whole agent
for the same behavior. No CrewAI Flow, no graph, no manager agent — three
roles and one Python `while` loop (`src/crew.py`) that gives the Writer at
most one more attempt if the Critic rejects the draft, then stops. That
loop is deliberately plain Python, not framework machinery, because
"try again once if rejected" doesn't need more than that.

## The comparison

Both conditions answer from the *same* retrieved passages (same search
tool, same question) and get judged by the *same* faithfulness check — one
direct LLM call per answer, asking "which of these claims isn't actually
backed up by the sources" — so neither side gets an easier check:

- **Baseline**: one LLM call, straight from the retrieved passages to an answer.
- **Crew**: Researcher → Writer → Critic, with up to 1 revision.

Real numbers are in [`COMPARISON.md`](./COMPARISON.md) — `notebooks/02_evaluation.ipynb`
has run, 100/100 questions, no failures.

## Notebooks (run in order, after `rag_router`'s notebooks 01–03 have run)

1. **`01_build_crew.ipynb`** — connects to `rag_router`'s Qdrant
   collections, builds the search tool and the crew, sanity-checks it on
   one question.
2. **`02_evaluation.ipynb`** — runs both conditions across a 100-question
   sample of `rag_router`'s eval set, logs hallucination rate, revision
   rate, and approval rate to DagsHub.

## Nothing local

Same standard as the other two projects: Qdrant Cloud (reused from
`rag_router`, read-only), MLflow hosted on DagsHub (reusing
`../rag_router/shared/tracking.py` directly rather than a third copy of
the same ~15 lines — this project already depends on `rag_router`'s
outputs, so treating it as fully independent would be redundancy without
a real benefit).

## Tests

```bash
cd fact_check_crew
PYTHONPATH=src pytest tests/ -v
```
20 tests: the retry-loop logic (mocked CrewAI calls, verifying it skips
revision when approved immediately, passes feedback through correctly,
and stops at `max_revisions` instead of looping forever), the search
tool's payload parsing (against a *real* LlamaIndex-populated Qdrant
collection, not an assumed shape — an earlier version of this assumed a
flat `text` payload key that doesn't actually exist), and the faithfulness
judge's JSON parsing, including a deliberate check that a judge which
failed to return parseable output is counted as *unknown*, not silently as
"found nothing wrong."
