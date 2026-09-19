<div align="center">

# fact_check_crew: Does a Verify-and-Revise Loop Catch Hallucinations?

`CrewAI` `gemini/gemini-3.1-flash-lite` `qdrant-client` `sentence-transformers` `pydantic` `MLflow` `pandas` `pytest` `llama-index-core`

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

A three-agent CrewAI system (Researcher, Writer, Critic) tests a specific question head-to-head against a single-pass baseline: does adding a verify-and-revise step actually catch unsupported claims that a plain "search then answer" call would let through? Both conditions retrieve from the same Qdrant collections, answer the same 100 questions, and get scored by the identical faithfulness judge, so neither side gets an easier check. Result: the crew's hallucination rate is a third lower than the baseline's (0.08 vs. 0.12), but the Writer only got sent back for revision on about 1 in 6 questions (`crew_avg_revisions` = 0.17) — most of the gain comes from having a Critic check the draft at all, not from the revision loop actually firing often. That's the real finding here: the architecture helps, but not for the reason you'd assume going in, and it costs roughly 2-3x the LLM calls per question to get it.

## Problem & motivation

The easy version of this project is "add a critique step and assume it helps" — ship the three-agent pipeline, call it more robust, move on. That skips the actual question: does the extra structure change the *outcome*, or does it just add latency and cost around the same answer? Answering that honestly requires a real baseline under the exact same conditions (same retrieved passages, same question, same judge), not a strawman single-call comparison that's easy to beat by construction.

There's a second, quieter trap: assuming the revision mechanism is where the value comes from, since "verify and revise" sounds like its benefit should come from the revising. The data here says otherwise — revisions only happened on 17% of questions, so if the hypothesis had been "the loop earns its cost by catching and fixing bad drafts," the numbers wouldn't support spending much confidence on that story. Most of the benefit is from a second pass of scrutiny existing before an answer goes out, whether or not it results in a rewrite — a materially different (and more useful) thing to know before deciding whether this architecture is worth its cost elsewhere.

## Approach

**Deliberately not standalone in its data.** This project reads `../rag_router`'s existing Qdrant collections (four Wikipedia domain indexes: sports, tech, history, English literature — read-only) and reuses its 400-question synthetic eval set as-is, sampling 100 of them. Building a second corpus and a second question set would have tested data-generation variance, not the thing actually in question here, which is the agent architecture.

### Three agents, one plain retry loop, no framework machinery

(`src/crew.py`): Researcher (searches, never answers from memory) → Writer (drafts only from what the Researcher found) → Critic (checks every claim in the draft against the Researcher's actual passages, approves or rejects with a specific reason). If the Critic rejects, the Writer gets exactly one more attempt with the Critic's feedback folded in, then whatever the Critic says on that attempt is final — a plain Python `while` loop (`run_crew_with_revision`), not a CrewAI Flow or a manager agent. An earlier draft had a fourth "Editor" role that just rubber-stamped whatever the Critic approved; folding that into the Critic's own approval removed a whole agent for the same behavior. The loop-control code is split from the actual CrewAI-calling code specifically so the retry logic (does it stop at `max_revisions`, does it pass feedback through correctly) can be unit-tested by mocking two functions, without needing a live LLM call to verify the control flow.

### Search tool

(`src/search_tool.py`): plain `qdrant-client` vector search across all four domain collections, embedding with `sentence-transformers` directly — no LlamaIndex query engine, since the Researcher just needs relevant passages, not a routed or synthesized answer, and pulling in a whole retrieval framework for that would be unjustified. One real, previously-undocumented gotcha found and fixed here: LlamaIndex's `QdrantVectorStore` doesn't store passage text as a flat `text` payload field — it's nested inside a JSON-stringified `_node_content` field (LlamaIndex's internal node serialization). `title`/`domain` are flat top-level keys, but `text` isn't. This was confirmed by building a real collection with `rag_router`'s ingest pipeline and inspecting the raw payload, not assumed from documentation — an earlier version of the code assumed a flat `text` key that doesn't actually exist.

### Runs CrewAI's synchronous `kickoff()` safely from inside a notebook

(`_kickoff_in_thread`): newer CrewAI versions detect a running asyncio event loop (which Jupyter/Colab kernels always have) and refuse to run `kickoff()` synchronously, raising instead and pointing at `kickoff_async()`. Rather than turn the project's public sync API into an async one — which the tests and `eval_factcheck.py`'s plain checkpointed for-loop both depend on staying sync — the call is isolated inside a fresh thread with no event loop of its own, so CrewAI's check never trips, whether the caller is a notebook cell or a plain script.

### The comparison

(`src/eval_factcheck.py`, run in `notebooks/02_evaluation.ipynb`): baseline is one LLM call straight from retrieved passages to an answer, no writer/critic separation, no revision — the actual bar the crew has to clear, not a strawman. Both the baseline's answer and the crew's final answer are scored by the identical `judge_faithfulness` function: one LLM call asking which claims in the answer aren't supported by the passages, returning an explicit list (empty means fully supported). Critically, the crew's answer is judged against the passages the crew's own Researcher actually retrieved, not a separately-fetched "shared" passage set — a mismatch there (different phrasing, different top-k) could bias the comparison without it being obvious. A judge response that fails to parse as JSON returns `None`, not an empty list, and `summarize()` excludes those from the hallucination rate rather than silently counting an unparseable judgment as "no unsupported claims found."

### Resumable, rate-limit-aware eval loop

100 questions means roughly 600-800 LLM calls (baseline call + judge call + crew's research/write/critique calls, plus a revision pass on some fraction of questions) — comfortably over the ~500/day free-tier ceiling this project family has already hit twice elsewhere in the portfolio. `run_comparison` checkpoints to disk every 5 questions and resumes on restart; a question that fails outright (after 3 retries with linear backoff on 429/503 transient errors) is logged and excluded from every downstream stat, not silently treated as "0 revisions, rejected." Resuming specifically re-attempts previously-failed questions rather than skipping them forever, and discards any checkpointed row that isn't in the current run's question sample, so resuming after a differently-sized run (e.g. an earlier 50-question test) can't leak extra rows into the final count.

## Data

- Reuses `../rag_router`'s existing Qdrant Cloud collections (four domains: sports, tech, history, English literature) and its 400-question synthetic eval set — no new corpus or question set built for this project.
- Evaluation sample: 100 of those 400 questions, fixed random seed 42.
- `notebooks/02_evaluation.ipynb` ran end to end: 100/100 questions scored, 0 failed outright, 0 judge-parsing failures on either condition.

## Results

From `COMPARISON.md`, `gemini-3.1-flash-lite`, seed 42, 100 questions:

| Metric | Value |
|---|---|
| `baseline_hallucination_rate` | 0.12 |
| `crew_hallucination_rate` | **0.08** |
| `crew_approval_rate` | 0.91 |
| `crew_avg_revisions` | 0.17 |
| `baseline_judge_failures` / `crew_judge_failures` | 0 / 0 |
| `n_failed` | 0 |

The crew's hallucination rate is a third lower than the baseline's (0.08 vs. 0.12) — the Critic step catches claims the single-pass baseline lets through. But `crew_avg_revisions` of 0.17 means the Writer was only sent back on roughly 1 in 6 questions, so most of the improvement isn't coming from the revision mechanism actually firing — it's coming from a Critic checking the draft at all, whether or not it sends it back. `crew_approval_rate` of 0.91 means the Critic approved 91% of drafts (first pass or after one revision) — it isn't rejecting so often that the typical final answer is a low-confidence fallback. Both `judge_failures` counts are 0, so neither hallucination rate is diluted by unscored answers.

**Cost isn't free.** The crew runs roughly 2-3x the LLM calls of the baseline per question (research + write + critique, plus a revision pass on ~17% of questions). A third fewer hallucinations for 2-3x the calls is a real tradeoff, not a free win.

## What I'd do differently / limitations

- **The next real question this result raises isn't "does the crew help" — it does — it's whether a cheaper single-pass-plus-critique, without the retry loop, captures most of the same benefit for less cost.** With revisions firing on only 17% of questions, the marginal value the loop itself adds over "one critique, no revision" isn't isolated by this eval and would need its own comparison arm to answer.
- **The faithfulness judge is the same model family (Gemini) judging its own or a similar model's output**, with no separate, stronger, or human-labeled ground truth to check the judge against. A systematic blind spot in the judge's own reasoning (confidently missing a specific class of unsupported claim) would lower both conditions' hallucination rates by the same unmeasured amount without showing up as a gap between them.
- **100 of the available 400 questions were evaluated, not all 400.** A larger sample would tighten confidence in the 0.08 vs. 0.12 gap; this run doesn't report a confidence interval or significance test on the difference.
- **`max_revisions=1` was fixed by config, not swept.** Whether 2 revisions would meaningfully lower the hallucination rate further, or just add cost for the same 91% approval ceiling, isn't tested here.
- **The README's stated test count (20) doesn't match what's actually in the `tests/` directory — 23 test functions across the three files** (`test_crew.py`: 5, `test_eval_factcheck.py`: 10, `test_search_tool.py`: 8). Small, harmless, but exactly the kind of number-drift this portfolio's other projects flag rather than silently leave uncorrected.
- **The `_kickoff_in_thread` workaround is a real fix for a real CrewAI/notebook interaction bug, but it's a workaround around a library version behavior, not a fix upstream** — a future CrewAI release could change that detection logic in a way this thread-isolation trick no longer needs, or no longer works around, and nothing here would notice automatically.
- **Cost is described qualitatively ("2-3x the LLM calls") rather than measured in tokens or dollars per question**, unlike some other projects in this portfolio that track exact per-call cost. A real cost comparison (e.g. total tokens across both conditions for the same 100 questions) would make the tradeoff in the summary concrete instead of approximate.

## Stack

- `CrewAI` (`Agent`, `Task`, `Crew`, `Process.sequential`) for the three-agent Researcher/Writer/Critic pipeline, `crewai.tools` for the search tool
- `gemini/gemini-3.1-flash-lite` via CrewAI's `litellm`-backed `LLM` class (temperature 0.2)
- `qdrant-client` for direct vector search against `rag_router`'s existing collections (no LlamaIndex query layer in this project's own code)
- `sentence-transformers` (`all-MiniLM-L6-v2`) for query embedding, matched to the model `rag_router` used to build the collections
- `pydantic` for the Critic's structured `CriticVerdict` output (`approved`, `feedback`)
- `MLflow`, hosted via DagsHub, reusing `../rag_router/shared/tracking.py` directly rather than a third copy of the same tracking code
- `pandas` for the eval-set sample and results handling
- `pytest`, 23 tests (retry-loop control flow with mocked CrewAI calls, search-tool payload parsing against a real LlamaIndex-populated Qdrant collection, faithfulness-judge JSON parsing including the "unparseable means unknown, not clean" case)
- Test-only dependency: `llama-index-core` / `llama-index-embeddings-huggingface`, used solely to build a realistic Qdrant fixture for `test_search_tool.py`, not used by any of the project's own `src/` code
-e 

---

# Supplementary document: `COMPARISON.md`

<div align="center">

# Baseline vs. Crew: Hallucination Rate

100 questions, `rag_router`'s eval set, `gemini-3.1-flash-lite`, seed 42

</div>

---

### Contents

- [Result](#result)
- [What to look for](#what-to-look-for)
- [Reading the result honestly](#reading-the-result-honestly)

## Result

| Metric | Value |
|---|---|
| `baseline_hallucination_rate` | 0.12 |
| `crew_hallucination_rate` | 0.08 |
| `crew_approval_rate` | 0.91 |
| `crew_avg_revisions` | 0.17 |
| `baseline_judge_failures` / `crew_judge_failures` | 0 / 0 |
| `n_failed` | 0 |

The crew's hallucination rate is a third lower than the single-pass baseline's (0.08 vs. 0.12) — the Critic step is catching claims the baseline lets through. `crew_avg_revisions` is 0.17, meaning the Writer only got sent back on roughly 1 in 6 questions. Most of that improvement isn't coming from the revision loop actually firing — it's coming from a Critic checking the draft at all before it goes out, whether or not it sends it back. The revision mechanism itself is a smaller piece of the gain than expected going in.

`crew_approval_rate` of 0.91 means the Critic approved 91% of drafts (first pass or after one revision), so it isn't rejecting so often that the "final answer" is usually a low-confidence fallback — it's approving most drafts, just checking them first.

Both `judge_failures` counts are 0, so the hallucination rates above aren't diluted by unscored answers — every question got a real faithfulness verdict either way.

## What to look for

| Metric | What it tells you |
|---|---|
| `baseline_hallucination_rate` | Fraction of single-pass answers with at least one claim the judge couldn't find support for in the retrieved passages |
| `crew_hallucination_rate` | Same check, applied to the crew's final answer (after 0 or 1 revisions) |
| `crew_approval_rate` | How often the Critic approved a draft at all (first pass or after one revision) — if this is low, the Critic may be too strict, or the corpus genuinely doesn't support confident answers to many of these questions |
| `crew_avg_revisions` | How often the Writer actually needed a second attempt |
| `*_judge_failures` | How many answers the faithfulness judge itself failed to score (returned unparseable output) — excluded from the hallucination rate, not counted as clean. Watch this: a high count here undermines trust in the rate itself, independent of which condition looks better |

## Reading the result honestly

The crew's hallucination rate is meaningfully lower than the baseline's, which is the expected finding — but `crew_avg_revisions` at 0.17 means revisions were rare. Most of that improvement is coming from *knowing a check exists*, not from the revision mechanism itself being exercised often. If this were pushed further, the next question isn't "does the crew help" (it does) — it's whether a cheaper single-pass-plus-critique without the retry loop gets most of the same benefit for less cost.

**Cost is not free here** — the crew costs roughly 2-3x the LLM calls of the baseline per question (research + write + critique, plus a revision pass on ~17% of questions). A third fewer hallucinations for 2-3x the calls is a real tradeoff, not a free win — worth stating plainly rather than glossing over.
