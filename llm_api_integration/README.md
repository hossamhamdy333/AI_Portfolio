<div align="center">

# LLM API Integration

`FastAPI` `google.generativeai` `requests` `pydantic` `MLflow` `python-dotenv` `PyYAML` `pytest` `scripts/benchmark_backends.py`

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

A FastAPI service that wraps the Gemini API the way you'd actually want an LLM wrapped before other services depend on it: streaming, tool calling, schema-validated structured output, retry logic, and per-request token/cost tracking — not just a `generate_content()` call behind an endpoint. The design claim underneath all of it — that the retry, validation, and tool-dispatch logic is provider-agnostic in shape — was originally untested; it's since been proven by adding Ollama and vLLM as real, switchable backends, and not one line of `tools.py`, `schemas.py`, or `tracking.py` had to change to make that work. Only `client.py` gained a sibling (`local_client.py`). 27 unit tests cover retries, schema validation, tool dispatch, cost math, and backend selection — deliberately scoped to what's actually deterministic, not to "is the model's answer good."

## Problem & motivation

The naive version of an "LLM integration" project is a single happy-path call to an API in a notebook. The real engineering problems live around that call, not in it: what happens when the call times out or hits a rate limit, what happens when the model doesn't return valid JSON despite being asked to, how do you know what a feature is actually costing per request, and how do you stream a response back to a client when you can't "retry" something they've already started reading (retrying from scratch would duplicate content they've already seen). This project is built around those specific failure modes rather than the happy path.

Gemini was picked mainly for practical reasons — free tier, no billing setup — but that creates an implicit claim worth actually testing: is the retry/validation/tool-dispatch design genuinely provider-agnostic, or does it quietly assume Gemini's SDK shape everywhere? Rather than leave that as an assertion, `local_client.py` adds Ollama and vLLM as real, switchable backends behind the same interface, which is the actual test of the claim, not just a restatement of it.

## Approach

### Two backend families behind one interface

(`backend_selection.py`, `client.py`, `local_client.py`): Gemini via `google.generativeai`'s SDK object, or a self-hosted Ollama/vLLM server via plain HTTP — both speak the same OpenAI-compatible `/v1/chat/completions` protocol, so one `local_client.py` module covers both instead of two near-identical files. `local_client.py`'s response objects duck-type Gemini's shape (`.text`, `.usage_metadata.prompt_token_count`, `.usage_metadata.candidates_token_count`), so `tracking.py`'s `extract_token_counts()` and every `response.text` read in `app.py` work unchanged regardless of which backend actually served the request. `select_backend()` is pulled out of `app.py` specifically because `app.py` has import-time side effects (init'ing MLflow, potentially opening a live connection), so testing the selection logic alone would otherwise mean triggering all of that just to check an if-statement.

### Structured output

(`POST /analyze`): sentiment analysis returned as a Pydantic-validated `SentimentResult` (`sentiment`, `confidence` bounded 0–1, `reasoning`), not raw model text. Two separate model instances are built at startup — one plain, one forced into `json_mode=True` via `response_mime_type="application/json"` — rather than a single instance with a mode flag flipped per call, so a bug in call ordering can't accidentally return JSON for a prose request or vice versa. `strip_markdown_fences()` is a defensive fallback for when the model wraps its JSON in ` ```json ` fences anyway, despite `json_mode`. A failed schema validation returns a `502`, not a `500` — the model's output failing validation is treated as a bad-upstream-response, not a bug in this service.

### Streaming

(`POST /chat/stream`): tokens streamed back via `StreamingResponse` as they're generated. Deliberately no retry logic here, unlike the other two endpoints — once a chunk has reached the client, retrying from scratch would duplicate content already seen. A mid-stream failure (dropped connection, a safety-filtered chunk) is caught and the stream ends cleanly with a `[response interrupted]` marker instead of an unhandled exception killing the connection. `stream_gemini()` yields the raw chunk object, not `chunk.text`, specifically so the caller can read `chunk.usage_metadata` off the final chunk — Gemini attaches aggregate usage there, not on every individual chunk.

### Tool calling

(`POST /chat/tools`): the model decides (as a JSON routing decision, not native function-calling) whether it needs `get_current_weather(city)` or `search_documents(query)`, and the server — not the model — actually executes the tool and feeds the result back for a grounded final answer. `run_tool()` raises on an unknown tool name rather than silently ignoring it, since that would hide a real bug (the model hallucinating a tool, or a typo in the tool schema sent to it). `search_documents` deliberately mirrors the retrieval interface used elsewhere in this portfolio's `semantic_search` project (query in, ranked docs out), so swapping the in-memory keyword match for a real Qdrant/FAISS lookup is a one-function change, not a redesign.

### Retries

: `call_gemini()` wraps every non-streaming call in exponential backoff (`config.yaml`: 3 attempts, 2s base, linear multiplier — `backoff_seconds * attempt`). Rate limits and timeouts are treated as expected background noise for any external API call, handled once in the wrapper rather than special-cased per call site. Retry settings are passed explicitly at every one of the three LLM call sites, rather than relying on a function default that happens to match `config.yaml`, so a future change to the retry policy in config can't silently stop applying somewhere it's called from a stale default.

### Cost/usage tracking

(`tracking.py`): every non-streaming call, and the final chunk of every stream, logs prompt/response token counts to MLflow via `log_usage()`, with real cost math (`compute_cost_usd()`, per-million-token rates read from `config.yaml`) instead of a hardcoded `$0`. `gemini-3.1-flash-lite`'s free tier means both rates default to `0.0`, but the calculation itself is correct and ready for a paid model's real pricing. Usage is logged immediately after each call returns, before downstream parsing or tool execution can fail — a call already cost real tokens the moment it returns, so logging it early means a later error can't silently drop that usage record. Tracking failures (e.g. the MLflow server being down) are caught and logged, never allowed to break the actual user-facing request.

### Live demo

(`notebooks/01_demo.ipynb`): spins up the actual FastAPI server and hits every endpoint for real, not against mocks.

## Data

There's no training or eval dataset here — this is a service-integration project, not a model-training one. What's fixed for demonstration purposes:

- `/analyze`: 5 hand-written test sentences spanning clear-positive, clear-negative, and two deliberately ambiguous/mixed cases, run against the live server in `01_demo.ipynb`.
- `/chat/tools`: a small in-memory 3-document corpus (`tools.py`'s `_DOCUMENTS`) backing `search_documents`, and 3 test prompts designed to exercise all three routing outcomes (weather tool, document-search tool, no tool needed).
- `scripts/benchmark_backends.py` generates its own load at request time (configurable concurrency and total-request count) rather than reading from a fixed dataset.

## Results

### Structured output

, from the live demo run: the clear-positive and clear-negative test sentences both returned confidence 1.00 in the correct direction. The two ambiguous sentences both landed as neutral but at different confidence levels — *"It was okay, nothing special but not bad either"* at 0.95, and *"The acting was good but the plot made no sense whatsoever"* at 0.90 — which tracks the real ambiguity of a mixed review better than forcing either into a hard positive/negative bucket.

### Tool routing

, from the same live run, all three outcomes exercised correctly:

| Prompt | Routed to | Result |
|---|---|---|
| "What is the weather like in Cairo right now?" | `get_current_weather` | Correct tool, answered from its output (24°C, clear) |
| "What is 2 + 2?" | *(no tool)* | Correctly answered directly, `tool_used: null` |
| "How does function calling work with language models?" | `search_documents` | Correctly routed, but the 3-document in-memory corpus has nothing on that topic — the tool returned "no matching documents found," and the model reported that honestly instead of fabricating an answer |

The third case is the more informative one: routing is the model's judgment call, not a deterministic function, so a correct routing decision paired with an honest "I don't know" (because the underlying data genuinely doesn't have the answer) is a better outcome to show than only the cases where a real answer happened to be waiting.

**Tests**: 27 unit tests, verified by counting `def test_*` functions directly across all 8 test files (matches the number the project's own docs state) — `test_client.py` (3: retry succeeds first try / after retries / raises after exhausting attempts), `test_client_json_parsing.py` (3: fence-stripping variants), `test_local_client.py` (3: `call_local`/`stream_local` against a real local HTTP server speaking the OpenAI-compatible protocol, not a mocked `requests.post`, plus a retry-contract match against `call_gemini`'s), `test_backend_selection.py` (5: each provider value, shared functions across Ollama/vLLM, unknown-provider error), `test_schemas.py` (6: valid/invalid `SentimentResult` and `ToolCallRequest` cases), `test_streaming.py` (1: raw chunks yielded, not `.text`), `test_tools.py` (3: weather stub, document match, unknown-tool error), `test_tracking.py` (3: zero-rate cost, nonzero-rate cost, config values actually reaching MLflow).

## What I'd do differently / limitations

- **Tool routing is a JSON-in-plain-text convention, not Gemini's native function-calling schema.** Wiring the actual function-calling API into the request would remove a class of failure this project's own demo surfaces (the model occasionally answering directly instead of routing when it should route, or vice versa) — this is the single highest-value change to make next.
- **No eval set for the tool-routing decision itself.** Every other tested component here has a deterministic right answer to assert against (does a retry retry, does a bad schema get rejected); routing correctness doesn't, and currently has no eval beyond the three prompts shown in the demo notebook.
- **`search_documents` is a 3-document in-memory keyword match**, not the real retrieval this project's interface is designed to be swapped for. The interface already matches `semantic_search`'s Qdrant-backed retrieval, so this is a one-function swap, not a redesign — just not done yet.
- **No request-level rate limiting on the FastAPI side itself** — only retry-on-failure against Gemini's own limits. A client hammering this service wouldn't be throttled by this service.
- **`google.generativeai` (the SDK this project is built on) is the older, deprecated Gemini SDK** — it raises a `FutureWarning` on every import, and Google's own deprecation notice says it won't support new Gemini models much longer. This was found while adding the Ollama/vLLM backends and deliberately not fixed at the same time — migrating to `google-genai` is a separate change that deserves its own pass, not something to bundle into a backend-switching change.
- **The backend-comparison script (`benchmark_backends.py`) reports latency/throughput only, not a dollar comparison.** Gemini's token cost is tracked via MLflow, but a self-hosted backend's real cost is whatever the serving hardware costs per hour, which the script doesn't estimate — an honest omission rather than a filled-in guess, but it means "which backend is cheaper" isn't actually answered here, only "which is faster."
- **Cost math defaults to $0.0 because the current model is on Gemini's free tier** — the calculation is correct and tested (`test_tracking.py` covers both zero-rate and nonzero-rate cases), but there's no live example in this repo of a nonzero cost actually being logged in production use.

## Stack

- `FastAPI` for the service, `uvicorn` to run it
- `google.generativeai` for the Gemini backend (flagged above as due for migration to `google-genai`)
- `requests` for the Ollama/vLLM OpenAI-compatible HTTP backend
- `pydantic` for request/response schema validation (`SentimentResult`, `ToolCallRequest`)
- `MLflow`, hosted via DagsHub, for per-request token/cost tracking
- `python-dotenv` for loading `GEMINI_API_KEY` from `.env` before the app reads it
- `PyYAML` for `config.yaml` (model name, backend selection, retry policy, pricing, ports)
- `pytest`, 27 tests across 8 files (retries, schema validation, tool dispatch, cost math, backend selection, streaming chunk shape)
- `scripts/benchmark_backends.py` for p50/p95 latency and throughput comparison across whichever backend is active, at configurable concurrency levels
