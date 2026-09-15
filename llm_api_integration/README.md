<div align="center">

# LLM API Integration

A FastAPI service that wraps the Gemini API properly. Not just a `generate_content()` call behind an endpoint, but streaming, tool calling, schema-validated structured output, retry logic, and per-request token/cost tracking — the way you'd actually want an LLM wrapped before other services start depending on it.

`Python` `FastAPI` `Pydantic` `google-generativeai` `MLflow`

</div>

---

### Contents

- [Pipeline](#pipeline)
- [Why this project, and why Gemini](#why-this-project-and-why-gemini)
- [What it actually does](#what-it-actually-does)
- [Seeing it actually run](#seeing-it-actually-run)
- [Switching backends](#switching-backends)
- [Testing philosophy](#testing-philosophy)
- [What's in the repo](#whats-in-the-repo)
- [Try it](#try-it)
- [Reliability details worth knowing](#reliability-details-worth-knowing)
- [What I'd improve with more time](#a-few-things-id-improve-with-more-time)

## Pipeline

```
Gemini API (google-generativeai) ─┐
Ollama / vLLM (OpenAI-compatible) ─┴─→ backend_selection.py  [picks one, based on config.yaml]
    → client.py / local_client.py   [build_model, call_*, stream_* — same shapes either way]
    → tools.py    [tool registry + dispatcher]
    → schemas.py  [Pydantic validation on everything the model returns]
    → tracking.py [token counts + cost logged to MLflow per request]
    → app.py      [FastAPI routes — thin, no business logic]
    → 27 unit tests covering retries, schema validation, tool dispatch, cost math, backend selection
```

## Why this project, and why Gemini

Most "LLM integration" demos are a single happy-path call to an API. The interesting engineering problems show up around that call: what happens when it times out, what happens when the model doesn't return valid JSON, how do you know what a feature is costing you, how do you stream a response when you can't "retry" something the client has already started reading. This project is built around those problems specifically. Gemini was the pick mainly for practical reasons (free tier, no billing setup), but everything here — retries, structured output validation, tool-call routing — is provider-agnostic in shape. That claim used to be untested; it's proven now — `local_client.py` adds Ollama and vLLM as real, switchable backends (see "Switching backends" below), and not one line in `tools.py`, `schemas.py`, or `tracking.py` had to change to make that work. Only `client.py` gained a sibling, exactly as this section originally predicted.

## What it actually does

**Structured output** (`/analyze`). Sentiment analysis returned as a Pydantic-validated `SentimentResult`, not raw model text. Gemini's `response_mime_type="application/json"` does most of the work; `strip_markdown_fences()` is a defensive fallback for when the model wraps its answer in ` ```json ` fences anyway. A failed validation returns a clean `502` instead of crashing.

**Streaming** (`/chat/stream`). Tokens streamed back as they're generated via `StreamingResponse`. This one deliberately has no retry logic, unlike the other two endpoints: once a chunk has reached the client, retrying from scratch would duplicate content they've already seen. A mid-stream failure (dropped connection, a safety-filtered chunk) is caught and the stream ends cleanly with a `[response interrupted]` marker instead of an unhandled exception killing the connection.

**Tool calling** (`/chat/tools`). The model decides whether it needs `get_current_weather(city)` or `search_documents(query)` to answer, returns that decision as JSON, and the server (not the model) actually executes the tool and feeds the result back for a grounded final answer. `search_documents` intentionally mirrors the retrieval interface from the semantic-search project, so swapping the in-memory keyword match for a real Qdrant or FAISS lookup would be a one-function change, not a redesign.

**Retries.** `call_gemini()` wraps every non-streaming call in exponential backoff (`config.yaml`: 3 attempts, 2s base). Rate limits and timeouts are treated as expected background noise for any external API, not edge cases worth special-casing per call site.

**Cost/usage tracking.** Every non-streaming call, and the final chunk of every stream, logs prompt and response token counts to MLflow via `log_usage()`, with real cost math (`compute_cost_usd()`, per-million-token rates from config) instead of a hardcoded `$0`. gemini-3.1-flash-lite's free tier means the rates default to `0.0`, but the calculation itself is correct and ready for whatever paid model gets swapped in later. Tracking failures are caught and logged rather than allowed to break the actual user-facing request.

## Seeing it actually run

`notebooks/01_demo.ipynb` spins up the FastAPI server and hits every endpoint for real. A few things worth pointing out from that run:

**`/analyze` on five test sentences** correctly separated clear positive/negative cases (confidence 1.00 both ways) from genuinely mixed ones: *"It was okay, nothing special but not bad either"* landed as neutral (0.95), and *"The acting was good but the plot made no sense whatsoever"* landed as neutral too, but at lower confidence (0.85), which fits the real ambiguity of a mixed review better than forcing it into positive or negative.

**`/chat/tools` on three prompts** is where the demo is most honest about reliability, not just the happy path:
- *"What is the weather like in Cairo right now?"* correctly routed to `get_current_weather` and answered using the tool's output.
- *"What is 2 + 2?"* correctly skipped the tools and answered directly.
- *"How does function calling work with language models?"* was supposed to route to `search_documents`, since it's a question the in-memory doc corpus can actually answer, but the model answered directly from its own knowledge instead and skipped the tool entirely. Routing decisions are the model's judgment call, not a deterministic function, and that's worth knowing going in rather than only demoing the cases that worked.

## Switching backends

`config.yaml`'s `backend.provider` picks one of three: `gemini` (default, API),
`ollama`, or `vllm` (both self-hosted, both speak the same OpenAI-compatible
protocol — that's why one `local_client.py` covers both instead of two
near-identical files).

**Ollama** — simplest to set up, right-sized for trying this out locally:
```bash
ollama serve
ollama pull llama3.1
# configs/config.yaml: backend.provider: "ollama"
```

**vLLM** — needs a GPU and the model in native HF/safetensors format (not
GGUF — vLLM runs its own PagedAttention scheduler over the original weights).
If you have a merged fine-tuned checkpoint from another project (e.g.
`customer_support_copilot`'s `notebooks/gguf_conversion.ipynb` produces one
as an intermediate step before quantizing further), point vLLM at that:
```bash
pip install vllm
vllm serve hossam3759180/support-copilot-merged --port 8001
# configs/config.yaml: backend.provider: "vllm"
```

**Comparing them** — restart the app between each `backend.provider` change,
then run the same benchmark against each:
```bash
python scripts/benchmark_backends.py --concurrency 1 5 10 20 --total-requests 40
```
Reports p50/p95 latency and throughput per concurrency level, plus which
backend/model actually answered (from `/health`). Gemini's token cost is
already tracked in MLflow via `tracking.py`; a self-hosted backend's cost is
whatever the serving hardware costs per hour, which this script doesn't
estimate — the honest comparison here is latency/throughput, not a dollar
figure.

## Testing philosophy

27 unit tests across `tests/`, deliberately scoped to what's actually deterministic:

| File | What it covers |
|---|---|
| `test_client.py` | retry succeeds first try / succeeds after retries / raises after max attempts exhausted |
| `test_client_json_parsing.py` | markdown-fence stripping (`json` fence, plain fence, already-clean JSON) |
| `test_local_client.py` | call_local/stream_local against a real local HTTP server speaking the OpenAI-compatible protocol (not a mocked `requests.post` — the actual JSON/SSE parsing is genuinely exercised), plus the retry contract matching call_gemini's |
| `test_backend_selection.py` | each `backend.provider` value picks the right functions/model/kwargs, Ollama and vLLM share the same functions, an unknown provider raises a clear error |
| `test_schemas.py` | valid `SentimentResult` parses, out-of-range confidence rejected, missing field rejected, valid tool call parses, null tool name is valid, missing arguments defaults to `{}` |
| `test_streaming.py` | `stream_gemini` yields raw chunks (so callers can read `usage_metadata`), not `.text` |
| `test_tools.py` | weather tool runs, document search finds a match, unknown tool name raises |
| `test_tracking.py` | zero rate gives zero cost, nonzero rate computes correctly, config values actually get applied to MLflow |

What's not tested: whether the model's actual answers are good. That's an evaluation problem, not a unit-test problem. There's no deterministic right answer to assert against for "is this a reasonable response to a prompt," so the tests stay focused on the parts of the system that do have one: does a retry actually retry, does a malformed schema actually get rejected, does an unknown tool name actually raise.

## What's in the repo

```
llm_api_integration/
├── notebooks/
│   ├── 00_setup.ipynb   # environment/repo setup
│   └── 01_demo.ipynb    # live server, hits every endpoint, MLflow dashboard, runs the test suite
├── src/
│   ├── config.py            # loads config.yaml into a dict
│   ├── client.py             # Gemini wrapper: build_model, call_gemini (retry), stream_gemini
│   ├── local_client.py       # Ollama/vLLM wrapper - same shapes as client.py, one HTTP protocol
│   ├── backend_selection.py  # picks gemini vs ollama vs vllm based on config.yaml
│   ├── tools.py      # tool definitions + TOOL_REGISTRY + dispatcher
│   ├── schemas.py    # Pydantic models: SentimentResult, ToolCallRequest
│   ├── tracking.py   # MLflow token/cost logging
│   └── app.py        # FastAPI routes — /analyze, /chat/stream, /chat/tools, /health
├── scripts/benchmark_backends.py  # p50/p95/throughput comparison across whichever backend is active
├── tests/            # 27 tests, see table above
├── configs/config.yaml   # model name, backend selection, temperature, retry policy, pricing, ports
└── requirements.txt
```

## Try it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
uvicorn src.app:app --reload --port 8000
```

`.env` is loaded automatically via `python-dotenv` before the app reads `GEMINI_API_KEY`, so there's no manual `export` needed.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely breathtaking — a masterpiece."}'
# → {"sentiment": "positive", "confidence": 1.0, "reasoning": "..."}

curl -X POST http://localhost:8000/chat/tools \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the weather like in Cairo right now?"}'
# → {"answer": "...", "tool_used": "get_current_weather", "tool_result": {"city": "Cairo", "temp_c": 24, "condition": "clear"}}

curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a vector database is in 3 sentences."}'
# → streamed plain text, chunk by chunk
```

Run the tests:

```bash
pytest tests/ -v
```

## Reliability details worth knowing

Two separate model instances are built at startup, one plain and one forced into `json_mode=True`, rather than a single instance with a mode flag flipped per call. This way a bug in call ordering can't accidentally return JSON for a prose request or vice versa.

Errors from the model come back as `502`s, not `500`s. Invalid JSON, an unknown tool name, wrong tool arguments: all of that is the model's output failing validation, not a bug in this service, so it's surfaced as a bad-upstream-response error rather than an unhandled server exception.

Usage is logged immediately after each call rather than batched at the end of a request. A call already happened and cost real tokens the moment it returns, so logging it before downstream parsing or tool execution can fail means a later error never silently drops that usage record.

Retry settings are passed explicitly on every one of the three LLM call sites instead of relying on a function default that happens to match `config.yaml`, so a future change to the retry policy in config can't silently stop applying somewhere.

## A few things I'd improve with more time

- Wire Gemini's native function-calling schema into the request itself instead of asking the model to emit a JSON routing decision as plain text: fewer moving parts, and it's exactly what the demo notebook shows can go wrong (the model answering directly instead of routing to `search_documents`)
- Add a lightweight eval set for the tool-routing decision specifically, since that's the one place in this system where "did it do the right thing" doesn't reduce to a schema check
- Swap `search_documents`' in-memory keyword match for the real Qdrant retrieval built in the semantic-search project, now that the interface already matches
- Add request-level rate limiting on the FastAPI side, not just retry-on-failure against Gemini's own limits
- Migrate off `google-generativeai` to `google-genai` — the old SDK now raises a `FutureWarning` on every import (`pip install` still works, it's not broken yet, but Google's own deprecation notice says it won't support new Gemini models much longer). Found this while adding the Ollama/vLLM backends; didn't fix it here since swapping SDKs is a separate, bigger change deserving its own careful pass, not something to bundle into a backend-switching change.
