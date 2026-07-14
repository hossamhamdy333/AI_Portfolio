# LLM API Integration

> FastAPI service wrapping the Gemini API — streaming, function calling,
> structured output, retries, and per-request token tracking.

---

## Why Gemini

Free tier, no billing setup required, and it supports everything this
project needs to demonstrate: streaming, function calling, and JSON-mode
structured output.

## What This Demonstrates

- **Streaming** — `/chat/stream` returns tokens as they're generated
- **Function/tool calling** — `/chat/tools` lets the model decide whether
  to call `get_current_weather` or `search_documents`, executes the tool
  itself, and returns a grounded final answer
- **Structured output** — `/analyze` returns sentiment validated against a
  Pydantic schema, not raw model text
- **Retry logic** — exponential backoff on transient API failures
- **Cost/usage tracking** — every call logs prompt/response token counts
  to MLflow

## Stack

FastAPI · google-generativeai (Gemini) · Pydantic · MLflow · pytest

## Structure

```
llm_api_integration/
├── configs/config.yaml     # model settings, retry policy — nothing hardcoded
├── src/
│   ├── config.py           # loads config.yaml
│   ├── client.py            # Gemini wrapper: build_model, call_gemini, stream_gemini
│   ├── tools.py              # 2 tool definitions + dispatcher
│   ├── schemas.py            # Pydantic models for structured output
│   ├── tracking.py           # MLflow token usage logging
│   └── app.py                 # FastAPI routes
├── tests/                     # tests what's actually testable: retries,
│                               # schema validation, tool dispatch — not
│                               # "is the LLM's answer good" (that's an eval, not a test)
├── requirements.txt
└── .env.example
```

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
uvicorn src.app:app --reload --port 8000
```

`.env` is loaded automatically via `python-dotenv` at app startup — no need
to `export` the key manually in your shell.

## Notes on Reliability

- **JSON mode:** `/analyze` and `/chat/tools` use Gemini's
  `response_mime_type="application/json"` so structured output is actually
  parseable, rather than relying on the model obeying a "respond only with
  JSON" instruction. `strip_markdown_fences()` is a defensive fallback in
  case it wraps the response in ` ```json ` fences anyway.
- **Cost tracking:** `log_usage()` computes real `cost_usd` from
  configurable per-million-token rates in `config.yaml` (both `0.0` by
  default since Gemini 2.0 Flash's free tier has no per-request charge) —
  not just token counts. Swap in real rates if you move to a paid model.
- **Error handling:** malformed model output (bad JSON, unknown tool name,
  wrong tool arguments) returns a `502` with a clear message, not an
  unhandled `500`.
- **Retry consistency:** all three LLM calls (`/analyze`, and both calls
  inside `/chat/tools`) use `config.yaml`'s retry settings explicitly —
  none rely on silently-matching function defaults.
- **Streaming has no retry, on purpose:** `/chat/stream` doesn't retry on
  failure like the other endpoints do. Once a chunk has been sent to the
  client, retrying from scratch would duplicate output they've already
  seen — there's no clean way to "retry" a partially-delivered stream.
  Instead, a mid-stream failure (network drop, a safety-filtered chunk)
  is caught, logged, and ends the stream with a short marker instead of
  crashing the connection with an unhandled exception.

## Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/analyze` | POST | `{"text": "..."}` → structured `SentimentResult` |
| `/chat/stream` | POST | `{"prompt": "..."}` → streamed plain-text response |
| `/chat/tools` | POST | `{"prompt": "..."}` → model picks a tool if needed, we execute it, return grounded answer |

## Running Tests

```bash
pytest tests/ -v
```

Tests cover retry behavior, schema validation, and tool dispatch — the
parts of this system that have a deterministic right answer. LLM output
quality itself is an evaluation problem, not something a unit test checks.

## Part of AI Engineering Portfolio

Reuses the retrieval pattern from `semantic_search/` (P3) inside the
`search_documents` tool. Precedes the RAG Chatbot (P6) and the Arabic AI
Agent (Mega Project) in the execution order.

→ [Full Portfolio](https://github.com/hossamhamdy333/AI_Portfolio)
