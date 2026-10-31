<div align="center">

# LLM Cost Router

**Demo:** no hosted version. Run it locally with Docker, see [Run it](#run-it).

`FastAPI` `LiteLLM` `Gemini` `Prometheus` `Docker` `pytest`

</div>

---

### Contents

- [Summary](#summary)
- [Problem & motivation](#problem--motivation)
- [Approach](#approach)
- [Data](#data)
- [Results](#results)
- [Run it](#run-it)
- [What I'd do differently / limitations](#what-id-do-differently--limitations)
- [Stack](#stack)

---

## Summary

A FastAPI service that scores how hard each question looks, sends it to
a cheap or a strong Gemini model, logs the cost of every request and
exposes Prometheus metrics. On 50 test questions the router cost $0.44
instead of $0.55 for always using the strong model, a 19.5% saving. On
the 30 questions it sent to the cheap model, a judge rated the cheap
answer as good as or better than the strong one in 27 (90%). All 34
tests pass.

![Test run and evaluation output on my laptop](docs/eval-run.png)

*The test run and the evaluation tables, straight from the terminal.*

## Problem & motivation

The strong model costs 6 times more per token than the cheap one: $1.50
against $0.25 per million input tokens, and $9.00 against $1.50 per
million output tokens. For a question like "What is the capital of
Japan?" the strong model is wasted money. For "Design a rate limiter and
explain the trade-offs" the cheap model may not be enough.

So the service needs two things. A way to tell the two kinds of question
apart before calling a model, and a way to check that the cheap answers
are not worse. The second one is why the repo has a full evaluation and
not only a router.

## Approach

The router gives each question a difficulty score from simple rules:

- a hard keyword such as compare, design, debug, explain or implement
  adds 3
- code adds 2 (a code fence, or `def`, `class`, `import`, `select`,
  `traceback`)
- a question over 60 words adds 2, over 25 words adds 1
- more than one question mark adds 1
- a question of 12 words or fewer with an easy keyword (what is, define,
  translate) subtracts 1

A score of 3 or more goes to the strong model. Anything lower goes to
the cheap one. A request can force a model with `force_tier`.

Models, called through LiteLLM:

- cheap: `gemini-3.1-flash-lite`
- strong: `gemini-3.5-flash`

Endpoints:

- `POST /ask` returns the answer, the model used, the score and reason,
  token counts, `cost_usd`, what the same tokens would have cost on the
  strong model, and the latency
- `GET /health`
- `GET /metrics` for Prometheus

Cost is the token counts from the API response times the list price.
Each request also writes one log line with the tier, tokens and cost.

| Metric | Meaning |
|---|---|
| `router_requests_total{tier}` | requests per tier |
| `router_errors_total{tier}` | failed model calls |
| `router_cost_usd_total{tier}` | cost at list price |
| `router_saved_usd_total` | estimated saving against always using the strong model |
| `router_tokens_total{tier,kind}` | input and output tokens |
| `router_latency_seconds{tier}` | model call latency |

## Data

50 questions in `data/questions.json`, made by `data/make_questions.py`.
30 are labelled easy ("Who wrote Hamlet?", "Convert 5 kilometers to
miles.") and 20 are labelled hard (compare two index types, debug a
Python function, design a rate limiter). The labels are mine.

Costs use Gemini list prices from October 2026. On the free tier the
real cost was $0, so the numbers are what the same usage would cost on a
paid plan. Hidden thinking tokens are billed as output and are included.

## Results

I ran every question on both models and compared three strategies.

| Strategy | Total cost (USD) | Saving vs always-strong | Avg score (1-10) | Avg answer length (words) |
|---|---|---|---|---|
| Always strong | 0.5456 | 0.0% | 10.00 | 311 |
| Always cheap | 0.0293 | 94.6% | 9.96 | 237 |
| Router | 0.4392 | 19.5% | 10.00 | 310 |

The 1-10 score is not useful here, because the judge gave almost every
answer a 10. The head-to-head comparison says more. A judge sees the
cheap and the strong answer in random order and picks the better one, or
a tie.

| Router sent the question to | Questions | Cheap better | Tie | Strong better |
|---|---|---|---|---|
| Cheap model | 30 | 2 | 25 | 3 |
| Strong model | 20 | 2 | 11 | 7 |

The router sent 20 questions to the strong model, but the strong answer
was clearly better on only 7 of them. So the router is cautious, and a
smarter one could save more. I did not train a classifier, because 50
labelled questions are too few to train and test on honestly.

To repeat the run: `python -m evaluation.run_eval`. Results are cached
in `evaluation/results.jsonl` and `evaluation/pairwise.jsonl`, and the
tables are written to `evaluation/summary.md`. The free tier allows only
20 strong-model requests a day, so the first run took several days. A
repeat on cached results makes no API calls.

## Run it

You need a free Gemini API key from Google AI Studio. Copy `.env.example` to `.env` and put your key in it.

```bash
docker build -t llm-cost-router .
docker run --rm -p 8000:8000 --env-file .env llm-cost-router
```

Without Docker:

```bash
pip install -r requirements.txt
set -a; source .env; set +a
uvicorn app.main:app --port 8000
```

Ask a question:

```bash
curl -s localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of Japan?"}'
```

```json
{"answer":"The capital of Japan is Tokyo.","tier":"cheap","model":"gemini/gemini-3.1-flash-lite","score":-1,"reason":"easy keyword","prompt_tokens":8,"completion_tokens":7,"cost_usd":1.25e-05,"strong_cost_usd":7.5e-05,"latency_ms":2768}
```

Run the tests (they mock the model calls, so no key is needed):

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The 34 tests cover the router rules, pricing, the API, the metrics and
the evaluation logic.

## What I'd do differently / limitations

- **The savings depend on the question mix.** My set is 40% hard. With
  mostly easy traffic the saving would be larger.
- **I tuned the router rules on the same 50 questions.** The first
  version matched my easy/hard labels on 37 of 50, and one change
  brought it to 50 of 50. Treat 100% as optimistic. A fresh set of
  questions would show how the rules really do.
- **The judge is a Gemini Flash-Lite model** (`gemini-3.5-flash-lite`),
  so it may favour Gemini answers. My first head-to-head prompt asked
  for completeness and favoured the longer strong answers (strong better
  on 17 of the first 27 easy questions). I changed it to ignore length
  and reran everything. The final prompt is in `evaluation/run_eval.py`.
- **I do not report latency.** My first measurement included rate-limit
  retry waits, so the numbers were wrong.
- **Both models are in the same Flash family.** The quality gap is
  smaller than it would be between Flash and Pro, which my free key
  could not use.
- **There is no fallback when the cheap model answers badly.**

## Stack

- `FastAPI` and `uvicorn` for the service
- `LiteLLM` to call the Gemini models
- `Gemini` (`gemini-3.1-flash-lite`, `gemini-3.5-flash`) for answers,
  and `gemini-3.5-flash-lite` as the judge
- `prometheus-client` for the metrics
- `Docker` for packaging
- `pytest` and `httpx`, 34 tests with the model calls mocked
