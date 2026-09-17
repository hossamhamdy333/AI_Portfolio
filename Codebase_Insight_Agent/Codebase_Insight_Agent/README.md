# Codebase Insight Agent

An agent that answers questions about my own [AI_Portfolio](https://github.com/hossamhamdy333/AI_Portfolio)
repo, grounded in each project's real README content, served as an **MCP server** so any
MCP client (Claude Desktop, Claude Code, or any other compliant client) can query it directly.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hossamhamdy333/AI_Portfolio/blob/main/Codebase_Insight_Agent/notebooks/01_indexing.ipynb)

## Structure — a real, sequential workflow, not five independent demos

Five notebooks, in order, each doing something the next one (or the live services)
actually depends on — not five separate demos of the same thing:

| Notebook | What it does | Depends on |
|---|---|---|
| `01_indexing.ipynb` | **Provisions the persistent index** — fetches every README, embeds it, writes it to Qdrant | Nothing — run this first |
| `02_router.ipynb` | Validates `config.PROJECT_DESCRIPTIONS`/`SIMILARITY_THRESHOLD` against real regression questions — a real pass/fail check, not a demo | Nothing — router doesn't need the index |
| `03_agent.ipynb` | Smoke-tests the actual agent against the persisted index — routing *and* answer correctness, LLM-judged against known facts | Notebook 1 |
| `04_evaluate.ipynb` | Measures what the correctness notebook 3 just confirmed actually costs vs. a naive baseline | Notebook 1 |
| `05_mcp_demo.ipynb` | Launches the real `mcp_server.py` and talks to it over genuine MCP protocol — the final end-to-end check | Notebook 1 |

**This isn't optional setup you can skip.** `mcp_server.py` and `web_app.py` both call
`portfolio.load_all_indexes()` at startup now, not `build_all_indexes()` — they load an
already-provisioned index, they don't build one themselves. Run notebook 1 with a real
`QDRANT_URL` before starting either service, or they'll fail immediately with a clear
error telling you to.

Two plain files hold the logic the notebooks share, so it's written once, not five times:

- `config.py` — project list, descriptions, model names, thresholds, `QDRANT_URL`/`QDRANT_API_KEY`
- `portfolio.py` — fetching, chunking, indexing, the router, the agent, and the
  persistence layer (`build_index`/`load_index`/`index_exists` — see "Provisioning the
  index" below)

`mcp_server.py` is the one piece that has to be a script rather than a notebook — an MCP
server is a long-running process talking over stdio, not something you run cell by cell.
It reuses `portfolio.py` directly.

## Provisioning the index (do this before anything else)

The index lives in **Qdrant Cloud** (free tier is enough), not locally and not
in-memory — that's what makes it actually persistent across notebook runs, Colab
sessions, and wherever `mcp_server.py`/`web_app.py` end up running, all pointing at the
same real data instead of each rebuilding their own throwaway copy.

1. Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) — takes a couple minutes
2. Copy the cluster URL and API key
3. Run `notebooks/01_indexing.ipynb`, paste both when it asks
4. Set the same two values as `QDRANT_URL`/`QDRANT_API_KEY` wherever `mcp_server.py` or
   `web_app.py` actually run (`.env` locally, or as container environment variables) — see
   `.env.example`

Skip steps 1-2 and leave `QDRANT_URL` blank for quick, throwaway local testing — the
notebook falls back to an in-memory Qdrant instead, but **nothing persists**: it's gone
the moment the notebook's kernel stops, and the live services elsewhere won't see it at
all. That's fine for kicking the tires, not for actually running this for real.

## Running it

**Easiest:** open the notebooks in Colab in order (badge above opens the first one),
starting with `01_indexing.ipynb`. Each one asks for `GOOGLE_API_KEY` (Gemini) and
`QDRANT_URL`/`QDRANT_API_KEY` via a prompt.

**Locally:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your-key-here
export QDRANT_URL=your-qdrant-cloud-url
export QDRANT_API_KEY=your-qdrant-api-key

# provision the index first (once, or after editing a README) - see the
# notebooks/ section above for why this can't be skipped
jupyter notebook notebooks/01_indexing.ipynb

# then, as an MCP server, e.g. for Claude Desktop:
python mcp_server.py
```

## Design notes

- **The router is embedding similarity, not an LLM call** — cheaper, faster, and one
  fewer place where output parsing could fail. Same choice already made in the
  `rag_router` project.
- **The critique → retry loop is what LangGraph earns its place over a single call** —
  the plan node decides single-project vs. multi-project, the critique node catches an
  ungrounded draft and forces a retry with feedback folded into the next query.
- **Each notebook rebuilds its own indexes.** A shared, persisted index store across
  notebooks would save time on repeated runs, but adds real complexity (Colab sessions
  don't share memory) for a demo that's meant to be readable start to finish.

## Remote deployment (connection token)

Running locally over stdio (the default, for Claude Desktop / Claude Code) needs no
token at all - whoever can launch this process on their own machine already has full
access, a token check there would protect nothing. That changes the moment this runs
as a real network service (Azure Container Apps, `MCP_TRANSPORT=streamable-http`):
anyone who finds the URL could connect and burn through the Gemini API quota with no
gate at all otherwise.

```bash
# Local (default) - no token, nothing to configure
python mcp_server.py

# Remote - gated by a shared secret
export MCP_TRANSPORT=streamable-http
export MCP_ACCESS_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export MCP_PUBLIC_URL=https://your-container-app-url.azurecontainerapps.io
python mcp_server.py
```

Any MCP client connecting to the remote URL needs to send `MCP_ACCESS_TOKEN`'s value as
a Bearer token. This is deliberately a single shared secret, not a full OAuth
client/token lifecycle - the problem being solved is "don't let a stranger who finds the
URL run up my API bill", not multi-tenant access control, so a shared secret is the
right-sized fix rather than standing up a real OAuth authorization server for one MCP
tool. See `tests/test_mcp_auth.py` for proof the gate actually rejects a wrong or
missing token, and that the default (no `MCP_ACCESS_TOKEN` set) stays exactly as simple
as local stdio use always was - no auth layer at all, not a verifier that happens to
always fail.

## Known gaps

- **Azure Monitor observability, dropped.** The earlier version wired up Application
  Insights tracing per MCP tool call. Removed here to keep `mcp_server.py` short; add it
  back (`azure-monitor-opentelemetry`, a `configure_azure_monitor()` call, a span around
  each `@mcp.tool()`) if this actually gets deployed somewhere long-running.
- **`04_evaluate.ipynb` still only measures cost, not a broader sense of answer
  quality.** Notebook 3 does now LLM-judge every regression question's answer against a
  known reference fact (a real correctness check, not just "did it route right"), but
  that's pass/fail on a fixed, small question set — a genuine quality *score* across a
  wider range of questions than 20 regression cases would still need real evaluation
  infrastructure this project doesn't have.
- **Deployment (Docker + Azure Container Apps) is built but not run.** `Dockerfile` and
  `.github/workflows/deploy-azure.yml` are ready; they need real Azure resources
  (Container Registry, Container App, resource group) and secrets configured first — see
  the setup comments at the bottom of that workflow file.
- **Three real bugs found and fixed while making the notebooks load-bearing, worth
  naming rather than quietly patching:**
  1. `Settings.embed_model`/`Settings.llm` (llama_index's global config) were never
     actually set anywhere — every real indexing/query call would have failed
     immediately with `ImportError: llama-index-embeddings-openai package not found`,
     since llama_index's un-configured default tries to resolve OpenAI's classes, not
     Gemini's. Fixed in `portfolio.py`'s `configure_llama_index()`.
  2. Rebuilding a project's index (`force=True`, e.g. after editing a README) used to
     silently **add** new points alongside the old ones instead of replacing them —
     verified directly (a 1-point collection became 2 points after a "refresh" with
     different content, not a clean swap). Fixed by dropping the collection before
     rebuilding; `tests/test_portfolio_persistence.py` checks this doesn't regress.
  3. Along the way, switched `llama-index-llms-gemini`/`llama-index-embeddings-gemini`
     to `llama-index-llms-google-genai`/`llama-index-embeddings-google-genai` — the
     former wraps the same deprecated `google.generativeai` SDK already flagged
     elsewhere in this portfolio (`llm_api_integration`'s README).

## The public website (`web_app.py`)

A second, separate front door onto the same `portfolio.py` logic `mcp_server.py`
uses — this one a plain public website, meant for a recruiter to click a link and
just start asking questions, no MCP client needed.

**No login for visitors, on purpose.** The whole point is zero friction — a login
wall on a link from a CV means most people just close the tab. What protects the
app instead:

- **Per-IP rate limiting** (`rate_limit.py`) — `RATE_LIMIT_MAX_REQUESTS` per
  `RATE_LIMIT_WINDOW_SECONDS` (10/hour by default), in-memory. Honest limitation
  worth stating plainly: this resets on a process restart and doesn't share state
  across multiple instances — fine for one Container App replica, would need Redis
  if this ever gets horizontally scaled.
- **Guardrails** (`guardrails.py`, the same module used in Azure RAG Assistant and
  the support copilot) — blocks prompt-injection attempts before they ever reach
  the agent, and moderates the output before it's returned. Benchmarked against
  the same kind of labeled adversarial set as Azure RAG Assistant: **19/20** on
  `tests/adversarial_prompts.json` (`tests/test_guardrails.py::test_adversarial_prompt_set_catch_rate`),
  with the one miss named in that test rather than hidden.

**One protected `/admin/*` route group** — just for you. Usage stats, recent
questions, what got blocked. There's deliberately no public `/auth/register` route
at all (`tests/test_web_app.py::test_no_public_registration_route_exists` checks
this stays true) — the only account is yours, created by:

```bash
python scripts/create_admin.py you@example.com
```
(prompts for a password rather than taking it as a CLI argument, so it doesn't end
up in your shell history)

### Running it locally

```bash
export GOOGLE_API_KEY=your-key-here
python scripts/create_admin.py you@example.com
uvicorn web_app:app --reload --port 8000
# open http://localhost:8000 - ask a question with no login,
# or click "Admin" in the corner to see the dashboard
```

### Deploying

This is a **separate Azure Container App** from the MCP server, built from
`Dockerfile.web` (not the plain `Dockerfile` the MCP server uses) even though both
live in this same folder and share `portfolio.py`/`config.py` — they're genuinely
different services with different jobs, so they get two separate deployments
instead of one container trying to do both. See
`.github/workflows/deploy-web-azure.yml`'s setup comments for the full walkthrough
— one new Container App, one new repo secret (`WEB_CONTAINER_APP_NAME`), and this
service's own `DATABASE_URL`/`JWT_SECRET_KEY` set on the container (not shared with
the MCP server's environment variables, which mean nothing here).

**Without a real `DATABASE_URL`** (Azure SQL), the query log and admin account live
in a local SQLite file that gets wiped on every redeploy — see the setup comments
for the Azure SQL connection string format, same as the other RBAC projects.

## If this lives inside another repo (e.g. as a subfolder of AI_Portfolio)

Everything here is self-contained and works identically regardless of what directory
contains this folder, except:

1. **`.github/workflows/deploy-azure.yml` AND `.github/workflows/deploy-web-azure.yml`
   must both move** to the surrounding repo's real top-level `.github/workflows/` —
   GitHub Actions only reads workflow files from the repo root, never from a
   subfolder. Both already have `appSourcePath:
   ${{ github.workspace }}/Codebase_Insight_Agent` baked in, so once moved they'll
   still build from this subfolder correctly.
2. **Claude Desktop's config path gets longer** but is otherwise unchanged — point it at
   wherever `mcp_server.py` ends up, e.g. `/path/to/AI_Portfolio/Codebase_Insight_Agent/venv/bin/python mcp_server.py`.
