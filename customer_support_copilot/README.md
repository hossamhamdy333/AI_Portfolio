<div align="center">

# AI Support Copilot

Your assistant for orders, refunds, accounts & shipping — a fine-tuned Llama-3 (QLoRA) model, grounded with RAG retrieval, behind real accounts and per-user chat history. Deployed on Azure Container Apps, running entirely on CPU.

**Live Application:** [support-copilot-app...azurecontainerapps.io](https://support-copilot-app.blackpebble-352cd42a.francecentral.azurecontainerapps.io)

`Python` `FastAPI` `llama.cpp` `QLoRA` `SQLAlchemy` `Azure SQL` `JWT Auth` `Docker` `GitHub Actions`

</div>

---

### Contents

- [Features](#features)
- [Stack](#stack)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
- [Tests](#tests)
- [Deployment](#deployment)
- [Why CPU instead of GPU](#why-cpu-instead-of-gpu)
- [What's genuinely NOT done here](#whats-genuinely-not-done-here)
- [Notes](#notes)

## Features

- Fine-tuned Llama-3 (8B, QLoRA) on the Bitext customer support dataset, retrieval-grounded against a knowledge base built from the same data
- Real accounts (email + password), JWT access + refresh tokens, two roles (`user`, `admin`)
- Claude-style multi-chat: each user can hold multiple named conversations, switch between them, delete them — not one endless thread per account
- Per-user chat history persisted to a real database, isolated at the query layer (a user can never see another user's conversations)
- Admin endpoints to list users, view any user's transcript, ban an account, and see aggregate stats — all gated by role, not just hidden in the UI
- Guardrails: PII redaction and prompt-injection detection on every message before it reaches the model
- Shared `GEMINI_API_KEY` for the optional faithfulness-check feature, metered against a daily cap so one demo project can't run up an open-ended bill
- CPU-only inference via a quantized GGUF model through `llama.cpp` — not the original `transformers` path, which was too slow to finish inside Azure's request timeout on this hardware tier

## Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI (Python) |
| LLM | Self-hosted, fine-tuned Llama-3-8B (QLoRA), quantized to GGUF, served via `llama-cpp-python` (an optional vLLM backend also exists — see `src/llm_backend.py`) |
| Retrieval | Chroma + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Database | Azure SQL (production) / SQLite (local dev) via SQLAlchemy |
| Auth | JWT access + refresh tokens, bcrypt password hashing |
| Hosting | Azure Container Apps, deployed as a Docker container |
| CI/CD | GitHub Actions → Azure Container Registry → Azure Container Apps |

The frontend is a single HTML/JS page served directly by the backend — no separate frontend service.

## Project structure

```
customer_support_copilot/
├── src/
│   ├── app.py                # API routes
│   ├── llm_backend.py         # llama.cpp (default) or vLLM generation backend
│   ├── retriever.py           # KB embedding + retrieval
│   ├── evaluate.py            # Gemini faithfulness check, rate-limited
│   ├── auth.py                # password hashing, JWT, refresh tokens
│   ├── oauth.py                # Google OAuth backend (not wired into the UI - see below)
│   ├── database.py            # SQLAlchemy engine/session
│   ├── models.py               # User, Conversation, ChatMessage, RefreshToken, GeminiUsage
│   ├── guardrails.py           # prompt-injection and PII checks
│   ├── config.py               # settings
│   ├── prepare_data.py         # builds training data + knowledge base
│   └── adversarial_prompts.json
├── scripts/
│   └── adversarial_report.py  # honest guardrails catch-rate report
├── notebooks/
│   ├── training.ipynb          # QLoRA fine-tune on Colab, pushes adapter to HF Hub
│   ├── gguf_conversion.ipynb   # converts + quantizes the adapter for fast CPU inference
│   └── eda.ipynb
├── frontend/index.html         # chat UI, sidebar, auth
├── tests/
├── data/kb_articles.jsonl
├── Dockerfile
└── .env.example
```

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py     # builds the knowledge base, run once
cp .env.example .env

uvicorn src.app:app --reload --port 8000
```

Open http://localhost:8000. Local dev uses `sqlite:///./dev.db` by default — zero setup.

## Tests

```bash
pytest -v
```

Covers auth (register/login/refresh/logout), guardrails (PII redaction, injection detection), OAuth backend logic, and — the one that actually matters most — that one user's chat history is never returned to a different logged-in user.

## Deployment

Deploys to Azure Container Apps via a GitHub Actions workflow (`.github/workflows/build-push-acr.yml`) that builds the image and pushes it to Azure Container Registry directly from GitHub's runners, rather than pushing from a local machine — this was the fix for repeatedly timing out on a large image upload over a slow home connection.

Environment variables required on the Container App:

```
DATABASE_URL
JWT_SECRET_KEY
GGUF_REPO            # default: hossam3759180/support-copilot-gguf
GGUF_FILENAME        # default: support-copilot-q4.gguf
```

Optional — only needed for the faithfulness-check feature:

```
GEMINI_API_KEY
ENABLE_EVAL          # default false
GEMINI_DAILY_LIMIT   # default 50
```

### Setting up Azure SQL (production)

**Data persistence matters here:** `DATABASE_URL` defaults to `sqlite:///./dev.db`, which lives inside the container's own filesystem — fine for local dev, but on Azure Container Apps that file resets to empty on every restart or scale-to-zero event, which happens routinely on the free tier. Every registered account and every conversation would be wiped without warning. Set up Azure SQL before relying on this for anything real.

1. Azure Portal → **Create a resource → Azure SQL Database**. The free serverless offer (100,000 vCore-seconds/month) covers this comfortably.
2. Under the server's **Networking** settings, allow Azure services to access the server (needed for Container Apps to reach it), and allowlist your own IP if you want to query it directly from the portal's Query Editor.
3. If the server was created with Entra-only authentication (a default on the free-offer flow), enable SQL authentication and set an admin password — via the portal's Microsoft Entra ID settings page, or directly:
   ```bash
   az sql server ad-only-auth disable --name <server> --resource-group <rg>
   az sql server update --name <server> --resource-group <rg> --admin-password "<strong password>"
   ```
4. Set `DATABASE_URL`:
   ```
   mssql+pyodbc://<user>:<password>@<server>.database.windows.net/<db>?driver=ODBC+Driver+18+for+SQL+Server
   ```
   If the password contains `@`, `%`, or other URL-reserved characters, either URL-encode them or just pick a password without them — simpler and avoids parsing errors.
5. The Dockerfile installs the Microsoft ODBC Driver 18 that `pyodbc` needs at the system level — `pip install` alone isn't enough for this driver, unlike `pymssql` on some of the other projects in this portfolio.

New tables added after the app was already deployed once (e.g. `conversations`, the `conversation_id` column on `chat_messages`) don't get created automatically by a redeploy — SQLAlchemy's `create_all()` only creates tables that don't exist yet, it never alters an existing one. Run the missing `CREATE TABLE` / `ALTER TABLE` once by hand via the portal's Query Editor when that happens.

### Generating a JWT secret

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Never reuse the default dev value in production — anyone with it can forge a valid token for any user, including an admin.

### Setting up vLLM (optional, needs a GPU)

`LLM_BACKEND=llamacpp` (the default) is what's actually deployed, chosen because no GPU workload profile is available on this Azure subscription/region. If that changes, `src/llm_backend.py` also supports `LLM_BACKEND=vllm` — an HTTP client to a separately-run vLLM server, which is the right tool once concurrent traffic matters (continuous batching, PagedAttention) rather than the one-request-at-a-time model `llama.cpp` serves. vLLM needs the original merged checkpoint, not the GGUF file — see `notebooks/gguf_conversion.ipynb`'s merge step for how to produce and push that.

## Why CPU instead of GPU

Azure for Students doesn't include a GPU workload profile in this account's available regions. Rather than block the whole project on that, the model was converted from raw `transformers` inference (too slow on CPU — an 8B model in float32 timed out on every request, past Azure Container Apps' ~4-minute request limit) to a quantized GGUF file run through `llama.cpp`, which is built for fast CPU inference. That change — plus pinning the thread count to the container's actual CPU allocation instead of `os.cpu_count()`'s host-wide count, which was oversubscribing threads — is what got responses answering in roughly 15-20 seconds instead of timing out.

## What's genuinely NOT done here

- **Google Sign-In exists in the backend (`src/oauth.py`, tested) but isn't exposed in the UI.** Getting it fully public requires Google's OAuth consent screen to be published, which in turn requires a privacy policy and terms-of-service page hosted on a domain you actually own — Azure's shared `azurecontainerapps.io` subdomain doesn't qualify as an authorized domain for that. Re-enabling it is a one-line frontend change once that's in place.
- No email verification — accounts work immediately on registration.
- No password reset flow — out of scope for a demo, would need an email provider wired in.
- The Gemini daily limit meters *call count*, not actual token spend — a user could still send unusually long messages within that count.
- Quantizing to GGUF (Q4_K_M) trades a small amount of answer quality for the CPU speed that makes this deployment usable at all — the original QLoRA adapter (before quantization) is the higher-fidelity version, still on Hugging Face Hub.
- Conversations created before the multi-chat feature was added aren't visible in the sidebar (their messages have no `conversation_id`) — still queryable by an admin via the transcript endpoint, just not surfaced in the normal chat UI.

## Notes

- Guardrails (`guardrails.py`) redact PII and block a heuristic set of prompt-injection patterns before a query reaches the model — see `scripts/adversarial_report.py` for the honest catch-rate against a real adversarial test set.
- `RefreshToken` rows are stored hashed, not raw, so logout can actually revoke a session — a JWT access token by itself can't be revoked once issued, which is why the access token is short-lived (20 minutes) and the refresh token is what's actually held onto.
- The knowledge base (`data/kb_articles.jsonl`) is deduplicated by intent from the same Bitext dataset used for fine-tuning, so the retriever and the model were trained on the same underlying domain rather than two unrelated sources.
