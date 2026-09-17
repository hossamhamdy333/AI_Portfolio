<div align="center">

# Azure RAG Assistant

A Retrieval-Augmented Generation chatbot. Upload documents, ask questions about them in plain language, and get answers grounded in the actual content — built with Gemini, Qdrant, and deployed on Azure.

**Live demo:** [rag-assistant-hossam...azurewebsites.net](https://rag-assistant-hossam-dhfjgrfwcaf6g9e6.francecentral-01.azurewebsites.net)

`Python` `FastAPI` `LangChain` `Gemini API` `Qdrant` `Azure Blob Storage` `Docker` `GitHub Actions`

</div>

---

### Contents

- [Features](#features)
- [Stack](#stack)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
- [Tests](#tests)
- [Deployment](#deployment)
- [Notes](#notes)

## Features

- Upload PDFs, images, or text files
- Documents are chunked, embedded, and indexed for semantic search
- Chat interface that retrieves relevant context before answering
- Delete a previously uploaded document (removes it from both the database and the vector index)
- Built-in calculator for arithmetic queries
- Uploaded files are archived to Azure Blob Storage
- Real accounts (email+password or "Continue with Google"), JWT auth, and per-user document isolation enforced at the retrieval layer
- Per-user and per-IP rate limiting on chat, upload, login, and registration

## Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI (Python) |
| LLM | Google Gemini (`gemini-3.1-flash-lite`) via LangChain |
| Vector search | Qdrant |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Storage | Azure Blob Storage |
| Hosting | Azure App Service, deployed as a Docker container |
| CI/CD | GitHub Actions → Docker Hub → Azure |

The frontend is a single HTML/JS page served directly by the backend — no separate frontend service.

## Project structure

```
Azure_RAG_Assistant/
├── backend/
│   ├── main.py              # API routes
│   ├── agent.py             # LangChain agent + tools
│   ├── text_processing.py   # document parsing, chunking, deletion
│   ├── azure_storage.py     # blob storage upload
│   ├── safe_math.py         # calculator tool
│   ├── auth.py              # password hashing, JWT, refresh tokens
│   ├── oauth.py             # Google OAuth login
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models.py            # User, Document, RefreshToken, RequestLog
│   ├── rate_limit.py        # per-user / per-IP rate limiting
│   ├── guardrails.py        # prompt-injection and PII checks
│   ├── observability.py     # optional tracing hook
│   ├── config.py            # settings
│   ├── static/index.html    # frontend
│   ├── tests/
│   └── Dockerfile
├── scripts/                  # load testing, adversarial testing, demos
├── docker-compose.yml        # local dev only
└── .env.example
```

## Running locally

```bash
cp .env.example .env   # add your Gemini, Hugging Face, and Qdrant keys
docker-compose up --build
```

Open http://localhost:8000. Local storage uses Azurite as a stand-in for Azure Blob Storage.

## Tests

```bash
cd backend
pip install -r requirements.txt
pytest -v
```

## Deployment

Deploys automatically to Azure App Service on every push to `main` via GitHub Actions.

**App Service plan tier matters for anything beyond a personal demo.** The
Free F1 tier caps the whole app at 60 CPU-minutes/day, shared across every
user - once that's used up, Azure stops serving requests until the next
day, regardless of how correct the code is. Move to a paid tier (B1 or
higher) before sharing this with real users; F1 is fine for testing and
your own use, not for anything meant to stay reliably up.

Environment variables required in the App Service:

```
GEMINI_API_KEY
HF_TOKEN
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION_NAME
AZURE_STORAGE_CONNECTION_STRING
AZURE_STORAGE_CONTAINER_NAME
DATABASE_URL
JWT_SECRET_KEY
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI
FRONTEND_URL
WEBSITES_PORT=8000
```

Optional - all have sensible defaults, only set these to override them:

```
RATE_LIMIT_ENABLED       # default true
CHAT_RATE_LIMIT_PER_HOUR     # default 30
UPLOAD_RATE_LIMIT_PER_HOUR   # default 10
LOGIN_RATE_LIMIT_PER_15MIN   # default 10
REGISTER_RATE_LIMIT_PER_HOUR # default 5
```

GitHub repository secrets required for the workflow:

```
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
AZURE_WEBAPP_PUBLISH_PROFILE
```

## Phase 3: Authentication, authorization, and a real database

Real accounts (email+password or "Continue with Google"), JWT access + refresh
tokens, two roles (`user`, `admin`), and per-user document isolation enforced at
the retrieval layer - not just hidden in the UI. Backed by a real relational
database instead of no persistence at all.

### Setting up Azure SQL (production)

**Data persistence matters here:** `DATABASE_URL` defaults to
`sqlite:///./dev.db`, which lives inside the container's own filesystem -
fine for local dev, but on Azure App Service that file resets to empty on
every redeploy (a fresh container image = a fresh, empty filesystem).
Every registered account and document record would be wiped on every push
to `main`. Set up Azure SQL before relying on this for anything real.

1. In the Azure Portal: **Create a resource → Azure SQL Database**. Pick
   the serverless compute tier for a demo (scales to near-zero cost when
   idle) - there's also a genuine free tier (one per subscription) if
   you're on Azure for Students.
2. Under the server's **Networking** settings, allow Azure services to
   access the server (needed for App Service to reach it).
3. Set `DATABASE_URL` to:
   ```
   mssql+pymssql://<user>:<password>@<server>.database.windows.net:1433/<db>
   ```
4. That's it - `pymssql` (already in `requirements.txt`) ships as a
   self-contained wheel with its native dependencies bundled in, so unlike
   `pyodbc` it needs no separate driver installed in the container or any
   Dockerfile changes.

For local dev, skip all of this - the default `sqlite:///./dev.db` needs
zero setup and is what the test suite uses.

### Setting up Google OAuth

1. [Google Cloud Console](https://console.cloud.google.com) → APIs & Services
   → Credentials → **Create Credentials → OAuth client ID** → Web application.
2. Authorized redirect URI: `http://localhost:8000/auth/google/callback` for
   local dev, or your real domain's equivalent for production.
3. Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`,
   `FRONTEND_URL` accordingly. Leave `GOOGLE_CLIENT_ID` blank to disable the
   button entirely - email+password login works either way.

### Generating a JWT secret

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Put the output in `JWT_SECRET_KEY`. Never reuse the default dev value in
production - anyone with this value can forge a valid token for any user,
including an admin.

### Setting up Ollama (LLM_BACKEND=local)

```bash
# install: https://ollama.com/download
ollama serve                # runs on localhost:11434
ollama pull llama3.1        # or any model you prefer
```

Then set `LLM_BACKEND=local` (Gemini stays the default otherwise -
`LOCAL_LLM_BASE_URL`/`LOCAL_LLM_MODEL` already point at Ollama's own
OpenAI-compatible endpoint by default).

### Promoting a user to admin

There's no API endpoint for this on purpose - "become admin" as a callable
route would be a privilege-escalation bug, not a feature. Do it directly:

```bash
python -c "
from database import SessionLocal
from models import User, Role
db = SessionLocal()
user = db.query(User).filter(User.email == 'you@example.com').first()
user.role = Role.admin
db.commit()
"
```

The user needs to log in again afterward - their current access token still
has the old role baked in until it's reissued (tokens expire every 20
minutes by default, so this resolves itself quickly either way).

### What's genuinely NOT done here

- No email verification - accounts work immediately on registration.
- No password reset flow - out of scope for a demo, would need an email
  provider wired in.
- No per-document sharing between users (Google-Docs style) - each user's
  documents are fully private to them, with no sharing feature at all.
- `/admin/stats` reports account/document counts, not real per-request API
  cost - that needs the LLM call itself to report token usage back, which
  isn't wired in yet.
- Rate limiting caps *request counts* (e.g. 30 chats/hour), not actual
  Gemini token spend - a user could still send unusually long messages
  within that count and cost more than a typical one. A real per-account
  dollar cap would need the LLM call to report token usage back per
  request, same gap as the `/admin/stats` point above.

## Notes

- `/health` is a pure liveness check (no DB call) so Azure's own health probe never restart-loops the app over a slow database. `/health/db` actually runs a query - hit this one directly when debugging login/register issues to immediately tell apart a DB problem from anything else.
- The connection pool uses `pool_pre_ping` and `pool_recycle` - without these, a connection that was opened before Azure SQL's serverless tier auto-paused sits in the pool looking fine and then fails or hangs the moment it's reused.
- **SQL Server gotcha that cost real debugging time:** a plain unique index on a nullable column (`google_id`, null for every non-Google account) works fine on SQLite/Postgres but not SQL Server - it treats every `NULL` as equal to every other `NULL`, so the second person to register with email+password ever collides with the first and gets a raw 500. Fixed with a filtered index (`mssql_where=google_id IS NOT NULL`) in `models.py`, scoped to the mssql dialect only. Worth knowing if you're moving anything from SQLite to Azure SQL - the test suite (SQLite) can't catch this class of bug at all, it only shows up against the real database.
- Qdrant needs an explicit payload index on `metadata.user_id` before it can filter by it - `agent.py` creates this automatically the moment it creates a fresh collection, so it only bites you if a collection was created before that line existed (fix: create the index once by hand, or drop and let it recreate).
- The calculator uses a restricted AST-based evaluator instead of `eval()`.
- If blob storage is unavailable, uploads still succeed and remain searchable — only the raw file backup is skipped.
- Deleting a document removes its vector chunks from Qdrant (filtered by a document id tagged onto each chunk, not by filename, so two uploads that happen to share a name can't collide) and its row from the database.
- Rate limiting (`rate_limit.py`) is database-backed rather than in-memory, so limits are enforced correctly even if this ever runs as more than one instance. Chat and upload are limited per-account; login and registration are limited per-IP, since there's no logged-in user yet at that point. Disable entirely with `RATE_LIMIT_ENABLED=false` (used by the test suite, since it legitimately calls these endpoints many times in a row).
- Guardrails (`guardrails.py`) redact PII and block a heuristic set of prompt-injection patterns before a query reaches the LLM - see `scripts/adversarial_report.py` for the honest catch-rate against a real adversarial test set (19/20, with the one miss listed).
- Observability (Arize Phoenix) is off by default - set `ENABLE_OBSERVABILITY=true` and see `scripts/regression_demo.py` for proof it actually catches a quality regression, not just happy-path traffic.
