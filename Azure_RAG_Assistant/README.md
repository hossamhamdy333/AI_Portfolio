<div align="center">

# Azure RAG Assistant

A Retrieval-Augmented Generation chatbot. Upload documents, ask questions about them in plain language, and get answers grounded in the actual content — built with Gemini, Qdrant, and deployed on Azure.

**Live demo:** [azure-rag-assistant...azurewebsites.net](https://azure-rag-assistant-b6hqawe7eef6euaf.francecentral-01.azurewebsites.net)

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
- Built-in calculator for arithmetic queries
- Uploaded files are archived to Azure Blob Storage

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
│   ├── text_processing.py   # document parsing and chunking
│   ├── azure_storage.py     # blob storage upload
│   ├── safe_math.py         # calculator tool
│   ├── config.py            # settings
│   ├── static/index.html    # frontend
│   ├── tests/
│   └── Dockerfile
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

Deploys automatically to Azure App Service on every push to `main` via GitHub Actions. Environment variables required in the App Service:

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

1. In the Azure Portal: **Create a resource → Azure SQL Database**. Pick the
   serverless compute tier for a demo (scales to near-zero cost when idle).
2. Under the server's **Networking** settings, allow Azure services to access
   the server (needed for App Service to reach it), and add your own IP for
   running migrations locally.
3. Set `DATABASE_URL` to:
   ```
   mssql+pyodbc://<user>:<password>@<server>.database.windows.net:1433/<db>?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes
   ```
4. Install the driver: `pip install pyodbc` **and** the Microsoft ODBC Driver
   18 itself (`pip install` alone does not install this - see
   [Microsoft's install docs](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)
   for your OS/container base image; the existing `Dockerfile` here is
   Debian-based, so it needs the `msodbcsql18` apt package added).
5. Uncomment `pyodbc` in `requirements.txt`.

For local dev, skip all of this - the default `DATABASE_URL` is
`sqlite:///./dev.db`, which needs zero setup and is what the test suite uses.

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

## Notes

- The calculator uses a restricted AST-based evaluator instead of `eval()`.
- If blob storage is unavailable, uploads still succeed and remain searchable — only the raw file backup is skipped.
- Guardrails (`guardrails.py`) redact PII and block a heuristic set of prompt-injection patterns before a query reaches the LLM - see `scripts/adversarial_report.py` for the honest catch-rate against a real adversarial test set (19/20, with the one miss listed).
- Observability (Arize Phoenix) is off by default - set `ENABLE_OBSERVABILITY=true` and see `scripts/regression_demo.py` for proof it actually catches a quality regression, not just happy-path traffic.
