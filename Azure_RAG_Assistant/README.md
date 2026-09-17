<div align="center">

# Azure RAG Assistant

A document Q&A chatbot: upload a file, ask questions about it in plain language, and get answers grounded in what's actually in the document — not a chatbot demo that happens to have a file upload button next to it.

**Live Application:** [rag-assistant-hossam...azurewebsites.net](https://rag-assistant-hossam-dhfjgrfwcaf6g9e6.francecentral-01.azurewebsites.net)

`Python` `FastAPI` `LangChain` `Gemini API` `Qdrant` `Azure SQL` `Azure Blob Storage` `Docker` `GitHub Actions`

</div>

---

### Contents

- [What it does](#what-it-does)
- [What this demonstrates](#what-this-demonstrates)
- [Stack](#stack)
- [Project structure](#project-structure)
- [Running it](#running-it)
- [What's genuinely not done](#whats-genuinely-not-done)

## What it does

Upload a PDF, image, or text file. It gets chunked, embedded, and indexed. Ask a question and the agent retrieves the relevant chunks before answering, so the answer is traceable back to a real part of your document instead of the model just guessing from memory. It can also do plain arithmetic without hallucinating the answer. Every user has a real account (email+password or Google), and one person's documents are never visible to another — enforced at the database query itself, not just hidden in the UI.

## What this demonstrates

- **RAG done for real** — retrieval happens before generation, and the per-user filter on that retrieval is checked by Qdrant itself, not something the frontend happens to respect (see `tests/test_agent_isolation.py`, which proves it can't be bypassed by asking about "all documents")
- **Guardrails with a measured catch rate, not an assumed one** — PII redaction and prompt-injection detection benchmarked against a real adversarial test set: 19/20, with the one miss documented rather than hidden (`scripts/adversarial_report.py`)
- **Real auth** — JWT access + refresh tokens, bcrypt, two roles, rate limiting on login/register/chat/upload that's backed by the database instead of an in-memory counter that would just reset itself on every deploy
- **Deployment engineering, including the parts that went wrong** — this ran into a genuine SQL Server gotcha in production that the test suite (SQLite) couldn't have caught: a unique index on a nullable column treats every `NULL` as a duplicate on SQL Server, so the second person to ever register crashed the whole request. Fixed with a filtered index, documented in the code, not swept under the rug.
- **Cost-aware defaults** — connection pooling tuned for Azure SQL's serverless auto-pause (a stale pooled connection from before a pause will otherwise hang a request instead of failing cleanly), separate `/health` and `/health/db` endpoints so a slow database can't false-positive Azure's own liveness probe into restart-looping the app

## Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI |
| LLM | Google Gemini via LangChain |
| Vector search | Qdrant |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Database | Azure SQL (production) / SQLite (local dev) via SQLAlchemy |
| File storage | Azure Blob Storage |
| Hosting | Azure App Service (Docker container) |
| CI/CD | GitHub Actions → Docker Hub → Azure |

The frontend is one HTML/JS page served by the backend directly — no separate frontend service.

## Project structure

```
Azure_RAG_Assistant/
├── backend/
│   ├── main.py              # API routes
│   ├── agent.py             # LangChain agent + retrieval tool
│   ├── text_processing.py   # chunking, embedding, deletion
│   ├── auth.py               # JWT, password hashing, refresh tokens
│   ├── oauth.py               # Google login
│   ├── database.py            # SQLAlchemy engine/session
│   ├── models.py               # User, Document, RequestLog
│   ├── rate_limit.py            # per-user / per-IP limits, DB-backed
│   ├── guardrails.py             # prompt-injection + PII checks
│   ├── azure_storage.py           # blob upload
│   ├── safe_math.py                # AST-based calculator, not eval()
│   ├── static/index.html            # frontend
│   └── tests/
├── scripts/adversarial_report.py     # honest guardrails catch-rate report
└── docker-compose.yml                 # local dev only
```

## Running it

```bash
cp .env.example .env   # Gemini, Hugging Face, and Qdrant keys
docker-compose up --build
```

Open http://localhost:8000. Local storage uses Azurite as a stand-in for Azure Blob Storage, and SQLite needs zero setup.

```bash
cd backend
pytest -v
```

## What's genuinely not done

- No email verification or password reset flow
- No sharing a document between users — everything is fully private, no sharing feature at all
- Rate limits cap request counts (e.g. 30 chats/hour), not actual token spend — a user could still send unusually long messages within that count and cost more than a typical one
- `/admin/stats` reports account and document counts, not real per-request API cost — that needs the LLM call to report token usage back, which isn't wired in
