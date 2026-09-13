<div align="center">

# AI Support Copilot

A production-shaped customer support chatbot: a self fine-tuned Llama-3 model, grounded with RAG so it answers from real policy instead of guessing, sitting behind real accounts, per-user chat history, and role-based access control — not a chatbot demo bolted onto a script.

**Live Application:** [support-copilot-app...azurecontainerapps.io](https://support-copilot-app.blackpebble-352cd42a.francecentral.azurecontainerapps.io)

`Python` `FastAPI` `QLoRA` `llama.cpp` `RAG` `SQLAlchemy` `Azure SQL` `JWT Auth` `Docker` `GitHub Actions`

</div>

---

### Contents

- [What it does](#what-it-does)
- [What this demonstrates](#what-this-demonstrates)
- [Architecture](#architecture)
- [Stack](#stack)
- [Project structure](#project-structure)
- [Running it](#running-it)
- [What's genuinely not done](#whats-genuinely-not-done)

## What it does

Ask it about a refund, an order, a shipping delay, or an account change, and it answers using a model fine-tuned specifically on customer support conversations — grounded against a knowledge base so the answer is pulled from actual policy, not invented. Every user gets their own account, their own conversation history (multiple named chats, switchable, deletable — not one endless thread), and every message passes through guardrails before it reaches the model or the user.

## What this demonstrates

- **Fine-tuning end to end** — QLoRA fine-tune of Llama-3-8B on the Bitext support dataset, trained and pushed to the Hugging Face Hub from a reproducible Colab pipeline
- **Making an 8B model fast on CPU, not just running it** — the naive `transformers` inference path timed out on every request under real deployment constraints (no GPU tier available); converting the model to a quantized GGUF file and serving it through `llama.cpp` — plus fixing a thread-oversubscription bug that was silently working against the fix — took response time from "never finishes" to ~15-20 seconds
- **Real auth, not a login form for show** — JWT access + refresh tokens (refresh tokens stored hashed so a session can actually be revoked, not just forgotten about), role-based access control enforced at the dependency layer on every admin route, per-user data isolation enforced in the query itself
- **RAG grounding** — Chroma + sentence-transformer embeddings retrieve the relevant policy before generation, so answers are traceable back to a source, not free-floating
- **Guardrails with a measured catch rate** — PII redaction and prompt-injection detection, benchmarked against an adversarial test set rather than assumed to work (`scripts/adversarial_report.py`)
- **Cost-aware infrastructure** — a single shared Gemini key for the optional faithfulness-check feature, metered against a daily cap in the database, so a portfolio demo can't silently run up an open-ended bill
- **Deployment engineering, not just a working laptop demo** — Docker image built and pushed by GitHub Actions (not from a local machine — that path kept timing out on upload), running on Azure Container Apps against a real Azure SQL database

## Architecture

```
User message
    │
    ▼
FastAPI (JWT auth, RBAC, guardrails)
    │
    ├──► Chroma + sentence-transformers ── retrieves relevant policy
    │
    └──► llama.cpp (quantized, fine-tuned Llama-3-8B) ── generates the answer
    │
    ▼
Persisted to the user's conversation (Azure SQL)
```

## Stack

| Layer | Tools |
|---|---|
| LLM | Self fine-tuned Llama-3-8B (QLoRA), quantized to GGUF, served via `llama-cpp-python` (a vLLM backend also exists for a future GPU deployment — see `src/llm_backend.py`) |
| Retrieval | Chroma + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Backend | FastAPI |
| Database | Azure SQL (production) / SQLite (local dev) via SQLAlchemy |
| Auth | JWT access + refresh tokens, bcrypt |
| Hosting | Azure Container Apps (Docker) |
| CI/CD | GitHub Actions → Azure Container Registry → Azure Container Apps |

## Project structure

```
customer_support_copilot/
├── src/
│   ├── app.py              # API routes: auth, conversations, chat, admin
│   ├── llm_backend.py       # llama.cpp (default) or vLLM generation backend
│   ├── retriever.py         # KB embedding + retrieval
│   ├── evaluate.py          # Gemini faithfulness check, rate-limited
│   ├── auth.py               # JWT, password hashing, refresh tokens
│   ├── guardrails.py         # prompt-injection and PII checks
│   ├── models.py              # User, Conversation, ChatMessage, GeminiUsage
│   └── prepare_data.py        # builds training data + knowledge base
├── scripts/adversarial_report.py   # honest guardrails catch-rate report
├── notebooks/
│   ├── training.ipynb          # QLoRA fine-tune, pushes adapter to HF Hub
│   └── gguf_conversion.ipynb   # quantizes for fast CPU inference
├── frontend/index.html         # chat UI, multi-conversation sidebar, auth
├── tests/
└── Dockerfile
```

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py     # builds the knowledge base, run once
cp .env.example .env
uvicorn src.app:app --reload --port 8000
```

```bash
pytest -v   # auth, guardrails, and the property that actually matters:
            # one user's chat history is never returned to another user
```

## What's genuinely not done

- Google Sign-In exists and is tested in the backend, but isn't exposed in the UI — publishing it needs a privacy policy hosted on a domain I own, which the current deployment doesn't have
- No email verification or password reset flow
- The Gemini rate limit meters call count, not actual token spend
- Quantizing to GGUF trades a small amount of answer quality for the CPU speed that makes this deployment viable at all — the un-quantized adapter is still on Hugging Face Hub
