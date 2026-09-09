<div align="center">

# AI Support Copilot

A customer support chatbot powered by a QLoRA-fine-tuned Llama-3 (8B) model, grounded with RAG retrieval over a knowledge base built from the Bitext customer support dataset. Deployed live on Azure Container Apps, running entirely on CPU.

**Live demo:** [support-copilot-app...azurecontainerapps.io](https://support-copilot-app.blackpebble-352cd42a.francecentral.azurecontainerapps.io)

`Python` `FastAPI` `Llama-3-8B (QLoRA)` `llama-cpp-python (GGUF)` `Sentence-Transformers` `ChromaDB` `Docker`

</div>

---

### Contents

- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Running it locally](#running-it-locally)
- [Deploying](#deploying)
- [Why CPU instead of GPU](#why-cpu-instead-of-gpu)
- [Accounts, roles, and a real database](#accounts-roles-and-a-real-database)
- [Setting up vLLM](#setting-up-vllm)

## How it works

```
User message (must be logged in)
    │
    ▼
FastAPI backend (src/app.py)
    │
    ├──► guardrails (src/guardrails.py)
    │       redact PII, block prompt-injection attempts, before
    │       anything reaches the model or gets logged
    │
    ├──► KBRetriever (src/retriever.py)
    │       embeds the query, finds the closest matching
    │       support article from the knowledge base
    │
    ├──► src/llm_backend.py
    │       llamacpp (default, CPU, quantized GGUF) or
    │       vllm (GPU, higher throughput) - one call either way
    │
    └──► saved to chat_messages, tied to the logged-in user
            (src/models.py) - so "my history" and an admin's
            "view this user's transcript" both mean something
```

The model started as a QLoRA fine-tune of Llama-3-8B, trained on Colab and pushed to Hugging Face Hub. To make it run fast enough on CPU-only cloud hardware (no GPU available on the free Azure for Students tier), it was converted to GGUF format and quantized — that's what takes response times from "times out after 4 minutes" down to about 15-20 seconds. vLLM is a second option for when GPU hardware IS available (see below) - the right tool specifically because this is a customer-facing bot where many people can be chatting at once.

## Project layout

| File | What it does |
|---|---|
| `src/prepare_data.py` | Downloads the Bitext dataset, builds the training set and the knowledge base (`data/kb_articles.jsonl`) |
| `notebooks/training.ipynb` | Fine-tunes Llama-3-8B with QLoRA on Colab, pushes the adapter to HF Hub |
| `notebooks/gguf_conversion.ipynb` | Converts the fine-tuned adapter to a quantized GGUF file (the format that runs fast on CPU) |
| `notebooks/eda.ipynb` | Exploratory analysis of the training dataset |
| `src/retriever.py` | Embeds and searches the knowledge base with `sentence-transformers` + Chroma |
| `src/llm_backend.py` | Picks llama.cpp (CPU) or vLLM (GPU) based on `LLM_BACKEND` - everything else calls `generate()` without caring which |
| `src/app.py` | The FastAPI backend — auth, per-user chat history, admin routes, wires guardrails + retriever + llm_backend together |
| `src/config.py` | All environment-driven settings in one place |
| `src/database.py`, `src/models.py` | SQLAlchemy — users, refresh tokens, chat messages |
| `src/auth.py`, `src/oauth.py` | Password hashing, JWT issuing/verification, Google OAuth |
| `src/guardrails.py` | PII redaction + prompt-injection detection, same layer as Azure RAG Assistant |
| `frontend/index.html` | The chat UI — now with login/register/Google, served directly by the backend |
| `src/evaluate.py` | Optional Gemini-based check for whether a response is actually supported by its retrieved context |
| `Dockerfile` | Builds the deployed image |

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build the knowledge base (only needed once)
python src/prepare_data.py

cp .env.example .env
# at minimum, set a real JWT_SECRET_KEY - see "Accounts, roles, and a
# real database" below for how to generate one

# Run the API + chat UI
uvicorn src.app:app --reload --port 8000
# open http://localhost:8000, register an account, and chat
```

## Deploying

This is deployed as a Docker container on **Azure Container Apps**, built and pushed via a GitHub Actions workflow (`.github/workflows/build-push-acr.yml`) rather than pushed from a local machine — this sidesteps large-image upload timeouts on a slow home connection.

Environment variables the container accepts:

| Variable | Purpose |
|---|---|
| `GGUF_REPO` | Hugging Face repo holding the quantized model (default: `hossam3759180/support-copilot-gguf`) |
| `GGUF_FILENAME` | File name within that repo (default: `support-copilot-q4.gguf`) |
| `GEMINI_API_KEY` | Only needed if `ENABLE_EVAL=1` |
| `ENABLE_EVAL` | Set to `1` to run the faithfulness check on every response (costs one extra Gemini call per message) |
| `LLM_BACKEND` | `llamacpp` (default) or `vllm` - see below |
| `VLLM_BASE_URL`, `VLLM_MODEL` | Only used when `LLM_BACKEND=vllm` |
| `DATABASE_URL` | `sqlite:///./dev.db` locally, or Azure SQL in production - see below |
| `JWT_SECRET_KEY` | **Must** be a real random secret in production |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `FRONTEND_URL` | Only needed for the "Continue with Google" button |

## Why CPU instead of GPU

Azure for Students doesn't include a GPU workload profile in this account's available regions. Rather than block the whole project on that, the model was converted from raw `transformers` inference (too slow on CPU — an 8B model in float32 would time out on every request) to a quantized GGUF file run through `llama.cpp`, which is specifically built for fast CPU inference.

## Accounts, roles, and a real database

Real accounts (email+password or "Continue with Google"), JWT access + refresh
tokens, two roles (`user`, `admin`), and every conversation is saved and tied
to the user who had it — none of that existed before.

### Setting up Azure SQL (production)

1. Azure Portal → **Create a resource → Azure SQL Database** (serverless
   compute tier for a demo — scales to near-zero cost when idle).
2. Allow Azure services through the server's firewall (Networking settings),
   plus your own IP if you need to connect directly.
3. `DATABASE_URL=mssql+pyodbc://<user>:<password>@<server>.database.windows.net:1433/<db>?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes`
4. `pip install pyodbc` (uncomment it in `requirements.txt`) **and** install
   the Microsoft ODBC Driver 18 itself in the container - `pip install` alone
   isn't enough, see
   [Microsoft's docs](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)
   for the `msodbcsql18` apt package this Debian-based `Dockerfile` needs added.

Local dev needs none of this - `sqlite:///./dev.db` is the default and needs
zero setup.

### Setting up Google OAuth

Same steps as Azure RAG Assistant's README: [Google Cloud Console](https://console.cloud.google.com)
→ Credentials → OAuth client ID → Web application → redirect URI
`http://localhost:8000/auth/google/callback` (or your real domain). Leave
`GOOGLE_CLIENT_ID` blank to disable the button - email+password still works.

### Promoting a user to admin

No API endpoint for this on purpose - it would be a privilege-escalation bug,
not a feature:

```bash
python -c "
from src.database import SessionLocal
from src.models import User, Role
db = SessionLocal()
user = db.query(User).filter(User.email == 'you@example.com').first()
user.role = Role.admin
db.commit()
"
```

Log in again afterward - the current access token still has the old role
baked in until it's reissued (tokens expire every 20 minutes by default).

### What's genuinely NOT done here

- No email verification, no password reset flow.
- `/admin/stats` counts accounts/messages, not real per-request API cost.
- Guardrails are regex heuristics (19/20 on the adversarial test set, see
  `scripts/adversarial_report.py`) - the one miss is listed, not hidden.

## Setting up vLLM

vLLM needs a GPU and the **original merged checkpoint** - not the GGUF file,
vLLM runs native HF/safetensors weights with its own PagedAttention scheduler.
`notebooks/gguf_conversion.ipynb` already produces this merged checkpoint as
an intermediate step before quantizing it further to GGUF - push that
intermediate checkpoint to its own HF Hub repo (e.g.
`hossam3759180/support-copilot-merged`) rather than only keeping the final
GGUF file.

```bash
pip install vllm
vllm serve hossam3759180/support-copilot-merged --port 8001
```

Then set `LLM_BACKEND=vllm` and restart the app - `VLLM_BASE_URL` already
points at `http://localhost:8001/v1` by default, matching the port above.

**Why vLLM here specifically, not everywhere:** llama.cpp serves one request
at a time per process - fine for a low-traffic dev tool, not for a
customer-facing bot where multiple people can be chatting at once. vLLM's
continuous batching is built exactly for that case. Compare the two directly
with the same load-test approach used in Azure RAG Assistant - hit `/chat`
at increasing concurrency under each backend and compare p50/p95 latency and
throughput.
