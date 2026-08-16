# Azure RAG Assistant

An agentic Retrieval-Augmented Generation (RAG) chatbot: upload documents,
ask questions about them in natural language, and get answers grounded in
your actual content — powered by Google Gemini, Qdrant vector search, and
deployed on Microsoft Azure.

**Live demo:** `https://azure-rag-assistant-b6hqawe7eef6euaf.francecentral-01.azurewebsites.net`

## What it does

- Upload a PDF, image, or text file through the web UI
- The document is chunked, embedded, and indexed in a vector database
- Ask questions in a chat interface — the agent searches your uploaded
  documents for relevant context before answering
- Also handles plain arithmetic via a built-in calculator tool
- Every upload is archived to Azure Blob Storage alongside the vector index

## Architecture

| Component | Technology |
|---|---|
| Backend API + UI | FastAPI, served as a single container (no separate frontend service) |
| LLM | Google Gemini (`gemini-3.1-flash-lite`) |
| Agent framework | LangChain (`create_agent`) |
| Embeddings | Hugging Face Inference API (`all-MiniLM-L6-v2`) |
| Vector database | Qdrant Cloud |
| Document storage | Azure Blob Storage |
| Hosting | Azure App Service (Web App for Containers) |
| CI/CD | GitHub Actions → Docker Hub → Azure, auto-deploys on push to `main` |

The frontend is a single static HTML/JS page (`backend/static/index.html`)
served directly by the FastAPI app — there's no separate frontend
deployment or framework.

## Project structure

```
Azure_RAG_Assistant/
├── README.md
├── .env.example              # template for local development
├── docker-compose.yml        # optional, local dev only (uses Azurite)
├── for-your-repo-root/       # deployment helper, see note below
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.py              # centralized settings (pydantic-settings)
    ├── main.py                # FastAPI app: "/", /health, /chat, /upload
    ├── agent.py                # LangChain agent: retrieval + calculator tools
    ├── text_processing.py      # document parsing, chunking, embedding
    ├── azure_storage.py        # Azure Blob Storage archival (best-effort)
    ├── safe_math.py             # AST-based safe calculator (no eval)
    ├── static/index.html        # the entire frontend
    └── tests/                   # pytest suite, runs in CI on every push
```

## Security notes

- The calculator tool uses an AST-based evaluator (`safe_math.py`), not
  Python's `eval()` — arithmetic expressions are parsed and restricted to a
  fixed set of numeric operators, so it cannot execute arbitrary code even
  from adversarial input.
- API endpoints are protected by a shared-secret header (`APP_API_KEY`),
  checked on every request when set.
- Document archival to Azure Blob Storage is best-effort: if storage is
  unreachable or misconfigured, uploads still succeed and remain fully
  searchable via the vector index — only the raw-file backup is skipped.

## Running the tests

```bash
cd backend
pip install -r requirements.txt
pytest -v
```

No real API keys are required — the test suite covers the calculator, the
storage fallback behavior, and core API routes using mocked credentials.

## Local development (optional)

A `docker-compose.yml` is included for local iteration, using
[Azurite](https://github.com/Azure/Azurite) (Microsoft's official local
Azure Storage emulator) in place of a real Azure account:

```bash
cp .env.example .env   # fill in your Gemini / Hugging Face / Qdrant keys
docker-compose up --build
# open http://localhost:8000
```

This is entirely optional — the production deployment on Azure doesn't
depend on it.

## Deployment

This project deploys as a single Docker container to Azure App Service,
with GitHub Actions building and pushing the image to Docker Hub on every
push to `main`. Required environment variables (set in Azure App Service →
Environment variables):

```
GEMINI_API_KEY
HF_TOKEN
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION_NAME
AZURE_STORAGE_CONNECTION_STRING
AZURE_STORAGE_CONTAINER_NAME
APP_API_KEY
WEBSITES_PORT=8000
```

Required GitHub repository secrets (Settings → Secrets and variables →
Actions), used by `.github/workflows/deploy.yml`:

```
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
AZURE_WEBAPP_PUBLISH_PROFILE
```

**Note on `for-your-repo-root/`:** if this project lives inside a monorepo
(a subfolder alongside other unrelated projects, as in this portfolio
repo), a thin proxy `Dockerfile` needs to sit at the *repository* root so
Azure's default build process can find it — see
`for-your-repo-root/README.md` for details. If deployed as its own
standalone repo instead, this isn't needed.
