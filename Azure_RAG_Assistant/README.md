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
APP_API_KEY
WEBSITES_PORT=8000
```

GitHub repository secrets required for the workflow:

```
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
AZURE_WEBAPP_PUBLISH_PROFILE
```

## Notes

- The calculator uses a restricted AST-based evaluator instead of `eval()`.
- API routes are protected by a shared-secret header (`APP_API_KEY`).
- If blob storage is unavailable, uploads still succeed and remain searchable — only the raw file backup is skipped.
