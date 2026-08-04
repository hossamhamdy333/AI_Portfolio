# DocuMind — Retrieval-Augmented Generation (RAG) Q&A System

Ask questions about your own documents (PDF / TXT / MD) and get answers grounded
in their content, with sources cited. Built with FastAPI, ChromaDB, local
sentence-transformer embeddings, and Gemini for generation.

**Why this project:** RAG is the single most in-demand pattern in applied AI
engineering. This project demonstrates the full stack: document processing,
chunking strategy, embeddings, vector search, prompt construction, an API
layer, and a usable front end — the same architecture used in production
support bots, internal knowledge assistants, and research tools.

## Architecture

```
 Upload doc            Ask question
     │                       │
     ▼                       ▼
 ┌─────────┐   embed   ┌──────────────┐   embed query   ┌──────────────┐
 │ ingest.py│ ────────► │  ChromaDB     │ ◄────────────── │ vectorstore.py│
 └─────────┘           │ (vector store)│                 └──────┬───────┘
                        └──────────────┘                        │ top-k chunks
                                                                  ▼
                                                          ┌───────────────┐
                                                          │   llm.py       │
                                                          │ (Gemini API)   │
                                                          └───────┬───────┘
                                                                  ▼
                                                             final answer
```

- **Embeddings**: local `sentence-transformers` model (`all-MiniLM-L6-v2`) —
  no API key or network call needed for this step, and it's free.
- **Vector store**: ChromaDB, persisted to disk under `data/chroma_db`.
- **Generation**: Gemini (free tier via Google AI Studio), grounded strictly in retrieved
  context via the system prompt in `app/llm.py`.
- **API**: FastAPI (`app/main.py`) — `/ingest`, `/query`, `/health`, `/reset`.
- **UI**: Streamlit chat interface (`ui/streamlit_app.py`) that talks to the API.

## Project layout

```
documind-rag/
├── app/
│   ├── config.py       # env-driven settings
│   ├── ingest.py        # load + chunk + embed + store documents
│   ├── vectorstore.py   # ChromaDB wrapper
│   ├── llm.py            # Gemini API wrapper (generation)
│   ├── rag.py             # retrieve -> generate orchestration
│   └── main.py             # FastAPI app
├── ui/
│   └── streamlit_app.py   # chat UI
├── scripts/
│   └── run_ingest.py       # CLI bulk-ingest helper
├── data/sample_docs/         # example document to try immediately
├── tests/test_rag.py           # unit tests for chunking logic
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## How to run it — step by step

### Option A: Run locally (recommended for first run)

**1. Install dependencies**
```bash
cd documind-rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your API key**
```bash
cp .env.example .env
```
Open `.env` and paste your Gemini API key into `GEMINI_API_KEY`
(get a **free** key, no credit card required, at https://aistudio.google.com/apikey).

**3. Ingest the sample documents** (or drop your own files into `data/sample_docs/` first)
```bash
python scripts/run_ingest.py data/sample_docs
```
You should see a per-file count of chunks indexed. The first run will
download the local embedding model (~90MB), which takes a minute.

**4. Start the API backend**
```bash
uvicorn app.main:app --reload --port 8000
```
Leave this running. Verify it's alive: open http://localhost:8000/health

**5. Start the UI** (in a new terminal, same venv activated)
```bash
streamlit run ui/streamlit_app.py
```
This opens a browser tab at http://localhost:8501. Upload more documents
from the sidebar if you like, then ask questions in the chat box.

**6. (Optional) Run the tests**
```bash
pytest tests/ -v
```

### Option B: Run with Docker (one command, no local Python setup)

```bash
cd documind-rag
cp .env.example .env        # then add your GEMINI_API_KEY
docker compose up --build
```
- API: http://localhost:8000
- UI: http://localhost:8501

Then ingest documents either through the UI's upload button, or by hitting
the API directly:
```bash
curl -X POST http://localhost:8000/ingest -F "file=@data/sample_docs/sample.txt"
```

---

## Trying it via the raw API (no UI)

```bash
# health check
curl http://localhost:8000/health

# ingest a document
curl -X POST http://localhost:8000/ingest -F "file=@data/sample_docs/sample.txt"

# ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does DocuMind decide which text is relevant to a question?"}'
```

---

## How to extend this for your CV / portfolio

Talking points that make this project stand out in interviews:
- **Chunking strategy**: explain the tradeoff between chunk size and context
  precision (implemented in `app/ingest.py::chunk_text`).
- **Evaluation**: add a script that measures retrieval precision@k against a
  hand-labeled set of question/answer pairs.
- **Swap the vector DB**: show you understand alternatives by porting
  `vectorstore.py` to Pinecone, Weaviate, or pgvector.
- **Add reranking**: insert a cross-encoder reranking step after initial
  retrieval to improve answer quality.
- **Add streaming**: stream the Gemini response token-by-token to the UI.
- **Add auth + multi-user**: namespace the Chroma collection per user/session.
- **Deploy it**: put the Docker image on Fly.io / Render / AWS and link a
  live demo on your CV — a deployed, working link is worth far more than a
  repo alone.
