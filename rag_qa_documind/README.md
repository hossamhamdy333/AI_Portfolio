# DocuMind — Retrieval-Augmented Generation (RAG) Q&A System

Ask questions about your own documents (PDF / TXT / MD) and get answers grounded
in their content, with sources cited. Built with FastAPI, ChromaDB, local
sentence-transformer embeddings, and Gemini for generation.

**Live demo:** https://documents-mind.streamlit.app/

**Why this project:** RAG is one of the most widely used patterns in applied AI
engineering. This project covers the full stack: document processing,
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
- **PDF extraction**: uses `pypdf` in layout mode, with a dictionary-based
  fallback (`wordninja`) that repairs words that still end up fused together
  with no spaces — a real issue with some LaTeX-generated academic PDFs.
- **Vector store**: ChromaDB, persisted to disk under `data/chroma_db`.
- **Generation**: Gemini (free tier via Google AI Studio), called through the
  Interactions API, grounded strictly in retrieved context via the system
  prompt in `app/llm.py`.
- **API**: FastAPI (`app/main.py`) — `/ingest`, `/query`, `/health`, `/reset`.
- **UI**: two interchangeable front ends —
  - `ui/streamlit_app.py`: talks to the FastAPI backend over HTTP. Use this
    for local development (two terminals) or Docker (two services).
  - `streamlit_app.py` (repo root): a standalone version that calls the RAG
    pipeline directly in-process, for platforms like Streamlit Community
    Cloud that only run a single process.

## Project layout

```
rag_qa_documind/
├── app/
│   ├── config.py       # env-driven settings
│   ├── ingest.py        # load + chunk + embed + store documents
│   ├── vectorstore.py   # ChromaDB wrapper
│   ├── llm.py            # Gemini API wrapper (generation)
│   ├── rag.py             # retrieve -> generate orchestration
│   └── main.py             # FastAPI app
├── ui/streamlit_app.py       # chat UI (talks to FastAPI backend)
├── streamlit_app.py            # standalone chat UI (Streamlit Cloud)
├── scripts/run_ingest.py         # CLI bulk-ingest helper
├── data/sample_docs/                # example document to try immediately
├── tests/test_rag.py                  # unit tests for chunking logic
├── notebooks/walkthrough.ipynb          # step-by-step notebook walkthrough
├── requirements.txt
├── .env.example
├── .streamlit/secrets.toml.example
├── Dockerfile
└── docker-compose.yml
```

---

## How to run it — step by step

### Option A: Run locally with FastAPI + Streamlit (two terminals)

**1. Install dependencies**
```bash
cd rag_qa_documind
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your API key**
```bash
cp .env.example .env
```
Open `.env` and paste your Gemini API key into `GEMINI_API_KEY`
(get a **free** key, no credit card required, at
https://aistudio.google.com/apikey).

**3. Ingest the sample document** (or drop your own files into
`data/sample_docs/` first)
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
cd rag_qa_documind
cp .env.example .env        # then add your GEMINI_API_KEY
docker compose up --build
```
- API: http://localhost:8000
- UI: http://localhost:8501

### Option C: Deploy for free to Streamlit Community Cloud (public link, no card)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **Create app**.
3. Repository: your repo. Branch: `main`. Main file path:
   `rag_qa_documind/streamlit_app.py` (the standalone root-level file, not
   the one in `ui/`).
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   GEMINI_MODEL = "gemini-3.1-flash-lite"
   ```
5. Deploy. You'll get a public URL like `https://your-app.streamlit.app`.

---

## Trying it via the raw API (no UI, local FastAPI setup only)

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

## Known limitations

- **Scanned/image-only PDFs won't work.** The app detects when a PDF extracts
  to almost no text and warns you, but it can't read text out of images —
  only PDFs with a real text layer (exported from Word, Google Docs, LaTeX,
  etc., not scanned photos of pages).
- **One shared index.** All uploaded documents go into the same vector index.
  Uploading unrelated documents together will hurt retrieval precision for
  questions about any one of them. Click **Clear index** before switching to
  a different document or topic.
- **Free-tier Gemini rate limits apply** if this is deployed publicly and
  gets meaningful traffic.

## How to extend this project

- **Evaluation**: extend the notebook's precision@k example into a real
  regression test suite against a hand-labeled set of question/answer pairs.
- **Add reranking**: insert a cross-encoder reranking step after initial
  retrieval to improve answer quality.
- **Add streaming**: stream the Gemini response token-by-token to the UI.
- **Add auth + multi-user**: namespace the Chroma collection per user/session
  instead of one shared index.
- **Support more providers**: `app/llm.py` currently calls Gemini only;
  adding an `LLM_PROVIDER` setting to switch between Gemini/OpenAI/Anthropic
  
