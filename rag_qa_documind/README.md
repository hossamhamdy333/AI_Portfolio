<<<<<<< HEAD
# DocuMind — Retrieval-Augmented Generation (RAG) Q&A System

Ask questions about your own documents (PDF / TXT / MD) and get answers grounded
in their content, with sources cited. Built with FastAPI, ChromaDB, local
sentence-transformer embeddings, and Claude for generation.

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
                                                          │ (Claude API)   │
                                                          └───────┬───────┘
                                                                  ▼
                                                             final answer
```

- **Embeddings**: local `sentence-transformers` model (`all-MiniLM-L6-v2`) —
  no API key or network call needed for this step, and it's free.
- **Vector store**: ChromaDB, persisted to disk under `data/chroma_db`.
- **Generation**: Claude (Anthropic API), grounded strictly in retrieved
  context via the system prompt in `app/llm.py`.
- **API**: FastAPI (`app/main.py`) — `/ingest`, `/query`, `/health`, `/reset`.
- **UI**: Streamlit chat interface (`ui/streamlit_app.py`) that talks to the API.

=======
# DocuMind

A small RAG (Retrieval-Augmented Generation) app. Upload a document, ask questions about it, get answers pulled straight from the text — with the source shown so you can check it's not making things up.

Built with FastAPI, ChromaDB, local sentence-transformer embeddings, and Gemini for generating the actual answers.

## How it works

1. You upload a document (PDF, TXT, or MD)
2. It gets split into small chunks and turned into embeddings (basically, numerical "meaning fingerprints") using a small model that runs locally — no API key needed for this part
3. Those embeddings get stored in ChromaDB, a lightweight local vector database
4. When you ask a question, it gets embedded too, and the system finds the chunks whose meaning is closest to your question
5. Those chunks get sent to Gemini along with your question, and Gemini writes an answer using only that context
6. You get the answer back with the source file named

```
 upload doc                    ask question
     │                              │
     ▼                              ▼
 chunk + embed  ──────────►  ChromaDB  ◄────────── embed the question
                                  │
                          top-k relevant chunks
                                  │
                                  ▼
                          Gemini generates
                          an answer from them
```

>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
## Project layout

```
documind-rag/
├── app/
<<<<<<< HEAD
│   ├── config.py       # env-driven settings
│   ├── ingest.py        # load + chunk + embed + store documents
│   ├── vectorstore.py   # ChromaDB wrapper
│   ├── llm.py            # Claude API wrapper (generation)
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
=======
│   ├── config.py       # settings, loaded from .env
│   ├── ingest.py        # loads, chunks, and embeds documents
│   ├── vectorstore.py   # talks to ChromaDB
│   ├── llm.py            # calls Gemini to generate the answer
│   ├── rag.py             # ties retrieval + generation together
│   └── main.py             # the FastAPI app
├── ui/streamlit_app.py       # the chat interface
├── scripts/run_ingest.py       # command-line way to bulk-load documents
├── data/sample_docs/             # a sample file so you can try it immediately
├── tests/test_rag.py               # a few tests for the chunking logic
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Running it

**1. Install everything**
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
```bash
cd documind-rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

<<<<<<< HEAD
**2. Set your API key**
```bash
cp .env.example .env
```
Open `.env` and paste your Anthropic API key into `ANTHROPIC_API_KEY`
(get one at https://console.anthropic.com/).

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
cp .env.example .env        # then add your ANTHROPIC_API_KEY
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
- **Add streaming**: stream the Claude response token-by-token to the UI.
- **Add auth + multi-user**: namespace the Chroma collection per user/session.
- **Deploy it**: put the Docker image on Fly.io / Render / AWS and link a
  live demo on your CV — a deployed, working link is worth far more than a
  repo alone.
=======
**2. Add your Gemini key**
```bash
cp .env.example .env
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no credit card needed) and paste it into `.env` as `GEMINI_API_KEY`.

**3. Load the sample document**
```bash
python scripts/run_ingest.py data/sample_docs
```
First run downloads a small embedding model (~90MB) — normal, one-time thing.

**4. Start the backend**
```bash
uvicorn app.main:app --reload --port 8000
```
Leave this running. Check it worked: [localhost:8000/health](http://localhost:8000/health)

**5. Start the interface** (new terminal, same venv)
```bash
streamlit run ui/streamlit_app.py
```
Opens at `localhost:8501`. Upload your own files from the sidebar and ask questions in the chat box.

Both terminals need to stay open while you're using it.

### Or with Docker, if you'd rather skip the setup

```bash
cp .env.example .env   # add your key first
docker compose up --build
```
API on port 8000, UI on port 8501.

## Trying it without the UI

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ingest -F "file=@data/sample_docs/sample.txt"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does this project decide what text is relevant?"}'
```

## Things worth adding if you want to take this further

- A proper eval script — measure whether the right chunks actually get retrieved for a set of test questions
- Swap ChromaDB for something like Pinecone or pgvector, just to show you understand the tradeoffs
- Add reranking after retrieval to improve which chunks get used
- Stream the answer back token-by-token instead of waiting for the whole thing
- Support multiple users/sessions instead of one shared document index
- Deploy it somewhere real (Render, Fly.io) and link the live version — a working demo beats a repo link every time
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
