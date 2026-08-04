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

## Project layout

```
documind-rag/
├── app/
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
```bash
cd documind-rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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
