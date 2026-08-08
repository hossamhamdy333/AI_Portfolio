# DocuMind — RAG Q&A over your own documents

Upload a PDF, TXT, or MD file, ask questions about it, and get answers pulled straight from the text, with the source shown so you can check it's not making things up.

**Live demo:** https://documents-mind.streamlit.app/

Built with FastAPI, ChromaDB, local sentence-transformer embeddings, and Gemini for generating the actual answers.

## How it works

1. Upload a document. It gets split into chunks and turned into embeddings using a small model that runs locally, no API key needed for this part.
2. Those embeddings go into ChromaDB, a lightweight vector database, stored on disk.
3. When you ask a question, it gets embedded too, and the system finds the chunks whose meaning is closest to it.
4. Those chunks get sent to Gemini along with your question, and Gemini answers using only that context.
5. You get the answer back with the source file named.

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

A few things worth knowing about the implementation:

**PDF text extraction is trickier than it sounds.** Some PDFs (LaTeX-generated ones especially) extract with no spaces between words. I handle this with layout-mode extraction plus a dictionary-based fallback (`wordninja`) that fixes any words still stuck together.

**Each visitor gets their own private document set.** On the live demo, everyone who uploads a file gets an isolated Chroma collection, keyed to a random session ID. Nobody sees anyone else's uploads.

**Each visitor uses their own Gemini API key.** The public deployment doesn't ship with a shared key. Visitors paste their own free key into the sidebar, and it's kept only in their browser session's memory, never saved server-side or shared with other visitors. There's no shared quota to protect, so there's no access gate either.

**There are two versions of the front end.** `ui/streamlit_app.py` talks to the FastAPI backend over HTTP, which is the setup for local dev (two terminals) or Docker (two services). `streamlit_app.py` at the repo root calls the pipeline directly in-process instead, which is what Streamlit Community Cloud needs since it only runs one process.

## Project layout

```
rag_qa_documind/
├── app/
│   ├── config.py          # settings, loaded from .env
│   ├── ingest.py          # loads, chunks, and embeds documents
│   ├── vectorstore.py     # talks to ChromaDB, handles session isolation
│   ├── llm.py             # calls Gemini to generate the answer
│   ├── rag.py             # ties retrieval + generation together
│   └── main.py            # the FastAPI app
├── ui/streamlit_app.py    # chat UI (talks to the FastAPI backend)
├── streamlit_app.py       # standalone chat UI (for Streamlit Cloud)
├── scripts/run_ingest.py  # command-line bulk-ingest helper
├── data/sample_docs/      # a sample file to try immediately
├── tests/
│   ├── test_rag.py           # tests for the chunking logic
│   └── test_vectorstore.py   # tests for session isolation
├── notebooks/walkthrough.ipynb  # step-by-step notebook
├── requirements.txt
├── .env.example
├── .streamlit/secrets.toml.example
├── Dockerfile
└── docker-compose.yml
```

## Running it

**1. Install everything**
```bash
cd rag_qa_documind
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Gemini key**
```bash
cp .env.example .env
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), no credit card needed, and paste it into `.env` as `GEMINI_API_KEY`.

**3. Load the sample document**
```bash
python scripts/run_ingest.py data/sample_docs
```
First run downloads a small embedding model (~90MB). That's normal and only happens once.

**4. Start the backend**
```bash
uvicorn app.main:app --reload --port 8000
```
Leave this running. Check it worked at [localhost:8000/health](http://localhost:8000/health).

**5. Start the interface** (new terminal, same venv)
```bash
streamlit run ui/streamlit_app.py
```
Opens at `localhost:8501`. Upload your own files from the sidebar and ask questions in the chat box. Both terminals need to stay open while you're using it.

**6. Run the tests** (optional)
```bash
pytest tests/ -v
```

### Or with Docker, if you'd rather skip the setup

```bash
cp .env.example .env   # add your key first
docker compose up --build
```
API on port 8000, UI on port 8501.

### Deploying it for free (public link, no credit card)

Streamlit Community Cloud works well for this and doesn't ask for a card.

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**.
3. Repository: your repo. Branch: `main`. Main file path: `rag_qa_documind/streamlit_app.py` (the standalone one at the repo root, not the one in `ui/`).
4. No secrets are required. Each visitor pastes in their own free Gemini API key in the app's sidebar, so there's nothing sensitive to configure. If you want to pin a non-default model, you can optionally set it under **Advanced settings → Secrets**:
   ```toml
   GEMINI_MODEL = "gemini-3.1-flash-lite"
   ```
5. Deploy. You'll get a URL like `https://your-app.streamlit.app`.

## Trying it without the UI

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ingest -F "file=@data/sample_docs/sample.txt"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does this project decide what text is relevant?"}'
```

Add an `X-Session-Id: your-id` header to any of these to scope them to a private collection instead of the shared default one.

## Known limitations

- **Scanned PDFs won't work.** If a PDF is just images of text with no real text layer, there's nothing to extract. The app warns you when this happens instead of failing silently.
- **Sessions isolate uploads between different visitors, not between unrelated documents you upload yourself.** If you upload several different documents in the same session, they all go into the same private index together, and retrieval precision for questions about any one of them can drop. Clear the index before switching topics.
- **Sessions don't persist across restarts.** They live only as long as the Chroma DB directory does. No accounts, no login, just a random ID.
- **Each visitor is subject to their own free-tier Gemini rate limits** since they bring their own key, so there's no shared quota to run out.

## Things worth adding if I take this further

- A proper eval script. The notebook has a tiny precision@k example that's worth building into a real regression suite.
- Reranking after retrieval to improve which chunks actually get used.
- Streaming the answer back token-by-token instead of waiting for the whole thing.
- Real per-user accounts if uploads should survive across visits, not just within one session.
- Support for other LLM providers, not just Gemini. `app/llm.py` would need an `LLM_PROVIDER` setting to switch between Gemini/OpenAI/Anthropic without touching `rag.py`.
- Swapping ChromaDB for something like Pinecone or pgvector, mostly to show I understand the tradeoffs.
