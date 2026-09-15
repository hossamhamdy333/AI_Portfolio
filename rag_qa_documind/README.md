<div align="center">

# DocuMind — RAG Q&A over your own documents

Upload a PDF, TXT, or MD file and ask questions about it. Answers are grounded in the actual text — every answer is shown with the source passage it came from, so you can check it isn't making things up.

**Live Application:** [documents-mind.streamlit.app](https://documents-mind.streamlit.app/)

`Python` `FastAPI` `ChromaDB` `Sentence-Transformers` `Gemini API` `SQLAlchemy` `JWT Auth` `Streamlit` `Docker`

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

Upload a document, it gets chunked and embedded locally, and stored in a private, per-account vector index. Ask a question and it's answered using only the chunks retrieved from that index — not the model's general knowledge — with the source file cited. Every account has its own private document set, and the whole thing runs behind real login, not a shared open URL.

## What this demonstrates

- **RAG grounded in real retrieval, not just a prompt template** — Chroma + `sentence-transformers` embeddings retrieve the relevant chunks before generation, and every answer is traceable back to the source passage it came from
- **Real auth, not a login form for show** — JWT access + refresh tokens and Google OAuth on the FastAPI backend, bcrypt password hashing, and per-account document isolation enforced at the Chroma collection level, not a client-supplied ID
- **Handling messy real-world PDFs** — LaTeX-exported PDFs extract with words glued together with no spaces; fixed with layout-mode extraction plus a dictionary-based fallback (`wordninja`) that catches whatever's still stuck together
- **Guardrails with a measured catch rate** — PII redaction and prompt-injection detection, benchmarked against an adversarial test set rather than assumed to work
- **Cost-aware infrastructure** — one shared Gemini key for the public deployment, metered against a daily per-account cap tracked in the database, so a portfolio demo can't be drained by one visitor or run up an open-ended bill
- **Two deployment shapes for one codebase** — a FastAPI backend + separate Streamlit UI for local dev and Docker, and a second standalone entrypoint that runs the same pipeline in-process for Streamlit Community Cloud, which only supports a single process

## Architecture

```
upload doc                              ask question
    │                                        │
    ▼                                        ▼
chunk + embed  ──────────►   Chroma   ◄────────── embed the question
(sentence-transformers)     (per-account            │
                              collection)    top-k relevant chunks
                                                     │
                                                     ▼
                                          Gemini generates the answer
                                          from those chunks only
```

## Stack

| Layer | Tools |
|---|---|
| Retrieval | ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| LLM | Gemini API, one shared key metered by a daily per-account cap |
| Backend | FastAPI |
| Database | SQLite (local dev) / Azure SQL (production) via SQLAlchemy |
| Auth | JWT access + refresh tokens, bcrypt, Google OAuth |
| Frontend | Streamlit — two entrypoints, see Architecture |
| Hosting | Streamlit Community Cloud |

## Project structure

```
rag_qa_documind/
├── app/
│   ├── ingest.py           # load, chunk, embed documents
│   ├── vectorstore.py      # Chroma, per-account isolation
│   ├── llm.py               # calls Gemini
│   ├── rag.py                # retrieval + generation
│   ├── database.py, models.py   # SQLAlchemy - users, tokens, documents
│   ├── auth.py, oauth.py    # password hashing, JWT, Google OAuth
│   ├── guardrails.py        # PII redaction + prompt-injection detection
│   └── main.py               # FastAPI app - auth, ingest, query, admin
├── ui/streamlit_app.py     # chat UI, talks to the FastAPI backend
├── streamlit_app.py        # standalone entrypoint (Streamlit Cloud)
├── scripts/run_ingest.py   # command-line bulk-ingest helper
├── tests/                  # auth, isolation, guardrails, retrieval
├── notebooks/walkthrough.ipynb
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── .env.example
```

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your Gemini key - free at aistudio.google.com/apikey
python scripts/run_ingest.py data/sample_docs
uvicorn app.main:app --reload --port 8000
```

```bash
streamlit run ui/streamlit_app.py   # new terminal, same venv - opens at localhost:8501
```

```bash
pytest tests/ -v   # auth, guardrails, and the property that matters most:
                    # one account's documents never leak into another's
```

Or skip the setup: `cp .env.example .env` then `docker compose up --build`.

## What's genuinely not done

- Scanned PDFs (images with no real text layer) don't extract — the app warns instead of failing silently
- A hard page reload logs you out on the standalone deployment — login lives in `st.session_state`, not a URL, which is the correct trade for account security but a real UX cost
- The daily Gemini cap meters question count, not actual token spend
- Guardrails are regex heuristics, not a trained moderation model
- Google Sign-In is wired for the FastAPI backend only — the standalone Streamlit Cloud app has no separate server to complete the OAuth redirect, so it stays email + password there
