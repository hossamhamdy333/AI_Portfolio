<div align="center">

# DocuMind — RAG Q&A over your own documents

Upload a PDF, TXT, or MD file, ask questions about it, and get answers pulled straight from the text, with the source shown so you can check it's not making things up.

**Live demo:** [documents-mind.streamlit.app](https://documents-mind.streamlit.app/)

`Python` `FastAPI` `ChromaDB` `Sentence-Transformers` `Gemini API` `Streamlit` `Docker`

</div>

---

### Contents

- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Running it](#running-it)
- [Trying it without the UI](#trying-it-without-the-ui)
- [Accounts and a real database](#accounts-and-a-real-database)
- [Known limitations](#known-limitations)
- [Things worth adding if I take this further](#things-worth-adding-if-i-take-this-further)

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

**PDF text extraction is trickier than it sounds.** Some PDFs (LaTeX-generated ones especially) extract with no spaces between words. This is handled with layout-mode extraction plus a dictionary-based fallback (`wordninja`) that fixes any words still stuck together.

**Every user gets their own private document set, tied to a real account.** Log in (email+password, or Google on the FastAPI backend) and your uploads are private to you — isolated at the Chroma collection level, keyed to your account ID rather than a random URL parameter the old version used. See "Accounts and a real database" below.

**Each visitor uses their own Gemini API key.** The public deployment doesn't ship with a shared key. Visitors paste their own free key into the sidebar, and it's kept only in their browser session's memory, never saved server-side or shared with other visitors. There's no shared quota to protect, so there's no access gate for the *Gemini key* — logging in is still required, though, since that's what protects each user's own document set from every other user, not the API quota.

**There are two versions of the front end.** `ui/streamlit_app.py` talks to the FastAPI backend over HTTP (JWT bearer tokens), which is the setup for local dev (two terminals) or Docker (two services). `streamlit_app.py` at the repo root calls the pipeline directly in-process instead (checking the same password-hash database directly, no tokens issued), which is what Streamlit Community Cloud needs since it only runs one process with no separate server to hold a token-issuing conversation with.

## Project layout

```
rag_qa_documind/
├── app/
│   ├── config.py          # settings, loaded from .env
│   ├── ingest.py          # loads, chunks, and embeds documents
│   ├── vectorstore.py     # talks to ChromaDB, handles per-user isolation
│   ├── llm.py             # calls Gemini to generate the answer
│   ├── rag.py             # ties retrieval + generation together
│   ├── database.py, models.py   # SQLAlchemy - users, refresh tokens, document log
│   ├── auth.py, oauth.py  # password hashing, JWT, Google OAuth (FastAPI backend)
│   ├── guardrails.py      # PII redaction + prompt-injection detection
│   └── main.py            # the FastAPI app - auth, per-user ingest/query, admin routes
├── ui/streamlit_app.py    # chat UI (talks to the FastAPI backend, JWT bearer tokens)
├── streamlit_app.py       # standalone chat UI (Streamlit Cloud, session-state login)
├── scripts/run_ingest.py  # command-line bulk-ingest helper
├── data/sample_docs/      # a sample file to try immediately
├── tests/
│   ├── test_rag.py           # tests for the chunking logic
│   ├── test_vectorstore.py   # tests for collection-naming logic
│   ├── test_auth.py          # register/login/refresh/logout/admin-gating
│   ├── test_oauth.py         # Google OAuth redirect + state validation
│   ├── test_isolation.py     # proves one user's documents can't leak to another
│   └── test_guardrails.py    # PII redaction + prompt-injection detection
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
4. Secrets required now that there are real accounts:
   ```toml
   JWT_SECRET_KEY = "..."   # generate with: python -c "import secrets; print(secrets.token_hex(32))"
   DATABASE_URL = "..."     # a real Azure SQL connection string - see "Accounts
                            # and a real database" below. Without this, accounts
                            # are stored in a local SQLite file that will NOT
                            # survive a Streamlit Cloud redeploy.
   GEMINI_MODEL = "gemini-3.1-flash-lite"   # optional, only if you want a non-default model
   ```
5. Deploy. You'll get a URL like `https://your-app.streamlit.app`.

## Trying it without the UI

```bash
# register + log in first, everything below needs the access token
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "a-real-password-123"}'
# -> {"access_token": "...", "refresh_token": "..."}

TOKEN="paste the access_token here"

curl http://localhost:8000/health -H "Authorization: Bearer $TOKEN"

curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/sample_docs/sample.txt"

curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does this project decide what text is relevant?"}'
```

## Accounts and a real database

Real accounts (email+password everywhere, plus Google OAuth on the FastAPI
backend specifically), JWT access+refresh tokens for the FastAPI backend, and
per-user document isolation - reusing the exact same session-keyed Chroma
collection mechanism that was already in `app/vectorstore.py`, just keyed by
a verified account ID instead of a client-supplied header or URL parameter.

### Setting up Azure SQL (production)

Same steps as the other two projects: Azure Portal → Create a resource →
Azure SQL Database (serverless tier for a demo), allow Azure services through
the firewall, then:
```
DATABASE_URL=mssql+pyodbc://<user>:<password>@<server>.database.windows.net:1433/<db>?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes
```
`pip install pyodbc` (uncomment it in `requirements.txt`) and install the
Microsoft ODBC Driver 18 in the container - `pip install` alone isn't enough.

Local dev needs none of this - `sqlite:///./dev.db` is the default.

**Important for the Streamlit Community Cloud deployment specifically:**
without a real `DATABASE_URL` secret, accounts live in a local SQLite file
that gets wiped on every redeploy - set this to a real Azure SQL string in
Secrets if accounts should actually persist.

### Setting up Google OAuth (FastAPI backend only)

Same as the other two projects: [Google Cloud Console](https://console.cloud.google.com)
→ Credentials → OAuth client ID → redirect URI
`http://localhost:8000/auth/google/callback`. Not available on the standalone
Streamlit Cloud deployment - see that file's docstring for why.

### Promoting a user to admin

```bash
python -c "
from app.database import SessionLocal
from app.models import User, Role
db = SessionLocal()
user = db.query(User).filter(User.email == 'you@example.com').first()
user.role = Role.admin
db.commit()
"
```

### What changed, and one real trade-off worth naming

The old version's random-session-ID-in-the-URL scheme survived a page reload
(the ID lived in the URL, not memory) but meant anyone who saw or guessed a
`?sid=...` link could open a stranger's document set - a real weakness for a
public deployment. Real accounts fix that, but login now lives in
`st.session_state`, which does NOT survive a hard page reload the way the URL
parameter did - you'll need to log in again after a refresh. That's the
correct trade to make (the old scheme's "survives a reload" was also its
security hole), but it's a genuine UX regression worth knowing about, not a
free win.

## Known limitations

- **Scanned PDFs won't work.** If a PDF is just images of text with no real text layer, there's nothing to extract. The app warns you when this happens instead of failing silently.
- **Accounts isolate uploads between different users, not between unrelated documents you upload yourself.** If you upload several different documents to your own account, they all go into the same private index together, and retrieval precision for questions about any one of them can drop. Clear your index before switching topics.
- **A hard page reload logs you out** (see "Accounts and a real database" above for why this is the right trade-off, not an oversight).
- **Each visitor is subject to their own free-tier Gemini rate limits** since they bring their own key, so there's no shared quota to run out.
- **Guardrails are regex heuristics**, not a trained moderation model (see `app/guardrails.py` and Azure RAG Assistant's `scripts/adversarial_report.py` for the honest catch rate on the same approach - 19/20, one listed miss).

## Things worth adding if I take this further

- A proper eval script. The notebook has a tiny precision@k example that's worth building into a real regression suite.
- Reranking after retrieval to improve which chunks actually get used.
- Streaming the answer back token-by-token instead of waiting for the whole thing.
- A "remember me" option that survives a page reload without going back to the old URL-parameter scheme's security hole (e.g. a short-lived signed cookie).
- Support for other LLM providers, not just Gemini. `app/llm.py` would need an `LLM_PROVIDER` setting to switch between Gemini/OpenAI/Anthropic without touching `rag.py`.
- Swapping ChromaDB for something like Pinecone or pgvector, mostly to show an understanding of the tradeoffs.
