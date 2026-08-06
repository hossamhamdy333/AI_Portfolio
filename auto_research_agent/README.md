# Auto Research Agent

Autonomous multi-agent system that takes an open-ended research task, searches the
web, pulls from an indexed knowledge base (PDFs, docs, spreadsheets, audio/video,
images), optionally runs Python for quick analysis, pauses for human review if
sources conflict, and writes a final Markdown report — served through a FastAPI
backend and a Streamlit UI.

**100% free stack. No card required anywhere.** Every paid tool from the original
blueprint was swapped for a free/open-source equivalent — see the table below.

## Architecture

```
docs/ (pdf, docx, xlsx, mp3/mp4, png/jpg)
        |
        v
ingestion/loaders.py  -->  ingestion/chunking.py  -->  Qdrant (local, on disk)
                                                              |
                                                              v
frontend/app.py (Streamlit)  <-->  api/main.py (FastAPI)  <-->  agents/graph.py (LangGraph)
                                          |                        |  Planner
                                          v                        |  Researcher (web + RAG)
                                agents/state.py (SQLite)           |  [human review if conflict]
                                                                    |  Analyst (code exec)
                                                                    |  Writer
observability/phoenix_setup.py  -- traces every LLM call, free, local UI
eval/evaluate_rag.py            -- Ragas scores for the retrieval pipeline
```

## What replaced what (paid -> free)

| Blueprint (paid)                  | This repo (free)                                      |
|------------------------------------|--------------------------------------------------------|
| GPT-4o / Claude 3.5 Sonnet         | **Groq** `llama-3.3-70b-versatile` (free tier, no card) |
| Whisper API                        | **faster-whisper** (runs locally, no API call)          |
| GPT-4o / LLaVA vision              | **Groq** `llama-3.2-11b-vision-preview` (free tier); `pytesseract` OCR as offline fallback |
| OpenAI `text-embedding-3-small`    | **sentence-transformers** `all-MiniLM-L6-v2` (local, free) |
| LlamaParse                         | `pypdf` / `python-docx` / `openpyxl` (free, local)      |
| Pinecone / managed Qdrant          | **Qdrant local mode** — a folder on disk, no server, no signup |
| Tavily / SerpAPI                   | **DuckDuckGo search** — fully free, no key at all       |
| Postgres / Redis                   | **SQLite** — one file, zero setup                        |
| LangSmith                          | **Arize Phoenix** — open source, runs 100% locally, no signup at all |

The **only** things that need a signup (still no card, ever):
- **Groq API key** — https://console.groq.com/keys (email only)
- **ngrok auth token** — only if you tunnel the FastAPI backend out of Colab for a live demo. https://dashboard.ngrok.com

## How the pieces connect (setup order)

1. **Install deps**
   ```bash
   pip install -r requirements.txt
   # also: sudo apt-get install ffmpeg tesseract-ocr   (for audio + OCR fallback)
   ```

2. **Set your key**
   ```bash
   cp .env.example .env
   # paste your GROQ_API_KEY into .env
   ```

3. **Start the backend** (this is the hub everything else talks to)
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   On first run it creates `./data/qdrant_db` (vector store) and
   `./data/agent_memory.sqlite` (agent state/memory) automatically.

4. **Start the frontend** (talks to the backend over HTTP)
   ```bash
   streamlit run frontend/app.py
   ```
   Open the URL it prints. Upload a file in the sidebar to index it, then run a task.

5. **(Optional) Observability** — before step 3, run once in the same process/session:
   ```python
   from observability.phoenix_setup import setup_observability
   setup_observability()  # opens a trace UI at localhost:6006
   ```

6. **(Optional) Evaluate retrieval quality**
   ```bash
   python eval/evaluate_rag.py
   ```
   Edit `SAMPLE_QA` in that file with real questions once you've indexed real docs.

### Running from Colab (no local machine)

Colab can run the backend and tunnel it out with `pyngrok`:
```python
!pip install -r requirements.txt -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
get_ipython().system_raw("uvicorn api.main:app --port 8000 &")
public_url = ngrok.connect(8000)
print(public_url)
```
Then point `BACKEND_URL` in `frontend/app.py` (run it on a second machine, or also
via a second ngrok tunnel) at that public URL.

### Running with Docker (needs Docker somewhere — Colab itself can't run Docker;
use your local machine, or the GNS3/Alpine server you used for the NIDS project)

```bash
docker compose -f docker/docker-compose.yml up --build
```
API on `:8000`, UI on `:8501`.

## Human-in-the-loop, concretely

The graph always pauses right before the **Analyst** node. The API checks the
`needs_human_review` flag the Researcher set:
- **No conflict found** -> API auto-resumes immediately, run finishes normally.
- **Conflict flagged** -> API returns `awaiting_human_review` with the research so
  far. The Streamlit UI shows it and waits for you to click Approve/Cancel, which
  calls `POST /resume/{thread_id}`.

## Honest limitations (put these in the repo, don't hide them)

- The Python code-exec tool is a restricted `exec()`, not a real sandbox. Fine for
  a personal project running your own tasks; don't expose it to untrusted public
  input without swapping in a real sandbox (subprocess + resource limits, or a
  container).
- Vision captioning only works if you pass a Groq client into `describe_image`;
  otherwise it silently falls back to OCR, which can't describe charts/trends,
  only read visible text.
- Qdrant local mode is single-process — fine for a demo/portfolio project, not for
  concurrent multi-user production traffic.

## CV bullets (accurate, not inflated)

- Architected a multi-agent research system (LangGraph) with Planner, Researcher,
  Analyst, and Writer agents, including a human-in-the-loop pause triggered when
  sources conflict.
- Built a multimodal ingestion pipeline (PDF/DOCX/XLSX/audio/video/image) into a
  local Qdrant vector store using free local embeddings.
- Exposed the pipeline via a FastAPI backend (REST + WebSocket streaming) and a
  Streamlit frontend; containerized both with Docker Compose.
- Added observability (Arize Phoenix, open-source) and automated RAG evaluation
  (Ragas: faithfulness, answer relevancy, context precision/recall).

## Demo checklist before pushing to GitHub

- [ ] Put 1-2 sample PDFs in `docs/` so the RAG path works out of the box
- [ ] Record a 1-2 min screen recording of the Streamlit demo, link it at the top
      of this README
- [ ] Run `eval/evaluate_rag.py` once with real questions, paste the scores here
