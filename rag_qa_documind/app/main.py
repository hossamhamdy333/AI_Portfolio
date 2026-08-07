"""
FastAPI service for DocuMind.

Endpoints:
  GET  /health         -> liveness check
  POST /ingest         -> upload a file (.txt/.md/.pdf) to be embedded & indexed
  POST /query          -> ask a question, get an answer grounded in indexed docs
  POST /reset           -> wipe the vector index (start fresh)

All endpoints accept an optional X-Session-Id header. When set, ingestion
and retrieval are scoped to that session's own private collection instead
of the single shared index -- pass a stable per-browser-session ID from
the UI (see ui/streamlit_app.py) so concurrent users of a shared
deployment don't see each other's uploaded documents. Omit the header to
keep the original single-shared-index behavior (e.g. for the CLI script).
"""
import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel

from app.ingest import ingest_file
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection

app = FastAPI(title="DocuMind RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health(x_session_id: Optional[str] = Header(default=None)):
    return {"status": "ok", "indexed_chunks": get_collection(x_session_id).count()}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None),
):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".txt", ".md", ".pdf"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_chunks = ingest_file(tmp_path, source_name=file.filename, session_id=x_session_id)
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "chunks_indexed": n_chunks}


@app.post("/query")
def query(req: QueryRequest, x_session_id: Optional[str] = Header(default=None)):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    return answer_question(req.question, top_k=req.top_k, session_id=x_session_id)


@app.post("/reset")
def reset(x_session_id: Optional[str] = Header(default=None)):
    reset_collection(x_session_id)
    return {"status": "index cleared"}
