"""
FastAPI service for DocuMind.

Endpoints:
  GET  /health         -> liveness check
  POST /ingest         -> upload a file (.txt/.md/.pdf) to be embedded & indexed
  POST /query          -> ask a question, get an answer grounded in indexed docs
  POST /reset           -> wipe the vector index (start fresh)
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingest import ingest_file
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection

app = FastAPI(title="DocuMind RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": get_collection().count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".txt", ".md", ".pdf"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_chunks = ingest_file(tmp_path, source_name=file.filename)
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "chunks_indexed": n_chunks}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    return answer_question(req.question, top_k=req.top_k)


@app.post("/reset")
def reset():
    reset_collection()
    return {"status": "index cleared"}
