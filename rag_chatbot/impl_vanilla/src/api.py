"""FastAPI serving layer for the RAG chatbot -- the piece flagged as missing
in this project's own README ('build the FastAPI /chat endpoint the config
is already set up for'). Wires together retrieval.py + generation.py exactly
as the eval loop already does, just behind an HTTP endpoint instead of a
notebook loop.

Qdrant runs in local embedded mode (no separate server) pointed at the same
Drive folder 04_rag_pipeline.ipynb writes to -- so this API reads whatever
was last built there. Run this from within the same Colab session/runtime
that has Drive mounted at /content/drive, right after running the pipeline
notebook (or after confirming that Drive path already has data from a
previous run).

Run: uvicorn src.api:app --host <serving.host> --port <serving.port>
(reads host/port from configs/config.yaml, same as everything else in this repo)
"""

import logging
import os

import yaml
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.generation import generate_answer
from src.retrieval import rerank_chunks, search_chunks

logger = logging.getLogger(__name__)

with open("configs/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Same path 04_rag_pipeline.ipynb's cell 10 writes to -- must match, since this
# API only has data if that notebook already ran and populated this Drive folder.
QDRANT_LOCAL_PATH = "/content/drive/MyDrive/AI_Portfolio_Qdrant"

app = FastAPI(title="RAG Chatbot API", version="1.0")

# Loaded once at startup, reused across requests -- these are the same
# objects the eval notebook builds once per run, just kept alive here.
_state = {}


@app.on_event("startup")
def load_models():
    _state["embedder"] = SentenceTransformer(CONFIG["retrieval"]["model_name"])
    _state["reranker"] = CrossEncoder(CONFIG["reranker"]["model_name"])
    if not os.path.isdir(QDRANT_LOCAL_PATH):
        raise RuntimeError(
            f"No Qdrant data found at {QDRANT_LOCAL_PATH} -- run 04_rag_pipeline.ipynb "
            "first to build and persist the collection."
        )
    _state["qdrant_client"] = QdrantClient(path=QDRANT_LOCAL_PATH)
    _state["gemini_client"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    logger.info("Models and Qdrant client loaded.")


class ChatRequest(BaseModel):
    question: str
    top_k_retrieve: int | None = None
    top_k_rerank: int | None = None


class Citation(BaseModel):
    source_num: int
    article_id: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k_retrieve = request.top_k_retrieve or CONFIG["rag"]["top_k_retrieve"]
    top_k_rerank = request.top_k_rerank or CONFIG["rag"]["top_k_rerank"]

    candidates = search_chunks(
        request.question,
        _state["qdrant_client"],
        CONFIG["qdrant"]["collection_name"],
        _state["embedder"],
        top_k=top_k_retrieve,
    )
    if not candidates:
        return ChatResponse(answer="لا توجد معلومات كافية للإجابة على هذا السؤال.", citations=[])

    reranked = rerank_chunks(request.question, candidates, _state["reranker"], top_k=top_k_rerank)

    answer_text, citations, _ = generate_answer(
        _state["gemini_client"],
        request.question,
        reranked,
        model_name=CONFIG["rag"]["llm_model"],
        temperature=CONFIG["rag"]["temperature"],
        max_output_tokens=CONFIG["rag"]["max_output_tokens"],
    )
    return ChatResponse(answer=answer_text, citations=citations)


@app.get("/health")
def health():
    return {"status": "ok"}
