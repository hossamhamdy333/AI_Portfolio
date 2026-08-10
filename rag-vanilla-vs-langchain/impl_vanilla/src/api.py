"""FastAPI serving layer for the RAG chatbot -- wires together
retrieval.py + generation.py exactly as 04_evaluation.ipynb's eval loop
does, just behind an HTTP endpoint instead of a notebook loop.

Connects to Qdrant Cloud (configs/config.yaml's qdrant.url), not a local
embedded instance -- so this reads whatever collection
04_evaluation.ipynb last built there, from anywhere, not just from within
the same Colab session that built it. That's also what makes deploying
this to Hugging Face Spaces (see ../Dockerfile) actually work: a Space's
filesystem doesn't persist the way Colab's Drive mount does, so a local
embedded Qdrant path wouldn't survive a Space restart even if reachable.

Local run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
Spaces run: see ../Dockerfile (listens on 7860, per configs/config.yaml's serving section)
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

# configs/config.yaml lives at the repo root, shared with impl_langchain --
# this file sits at impl_vanilla/src/api.py, so repo root is two levels up.
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(title="RAG Chatbot API", version="1.0")

# Loaded once at startup, reused across requests -- these are the same
# objects the eval notebook builds once per run, just kept alive here.
_state = {}


@app.on_event("startup")
def load_models():
    _state["embedder"] = SentenceTransformer(CONFIG["retrieval"]["model_name"])
    _state["reranker"] = CrossEncoder(CONFIG["reranker"]["model_name"])

    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise RuntimeError(
            "QDRANT_API_KEY is not set. Get one from your free cluster at "
            "https://cloud.qdrant.io and set it as an env var (or a Space secret)."
        )
    _state["qdrant_client"] = QdrantClient(url=CONFIG["qdrant"]["url"], api_key=qdrant_api_key)
    _state["gemini_client"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    logger.info("Models and Qdrant Cloud client loaded (chunking strategy: %s).", CONFIG["chunking"]["strategy"])


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
    return {"status": "ok", "chunking_strategy": CONFIG["chunking"]["strategy"]}
