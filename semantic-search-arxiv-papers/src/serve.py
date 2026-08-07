"""
FastAPI serving endpoint for semantic search.

Two-stage retrieval: bi-encoder (Qdrant Cloud) -> cross-encoder rerank.
All infra connection details come from environment variables (see
.env.example) so nothing here points at localhost or bakes in secrets.
Model names, top_k, etc. are read from configs/config.yaml, the same file
the notebooks use, so serving can never silently drift from what was
actually benchmarked.
"""
import os
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional locally; in real deployments (Streamlit
    # Cloud, HF Spaces, Render, etc.) env vars / secrets are injected
    # directly by the platform, so this is never required there.
    pass

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME    = config["retrieval"]["model_name"]       # e.g. BAAI/bge-base-en-v1.5
RERANKER_NAME = config["reranker"]["model_name"]
COLLECTION    = config["qdrant"]["collection_name"]
TOP_K         = config["retrieval"]["top_k"]
RERANK_K      = config["retrieval"]["rerank_top_k"]

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is not set. This app expects a hosted Qdrant instance "
        "(Qdrant Cloud free tier works well) - see .env.example for the "
        "variables you need to set before starting the server."
    )

app      = FastAPI(title="Semantic Search over ArXiv ML Papers")
model    = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANKER_NAME)
client   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


class SearchRequest(BaseModel):
    query: str


@app.get("/health")
async def health():
    """Basic liveness check - handy for free-tier hosts that ping to keep the app awake."""
    return {"status": "ok", "model": MODEL_NAME, "collection": COLLECTION}


@app.post("/search")
async def search(request: SearchRequest):
    # Stage 1: bi-encoder retrieval
    query_vec = model.encode(
        [request.query], normalize_embeddings=True, convert_to_numpy=True
    )[0]

    hits = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec.tolist(),
        limit=TOP_K,
    )

    # Stage 2: cross-encoder reranking
    candidates = [(hit.payload["abstract"], hit.payload) for hit in hits]
    pairs      = [[request.query, abstract] for abstract, _ in candidates]
    scores     = reranker.predict(pairs)

    sorted_hits = sorted(
        zip(scores, [payload for _, payload in candidates]),
        key=lambda x: x[0],
        reverse=True
    )[:RERANK_K]

    return {
        "query"  : request.query,
        "results": [
            {"title": p["title"], "score": float(s), "abstract": p["abstract"][:300]}
            for s, p in sorted_hits
        ]
    }
