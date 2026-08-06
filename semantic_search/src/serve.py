"""
FastAPI serving endpoint for semantic search.
Async upload triggers Celery background worker for embedding + upsert.
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
import numpy as np

app    = FastAPI()
model  = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = QdrantClient(host="localhost", port=6333)

COLLECTION = "arxiv_papers"
TOP_K      = 10
RERANK_K   = 3


class SearchRequest(BaseModel):
    query: str


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

    # Sort by score, return top RERANK_K
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
