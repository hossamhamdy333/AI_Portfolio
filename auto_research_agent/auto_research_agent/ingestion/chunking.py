"""
Chunking + embedding + local Qdrant storage.
Qdrant runs in local (embedded) mode here — a folder on disk, no server, no signup, no card.
"""
import os
import uuid
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "knowledge_base"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_qdrant(path: str) -> QdrantClient:
    os.makedirs(path, exist_ok=True)
    client = QdrantClient(path=path)
    dim = get_embedder().get_sentence_embedding_dimension()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return client


def chunk_documents(raw_docs: List[Dict], chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """Splits loaded docs into chunks, keeping source/page/modality metadata attached."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in raw_docs:
        for piece in splitter.split_text(doc["text"]):
            chunks.append({
                "text": piece,
                "source": doc["source"],
                "page": doc.get("page"),
                "modality": doc.get("modality", "text"),
            })
    return chunks


def index_chunks(client: QdrantClient, chunks: List[Dict]) -> int:
    if not chunks:
        return 0
    embedder = get_embedder()
    vectors = embedder.encode([c["text"] for c in chunks], show_progress_bar=False)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={"text": c["text"], "source": c["source"], "page": c["page"], "modality": c["modality"]},
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(client: QdrantClient, query: str, k: int = 4) -> List[Dict]:
    embedder = get_embedder()
    vec = embedder.encode(query).tolist()
    hits = client.query_points(collection_name=COLLECTION, query=vec, limit=k).points
    return [
        {"text": h.payload["text"], "source": h.payload["source"],
         "page": h.payload["page"], "score": h.score}
        for h in hits
    ]


def ingest_file(client: QdrantClient, path: str, groq_client=None) -> int:
    """One-call pipeline: load -> chunk -> embed -> store."""
    from ingestion.loaders import load_any
    raw = load_any(path, groq_client=groq_client)
    chunks = chunk_documents(raw)
    return index_chunks(client, chunks)
