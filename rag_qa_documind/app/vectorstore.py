"""
Vector store layer. Uses ChromaDB (persisted to disk) with a local
sentence-transformers embedding function, so ingestion/retrieval never
needs an API key or network call for embeddings.
"""
import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_db_dir)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
<<<<<<< HEAD
            model_name=settings.embedding_model
=======
            model_name=settings.embedding_model,
            model_kwargs={"low_cpu_mem_usage": False},
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
        )
        _collection = get_client().get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def query(question: str, top_k: int = None):
    """Return top_k most relevant chunks for a question."""
    top_k = top_k or settings.top_k
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, count),
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source"), "score": 1 - dist})
    return hits


def reset_collection():
    """Delete and recreate the collection (useful for a clean re-ingest)."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    _collection = None
    return get_collection()
