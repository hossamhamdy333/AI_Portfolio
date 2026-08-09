"""
Vector store layer. Uses ChromaDB (persisted to disk) with a local
sentence-transformers embedding function, so ingestion/retrieval never
needs an API key or network call for embeddings.

Collections are optionally namespaced per session_id so that concurrent
visitors to a shared public deployment don't see each other's uploaded
documents. Pass session_id=None (the default) to use the single shared
collection -- this is what CLI tools like scripts/run_ingest.py do, and
what every function here did before session support was added.
"""
import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

_client = None
_embed_fn = None
_collections = {}  # collection_name -> Chroma collection object


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_db_dir)
    return _client


def _get_embed_fn():
    """Lazily build the embedding function once and reuse it for every
    collection, instead of reloading the sentence-transformers model per
    session."""
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            model_kwargs={"low_cpu_mem_usage": False},
        )
    return _embed_fn


def _collection_name(session_id: str = None) -> str:
    """Build the Chroma collection name for a given session.

    session_id=None -> the single shared/default collection (unchanged
    behavior for local dev, Docker, and the CLI ingestion script).
    session_id="abc123" -> "<base>_sess_abc123", isolated from every
    other session so concurrent public users never see each other's docs.
    """
    base = settings.chroma_collection
    if not session_id:
        return base
    safe_id = "".join(ch for ch in session_id if ch.isalnum())[:32]
    return f"{base}_sess_{safe_id}" if safe_id else base


def get_collection(session_id: str = None):
    name = _collection_name(session_id)
    if name not in _collections:
        _collections[name] = get_client().get_or_create_collection(
            name=name,
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[name]


def query(question: str, top_k: int = None, session_id: str = None):
    """Return top_k most relevant chunks for a question."""
    top_k = top_k or settings.top_k
    collection = get_collection(session_id)
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


def reset_collection(session_id: str = None):
    """Delete and recreate the collection (useful for a clean re-ingest)."""
    name = _collection_name(session_id)
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    _collections.pop(name, None)
    return get_collection(session_id)
