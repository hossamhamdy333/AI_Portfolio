"""
Retrieval: FAISS and Qdrant search with SBERT encoding.
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import faiss
import numpy as np


def load_model(model_name):
    return SentenceTransformer(model_name)


def encode_query(model, query):
    """Encode a single query string into a normalized vector."""
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True)


def search_faiss(index, query_embedding, top_k):
    """Search FAISS index, return (scores, indices)."""
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]


def search_qdrant(client, collection_name, query_vector, top_k):
    """Search Qdrant collection, return list of payloads."""
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
    )
    return [hit.payload for hit in results]
