"""
Dense retrieval: encode query, search FAISS index, return ranked results.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def load_model(model_name):
    return SentenceTransformer(model_name)


def encode_query(model, query):
    """Encode a single query string into a normalized vector."""
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True)


def search(index, query_embedding, top_k):
    """Search FAISS index, return (scores, indices)."""
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]
