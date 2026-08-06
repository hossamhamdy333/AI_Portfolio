"""
Ingest pipeline: text -> chunks -> embeddings -> Qdrant upsert.
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import numpy as np


def chunk_text(text, chunk_size=256, overlap=32):
    """Split text into overlapping word chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(model, texts, batch_size=64):
    """Encode a list of texts into normalized vectors."""
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def upsert_to_qdrant(client, collection_name, embeddings, payloads, start_id=0):
    """Upsert embeddings + metadata to Qdrant."""
    points = [
        PointStruct(id=start_id + i, vector=emb.tolist(), payload=payload)
        for i, (emb, payload) in enumerate(zip(embeddings, payloads))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)
