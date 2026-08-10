from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def build_collection(qdrant_client, collection_name, embedding_dim):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )


def upsert_chunks(qdrant_client, collection_name, chunk_df, embedder, batch_size=64):
    chunk_texts = chunk_df["chunk_text"].tolist()
    chunk_vectors = embedder.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True)

    points = [
        PointStruct(
            id=i,
            vector=chunk_vectors[i].tolist(),
            payload={
                "chunk_id": row["chunk_id"],
                "article_id": row["article_id"],
                "chunk_text": row["chunk_text"],
            }
        )
        for i, (_, row) in enumerate(chunk_df.iterrows())
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    return len(points)
