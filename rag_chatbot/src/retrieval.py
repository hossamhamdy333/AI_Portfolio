def search_chunks(query, qdrant_client, collection_name, embedder, top_k=10):
    query_vector = embedder.encode([query])[0].tolist()
    results = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
    return [
        {"chunk_id": r.payload["chunk_id"], "article_id": r.payload["article_id"], "chunk_text": r.payload["chunk_text"], "score": r.score}
        for r in results
    ]


def rerank_chunks(query, candidates, reranker, top_k=3):
    pairs = [[query, c["chunk_text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(score)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
