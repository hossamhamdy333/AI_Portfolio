import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "impl_vanilla"))

from shared.metrics import mean_mrr_ndcg
from src.retrieval import rerank_chunks, search_chunks

COLLECTION = "gate"
TOLERANCE = 0.02


def build_index(chunks, embedder):
    client = QdrantClient(":memory:")
    dim = embedder.get_sentence_embedding_dimension()
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    vectors = embedder.encode(chunks["chunk_text"].tolist(), batch_size=32)
    points = []
    for i, row in enumerate(chunks.itertuples()):
        payload = {
            "chunk_id": str(row.chunk_id),
            "article_id": str(row.article_id),
            "chunk_text": row.chunk_text,
        }
        points.append(PointStruct(id=i, vector=vectors[i].tolist(), payload=payload))
    client.upsert(COLLECTION, points=points)
    return client


def run(cfg, chunks, questions, embedder, reranker):
    client = build_index(chunks, embedder)

    top_k = cfg["rag"]["top_k_retrieve"]
    top_k_rerank = cfg["rag"]["top_k_rerank"]
    log = []
    for row in questions.itertuples():
        found = search_chunks(row.question, client, COLLECTION, embedder, top_k=top_k)
        top = rerank_chunks(row.question, found, reranker, top_k=top_k_rerank)
        log.append({
            "ranked_ids": [c["article_id"] for c in top],
            "correct_id": str(row.article_id),
        })

    scores = mean_mrr_ndcg(log, k=10)
    hit_rate = float(np.mean([r["correct_id"] in r["ranked_ids"] for r in log]))
    return {
        "mrr": round(scores["mrr"], 4),
        "ndcg_at_10": round(scores["ndcg_at_k"], 4),
        "hit_rate": round(hit_rate, 4),
    }


def failed_metrics(scores, baseline, tolerance=TOLERANCE):
    return [name for name, base in baseline.items() if scores[name] < base - tolerance]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    chunks = pd.read_parquet(HERE / "chunks.parquet")
    questions = pd.read_parquet(HERE / "questions.parquet")
    print("chunks:", len(chunks), "questions:", len(questions))

    from sentence_transformers import CrossEncoder, SentenceTransformer
    embedder = SentenceTransformer(cfg["retrieval"]["model_name"])
    reranker = CrossEncoder(cfg["reranker"]["model_name"])

    scores = run(cfg, chunks, questions, embedder, reranker)
    print("scores:", json.dumps(scores))

    baseline_path = HERE / "baseline.json"
    if args.write_baseline:
        baseline_path.write_text(json.dumps(scores, indent=2))
        print("baseline written")
        return

    if not baseline_path.exists():
        print("No eval/baseline.json yet. Copy the scores line above into it.")
        sys.exit(1)

    baseline = json.loads(baseline_path.read_text())
    for name, base in baseline.items():
        print(f"{name}: now {scores[name]} | baseline {base}")

    failed = failed_metrics(scores, baseline)
    if failed:
        print("GATE FAILED, dropped more than", TOLERANCE, "on:", ", ".join(failed))
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
