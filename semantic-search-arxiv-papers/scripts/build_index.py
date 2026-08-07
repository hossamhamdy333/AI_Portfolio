"""
One-time (or re-run-when-data-changes) script that populates the persistent
Qdrant Cloud collection used by src/serve.py and streamlit_app.py.

This was the missing piece in the original project: notebook 04 only ever
built an in-memory Qdrant collection for offline evaluation, so there was
no actual path to get vectors into a real, persistent, queryable index.

Usage:
    python scripts/build_index.py

Requires QDRANT_URL / QDRANT_API_KEY in the environment (see .env.example).
Reads everything else (model name, chunking, embedding dim) from
configs/config.yaml so this can never drift from what's benchmarked.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingest import chunk_text, embed_texts, upsert_to_qdrant  # noqa: E402
from retrieval import load_model  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError(
            "QDRANT_URL is not set. Create a free cluster at https://cloud.qdrant.io "
            "and set QDRANT_URL / QDRANT_API_KEY - see .env.example."
        )

    data_path = ROOT / "data" / "arxiv_subset.parquet"
    if not data_path.exists():
        raise RuntimeError(
            f"{data_path} not found. Run `dvc pull` first (see README for the "
            "Google Drive DVC remote setup)."
        )

    print(f"Loading data from {data_path} ...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} rows")

    model_name    = config["retrieval"]["model_name"]
    embedding_dim = config["retrieval"]["embedding_dim"]
    collection    = config["qdrant"]["collection_name"]
    chunk_size    = config["chunking"]["chunk_size"]
    overlap       = config["chunking"]["overlap"]

    print(f"Loading embedding model {model_name} ...")
    model = load_model(model_name)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        print(f"Collection '{collection}' already exists - recreating it.")
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection}' (dim={embedding_dim}, cosine).")

    # Build chunks + payloads
    all_chunks, all_payloads = [], []
    for _, row in df.iterrows():
        chunks = chunk_text(row["abstract"], chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_payloads.append({
                "title": row["title"],
                "abstract": chunk,
            })

    print(f"Embedding {len(all_chunks):,} chunks ...")
    embeddings = embed_texts(model, all_chunks)

    print("Upserting to Qdrant Cloud ...")
    batch_size = 256
    total = 0
    for start in range(0, len(embeddings), batch_size):
        end = start + batch_size
        n = upsert_to_qdrant(
            client, collection,
            embeddings[start:end], all_payloads[start:end],
            start_id=start,
        )
        total += n
        print(f"  {total:,} / {len(embeddings):,}", end="\r")

    print(f"\nDone. {total:,} vectors indexed in '{collection}'.")


if __name__ == "__main__":
    main()
