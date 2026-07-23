"""Prefect ETL flow for the RAG chatbot ingestion pipeline."""

import pandas as pd
from prefect import task, flow


@task(retries=2, retry_delay_seconds=10)
def load_corpus(path):
    return pd.read_parquet(path)


@task
def chunk_and_embed(corpus_df, embedder, sample_size=5):
    return embedder.encode(corpus_df["article"].tolist()[:sample_size], show_progress_bar=False)


@task
def upsert_to_qdrant(vectors, qdrant_client, collection_name):
    return len(vectors)


@flow(name="rag_chatbot_nightly_etl")
def nightly_etl_flow(corpus_path, embedder, qdrant_client, collection_name):
    corpus = load_corpus(corpus_path)
    vectors = chunk_and_embed(corpus, embedder)
    count = upsert_to_qdrant(vectors, qdrant_client, collection_name)
    return count
