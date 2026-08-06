"""Build one VectorStoreIndex per topic domain.

Expects data/processed/<domain>.parquet (built by build_corpus.py, columns:
title, text, domain, article_id) for each domain in DOMAINS. Each domain gets
its own persisted index directory so the router can load them independently.
"""

from pathlib import Path

import pandas as pd
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

DOMAINS = ["sports", "tech", "history", "english_literature"]


def load_domain_documents(domain, data_dir):
    """Read data/processed/<domain>.parquet into LlamaIndex Document objects."""
    df = pd.read_parquet(Path(data_dir) / f"{domain}.parquet")
    return [
        Document(text=row["text"], id_=row["article_id"], metadata={"title": row["title"], "domain": domain})
        for _, row in df.iterrows()
    ]


def build_domain_index(domain, data_dir, index_dir, embed_model):
    """Build and persist a VectorStoreIndex for one domain."""
    Settings.embed_model = embed_model

    documents = load_domain_documents(domain, data_dir)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(Path(index_dir) / domain))
    return index, len(documents)


def build_all_indexes(data_dir, index_dir, embed_model):
    """Build indexes for every domain in DOMAINS. Returns dict of domain -> index."""
    indexes = {}
    for domain in DOMAINS:
        index, n_docs = build_domain_index(domain, data_dir, index_dir, embed_model)
        indexes[domain] = index
        print(f"Built index for '{domain}': {n_docs} documents")
    return indexes


def load_domain_index(domain, index_dir, embed_model):
    """Load a previously persisted index for one domain."""
    Settings.embed_model = embed_model
    storage_context = StorageContext.from_defaults(persist_dir=str(Path(index_dir) / domain))
    return load_index_from_storage(storage_context)


def load_all_indexes(index_dir, embed_model):
    return {domain: load_domain_index(domain, index_dir, embed_model) for domain in DOMAINS}
