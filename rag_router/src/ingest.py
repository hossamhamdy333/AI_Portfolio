"""Build one VectorStoreIndex per topic domain, backed by Qdrant Cloud.

Expects data/processed/<domain>.parquet (built by build_corpus.py, columns:
title, text, domain, article_id) for each domain in DOMAINS. Each domain
gets its own Qdrant collection (configs/config.yaml's
qdrant.collection_prefix + "_" + domain) instead of a locally
disk-persisted index -- the original version used
`storage_context.persist(persist_dir=...)` to write JSON index files
straight into the repo's working directory, which then got swept into git
directly (not DVC) by a blind `git add .` in the notebook, permanently
bloating the repo's history. A hosted collection means nothing about
serving this index depends on the exact Colab session that built it,
and nothing about it needs committing to git or DVC at all.
"""

from pathlib import Path

import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def load_domain_documents(domain, data_dir):
    """Read data/processed/<domain>.parquet into LlamaIndex Document objects."""
    df = pd.read_parquet(Path(data_dir) / f"{domain}.parquet")
    return [
        Document(text=row["text"], id_=row["article_id"], metadata={"title": row["title"], "domain": domain})
        for _, row in df.iterrows()
    ]


def _collection_name(domain, collection_prefix):
    return f"{collection_prefix}_{domain}"


def _collection_exists(qdrant_client, collection_name):
    try:
        return qdrant_client.collection_exists(collection_name)
    except Exception:
        return False


def build_domain_index(domain, data_dir, qdrant_client, collection_prefix, embed_model, force_rebuild=False):
    """Build (or reconnect to) one domain's Qdrant-backed index.

    Skips re-embedding if the collection already has points, same
    force_rebuild guard rag_router's sibling project (impl_langchain) uses
    for its Chroma-backed retriever -- re-running this notebook shouldn't
    silently duplicate every document's vectors in the collection.

    force_rebuild=True actually clears the collection first (recreate,
    not add-on-top) -- an earlier version of this function let
    force_rebuild fall through to the same add_documents() call the
    normal build path uses, which just appended a second copy of every
    vector on top of the existing ones instead of starting clean. Caught
    by the test suite, not by reading the code.
    """
    collection_name = _collection_name(domain, collection_prefix)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

    collection_already_exists = _collection_exists(qdrant_client, collection_name)

    if collection_already_exists and not force_rebuild:
        count = qdrant_client.count(collection_name).count
        if count > 0:
            print(f"'{domain}': collection '{collection_name}' already has {count} points -- reconnecting, not rebuilding.")
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
            return index, count

    if collection_already_exists and force_rebuild:
        qdrant_client.delete_collection(collection_name)
        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

    documents = load_domain_documents(domain, data_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    return index, len(documents)


def build_all_indexes(data_dir, domains, qdrant_client, collection_prefix, embed_model, force_rebuild=False):
    """Build (or reconnect to) indexes for every domain. Returns dict of domain -> index."""
    indexes = {}
    for domain in domains:
        index, n_docs = build_domain_index(domain, data_dir, qdrant_client, collection_prefix, embed_model, force_rebuild)
        indexes[domain] = index
        print(f"'{domain}': {n_docs} documents")
    return indexes


def load_domain_index(domain, qdrant_client, collection_prefix, embed_model):
    """Reconnect to an already-built domain index, no re-embedding."""
    collection_name = _collection_name(domain, collection_prefix)
    if not _collection_exists(qdrant_client, collection_name):
        raise RuntimeError(
            f"No Qdrant collection '{collection_name}' found -- run "
            "notebooks/02_ingest_and_router.ipynb first to build it."
        )
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def load_all_indexes(domains, qdrant_client, collection_prefix, embed_model):
    return {domain: load_domain_index(domain, qdrant_client, collection_prefix, embed_model) for domain in domains}
