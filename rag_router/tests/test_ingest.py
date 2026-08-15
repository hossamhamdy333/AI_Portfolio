"""Tests for ingest.py.

Uses qdrant_client's in-memory mode (":memory:") and LlamaIndex's built-in
MockEmbedding -- both are real code paths (not mocks of *our* code), just
pointed at local/fake backends instead of Qdrant Cloud and a downloaded
HuggingFace model, so these run without network access. What's under test
-- build once, reconnect without duplicating -- is the same either way,
since QdrantVectorStore doesn't behave differently based on what backend
the client happens to be talking to.
"""

import os

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from ingest import build_domain_index, load_domain_index


@pytest.fixture
def embed_model():
    from llama_index.core.embeddings import MockEmbedding
    return MockEmbedding(embed_dim=384)


@pytest.fixture
def tiny_corpus(tmp_path):
    df = pd.DataFrame([
        {"title": "Football", "text": "Football is a popular sport.", "domain": "sports", "article_id": "sports_0"},
        {"title": "Basketball", "text": "Basketball uses a hoop.", "domain": "sports", "article_id": "sports_1"},
    ])
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    df.to_parquet(data_dir / "sports.parquet")
    return str(data_dir)


def test_build_domain_index_creates_collection(tiny_corpus, embed_model):
    client = QdrantClient(":memory:")
    index, n = build_domain_index("sports", tiny_corpus, client, "test", embed_model)
    assert n == 2
    assert client.count("test_sports").count == 2


def test_build_domain_index_reconnect_does_not_duplicate(tiny_corpus, embed_model):
    client = QdrantClient(":memory:")
    build_domain_index("sports", tiny_corpus, client, "test", embed_model)
    count_after_first = client.count("test_sports").count

    # Calling again should reconnect, not re-embed and re-add.
    build_domain_index("sports", tiny_corpus, client, "test", embed_model)
    count_after_second = client.count("test_sports").count

    assert count_after_first == count_after_second == 2


def test_build_domain_index_force_rebuild_clears_and_reembeds(tiny_corpus, embed_model):
    client = QdrantClient(":memory:")
    build_domain_index("sports", tiny_corpus, client, "test", embed_model)
    build_domain_index("sports", tiny_corpus, client, "test", embed_model, force_rebuild=True)
    # force_rebuild clears the collection first -- should still be 2, not 4.
    # (An earlier version of this function let force_rebuild fall through
    # to the same add-on-top path the normal reconnect case uses, which
    # silently duplicated every vector instead of starting clean.)
    assert client.count("test_sports").count == 2


def test_load_domain_index_raises_if_collection_missing(embed_model):
    client = QdrantClient(":memory:")
    with pytest.raises(RuntimeError, match="No Qdrant collection"):
        load_domain_index("sports", client, "nonexistent", embed_model)


def test_load_domain_index_reconnects_successfully(tiny_corpus, embed_model):
    client = QdrantClient(":memory:")
    build_domain_index("sports", tiny_corpus, client, "test", embed_model)
    index = load_domain_index("sports", client, "test", embed_model)
    assert index is not None
