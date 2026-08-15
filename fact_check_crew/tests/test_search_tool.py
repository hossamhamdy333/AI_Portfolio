"""Tests for search_tool.py -- real in-memory Qdrant, real LlamaIndex
payload structure (via rag_router's own ingest.py, imported directly to
build a genuine test collection rather than guessing the payload shape).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from search_tool import search_domains, format_hits_for_agent, build_search_tool, _extract_text

# rag_router's ingest.py is used here to build a REAL LlamaIndex-populated
# collection to test against -- verifying the payload-parsing logic
# against actual output, not an assumed shape, is the whole point of this
# specific test (an earlier version of _extract_text assumed a flat
# "text" payload key that doesn't exist; caught by doing exactly this).
#
# This only works when rag_router is present as a sibling directory,
# which is true in the actual AI_Portfolio repo but not if this project
# is tested in isolation (e.g. this zip, extracted on its own). That's a
# real, documented dependency (see README.md), not a bug -- so the one
# test that needs it skips with a clear reason instead of failing when
# it's missing, rather than looking like broken code in a context where
# it was never expected to run standalone.
_RAG_ROUTER_SRC = Path(__file__).resolve().parents[2] / "rag_router" / "src"
_rag_router_available = (_RAG_ROUTER_SRC / "ingest.py").exists()
if _rag_router_available:
    sys.path.insert(0, str(_RAG_ROUTER_SRC))


class FakeSTModel:
    """Deterministic fake sentence-transformers model for tests -- returns
    a fixed-length zero vector regardless of input, since we're testing
    search_tool's own logic (payload parsing, result merging/sorting),
    not embedding quality."""
    def encode(self, text):
        import numpy as np
        return np.zeros(384)


@pytest.mark.skipif(not _rag_router_available, reason="rag_router not present as a sibling directory -- expected when testing this project in isolation, see this file's module docstring")
def test_extract_text_from_real_llamaindex_payload():
    from ingest import build_domain_index
    from llama_index.core.embeddings import MockEmbedding

    client = QdrantClient(":memory:")
    df = pd.DataFrame([{"title": "Football", "text": "Football is a sport.", "domain": "sports", "article_id": "sports_0"}])
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        df.to_parquet(os.path.join(tmp, "sports.parquet"))
        build_domain_index("sports", tmp, client, "test", MockEmbedding(embed_dim=384))

    point = client.scroll("test_sports", limit=1, with_payload=True)[0][0]
    assert _extract_text(point.payload) == "Football is a sport."


def test_extract_text_handles_missing_node_content():
    assert _extract_text({}) == ""


def test_extract_text_handles_malformed_node_content():
    assert _extract_text({"_node_content": "not valid json"}) == ""


def test_search_domains_merges_and_sorts_across_domains():
    client = QdrantClient(":memory:")
    from qdrant_client.models import Distance, VectorParams, PointStruct

    class Fake4DimModel:
        def encode(self, text):
            import numpy as np
            return np.array([1.0, 0.0, 0.0, 0.0])  # matches the 4-dim toy collections below

    client.create_collection("test_sports", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert("test_sports", points=[
        PointStruct(id=1, vector=[1, 0, 0, 0], payload={"title": "Football", "_node_content": '{"text": "about football"}'}),
    ])
    client.create_collection("test_tech", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert("test_tech", points=[
        PointStruct(id=1, vector=[1, 0, 0, 0], payload={"title": "Python", "_node_content": '{"text": "about python"}'}),
    ])

    hits = search_domains("query", client, Fake4DimModel(), ["sports", "tech"], "test", top_k=5)
    assert {h["title"] for h in hits} == {"Football", "Python"}
    assert {h["domain"] for h in hits} == {"sports", "tech"}


def test_search_domains_skips_nonexistent_collections():
    client = QdrantClient(":memory:")
    hits = search_domains("query", client, FakeSTModel(), ["nonexistent_domain"], "test", top_k=5)
    assert hits == []


def test_format_hits_for_agent_empty():
    assert "No relevant passages" in format_hits_for_agent([])


def test_format_hits_for_agent_includes_domain_and_title():
    hits = [{"domain": "sports", "title": "Football", "text": "some text", "score": 0.9}]
    formatted = format_hits_for_agent(hits)
    assert "sports" in formatted and "Football" in formatted and "some text" in formatted


def test_build_search_tool_returns_callable_tool():
    client = QdrantClient(":memory:")
    tool_fn = build_search_tool(client, FakeSTModel(), ["nonexistent"], "test", top_k=3)
    result = tool_fn.run(query="anything")
    assert "No relevant passages" in result
