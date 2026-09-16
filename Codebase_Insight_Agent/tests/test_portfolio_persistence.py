"""
Tests for the persistence logic added to portfolio.py: index_exists,
build_all_indexes' skip-if-already-built behavior, and load_all_indexes'
clear error when the index hasn't been provisioned yet.

Uses a real (if in-memory/ephemeral) Qdrant client and a fake embedding
function - no real Gemini network calls needed, since these tests are
about the persistence bookkeeping logic, not embedding quality.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GOOGLE_API_KEY", "test")

import pytest
from qdrant_client import QdrantClient

import config
import portfolio


@pytest.fixture(autouse=True)
def isolated_qdrant_client():
    """A fresh in-memory Qdrant client per test, and configure_llama_index
    mocked out entirely - these tests never call a real embedding/LLM
    API, they're purely about the exists/skip/load bookkeeping."""
    client = QdrantClient(":memory:")
    with patch("portfolio.get_qdrant_client", return_value=client), \
         patch("portfolio.configure_llama_index"):
        yield client


def _fake_build_index(project_name, client=None):
    """A stand-in for portfolio.build_index that doesn't fetch a real
    README or call a real embedding API - just creates a real Qdrant
    collection with one point, which is all index_exists/load_index
    care about. Matches the real build_index's drop-before-create
    behavior, since that's exactly the bug this test file also checks
    for (test_force_rebuild_does_not_accumulate_duplicate_points)."""
    from qdrant_client.models import VectorParams, Distance, PointStruct

    client = client or portfolio.get_qdrant_client()
    name = portfolio.collection_name(project_name)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=8, distance=Distance.COSINE))
    client.upsert(name, points=[PointStruct(id=1, vector=[0.1] * 8, payload={"project": project_name})])


def test_index_exists_is_false_before_building():
    assert portfolio.index_exists("some_project") is False


def test_index_exists_is_true_after_building():
    _fake_build_index("some_project")
    assert portfolio.index_exists("some_project") is True


def test_build_all_indexes_skips_already_built_projects():
    with patch("portfolio.build_index", side_effect=_fake_build_index) as mock_build, \
         patch("portfolio.load_index", return_value="fake-index-object"), \
         patch("config.PROJECTS", ["project_a", "project_b"]):
        portfolio.build_all_indexes()
        assert mock_build.call_count == 2

        # second call: both already exist, neither should be rebuilt
        mock_build.reset_mock()
        portfolio.build_all_indexes(force=False)
        assert mock_build.call_count == 0


def test_build_all_indexes_force_true_rebuilds_everything():
    with patch("portfolio.build_index", side_effect=_fake_build_index) as mock_build, \
         patch("portfolio.load_index", return_value="fake-index-object"), \
         patch("config.PROJECTS", ["project_a", "project_b"]):
        portfolio.build_all_indexes()
        mock_build.reset_mock()

        portfolio.build_all_indexes(force=True)
        assert mock_build.call_count == 2  # rebuilt despite already existing


def test_load_all_indexes_raises_a_clear_error_when_nothing_is_provisioned():
    with patch("config.PROJECTS", ["never_built_project"]):
        with pytest.raises(RuntimeError, match="never_built_project"):
            portfolio.load_all_indexes()


def test_load_all_indexes_error_names_only_the_missing_projects():
    with patch("portfolio.build_index", side_effect=_fake_build_index), \
         patch("config.PROJECTS", ["built_project", "missing_project"]):
        portfolio.build_index("built_project")

        with pytest.raises(RuntimeError) as exc_info:
            portfolio.load_all_indexes()

        assert "missing_project" in str(exc_info.value)
        assert "built_project" not in str(exc_info.value)


def test_rebuilding_does_not_accumulate_duplicate_points(isolated_qdrant_client):
    """The actual bug found and fixed: rebuilding a project's index used
    to silently ADD new points alongside the old ones rather than
    replacing them - verified directly (1 point became 2, not a clean
    1-for-1 swap) before this fix. This test calls the REAL
    portfolio.build_index (not the _fake_ stand-in used elsewhere in this
    file), with only get_readme and the embedding step faked, so the
    collection-dropping behavior itself is genuinely exercised."""
    from llama_index.core import Settings
    from llama_index.core.embeddings import BaseEmbedding

    class _FakeEmbedding(BaseEmbedding):
        def _get_query_embedding(self, query):
            return [0.1] * 8

        def _get_text_embedding(self, text):
            return [0.1] * 8

        async def _aget_query_embedding(self, query):
            return [0.1] * 8

    Settings.embed_model = _FakeEmbedding(embed_dim=8)

    with patch("portfolio.get_readme", return_value="original readme content"):
        portfolio.build_index("some_project", client=isolated_qdrant_client)
    count_after_first_build = isolated_qdrant_client.count(portfolio.collection_name("some_project")).count

    with patch("portfolio.get_readme", return_value="completely different, updated readme content"):
        portfolio.build_index("some_project", client=isolated_qdrant_client)
    count_after_rebuild = isolated_qdrant_client.count(portfolio.collection_name("some_project")).count

    assert count_after_rebuild == count_after_first_build  # replaced, not accumulated
