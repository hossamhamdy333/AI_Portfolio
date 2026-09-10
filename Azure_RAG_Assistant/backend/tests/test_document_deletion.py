"""
Proves delete_document_chunks() (text_processing.py) removes exactly the
target document's vector chunks and nothing else - specifically the case
that motivated tagging chunks with document_id instead of just filename:
two different uploads that happen to share a filename must not bleed into
each other on delete.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from text_processing import delete_document_chunks
from config import settings


def _seeded_client():
    """A fresh in-memory Qdrant collection with three chunks across two
    documents for the same user, plus one chunk for a different user -
    same collection name the real app uses, so delete_document_chunks()
    (which reads settings.QDRANT_COLLECTION_NAME internally) finds it."""
    client = QdrantClient(":memory:")
    client.create_collection(
        settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    client.upsert(settings.QDRANT_COLLECTION_NAME, points=[
        PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"metadata": {"user_id": 1, "document_id": 100, "source": "report.pdf"}}),
        PointStruct(id=2, vector=[0.1, 0.2, 0.3, 0.4], payload={"metadata": {"user_id": 1, "document_id": 100, "source": "report.pdf"}}),
        # A second upload that happens to share the exact same filename as
        # document_id=100 - this is exactly the case filename-based
        # deletion would have gotten wrong.
        PointStruct(id=3, vector=[0.1, 0.2, 0.3, 0.4], payload={"metadata": {"user_id": 1, "document_id": 200, "source": "report.pdf"}}),
        # A different user's document entirely.
        PointStruct(id=4, vector=[0.1, 0.2, 0.3, 0.4], payload={"metadata": {"user_id": 2, "document_id": 300, "source": "report.pdf"}}),
    ])
    return client


def test_deletes_only_the_target_documents_chunks():
    client = _seeded_client()
    delete_document_chunks(client, document_id=100, user_id=1)

    remaining = client.scroll(settings.QDRANT_COLLECTION_NAME, limit=10)[0]
    remaining_document_ids = {p.payload["metadata"]["document_id"] for p in remaining}

    assert 100 not in remaining_document_ids
    assert remaining_document_ids == {200, 300}


def test_does_not_touch_a_different_document_with_the_same_filename():
    """The specific regression this design prevents: document_id=200 has
    the identical filename as the deleted document_id=100, and must
    survive untouched."""
    client = _seeded_client()
    delete_document_chunks(client, document_id=100, user_id=1)

    remaining = client.scroll(settings.QDRANT_COLLECTION_NAME, limit=10)[0]
    assert any(p.payload["metadata"]["document_id"] == 200 for p in remaining)


def test_does_not_touch_another_users_document():
    client = _seeded_client()
    delete_document_chunks(client, document_id=100, user_id=1)

    remaining = client.scroll(settings.QDRANT_COLLECTION_NAME, limit=10)[0]
    assert any(p.payload["metadata"]["document_id"] == 300 for p in remaining)


def test_deleting_a_nonexistent_document_id_is_a_harmless_no_op():
    client = _seeded_client()
    before = client.count(settings.QDRANT_COLLECTION_NAME).count
    delete_document_chunks(client, document_id=999999, user_id=1)
    after = client.count(settings.QDRANT_COLLECTION_NAME).count
    assert before == after
