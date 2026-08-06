import shutil
import tempfile

import pytest

from ingestion.chunking import chunk_documents, get_qdrant, index_chunks, search


def test_chunk_documents_splits_long_text_and_keeps_metadata():
    raw_docs = [
        {"text": "sentence one. " * 200, "source": "report.pdf", "page": 3, "modality": "text"},
    ]
    chunks = chunk_documents(raw_docs, chunk_size=200, overlap=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c["source"] == "report.pdf"
        assert c["page"] == 3
        assert c["modality"] == "text"
        assert len(c["text"]) <= 250  # allows a little slack over chunk_size


def test_chunk_documents_short_text_stays_one_chunk():
    raw_docs = [{"text": "short doc", "source": "a.docx", "page": None, "modality": "text"}]
    chunks = chunk_documents(raw_docs)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short doc"


def test_chunk_documents_empty_list_returns_empty():
    assert chunk_documents([]) == []


@pytest.fixture
def qdrant_client():
    tmp_dir = tempfile.mkdtemp()
    client = get_qdrant(tmp_dir)
    yield client
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_index_and_search_roundtrip(qdrant_client):
    chunks = [
        {"text": "cats are small furry animals", "source": "animals.txt", "page": 1, "modality": "text"},
        {"text": "the stock market fell sharply today", "source": "finance.txt", "page": 1, "modality": "text"},
    ]
    n = index_chunks(qdrant_client, chunks)
    assert n == 2

    results = search(qdrant_client, "tell me about cats", k=2)
    assert len(results) == 2
    for r in results:
        assert "text" in r and "source" in r and "score" in r


def test_index_chunks_empty_list_returns_zero(qdrant_client):
    assert index_chunks(qdrant_client, []) == 0


def test_search_empty_collection_returns_empty_list(qdrant_client):
    assert search(qdrant_client, "anything") == []
