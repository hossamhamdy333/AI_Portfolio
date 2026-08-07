import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vectorstore import _collection_name  # noqa: E402


def test_collection_name_default_has_no_session_suffix():
    from app.config import settings

    assert _collection_name(None) == settings.chroma_collection
    assert _collection_name() == settings.chroma_collection


def test_collection_name_differs_per_session():
    a = _collection_name("session-one")
    b = _collection_name("session-two")
    assert a != b
    assert a != _collection_name(None)


def test_collection_name_is_sanitized_and_bounded():
    messy_id = "weird/id with spaces!!" + "x" * 100
    name = _collection_name(messy_id)
    assert "/" not in name
    assert " " not in name
    assert "!" not in name
    assert len(name) <= 63


def test_collection_name_same_session_id_is_stable():
    assert _collection_name("abc123") == _collection_name("abc123")
