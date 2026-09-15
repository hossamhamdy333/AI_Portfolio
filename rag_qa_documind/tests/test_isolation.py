"""
Proves that two different users' uploaded documents never mix, using
the exact real ingest_file / vectorstore.query functions from the app -
just with a deterministic fake embedding function swapped in for the
real sentence-transformers model, so this test needs no network access
and no multi-hundred-MB model download.
"""

import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["DATABASE_URL"] = "sqlite:///./test_isolation.db"

import numpy as np


class _FakeEmbeddingFunction:
    """Same interface chromadb expects from a real embedding function:
    __call__(input: list[str]) -> list[ndarray], plus name()/get_config()/
    build_from_config() for chromadb's newer embedding-function registry
    validation. Deterministic and needs no network call - text -> a
    fixed-size vector derived from its hash, so identical text always
    gets an identical vector."""

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def embed_query(self, input):
        return self.__call__(input)

    def _embed(self, text):
        seed = hash(text) % (2**32)
        return np.array([((seed >> i) % 1000) / 1000 for i in range(384)], dtype=np.float32)

    @staticmethod
    def name():
        return "fake_test_embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return _FakeEmbeddingFunction()

    @staticmethod
    def is_legacy():
        return False


def test_one_users_document_is_invisible_to_another_user():
    import app.vectorstore as vectorstore_module
    from app.config import settings

    # settings is a singleton created the first time app.config is
    # imported anywhere in the test session - by the time this test runs,
    # some earlier test file may have already imported it, so setting the
    # CHROMA_DB_DIR env var here would silently do nothing. Patching the
    # already-created settings OBJECT directly (not the env var) is what
    # actually takes effect regardless of import order, and resetting
    # vectorstore's cached client/collections forces a fresh one to be
    # opened against the new directory instead of reusing an old handle.
    with patch.object(settings, "chroma_db_dir", tempfile.mkdtemp()), \
         patch.object(vectorstore_module, "_client", None), \
         patch.object(vectorstore_module, "_collections", {}), \
         patch("app.vectorstore._get_embed_fn", return_value=_FakeEmbeddingFunction()):

        from app.ingest import ingest_file
        from app.vectorstore import query as vectorstore_query

        alice_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        alice_file.write("Alice's contract says the refund window is 30 days.")
        alice_file.close()

        bob_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        bob_file.write("Bob's contract says the refund window is 90 days.")
        bob_file.close()

        try:
            ingest_file(alice_file.name, source_name="alice.txt", session_id="user-1")
            ingest_file(bob_file.name, source_name="bob.txt", session_id="user-2")

            alice_results = vectorstore_query("refund window", session_id="user-1")
            bob_results = vectorstore_query("refund window", session_id="user-2")

            assert len(alice_results) == 1
            assert "Alice" in alice_results[0]["text"]
            assert all("Bob" not in r["text"] for r in alice_results)

            assert len(bob_results) == 1
            assert "Bob" in bob_results[0]["text"]
            assert all("Alice" not in r["text"] for r in bob_results)
        finally:
            os.unlink(alice_file.name)
            os.unlink(bob_file.name)
