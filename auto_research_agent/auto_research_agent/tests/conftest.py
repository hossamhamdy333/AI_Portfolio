"""
Shared test setup.

We stub `sentence_transformers` because it pulls in torch (huge download, not
needed to test our own logic — we're not testing the embedding model itself,
just that we call it correctly). Every other dependency (qdrant-client,
langgraph, langchain-groq, ddgs, python-docx, openpyxl, pypdf...) is real.
"""
import sys
import types
import numpy as np


class _FakeSentenceTransformer:
    """Deterministic fake embedder: same text -> same vector, different text ->
    different vector. Real enough to test chunk/index/search wiring without torch."""

    def __init__(self, *args, **kwargs):
        self._dim = 16

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, text, show_progress_bar=False):
        if isinstance(text, str):
            return self._vec(text)
        return np.array([self._vec(t) for t in text])

    def _vec(self, text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self._dim)


# Runs at collection time (before any test module is imported), since
# ingestion/chunking.py imports sentence_transformers at module level.
_fake_module = types.ModuleType("sentence_transformers")
_fake_module.SentenceTransformer = _FakeSentenceTransformer
sys.modules["sentence_transformers"] = _fake_module
