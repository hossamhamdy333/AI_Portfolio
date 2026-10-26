import os

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")

import numpy as np

import config


class _FakeModel:
    def embed(self, texts):
        for text in texts:
            yield np.full(384, float(len(text)), dtype=np.float32)


def _local():
    emb = config.LocalEmbeddings("fake-model")
    emb._model = _FakeModel()
    return emb


def test_documents_come_back_as_plain_float_lists_in_order():
    vectors = _local().embed_documents(["a", "abc"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0][0] == 1.0 and vectors[1][0] == 3.0
    assert isinstance(vectors[0], list)


def test_query_is_one_vector():
    vector = _local().embed_query("hello")
    assert len(vector) == 384 and vector[0] == 5.0


def test_default_backend_needs_no_hf_token():
    assert config.settings.EMBEDDING_BACKEND == "local"
    assert isinstance(config.embeddings._inner, config.LocalEmbeddings)
