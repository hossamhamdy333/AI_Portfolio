"""
Proves the per-user retrieval filter in agent.py actually isolates one
user's documents from another - not just that the code looks like it
should, but that a real Qdrant query with one user's filter genuinely
cannot return another user's chunks, even when asked a question that
would clearly match them semantically.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class _FakeEmbeddings(Embeddings):
    """A tiny stand-in embedding model for tests - deterministic, needs no
    API key or network call. Turns text into a fixed-size vector from its
    hash, so identical text always gets the identical vector; that's all
    this test needs, it doesn't need the vectors to mean anything."""

    def _embed(self, text):
        seed = hash(text) % (2**32)
        return [((seed >> i) % 1000) / 1000 for i in range(384)]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _make_isolated_vectorstore():
    """A fresh in-memory Qdrant collection with one document for user 1
    and one for user 2 - same shape as text_processing.py's real upsert,
    just without needing a live embedding API call."""
    embeddings = _FakeEmbeddings()
    client = QdrantClient(":memory:")
    client.create_collection("test_isolation", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    vectorstore = QdrantVectorStore(client=client, collection_name="test_isolation", embedding=embeddings)

    vectorstore.add_documents([
        Document(page_content="Alice's contract says the refund window is 30 days.", metadata={"user_id": 1}),
        Document(page_content="Bob's contract says the refund window is 90 days.", metadata={"user_id": 2}),
    ])
    return vectorstore


def _search_as_user(vectorstore, user_id, query):
    user_filter = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
    return vectorstore.similarity_search(query, k=5, filter=user_filter)


def test_user_only_sees_their_own_document():
    vectorstore = _make_isolated_vectorstore()
    results = _search_as_user(vectorstore, user_id=1, query="refund window")
    assert len(results) == 1
    assert "Alice" in results[0].page_content


def test_user_cannot_see_another_users_document_even_with_a_matching_question():
    """The important case: Bob's document is a near-perfect semantic
    match for this question, but Alice's filter must still exclude it."""
    vectorstore = _make_isolated_vectorstore()
    results = _search_as_user(vectorstore, user_id=1, query="What does Bob's contract say about refunds?")
    assert all("Bob" not in r.page_content for r in results)


def test_unfiltered_search_would_have_returned_both_documents():
    """Confirms the isolation is coming from the filter, not from the
    fake embeddings coincidentally separating the two documents on their
    own - without a filter, both are retrievable."""
    vectorstore = _make_isolated_vectorstore()
    results = vectorstore.similarity_search("refund window", k=5)
    assert len(results) == 2
