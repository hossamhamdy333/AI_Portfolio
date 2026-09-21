"""
Covers the reliability fixes: retrying flaky embedding calls, waking a paused
database, always answering in JSON, keeping the server responsive while a slow
chat/upload runs, and making sure Qdrant's filter indexes always exist.
"""

import asyncio
import inspect
import os
import sqlite3
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_resilience.db")

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

import agent
import database
import main
from config import ResilientEmbeddings
from database import init_db

init_db()  # TestClient(app) without `with` doesn't run the lifespan


# ------------------------------------------------------------ embeddings

class _FakeInner:
    """Stands in for the Hugging Face endpoint. Each text -> [len(text)]."""

    def __init__(self, fail_first=0):
        self.calls = 0
        self.fail_first = fail_first

    def _maybe_fail(self):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("503: model is loading")

    def embed_documents(self, texts):
        self._maybe_fail()
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        self._maybe_fail()
        return [float(len(text))]


def _embedder(inner, **kw):
    return ResilientEmbeddings(inner, base_delay=0.001, **kw)


def test_embed_documents_keeps_order_across_many_parallel_batches():
    texts = ["x" * i for i in range(1, 101)]  # 100 texts -> 4 batches of 32
    result = _embedder(_FakeInner(), batch_size=32).embed_documents(texts)
    assert result == [[float(len(t))] for t in texts]


def test_embed_documents_splits_into_batches():
    inner = _FakeInner()
    _embedder(inner, batch_size=10).embed_documents(["a"] * 25)
    assert inner.calls == 3


def test_embed_documents_empty_list():
    assert _embedder(_FakeInner()).embed_documents([]) == []


def test_transient_failures_are_retried():
    inner = _FakeInner(fail_first=2)
    assert _embedder(inner, attempts=4).embed_query("abcd") == [4.0]
    assert inner.calls == 3


def test_a_failing_batch_is_retried_without_losing_order():
    inner = _FakeInner(fail_first=1)
    texts = ["x" * i for i in range(1, 70)]
    assert _embedder(inner, batch_size=32).embed_documents(texts) == [[float(len(t))] for t in texts]


def test_gives_up_after_the_configured_attempts():
    inner = _FakeInner(fail_first=99)
    with pytest.raises(RuntimeError):
        _embedder(inner, attempts=3).embed_query("hi")
    assert inner.calls == 3


# ------------------------------------------------------------ database wake-up

def _engine_with_flaky_connect(fail_times, message):
    calls = {"n": 0}
    real_connect = sqlite3.connect

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise sqlite3.OperationalError(message)
        return real_connect(*a, **k)

    engine = create_engine("sqlite://")
    engine.dialect.dbapi = types.SimpleNamespace(**{**vars(sqlite3), "connect": flaky, "Error": sqlite3.Error})
    return engine, calls


def test_connect_is_retried_while_the_database_wakes_up():
    engine, calls = _engine_with_flaky_connect(2, "Database is not currently available (40613)")
    database.install_connect_retry(engine, attempts=5, delay=0.001)
    with engine.connect() as conn:
        assert conn.execute(text("select 1")).scalar() == 1
    assert calls["n"] == 3


def test_a_wrong_password_is_not_retried():
    engine, calls = _engine_with_flaky_connect(99, "Login failed for user 'x' (18456)")
    database.install_connect_retry(engine, attempts=5, delay=0.001)
    with pytest.raises(Exception):
        engine.connect()
    assert calls["n"] == 1


# ------------------------------------------------------------ JSON errors

def test_unexpected_errors_come_back_as_json():
    async def boom():
        raise RuntimeError("kaboom")

    main.app.add_api_route("/__boom", boom)
    resp = TestClient(main.app, raise_server_exceptions=False).get("/__boom")
    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    assert "detail" in resp.json()


# ------------------------------------------------------------ chat / upload

def _token(client, email):
    r = client.post("/auth/register", json={"email": email, "password": "a-real-password-123"})
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def test_chat_and_upload_run_in_worker_threads_not_on_the_event_loop():
    # A plain `def` endpoint is run in a thread pool by FastAPI; an `async def`
    # one that does slow blocking work freezes the whole server meanwhile.
    assert not inspect.iscoroutinefunction(main.chat)
    assert not inspect.iscoroutinefunction(main.upload_document)


def test_the_server_stays_responsive_while_a_slow_chat_runs(monkeypatch):
    monkeypatch.setattr(main, "run_agent", lambda q, user_id: (time.sleep(1.0), "done")[1])

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            reg = await c.post("/auth/register", json={"email": "slowchat@example.com", "password": "a-real-password-123"})
            headers = {"Authorization": "Bearer " + reg.json()["access_token"]}
            t0 = time.perf_counter()
            chat = asyncio.create_task(c.post("/chat", json={"query": "hello there"}, headers=headers))
            await asyncio.sleep(0.2)  # the health check is *meant* to go out 0.2s after the chat starts
            health = await c.get("/health")
            # Measured from when it was meant to be sent: if the chat froze the
            # server, even the wake-up from sleep is late, and this shows it.
            health_took = time.perf_counter() - (t0 + 0.2)
            chat_resp = await chat
            return health.status_code, health_took, chat_resp

    status, health_took, chat_resp = asyncio.run(scenario())
    assert status == 200
    assert health_took < 0.5, f"/health had to wait {health_took:.2f}s for the slow chat"
    assert chat_resp.status_code == 200 and chat_resp.json()["answer"] == "done"


def test_upload_prepares_qdrant_before_indexing(monkeypatch):
    order = []
    monkeypatch.setattr(main, "get_qdrant_client", lambda: order.append("qdrant-ready"))
    monkeypatch.setattr(main, "process_and_upsert", lambda *a, **k: (order.append("indexed"), 3)[1])
    monkeypatch.setattr(main, "upload_to_blob_storage", lambda *a, **k: None)

    client = TestClient(main.app)
    token = _token(client, "uploader@example.com")
    resp = client.post("/upload", headers={"Authorization": "Bearer " + token},
                       files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["chunks"] == 3
    assert order == ["qdrant-ready", "indexed"]


def test_upload_error_is_reported_as_json(monkeypatch):
    monkeypatch.setattr(main, "get_qdrant_client", lambda: None)
    def fail(*a, **k):
        raise RuntimeError("embedding service down")
    monkeypatch.setattr(main, "process_and_upsert", fail)

    client = TestClient(main.app, raise_server_exceptions=False)
    token = _token(client, "uploader2@example.com")
    resp = client.post("/upload", headers={"Authorization": "Bearer " + token},
                       files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 500
    assert "embedding service down" in resp.json()["detail"]


# ------------------------------------------------------------ /warm

class _CountingSession:
    queries = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, _q):
        _CountingSession.queries += 1


def test_warm_pings_the_database_but_only_once_per_interval(monkeypatch):
    _CountingSession.queries = 0
    monkeypatch.setattr(main, "SessionLocal", _CountingSession)
    monkeypatch.setattr(main, "_last_warm", 0.0)
    client = TestClient(main.app)

    assert client.get("/warm").json() == {"db": "ok"}
    for _ in range(20):
        assert client.get("/warm").json() == {"db": "recent"}
    assert _CountingSession.queries == 1

    monkeypatch.setattr(main, "_last_warm", time.monotonic() - 301)  # pretend 5+ minutes passed
    assert client.get("/warm").json() == {"db": "ok"}
    assert _CountingSession.queries == 2


def test_warm_allows_an_immediate_retry_after_a_failure(monkeypatch):
    class Broken(_CountingSession):
        def execute(self, _q):
            raise RuntimeError("database is paused")

    monkeypatch.setattr(main, "_last_warm", 0.0)
    monkeypatch.setattr(main, "SessionLocal", Broken)
    client = TestClient(main.app)
    assert client.get("/warm").status_code == 503

    _CountingSession.queries = 0
    monkeypatch.setattr(main, "SessionLocal", _CountingSession)
    assert client.get("/warm").json() == {"db": "ok"}


# ------------------------------------------------------------ Qdrant indexes

class _FakeQdrant:
    def __init__(self, **kwargs):
        self.indexed = []
        self.created = False

    def collection_exists(self, name):
        return True  # the upload already created it, WITHOUT indexes

    def create_collection(self, **kwargs):
        self.created = True

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexed.append(field_name)


def test_filter_indexes_are_ensured_even_when_the_collection_already_exists(monkeypatch):
    monkeypatch.setattr(agent, "QdrantClient", _FakeQdrant)
    monkeypatch.setattr(agent, "_qdrant_client", None)
    client = agent.get_qdrant_client()
    assert client.created is False
    assert client.indexed == ["metadata.user_id", "metadata.document_id"]
    assert agent.get_qdrant_client() is client       # shared, not rebuilt
    assert client.indexed == ["metadata.user_id", "metadata.document_id"]  # and not re-indexed


def test_an_index_error_does_not_crash_startup(monkeypatch):
    class Grumpy(_FakeQdrant):
        def create_payload_index(self, **kwargs):
            raise RuntimeError("already exists")

    monkeypatch.setattr(agent, "QdrantClient", Grumpy)
    monkeypatch.setattr(agent, "_qdrant_client", None)
    assert agent.get_qdrant_client() is not None
