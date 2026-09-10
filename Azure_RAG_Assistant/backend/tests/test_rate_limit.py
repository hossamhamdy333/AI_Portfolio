import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ["DATABASE_URL"] = "sqlite:///./test_rate_limit.db"  # dedicated file, not shared with other test modules

import pytest
from fastapi import HTTPException

import config
from database import SessionLocal, init_db
from rate_limit import enforce_rate_limit

init_db()


@pytest.fixture
def db(monkeypatch):
    # Force this specific setting via monkeypatch rather than relying on
    # os.environ being read before config.py's first import somewhere else
    # in the test session - pydantic-settings only reads the environment
    # once, the first time Settings() is constructed anywhere in the
    # process, so whichever test module happens to import config.py first
    # (alphabetically, test_auth.py) wins and every later os.environ
    # assignment for the same variable is silently a no-op. monkeypatch
    # mutates the already-constructed settings object directly instead, so
    # it works regardless of import order. (Confirmed this the hard way -
    # an earlier version of this file relied on os.environ alone and
    # intermittently failed depending on which order pytest collected
    # test files in.)
    monkeypatch.setattr(config.settings, "RATE_LIMIT_ENABLED", True)
    session = SessionLocal()
    yield session
    session.close()


def test_allows_requests_under_the_limit(db):
    for _ in range(3):
        enforce_rate_limit(db, key="test:under-limit", endpoint="chat", max_requests=5, window_minutes=60)
    # no exception raised - all 3 of 5 allowed requests went through


def test_blocks_requests_over_the_limit(db):
    for _ in range(3):
        enforce_rate_limit(db, key="test:over-limit", endpoint="chat", max_requests=3, window_minutes=60)

    with pytest.raises(HTTPException) as exc_info:
        enforce_rate_limit(db, key="test:over-limit", endpoint="chat", max_requests=3, window_minutes=60)
    assert exc_info.value.status_code == 429


def test_different_keys_have_independent_limits(db):
    # Using up user A's limit should not affect user B's - each key is
    # counted separately, same principle as the per-user document
    # isolation elsewhere in this project.
    for _ in range(3):
        enforce_rate_limit(db, key="test:user-a", endpoint="chat", max_requests=3, window_minutes=60)

    with pytest.raises(HTTPException):
        enforce_rate_limit(db, key="test:user-a", endpoint="chat", max_requests=3, window_minutes=60)

    # user B has made zero requests so far and should not be blocked
    enforce_rate_limit(db, key="test:user-b", endpoint="chat", max_requests=3, window_minutes=60)


def test_different_endpoints_have_independent_limits(db):
    # Using up the "chat" limit for a key should not affect the same key's
    # "upload" limit - they're tracked separately.
    for _ in range(3):
        enforce_rate_limit(db, key="test:multi-endpoint", endpoint="chat", max_requests=3, window_minutes=60)

    with pytest.raises(HTTPException):
        enforce_rate_limit(db, key="test:multi-endpoint", endpoint="chat", max_requests=3, window_minutes=60)

    # same key, different endpoint - should not be blocked
    enforce_rate_limit(db, key="test:multi-endpoint", endpoint="upload", max_requests=3, window_minutes=60)


def test_disabled_rate_limit_never_blocks(db, monkeypatch):
    monkeypatch.setattr(config.settings, "RATE_LIMIT_ENABLED", False)

    # Far more requests than max_requests allows - should never raise
    # while the feature is toggled off.
    for _ in range(10):
        enforce_rate_limit(db, key="test:disabled", endpoint="chat", max_requests=1, window_minutes=60)
