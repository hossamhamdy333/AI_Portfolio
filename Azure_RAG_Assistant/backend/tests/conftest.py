import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")

import pytest
import config


@pytest.fixture(autouse=True)
def _rate_limiting_off_by_default(monkeypatch):
    """
    Every test in this suite gets RATE_LIMIT_ENABLED=False unless it
    explicitly turns it back on itself (see test_rate_limit.py's own `db`
    fixture, which does exactly that).

    This has to be a monkeypatch, not an os.environ assignment before
    import: pydantic-settings only reads the environment once, the first
    time Settings() is constructed anywhere in the test process - whichever
    test module pytest happens to import first "wins", and every other
    module's os.environ assignment for the same variable is silently a
    no-op from then on. That made an earlier version of this test suite
    order-dependent (working or breaking depending on alphabetical file
    collection order, purely by accident) - autouse + monkeypatch fixes
    this for good, since it mutates the already-constructed settings
    object at test-run time instead of relying on import-time timing.
    """
    monkeypatch.setattr(config.settings, "RATE_LIMIT_ENABLED", False)
