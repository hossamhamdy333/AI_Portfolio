"""
Runs before any test module is imported (pytest always imports conftest.py
first), which is what makes the DATABASE_URL override below actually work.

Without this file, test_web_app.py's own `os.environ["DATABASE_URL"] = ...`
line was too late whenever another test file (e.g. test_portfolio_persistence.py,
collected first alphabetically) imported `config` before it ran - config.py
and database.py both read DATABASE_URL from the environment exactly once, at
import time, so the override was silently a no-op. The real bug this caused:
test_web_app.py was writing into the repo's real dev.db (the default) instead
of a throwaway test database, so a second test run failed on a UNIQUE
constraint - the admin user from the previous run was still there.

Setting it here, before pytest imports any test module, closes that gap.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_web_app.db")
os.environ.setdefault("GOOGLE_API_KEY", "test")


def pytest_sessionstart(session):
    # Always start from a clean test database, so re-running the suite
    # twice in a row doesn't hit stale rows from the previous run.
    test_db = os.path.join(os.path.dirname(__file__), "..", "test_web_app.db")
    if os.path.exists(test_db):
        os.remove(test_db)
