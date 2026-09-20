"""
Database connection. One connection string, two engines it can point at:

  Local dev / testing:  sqlite:///./dev.db
  Production (Azure):   mssql+pyodbc://<user>:<password>@<server>.database.windows.net/<db>?driver=ODBC+Driver+18+for+SQL+Server

Swapping between them is just changing DATABASE_URL - see README's
"Setting up Azure SQL" section for the production side of this.
"""

import logging
import time

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

from src.config import settings

logger = logging.getLogger(__name__)

url = settings.DATABASE_URL

if url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif url.startswith("mssql"):
    # pyodbc login timeout (seconds). A paused serverless Azure SQL database
    # can take a long time to answer its first login.
    connect_args = {"timeout": 30}
else:
    connect_args = {}

engine = create_engine(
    url,
    connect_args=connect_args,
    # Azure silently drops idle connections. Test a pooled connection before
    # using it, and never reuse one older than 30 minutes, so the first
    # request after a quiet period doesn't grab a dead connection and 500.
    pool_pre_ping=True,
    pool_recycle=1800,
)


def install_connect_retry(target_engine, attempts: int = 8, delay: float = 8.0):
    """Retry the initial database connection while Azure SQL serverless wakes
    up from auto-pause.

    A paused database refuses logins (error 40613, "not currently
    available") for roughly 30-60 seconds while it resumes. Without a retry,
    the first login/chat after an idle period fails with a 500. With it, that
    first request just takes a bit longer and then succeeds. A wrong
    password (error 18456) is NOT retried - that would only delay the error.
    """
    @event.listens_for(target_engine, "do_connect")
    def _connect_with_retry(dialect, conn_rec, cargs, cparams):
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return dialect.dbapi.connect(*cargs, **cparams)
            except dialect.dbapi.Error as exc:
                if "18456" in str(exc):  # login failed: bad credentials
                    raise
                last_exc = exc
                logger.warning(
                    "Database connect attempt %d/%d failed (database may be waking up): %s",
                    attempt, attempts, str(exc)[:160],
                )
                if attempt < attempts:
                    time.sleep(delay)
        raise last_exc


if url.startswith("mssql"):
    install_connect_retry(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency - yields one session per request, always closed
    afterward even if the request raises an exception."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates every table that doesn't exist yet. Safe to call on every
    app startup. Fine for a project this size; a real production system
    with an evolving schema would use Alembic migrations instead."""
    import src.models  # noqa: F401 - importing registers the models with Base
    Base.metadata.create_all(bind=engine)
