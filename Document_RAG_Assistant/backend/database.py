"""
Database connection. One connection string, two engines it can point at:

  Local dev / testing:  sqlite:///./dev.db
  Production (Azure):   mssql+pymssql://<user>:<password>@<server>.database.windows.net:1433/<db>

Swapping between them is just changing DATABASE_URL - nothing else in the
code needs to change, since SQLAlchemy's Core/ORM layer is the same either
way. pymssql ships as a self-contained wheel with its native dependencies
bundled in, so no extra system packages need installing in the container
for this to work (unlike pyodbc, which needs Microsoft's ODBC driver
installed separately) - see the README's "Setting up Azure SQL" section
for the actual account setup.
"""

import time

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

from config import settings, logger

# check_same_thread=False is only needed for SQLite (FastAPI can call the
# same connection from different threads); SQL Server doesn't need this
# argument at all, so it's added conditionally instead of always.
#
# For SQL Server specifically: pymssql's query timeout defaults to 0,
# meaning *unlimited* - if Azure SQL's serverless tier is mid-wake-up from
# auto-pause, a query can hang indefinitely with no error at all, which is
# worse than a clear failure the frontend can show and let the user retry.
# 30s comfortably covers a normal serverless wake-up while still failing
# fast if something is actually wrong, rather than hanging forever.
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif settings.DATABASE_URL.startswith("mssql"):
    connect_args = {"timeout": 30, "login_timeout": 30}
else:
    connect_args = {}

# pool_pre_ping: issues a cheap "is this connection still alive" check
# before handing it to a request. Without this, a connection that was
# opened before Azure SQL auto-paused (and is now dead) can sit in the
# pool and get reused, causing the *next* request through it to hang or
# fail strangely instead of transparently reconnecting.
# pool_recycle: forces connections older than this to be discarded and
# reopened, as a second safety net against the same class of staleness.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=280,
)


def install_connect_retry(target_engine, attempts: int = 6, delay: float = 6.0):
    """Retry the initial database connection while Azure SQL serverless wakes
    up from auto-pause.

    init_db() below already survives a paused database at *startup*, but a
    paused database at *request time* (nobody used the app for an hour, then
    someone logs in) had no protection: the first login failed with an
    unhandled error and returned a plain-text 500. A resuming database
    refuses logins for roughly 30-60 seconds; retrying the connection
    makes that first request just take longer and then succeed. A wrong
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


if settings.DATABASE_URL.startswith("mssql"):
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


def init_db(max_attempts: int = 10, initial_delay_seconds: float = 3.0) -> None:
    """
    Creates every table that doesn't exist yet. Safe to call on every app
    startup - it's a no-op for tables that already exist. Fine for a
    project this size; a real production system with an evolving schema
    would use Alembic migrations instead of this, so schema changes don't
    require manually diffing what create_all() would do.

    Retries with exponential backoff on connection failures rather than
    crashing the whole container on the first attempt. This matters
    specifically for Azure SQL's serverless tier, which auto-pauses when
    idle and takes some seconds to wake back up on the next connection -
    without a retry here, a container that starts right as the database
    is waking up would crash before the database ever became reachable,
    even though it would have connected fine a few seconds later. Also
    covers any other transient network hiccup at startup.
    """
    import models  # noqa: F401 - importing registers the models with Base

    delay = initial_delay_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            _repair_google_id_index()
            return
        except OperationalError:
            if attempt == max_attempts:
                logger.exception("Database still unreachable after %d attempts - giving up", max_attempts)
                raise
            logger.warning(
                "Database not reachable yet (attempt %d/%d) - retrying in %.0fs. "
                "Normal on first startup after Azure SQL serverless auto-pause.",
                attempt, max_attempts, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 15)


def _repair_google_id_index() -> None:
    """Safety net for a database whose tables were created BEFORE the
    filtered index in models.py existed. create_all() never alters an
    existing table, so such a database still has a plain unique index on
    users.google_id - which on SQL Server allows only ONE NULL, so every
    email/password registration after the first fails. This swaps it for the
    filtered one. Idempotent: does nothing once the filtered index is in
    place, and only runs against SQL Server."""
    if not settings.DATABASE_URL.startswith("mssql"):
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                IF EXISTS (SELECT 1 FROM sys.indexes
                           WHERE name = 'ix_users_google_id'
                             AND object_id = OBJECT_ID('users') AND has_filter = 0)
                    DROP INDEX ix_users_google_id ON users;
                IF NOT EXISTS (SELECT 1 FROM sys.indexes
                               WHERE name = 'ix_users_google_id'
                                 AND object_id = OBJECT_ID('users'))
                    CREATE UNIQUE INDEX ix_users_google_id ON users (google_id)
                    WHERE google_id IS NOT NULL;
            """))
    except Exception:
        # Never stop the app from starting over this; log it instead.
        logger.exception("Could not verify the users.google_id index")
