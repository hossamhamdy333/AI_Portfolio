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

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import settings

# check_same_thread=False is only needed for SQLite (FastAPI can call the
# same connection from different threads); SQL Server doesn't need this
# argument at all, so it's added conditionally instead of always.
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
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
    app startup - it's a no-op for tables that already exist. Fine for a
    project this size; a real production system with an evolving schema
    would use Alembic migrations instead of this, so schema changes don't
    require manually diffing what create_all() would do."""
    import models  # noqa: F401 - importing registers the models with Base
    Base.metadata.create_all(bind=engine)
