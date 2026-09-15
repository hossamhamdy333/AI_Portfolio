"""
Database connection. One connection string, two engines it can point at:

  Local dev / testing:  sqlite:///./dev.db
  Production (Azure):   mssql+pyodbc://<user>:<password>@<server>.database.windows.net/<db>?driver=ODBC+Driver+18+for+SQL+Server

See README's "Accounts and a real database" section for the production side.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency - yields one session per request, always closed
    afterward even if the request raises an exception. The standalone
    streamlit_app.py doesn't use this - it opens/closes its own session
    directly per Streamlit rerun, see that file."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates every table that doesn't exist yet. Safe to call on every
    startup. Fine for a project this size; a real production system with
    an evolving schema would use Alembic migrations instead."""
    import app.models  # noqa: F401 - importing registers the models with Base
    Base.metadata.create_all(bind=engine)
