"""
Database connection for web_app.py. One connection string, two engines it
can point at:

  Local dev / testing:  sqlite:///./dev.db
  Production (Azure):   mssql+pyodbc://<user>:<password>@<server>.database.windows.net/<db>?driver=ODBC+Driver+18+for+SQL+Server

Nothing here is used by mcp_server.py or the notebooks - only web_app.py.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import config

if config.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    # pyodbc login timeout (seconds). Azure SQL serverless can take a while
    # to resume from auto-pause, so don't give up after the 15s default.
    connect_args = {"timeout": 30}

engine = create_engine(
    config.DATABASE_URL,
    connect_args=connect_args,
    # Azure silently drops idle TCP connections (SQL gateway / Container Apps
    # load balancer, ~4-30 min). Without these, the first request after an
    # idle period grabs a dead pooled connection and 500s; the next one works
    # because SQLAlchemy has discarded it by then.
    pool_pre_ping=True,   # test the connection before using it, reconnect if dead
    pool_recycle=1800,    # never reuse a connection older than 30 min
)
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
    startup."""
    import models  # noqa: F401 - importing registers the models with Base
    Base.metadata.create_all(bind=engine)
