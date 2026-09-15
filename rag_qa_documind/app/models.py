"""
Three tables:

  User          - one row per account. password_hash is nullable because
                  a Google-only account (FastAPI backend path only) never
                  sets a password.
  RefreshToken  - server-side record of each refresh token (hashed, not
                  raw) so logout can actually revoke it. Only used by the
                  FastAPI backend's JWT flow - the standalone Streamlit
                  app doesn't issue tokens to itself, it just checks
                  st.session_state each rerun (see streamlit_app.py).
  Document      - a log of what each user uploaded, for the "my
                  documents" list and an admin's view - the actual chunk
                  content still lives in Chroma, this is just metadata.
"""

from datetime import datetime, timezone
import enum

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship

from app.database import Base


def utcnow():
    """Naive UTC datetime - SQLite silently drops timezone info on
    anything stored in a DateTime column, so storing a timezone-aware
    value and later comparing it against one raises "can't compare
    offset-naive and offset-aware datetimes". Staying naive-but-always-UTC
    everywhere avoids the mismatch, on SQLite and SQL Server alike."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Role(str, enum.Enum):
    user = "user"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    google_id = Column(String(255), unique=True, nullable=True, index=True)
    role = Column(Enum(Role), nullable=False, default=Role.user)
    is_active = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    documents = relationship("Document", back_populates="owner")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    chunks_indexed = Column(Integer, nullable=False, default=0)
    uploaded_at = Column(DateTime, nullable=False, default=utcnow)

    owner = relationship("User", back_populates="documents")
