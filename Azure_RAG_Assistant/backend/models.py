"""
Two tables:

  User      - one row per person with an account. password_hash is
              nullable because a Google-only account never sets a
              password - a user can have a password, a google_id, or
              both (if they later link Google to a password account).

  Document  - one row per uploaded file, so a document has an owner and
              the retriever can be filtered to "only this user's
              documents" instead of trusting the UI to hide the rest.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from database import Base


def utcnow():
    """Naive UTC datetime - SQLite silently drops timezone info on
    anything stored in a DateTime column, so storing a timezone-aware
    value here and later comparing it against one (e.g. in auth.py's
    expiry check) raises "can't compare offset-naive and offset-aware
    datetimes". Staying naive-but-always-UTC everywhere avoids the
    mismatch entirely, on SQLite and on SQL Server alike."""
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
    is_active = Column(Integer, nullable=False, default=1)  # 0/1, not a real bool column in every DB dialect
    created_at = Column(DateTime, nullable=False, default=utcnow)

    documents = relationship("Document", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    blob_url = Column(String(1000), nullable=True)
    chunks_indexed = Column(Integer, nullable=False, default=0)
    uploaded_at = Column(DateTime, nullable=False, default=utcnow)

    owner = relationship("User", back_populates="documents")


class RefreshToken(Base):
    """
    A refresh token is stored here (hashed, not raw) so logout can actually
    revoke it. A JWT access token alone can't be revoked once issued - it's
    just a signed piece of data anyone with the secret can verify, valid
    until it expires no matter what the server does. Keeping refresh
    tokens server-side is what makes "log out everywhere" possible.
    """
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)  # sha256 hex digest
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utcnow)
