"""
Three tables:

  User         - one row per account. password_hash is nullable because a
                 Google-only account never sets a password.
  RefreshToken - server-side record of each refresh token (hashed, not
                 raw) so logout can actually revoke it - a JWT access
                 token alone can't be revoked once issued.
  ChatMessage  - every message either side of a conversation sent, tied
                 to the user who sent/received it. This is what makes
                 "see your own chat history" and "admin views a user's
                 transcript" mean something - before this, nothing about
                 a conversation was ever saved anywhere.
"""

from datetime import datetime, timezone
import enum

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship

from src.database import Base


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


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    google_id = Column(String(255), unique=True, nullable=True, index=True)
    role = Column(Enum(Role), nullable=False, default=Role.user)
    is_active = Column(Integer, nullable=False, default=1)  # 0/1, not a real bool column in every DB dialect
    created_at = Column(DateTime, nullable=False, default=utcnow)

    messages = relationship("ChatMessage", back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)  # sha256 hex digest
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    user = relationship("User", back_populates="messages")
