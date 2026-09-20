"""
Five tables:

  User          - one row per account.
  RefreshToken  - hashed refresh tokens so logout can revoke them.
  Conversation  - one row per chat thread a user has started, like a
                  Claude conversation. A user can have many.
  ChatMessage   - one row per message, tied to the conversation it
                  belongs to (and denormalized onto the user too, so
                  admin/user-history queries don't need a join).
  GeminiUsage   - daily counter metering the shared GEMINI_API_KEY.
"""

from datetime import datetime, timezone
import enum

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Text, Index, text
from sqlalchemy.orm import relationship

from src.database import Base


def utcnow():
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
    # NOT unique=True: on SQL Server a plain unique index allows only ONE NULL,
    # and every email/password account has google_id NULL, so a second
    # registration would fail. The filtered index in __table_args__ only
    # enforces uniqueness for accounts that actually have a Google id.
    google_id = Column(String(255), nullable=True)
    role = Column(Enum(Role), nullable=False, default=Role.user)
    is_active = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    conversations = relationship("Conversation", back_populates="user")
    messages = relationship("ChatMessage", back_populates="user")

    __table_args__ = (
        Index(
            "ix_users_google_id", "google_id", unique=True,
            mssql_where=text("google_id IS NOT NULL"),
            postgresql_where=text("google_id IS NOT NULL"),
            sqlite_where=text("google_id IS NOT NULL"),
        ),
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New chat")
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="messages")


class GeminiUsage(Base):
    __tablename__ = "gemini_usage"

    id = Column(Integer, primary_key=True)
    date = Column(String(10), nullable=False, unique=True, index=True)
    count = Column(Integer, nullable=False, default=0)
