"""
Three tables:

  AdminUser     - there's exactly one of these in practice (you). No
                  public registration endpoint exists at all - see
                  scripts/create_admin.py for how this row gets created.
                  A separate table from a generic "User" on purpose:
                  this site has no concept of visitor accounts, so
                  naming it AdminUser instead of User says that plainly
                  instead of implying a multi-tenant system that isn't
                  here.
  RefreshToken  - server-side record of each refresh token (hashed, not
                  raw) so logout can actually revoke it.
  QueryLog      - every question asked through /ask, whether it was
                  blocked, and by which IP - what the admin dashboard
                  reads from.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean

from database import Base


def utcnow():
    """Naive UTC datetime - SQLite silently drops timezone info on
    anything stored in a DateTime column, so storing a timezone-aware
    value and later comparing it against one raises "can't compare
    offset-naive and offset-aware datetimes". Staying naive-but-always-UTC
    everywhere avoids the mismatch, on SQLite and SQL Server alike."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    admin_id = Column(Integer, ForeignKey("admin_users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class QueryLog(Base):
    __tablename__ = "query_log"

    id = Column(Integer, primary_key=True)
    ip_address = Column(String(64), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    blocked = Column(Boolean, nullable=False, default=False)
    block_reason = Column(String(255), nullable=True)
    rate_limited = Column(Boolean, nullable=False, default=False)
    target_projects = Column(String(500), nullable=True)  # comma-separated, for a quick glance in the dashboard
    created_at = Column(DateTime, nullable=False, default=utcnow, index=True)
