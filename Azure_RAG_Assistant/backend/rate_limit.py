"""
Database-backed rate limiting.

Counts recent rows in RequestLog instead of tracking counters in memory,
so limits are enforced correctly even if this app ever runs as more than
one instance - separate in-memory counters per instance would each have
their own wrong view of "how many requests has this user made"; a shared
database does not have that problem.

This isn't Redis-fast under very high request volume (a few indexed COUNT
queries per request), but for the traffic level this project is actually
built for, that cost is negligible. If this ever needs to scale well
beyond that, swapping this module's internals for Redis (INCR + TTL) is a
contained change - the call sites in main.py wouldn't need to change at
all, since they only call enforce_rate_limit().
"""

from datetime import timedelta

from fastapi import HTTPException, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from models import RequestLog, utcnow


def enforce_rate_limit(db: Session, key: str, endpoint: str, max_requests: int, window_minutes: int) -> None:
    """
    Raises 429 if `key` has already made >= max_requests to `endpoint`
    within the last `window_minutes`. Otherwise records this request as
    one of them (so the count includes the current attempt) and returns
    normally.

    No-ops entirely if settings.RATE_LIMIT_ENABLED is False (used by the
    test suite - see config.py's comment on that setting for why).
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    cutoff = utcnow() - timedelta(minutes=window_minutes)
    count = (
        db.query(func.count(RequestLog.id))
        .filter(RequestLog.key == key, RequestLog.endpoint == endpoint, RequestLog.created_at >= cutoff)
        .scalar()
    )
    if count >= max_requests:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded: max {max_requests} requests per {window_minutes} minutes for this action. Try again later.",
        )
    db.add(RequestLog(key=key, endpoint=endpoint))
    db.commit()


def client_ip(request: Request) -> str:
    """
    Best-effort client IP for rate-limiting unauthenticated endpoints
    (login/register) where there's no user_id to key on yet.

    Azure App Service sits in front of the app as a reverse proxy, so
    request.client.host would just be Azure's internal proxy address, not
    the real caller - X-Forwarded-For is what Azure sets to the actual
    client IP. Falls back to request.client.host for local dev, where
    there's no proxy in front and X-Forwarded-For is never set.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
