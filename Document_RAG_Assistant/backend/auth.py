"""
Everything auth-related that isn't a route: hashing passwords, issuing
and checking JWTs, and the two FastAPI dependencies every protected route
uses - get_current_user (must be logged in) and require_admin (must be
an admin).
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models import User, RefreshToken, Role, utcnow

bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, role: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "role": role, "exp": expires_at, "type": "access"}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(db: Session, user_id: int) -> str:
    """Generates a random opaque token, stores its hash (never the raw
    value) in the database, and returns the raw value to send to the
    client. Only the raw value can be checked against the stored hash -
    it can't be reversed back out of the database."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    db.add(RefreshToken(user_id=user_id, token_hash=token_hash, expires_at=expires_at))
    db.commit()
    return raw_token


def verify_refresh_token(db: Session, raw_token: str) -> RefreshToken | None:
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    record = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

    if record is None or record.revoked or record.expires_at < utcnow():
        return None
    return record


def revoke_refresh_token(db: Session, raw_token: str) -> None:
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).update({"revoked": 1})
    db.commit()


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError (caught by get_current_user below) on any
    problem - expired, wrong signature, malformed, wrong type."""
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return payload


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise unauthorized

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None or not user.is_active:
        raise unauthorized
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != Role.admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user
