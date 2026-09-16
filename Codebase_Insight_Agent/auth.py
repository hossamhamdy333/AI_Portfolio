"""
Password hashing, JWT issuing/verification, and the admin-required
FastAPI dependency. No "regular user" concept exists here at all - the
only account type is AdminUser, and there's no public registration route
(see scripts/create_admin.py for how that one account gets created).
"""

import hashlib
import secrets
from datetime import timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

import config
from database import get_db
from models import AdminUser, RefreshToken, utcnow

bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(admin_id: int) -> str:
    expires_at = utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(admin_id), "type": "access", "exp": expires_at}
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def create_refresh_token(db: Session, admin_id: int) -> str:
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = utcnow() + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)

    db.add(RefreshToken(admin_id=admin_id, token_hash=token_hash, expires_at=expires_at))
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
    payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return payload


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AdminUser:
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise unauthorized

    admin = db.query(AdminUser).filter(AdminUser.id == int(payload["sub"])).first()
    if admin is None or not admin.is_active:
        raise unauthorized
    return admin
