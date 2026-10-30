"""
Google login, the plain way: redirect to Google, Google redirects back
with a code, we exchange that code for the user's email, then issue our
own access+refresh tokens exactly like a normal login would.

Deliberately not using google-auth or authlib here - it's three HTTP
calls and one state check, adding a whole library for that would hide
what's actually happening for no real benefit.
"""

import secrets

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from fastapi import Depends

from config import settings
from database import get_db
from models import User, Role
from auth import create_access_token, create_refresh_token

router = APIRouter(prefix="/auth/google", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


@router.get("/login")
def google_login():
    """Sends the user to Google's consent screen. The `state` value is a
    CSRF check - Google sends it back unchanged on the callback, and we
    check it matches what we set, so a request can't be forged by
    tricking a logged-in user into visiting a crafted callback URL."""
    state = secrets.token_urlsafe(16)

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())

    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{query}")
    response.set_cookie("oauth_state", state, httponly=True, max_age=600)
    return response


@router.get("/callback")
def google_callback(request: Request, code: str, state: str, db: Session = Depends(get_db)):
    expected_state = request.cookies.get("oauth_state")
    if not expected_state or state != expected_state:
        raise HTTPException(400, "OAuth state mismatch - possible CSRF, or the request just took too long")

    token_response = httpx.post(GOOGLE_TOKEN_URL, data={
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "code": code,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    })
    token_response.raise_for_status()
    google_access_token = token_response.json()["access_token"]

    userinfo_response = httpx.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {google_access_token}"},
    )
    userinfo_response.raise_for_status()
    userinfo = userinfo_response.json()

    user = db.query(User).filter(User.google_id == userinfo["sub"]).first()
    if user is None:
        # A password account with the same email already exists - link
        # Google to it instead of creating a second, separate account for
        # the same person.
        user = db.query(User).filter(User.email == userinfo["email"]).first()
        if user is not None:
            user.google_id = userinfo["sub"]
        else:
            user = User(email=userinfo["email"], google_id=userinfo["sub"], role=Role.user)
            db.add(user)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(db, user.id)

    # Hands both tokens back to the frontend via the URL, then the
    # frontend JS (see static/index.html's init()) reads them off the URL
    # on page load and stores them - the access token in memory only, the
    # refresh token in localStorage (see index.html for why the split).
    redirect_url = f"{settings.FRONTEND_URL}/?access_token={access_token}&refresh_token={refresh_token}"
    response = RedirectResponse(redirect_url)
    response.delete_cookie("oauth_state")
    return response
