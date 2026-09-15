"""
Google login for the FastAPI backend (app/main.py + ui/streamlit_app.py
two-process setup) only. The standalone streamlit_app.py deployed on
Streamlit Community Cloud has no separate server for Google to redirect
back to over HTTP - it's one Python process rendering the UI directly -
so it stays email+password only (see that file's login form). Adding
OAuth there would mean wiring up Streamlit's own newer built-in auth
support (st.login), which is a big enough departure from this redirect-
based flow that it isn't duplicated here without a real reason to.
"""

import secrets

import httpx
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User, Role
from app.auth import create_access_token, create_refresh_token

router = APIRouter(prefix="/auth/google", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


@router.get("/login")
def google_login():
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
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
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "code": code,
        "redirect_uri": settings.google_redirect_uri,
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

    redirect_url = f"{settings.frontend_url}/?access_token={access_token}&refresh_token={refresh_token}"
    response = RedirectResponse(redirect_url)
    response.delete_cookie("oauth_state")
    return response
