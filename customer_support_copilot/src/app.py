"""
FastAPI backend for the AI Support Copilot.

This is the app that's actually deployed -- a container on Azure
Container Apps. Generation goes through src/llm_backend.py, which can be
either the original CPU-only llama.cpp GGUF setup or vLLM (see that
module's docstring for when each makes sense).
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func
import os

from src.retriever import KBRetriever
from src.evaluate import evaluate_faithfulness
from src import llm_backend
from src.guardrails import guard_input, guard_output
from src.config import settings
from src.database import get_db, init_db
from src.models import User, ChatMessage, MessageRole, Role
from src.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    verify_refresh_token, revoke_refresh_token,
    get_current_user, require_admin,
)
from src import oauth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior customer support agent for a premium brand. "
    "Reply politely, professionally, and resolve the user's issue based "
    "ONLY on the provided context."
)

# Populated at startup (see lifespan below).
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever

    init_db()

    logger.info("Loading LLM backend (%s)...", settings.LLM_BACKEND)
    llm_backend.load_model()
    logger.info("LLM backend ready.")

    logger.info("Building KB retriever...")
    _retriever = KBRetriever()
    logger.info("Retriever ready.")

    yield


app = FastAPI(title="AI Support Copilot API", lifespan=lifespan)
app.include_router(oauth.router)


# ---------------------------------------------------------------- schemas

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    context: str
    blocked: bool = False


# ---------------------------------------------------------------- auth

@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    existing = db.query(User).filter(User.email == body.email).first()
    if existing is not None:
        raise HTTPException(409, "An account with this email already exists")

    user = User(email=body.email, password_hash=hash_password(body.password), role=Role.user)
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(db, user.id)
    return {"access_token": access_token, "refresh_token": refresh_token}


@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    unauthorized = HTTPException(401, "Incorrect email or password")

    user = db.query(User).filter(User.email == body.email).first()
    if user is None or user.password_hash is None:
        raise unauthorized
    if not verify_password(body.password, user.password_hash):
        raise unauthorized
    if not user.is_active:
        raise HTTPException(403, "This account has been deactivated")

    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(db, user.id)
    return {"access_token": access_token, "refresh_token": refresh_token}


@app.post("/auth/refresh")
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    record = verify_refresh_token(db, body.refresh_token)
    if record is None:
        raise HTTPException(401, "Invalid, expired, or already-used refresh token")

    user = db.query(User).filter(User.id == record.user_id).first()
    if user is None or not user.is_active:
        raise HTTPException(401, "Account no longer active")

    return {"access_token": create_access_token(user.id, user.role.value)}


@app.post("/auth/logout")
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    revoke_refresh_token(db, body.refresh_token)
    return {"status": "logged out"}


@app.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "role": user.role.value}


# ---------------------------------------------------------------- chat (per-user, persisted)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not llm_backend.is_ready() or _retriever is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    input_guard = guard_input(request.query)
    if input_guard["blocked"]:
        logger.warning("Blocked a suspected prompt injection from user %d: %s", user.id, input_guard["injection_match"])
        return ChatResponse(response="I can't process that request.", context="", blocked=True)

    if input_guard["pii_redactions"]:
        logger.info("Redacted %d PII match(es) from user %d's message.", input_guard["pii_redactions"], user.id)

    try:
        context = _retriever.retrieve(input_guard["redacted_text"])

        prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Context: {context}
Query: {input_guard['redacted_text']}
<|assistant|>
"""
        ai_text = llm_backend.generate(prompt)

        output_guard = guard_output(ai_text)
        if output_guard["blocked"]:
            logger.warning("Blocked disallowed output for user %d: %s", user.id, output_guard["match"])
        final_text = output_guard["text"]

        if settings.ENABLE_EVAL:
            eval_result = evaluate_faithfulness(request.query, context, final_text, db)
            if eval_result and not eval_result.get("is_faithful", True):
                logger.warning("Faithfulness check flagged this response: %s", eval_result.get("reason"))

        db.add(ChatMessage(user_id=user.id, role=MessageRole.user, content=request.query))
        db.add(ChatMessage(user_id=user.id, role=MessageRole.assistant, content=final_text))
        db.commit()

        return ChatResponse(response=final_text, context=context, blocked=output_guard["blocked"])

    except Exception as e:
        logger.exception("Chat generation failed for user %d", user.id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
def my_chat_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """A user's own conversation history - not anyone else's."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": m.role.value, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages]


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm_backend.is_ready(), "backend": settings.LLM_BACKEND}


# ---------------------------------------------------------------- admin only

@app.get("/admin/users")
def admin_list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {"id": u.id, "email": u.email, "role": u.role.value, "is_active": bool(u.is_active), "created_at": u.created_at.isoformat()}
        for u in users
    ]


@app.get("/admin/users/{user_id}/transcript")
def admin_view_transcript(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """The full conversation history for one user - for support/QA
    purposes, not something a regular user can see about anyone but
    themselves (see /chat/history above)."""
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(404, "User not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": m.role.value, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages]


@app.post("/admin/users/{user_id}/ban")
def admin_ban_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(404, "User not found")
    if target.id == admin.id:
        raise HTTPException(400, "Can't ban your own account")

    target.is_active = 0
    db.commit()
    return {"status": "banned", "user_id": user_id}


@app.get("/admin/stats")
def admin_stats(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return {
        "total_users": db.query(func.count(User.id)).scalar(),
        "active_users": db.query(func.count(User.id)).filter(User.is_active == 1).scalar(),
        "total_messages": db.query(func.count(ChatMessage.id)).scalar(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
