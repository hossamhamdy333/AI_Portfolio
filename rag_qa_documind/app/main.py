"""
FastAPI service for DocuMind.

Endpoints:
  POST /auth/register, /auth/login, /auth/refresh, /auth/logout, GET /auth/me
  GET  /health         -> liveness check
  POST /ingest         -> upload a file (.txt/.md/.pdf) to be embedded & indexed
  POST /query          -> ask a question, get an answer grounded in indexed docs
  POST /reset          -> wipe YOUR OWN vector index (start fresh)
  GET  /documents       -> list of files you've uploaded
  /admin/*             -> admin-only routes (list users, view a user's uploads, ban)

Every route below /auth requires a valid access token. Isolation reuses
the exact mechanism that was already here (app/vectorstore.py's
session-keyed Chroma collections) - the only change is WHO decides the
key: it used to be an arbitrary client-supplied X-Session-Id header
(trust-the-caller), now it's str(user.id) from a verified JWT, so a
caller can no longer just pick a different session_id to read someone
else's documents.
"""
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timezone

from app.ingest import ingest_file
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection
from app.guardrails import guard_input, guard_output
from app.config import settings
from app.database import get_db, init_db
from app.models import User, Document, Role
from app.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    verify_refresh_token, revoke_refresh_token,
    get_current_user, require_admin,
)
from app import oauth


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="DocuMind RAG API", version="2.0.0", lifespan=lifespan)
app.include_router(oauth.router)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


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


# ---------------------------------------------------------------- documents (per-user)

@app.get("/health")
def health(user: User = Depends(get_current_user)):
    return {"status": "ok", "indexed_chunks": get_collection(str(user.id)).count()}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".txt", ".md", ".pdf"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_chunks = ingest_file(tmp_path, source_name=file.filename, session_id=str(user.id))
    finally:
        os.remove(tmp_path)

    db.add(Document(user_id=user.id, filename=file.filename, chunks_indexed=n_chunks))
    db.commit()

    return {"filename": file.filename, "chunks_indexed": n_chunks}


@app.post("/query")
def query(req: QueryRequest, user: User = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    input_guard = guard_input(req.question)
    if input_guard["blocked"]:
        return {"question": req.question, "answer": "I can't process that request.", "sources": [], "blocked": True}

    result = answer_question(input_guard["redacted_text"], top_k=req.top_k, session_id=str(user.id))

    output_guard = guard_output(result["answer"])
    result["answer"] = output_guard["text"]
    result["blocked"] = output_guard["blocked"]
    return result


@app.post("/reset")
def reset(user: User = Depends(get_current_user)):
    """Wipes YOUR OWN index only - the old version of this endpoint wiped
    the entire shared index when no session header was given, reachable
    by anyone with no auth at all. Requiring login here removes that
    unauthenticated-anonymous-caller path entirely."""
    reset_collection(str(user.id))
    return {"status": "index cleared"}


@app.get("/documents")
def list_my_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.user_id == user.id).order_by(Document.uploaded_at.desc()).all()
    return [{"id": d.id, "filename": d.filename, "chunks_indexed": d.chunks_indexed, "uploaded_at": d.uploaded_at.isoformat()} for d in docs]


# ---------------------------------------------------------------- admin only

@app.get("/admin/users")
def admin_list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {"id": u.id, "email": u.email, "role": u.role.value, "is_active": bool(u.is_active), "created_at": u.created_at.isoformat()}
        for u in users
    ]


@app.get("/admin/users/{user_id}/documents")
def admin_view_user_documents(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(404, "User not found")

    docs = db.query(Document).filter(Document.user_id == user_id).all()
    return [{"id": d.id, "filename": d.filename, "uploaded_at": d.uploaded_at.isoformat()} for d in docs]


@app.post("/admin/users/{user_id}/deactivate")
def admin_deactivate_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(404, "User not found")
    if target.id == admin.id:
        raise HTTPException(400, "Can't deactivate your own account")

    target.is_active = 0
    db.commit()
    return {"status": "deactivated", "user_id": user_id}


@app.get("/admin/stats")
def admin_stats(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return {
        "total_users": db.query(func.count(User.id)).scalar(),
        "active_users": db.query(func.count(User.id)).filter(User.is_active == 1).scalar(),
        "total_documents": db.query(func.count(Document.id)).scalar(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
