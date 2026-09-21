import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func

from azure_storage import upload_to_blob_storage
from text_processing import process_and_upsert, delete_document_chunks
from agent import run_agent, get_qdrant_client, split_sources
from guardrails import guard_input, guard_output
from config import settings, logger
from database import get_db, init_db, SessionLocal
from models import User, Document, Role
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    verify_refresh_token, revoke_refresh_token,
    get_current_user, require_admin,
)
from rate_limit import enforce_rate_limit, client_ip
import oauth


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # creates tables that don't exist yet - safe to call every startup

    # Off by default - tests and local runs never require a running
    # Phoenix collector unless you explicitly turn this on.
    if settings.ENABLE_OBSERVABILITY:
        from observability import setup_observability

        setup_observability()
        logger.info("Observability enabled - tracing to Phoenix.")
    yield


app = FastAPI(title="Azure RAG Assistant API", lifespan=lifespan)
app.include_router(oauth.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Always answer with JSON. Without this, an unexpected error (for
    example the database being briefly unreachable) came back as the plain
    text 'Internal Server Error', which the page could not read and showed
    as "Unexpected token 'I' ... is not valid JSON"."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on my side. Please try again."},
    )

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "static"))

# CORS: credentials + wildcard origins is invalid/insecure, so we don't allow
# credentials here. Tighten allow_origins to your real frontend URL(s) in
# production instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------------------------------------------------------------- pages

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the built-in single-page chat UI. No separate frontend
    deployment needed - visiting the backend's own URL is the app."""
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/health")
async def health():
    """Cheap liveness check - deliberately does NOT touch the database, so
    Azure's own health probe can't be taken down by a slow/paused DB and
    restart-loop the whole container over something that isn't a crash."""
    return {"status": "ok"}


# The page calls /warm as soon as it loads, so a paused database starts waking
# up while the visitor is still typing their email and password. A real query
# is sent at most once every few minutes, so a flood of page loads (or a
# crawler) can't keep the database awake around the clock and bill for it.
_WARM_MIN_INTERVAL = 300  # seconds
_last_warm = 0.0
_warm_lock = threading.Lock()


@app.get("/warm")
def warm():
    global _last_warm
    from sqlalchemy import text

    with _warm_lock:
        now = time.monotonic()
        if _last_warm and now - _last_warm < _WARM_MIN_INTERVAL:
            return {"db": "recent"}
        _last_warm = now
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {"db": "ok"}
    except Exception:
        with _warm_lock:
            _last_warm = 0.0  # failed: let the next visitor try again right away
        logger.warning("Warm-up ping could not reach the database yet", exc_info=True)
        return JSONResponse(status_code=503, content={"db": "unavailable"})


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    """Separate, explicit readiness check for the database itself. Hit
    this directly when debugging - if it hangs or errors, the problem is
    DB connectivity/credentials, not the app. If it returns fast, login
    hanging is something else."""
    from sqlalchemy import text

    db.execute(text("SELECT 1"))
    return {"status": "ok", "database_url_scheme": settings.DATABASE_URL.split("://")[0]}


# ---------------------------------------------------------------- auth

@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    enforce_rate_limit(
        db, key=f"ip:{client_ip(request)}", endpoint="register",
        max_requests=settings.REGISTER_RATE_LIMIT_PER_HOUR, window_minutes=60,
    )

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
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    enforce_rate_limit(
        db, key=f"ip:{client_ip(request)}", endpoint="login",
        max_requests=settings.LOGIN_RATE_LIMIT_PER_15MIN, window_minutes=15,
    )

    unauthorized = HTTPException(401, "Incorrect email or password")

    user = db.query(User).filter(User.email == body.email).first()
    if user is None or user.password_hash is None:
        raise unauthorized  # None: a Google-only account has no password to check against
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


# ---------------------------------------------------------------- chat + upload (per-user)

@app.post("/chat")
def chat(request: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    enforce_rate_limit(
        db, key=f"user:{user.id}", endpoint="chat",
        max_requests=settings.CHAT_RATE_LIMIT_PER_HOUR, window_minutes=60,
    )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    input_guard = guard_input(request.query)
    if input_guard["blocked"]:
        logger.warning("Blocked a suspected prompt injection from user %d: %s", user.id, input_guard["injection_match"])
        return {"answer": "I can't process that request.", "blocked": True}

    if input_guard["pii_redactions"]:
        logger.info("Redacted %d PII match(es) from user %d's query before sending to the LLM.", input_guard["pii_redactions"], user.id)

    try:
        answer = run_agent(input_guard["redacted_text"], user_id=user.id)
    except Exception as e:
        logger.exception("Chat request failed for user %d", user.id)
        raise HTTPException(status_code=500, detail=str(e))

    output_guard = guard_output(answer)
    if output_guard["blocked"]:
        logger.warning("Blocked disallowed output for user %d: %s", user.id, output_guard["match"])

    answer_text, sources = split_sources(output_guard["text"])
    return {"answer": answer_text, "sources": sources, "blocked": output_guard["blocked"]}


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(
        db, key=f"user:{user.id}", endpoint="upload",
        max_requests=settings.UPLOAD_RATE_LIMIT_PER_HOUR, window_minutes=60,
    )

    document = None
    try:
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Created (and flushed, not yet committed) before indexing so its id
        # exists to tag each vector chunk with - that's what lets a single
        # document be found and deleted later without touching a different
        # upload that happens to share the same filename. If anything below
        # fails, this row is never committed, so it never actually persists.
        # Makes sure the vector collection and its filter indexes exist BEFORE
        # the first document is indexed (a collection created by the upload
        # itself has no indexes, and the next chat would then fail).
        get_qdrant_client()

        document = Document(user_id=user.id, filename=file.filename, blob_url=None, chunks_indexed=0)
        db.add(document)
        db.flush()

        chunks = process_and_upsert(file_bytes, file.filename, user_id=user.id, document_id=document.id)
        blob_url = upload_to_blob_storage(file_bytes, file.filename)  # best-effort, may be None

        document.blob_url = blob_url
        document.chunks_indexed = chunks
        db.commit()

        return {"status": "success", "document_id": document.id, "blob_url": blob_url, "chunks": chunks}
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Upload failed for user %d", user.id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_my_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """A user's own uploaded documents - not anyone else's, enforced by
    the same user_id filter as /chat and /upload above."""
    docs = db.query(Document).filter(Document.user_id == user.id).order_by(Document.uploaded_at.desc()).all()
    return [{"id": d.id, "filename": d.filename, "uploaded_at": d.uploaded_at.isoformat(), "chunks_indexed": d.chunks_indexed} for d in docs]


@app.delete("/documents/{document_id}")
def delete_document(document_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Removes a document's vector chunks from Qdrant and its row from the
    database. Owner-only (or admin) - checked explicitly here rather than
    relying on the query filter alone, so a non-owner gets a clear 403
    instead of a misleading 404 that could be mistaken for "already
    deleted" during debugging."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise HTTPException(404, "Document not found")
    if document.user_id != user.id and user.role != Role.admin:
        raise HTTPException(403, "You don't own this document")

    delete_document_chunks(get_qdrant_client(), document_id=document_id, user_id=document.user_id)

    db.delete(document)
    db.commit()
    return {"status": "deleted", "document_id": document_id}


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
    """For support/debugging - an admin can see what a user uploaded, a
    regular user still cannot see this about anyone but themselves."""
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
    """A rough usage overview - number of accounts and documents. Not a
    real per-request cost tracker (that needs the LLM call itself to
    report token counts back, which is a separate piece of work), just
    what's cheaply knowable from the database alone right now."""
    return {
        "total_users": db.query(func.count(User.id)).scalar(),
        "active_users": db.query(func.count(User.id)).filter(User.is_active == 1).scalar(),
        "total_documents": db.query(func.count(Document.id)).scalar(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
