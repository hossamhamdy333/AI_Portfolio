import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func

from azure_storage import upload_to_blob_storage
from text_processing import process_and_upsert
from agent import run_agent
from guardrails import guard_input, guard_output
from config import settings, logger
from database import get_db, init_db
from models import User, Document, Role
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    verify_refresh_token, revoke_refresh_token,
    get_current_user, require_admin,
)
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
    return {"status": "ok"}


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
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
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

    return {"answer": output_guard["text"], "blocked": output_guard["blocked"]}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        chunks = process_and_upsert(file_bytes, file.filename, user_id=user.id)
        blob_url = upload_to_blob_storage(file_bytes, file.filename)  # best-effort, may be None

        db.add(Document(user_id=user.id, filename=file.filename, blob_url=blob_url, chunks_indexed=chunks))
        db.commit()

        return {"status": "success", "blob_url": blob_url, "chunks": chunks}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed for user %d", user.id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_my_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """A user's own uploaded documents - not anyone else's, enforced by
    the same user_id filter as /chat and /upload above."""
    docs = db.query(Document).filter(Document.user_id == user.id).order_by(Document.uploaded_at.desc()).all()
    return [{"id": d.id, "filename": d.filename, "uploaded_at": d.uploaded_at.isoformat(), "chunks_indexed": d.chunks_indexed} for d in docs]


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
