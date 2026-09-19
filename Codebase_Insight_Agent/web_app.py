"""
The public, recruiter-facing website. Wraps portfolio.py directly - same
shared logic mcp_server.py uses, different front door.

Design, in one sentence each:
  - No login for visitors: the whole point is zero friction for someone
    clicking a link on a CV.
  - Rate limiting (rate_limit.py) instead of a login wall: stops one
    visitor from burning the Gemini quota for everyone else.
  - Guardrails (guardrails.py, same module as Azure_RAG_Assistant /
    customer_support_copilot): stops prompt-injection turning this into
    a general-purpose chatbot or saying something embarrassing on a
    page with your name on it.
  - One protected /admin/* route group: for you, to see what's being
    asked and what got blocked. Nobody else gets an account at all -
    see scripts/create_admin.py for how that one account is created.

Run it with: uvicorn web_app:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

import config
import portfolio
from guardrails import guard_input, guard_output
from rate_limit import is_rate_limited
from database import get_db, init_db
from models import AdminUser, QueryLog
from auth import (
    verify_password,
    create_access_token, create_refresh_token,
    verify_refresh_token, revoke_refresh_token,
    require_admin,
)

# Built once at startup, shared across every request - see lifespan below.
indexes = None
router = None
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexes, router, agent

    init_db()

    # load_all_indexes(), not build_all_indexes() - requires
    # notebooks/01_indexing.ipynb to have been run first against a real
    # QDRANT_URL. See portfolio.py's load_all_indexes() docstring and
    # the README's "Provisioning the index" section.
    print("Loading indexes...")
    indexes = portfolio.load_all_indexes()
    router = portfolio.build_router()
    agent = portfolio.build_agent(indexes, router)
    print(f"Ready. {len(indexes)} project(s) loaded.")

    yield


app = FastAPI(title="Codebase Insight Agent - public site", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(max_length=500)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


def get_client_ip(request: Request) -> str:
    """Azure Container Apps sits behind a reverse proxy, so the real
    visitor IP is in X-Forwarded-For (its first entry - the rest are
    intermediate proxies), not request.client.host, which would just be
    the proxy's own address. Falls back to request.client.host for local
    dev, where there's no proxy in front at all."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------- public

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok", "ready": agent is not None}


@app.post("/ask")
async def ask(body: AskRequest, request: Request, db: Session = Depends(get_db)):
    ip = get_client_ip(request)

    if is_rate_limited(ip):
        db.add(QueryLog(ip_address=ip, question=body.question, rate_limited=True))
        db.commit()
        raise HTTPException(
            429,
            f"Rate limit reached ({config.RATE_LIMIT_MAX_REQUESTS} questions per "
            f"{config.RATE_LIMIT_WINDOW_SECONDS // 60} minutes). Try again shortly.",
        )

    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    input_guard = guard_input(body.question)
    if input_guard["blocked"]:
        db.add(QueryLog(
            ip_address=ip, question=body.question, blocked=True,
            block_reason=input_guard["injection_match"],
        ))
        db.commit()
        return {"answer": "I can't process that request.", "blocked": True}

    if agent is None:
        raise HTTPException(503, "Still starting up, try again in a moment.")

    # portfolio.ask() is synchronous and ends up calling llama_index's
    # Gemini client, which internally does asyncio.run() - that blows up
    # if called directly from here, since this endpoint is already
    # running inside uvicorn's event loop. run_in_threadpool moves the
    # whole blocking call to a worker thread, which has no event loop of
    # its own, so llama_index's asyncio.run() works fine there.
    result = await run_in_threadpool(portfolio.ask, agent, input_guard["redacted_text"])

    output_guard = guard_output(result["answer"])

    db.add(QueryLog(
        ip_address=ip,
        question=body.question,
        answer=output_guard["text"],
        blocked=output_guard["blocked"],
        block_reason=output_guard["match"] if output_guard["blocked"] else None,
        target_projects=",".join(result["projects"]),
    ))
    db.commit()

    return {"answer": output_guard["text"], "blocked": output_guard["blocked"], "projects": result["projects"]}


# ---------------------------------------------------------------- admin auth

@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    unauthorized = HTTPException(401, "Incorrect email or password")

    admin = db.query(AdminUser).filter(AdminUser.email == body.email).first()
    if admin is None or not verify_password(body.password, admin.password_hash):
        raise unauthorized
    if not admin.is_active:
        raise HTTPException(403, "This account has been deactivated")

    access_token = create_access_token(admin.id)
    refresh_token = create_refresh_token(db, admin.id)
    return {"access_token": access_token, "refresh_token": refresh_token}


@app.post("/auth/refresh")
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    record = verify_refresh_token(db, body.refresh_token)
    if record is None:
        raise HTTPException(401, "Invalid, expired, or already-used refresh token")
    return {"access_token": create_access_token(record.admin_id)}


@app.post("/auth/logout")
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    revoke_refresh_token(db, body.refresh_token)
    return {"status": "logged out"}


# ---------------------------------------------------------------- admin only

@app.get("/admin/stats")
def admin_stats(admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)):
    return {
        "total_questions": db.query(func.count(QueryLog.id)).scalar(),
        "blocked_count": db.query(func.count(QueryLog.id)).filter(QueryLog.blocked.is_(True)).scalar(),
        "rate_limited_count": db.query(func.count(QueryLog.id)).filter(QueryLog.rate_limited.is_(True)).scalar(),
        "unique_visitors": db.query(func.count(func.distinct(QueryLog.ip_address))).scalar(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/admin/logs")
def admin_logs(
    limit: int = 50,
    offset: int = 0,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    logs = (
        db.query(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .offset(offset)
        .limit(min(limit, 200))  # hard cap so a bad ?limit= can't force one giant query
        .all()
    )
    return [
        {
            "id": q.id,
            "ip_address": q.ip_address,
            "question": q.question,
            "answer": q.answer,
            "blocked": q.blocked,
            "block_reason": q.block_reason,
            "rate_limited": q.rate_limited,
            "target_projects": q.target_projects,
            "created_at": q.created_at.isoformat(),
        }
        for q in logs
    ]
