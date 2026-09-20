"""
FastAPI backend for the AI Support Copilot.

Supports multiple named conversations per user, like Claude's chat
sidebar, instead of one endless thread per account.
"""

import json
import re
import logging
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
import os

from src.retriever import KBRetriever
from src.evaluate import evaluate_faithfulness
from src import llm_backend
from src.guardrails import guard_input, guard_output
from src.config import settings
from src.database import get_db, init_db, SessionLocal
from src.models import User, Conversation, ChatMessage, MessageRole, Role
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

_retriever = None

# llama.cpp's model object isn't safe to call from two threads at once. The
# model work runs in a worker thread (below) so it no longer freezes the whole
# server while it generates; this lock keeps generations one at a time.
_model_lock = threading.Lock()

# The model is deterministic (temperature 0), so the same question always gets
# the same answer. Remembering recent answers makes a repeated question
# instant instead of another 30+ seconds of CPU inference. In memory only: it
# resets on restart, and the answer still goes through the output guard every
# time it is served.
#
# A REWORDED question ("where is my order please") also hits the cache, but
# only if it retrieves the same knowledge-base article AND its meaning is very
# close (embedding similarity >= settings.CACHE_SIMILARITY). Requiring the same
# article means it can never serve an answer written for a different topic.
_ANSWER_CACHE_MAX = 200
# key -> (context, answer, question vector or None, hit_token_cap)
_answer_cache: "OrderedDict[str, tuple]" = OrderedDict()
_answer_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    return " ".join(text.lower().split())


def _cache_get(key: str):
    with _answer_cache_lock:
        hit = _answer_cache.get(key)
        if hit is not None:
            _answer_cache.move_to_end(key)
        return hit


def _cache_put(key: str, value) -> None:
    with _answer_cache_lock:
        _answer_cache[key] = value
        _answer_cache.move_to_end(key)
        while len(_answer_cache) > _ANSWER_CACHE_MAX:
            _answer_cache.popitem(last=False)


def _embed(text: str):
    """Unit-length embedding of the question, or None if unavailable."""
    try:
        return _retriever.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    except Exception:
        return None


def _semantic_get(context: str, vec):
    """A cached (answer, hit_token_cap) for a reworded version of this question."""
    if vec is None:
        return None
    best, best_sim = None, settings.CACHE_SIMILARITY
    with _answer_cache_lock:
        for cached_ctx, cached_answer, cached_vec, cached_cut in _answer_cache.values():
            if cached_vec is None or cached_ctx != context:
                continue
            sim = float((vec * cached_vec).sum())
            if sim >= best_sim:
                best, best_sim = (cached_answer, cached_cut), sim
    return best


_SENTENCE_END = re.compile(r"(?<!\d)[.!?](?=\s|$)")


def _tidy(answer: str, hit_cap: bool) -> str:
    """If the answer was cut off by the token cap, end it at the last complete
    sentence instead of mid-word. Leaves it alone if that would throw away
    more than half of it, and never touches an answer that finished on its own."""
    a = answer.strip()
    if not hit_cap or not a or a[-1] in ".!?\"')":
        return a
    ends = [m.end() for m in _SENTENCE_END.finditer(a)]
    if ends and ends[-1] >= 30 and ends[-1] >= len(a) / 2:
        return a[: ends[-1]]
    return a


def _build_prompt(context: str, text: str) -> str:
    return f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Context: {context}
Query: {text}
<|assistant|>
"""


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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Always answer with JSON. Without this, an unexpected error comes back
    as the plain text 'Internal Server Error', which the frontend can't
    parse ("Unexpected token 'I' ... is not valid JSON")."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on my side. Please try again."},
    )


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
    conversation_id: int


class ChatResponse(BaseModel):
    response: str
    context: str
    blocked: bool = False


class NewConversationResponse(BaseModel):
    id: int
    title: str


# ---------------------------------------------------------------- auth

@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if db.query(User).filter(User.email == body.email).first() is not None:
        raise HTTPException(409, "An account with this email already exists")

    user = User(email=body.email, password_hash=hash_password(body.password), role=Role.user)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with this email already exists")
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


# ---------------------------------------------------------------- conversations

@app.post("/conversations", response_model=NewConversationResponse, status_code=201)
def create_conversation(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convo = Conversation(user_id=user.id, title="New chat")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return NewConversationResponse(id=convo.id, title=convo.title)


@app.get("/conversations")
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convos = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in convos]


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user.id).first()
    if convo is None:
        raise HTTPException(404, "Conversation not found")
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": m.role.value, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages]


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user.id).first()
    if convo is None:
        raise HTTPException(404, "Conversation not found")
    db.delete(convo)
    db.commit()
    return {"status": "deleted"}


# ---------------------------------------------------------------- chat

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not llm_backend.is_ready() or _retriever is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    convo = db.query(Conversation).filter(Conversation.id == request.conversation_id, Conversation.user_id == user.id).first()
    if convo is None:
        raise HTTPException(404, "Conversation not found")

    input_guard = guard_input(request.query)
    if input_guard["blocked"]:
        logger.warning("Blocked a suspected prompt injection from user %d: %s", user.id, input_guard["injection_match"])
        return ChatResponse(response="I can't process that request.", context="", blocked=True)

    if input_guard["pii_redactions"]:
        logger.info("Redacted %d PII match(es) from user %d's message.", input_guard["pii_redactions"], user.id)

    try:
        def _run_model(text: str):
            key = _cache_key(text)
            hit = _cache_get(key)
            if hit is not None:
                return hit[0], hit[1], hit[3]
            with _model_lock:
                ctx = _retriever.retrieve(text)
                vec = _embed(text)
                similar = _semantic_get(ctx, vec)
                if similar is not None:
                    out, cut = similar
                else:
                    out = llm_backend.generate(_build_prompt(ctx, text))
                    cut = getattr(llm_backend, "last_finish_reason", "stop") == "length"
            _cache_put(key, (ctx, out, vec, cut))
            return ctx, out, cut

        context, ai_text, hit_cap = await run_in_threadpool(_run_model, input_guard["redacted_text"])
        ai_text = _tidy(ai_text, hit_cap)

        output_guard = guard_output(ai_text)
        if output_guard["blocked"]:
            logger.warning("Blocked disallowed output for user %d: %s", user.id, output_guard["match"])
        final_text = output_guard["text"]

        if settings.ENABLE_EVAL:
            eval_result = evaluate_faithfulness(request.query, context, final_text, db)
            if eval_result and not eval_result.get("is_faithful", True):
                logger.warning("Faithfulness check flagged this response: %s", eval_result.get("reason"))

        db.add(ChatMessage(conversation_id=convo.id, user_id=user.id, role=MessageRole.user, content=request.query))
        db.add(ChatMessage(conversation_id=convo.id, user_id=user.id, role=MessageRole.assistant, content=final_text))
        convo.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Auto-title a fresh conversation from its first message.
        if convo.title == "New chat":
            convo.title = request.query.strip()[:50]

        db.commit()

        return ChatResponse(response=final_text, context=context, blocked=output_guard["blocked"])

    except Exception as e:
        logger.exception("Chat generation failed for user %d", user.id)
        raise HTTPException(status_code=500, detail=str(e))


def _ndjson(obj: dict) -> str:
    return json.dumps(obj) + "\n"


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Same as /chat, but the answer is sent piece by piece (one JSON object
    per line) so the page can show it while it is still being written.
    Events: {"type":"token","text":...} repeated, then one
    {"type":"final","response":...,"context":...,"blocked":...} (the
    guarded text, which replaces what was shown), or {"type":"error",...}."""
    if not llm_backend.is_ready() or _retriever is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    convo = db.query(Conversation).filter(Conversation.id == request.conversation_id, Conversation.user_id == user.id).first()
    if convo is None:
        raise HTTPException(404, "Conversation not found")
    user_id, convo_id = user.id, convo.id

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    input_guard = guard_input(request.query)
    if input_guard["blocked"]:
        logger.warning("Blocked a suspected prompt injection from user %d: %s", user_id, input_guard["injection_match"])
        blocked = _ndjson({"type": "final", "response": "I can't process that request.", "context": "", "blocked": True})
        return StreamingResponse(iter([blocked]), media_type="application/x-ndjson", headers=headers)

    if input_guard["pii_redactions"]:
        logger.info("Redacted %d PII match(es) from user %d's message.", input_guard["pii_redactions"], user_id)

    text = input_guard["redacted_text"]
    query = request.query

    def events():
        try:
            key = _cache_key(text)
            hit = _cache_get(key)
            if hit is not None:
                ctx, ai_text, _vec, hit_cap = hit
                yield _ndjson({"type": "token", "text": ai_text})
            else:
                parts, started, similar = [], False, None
                with _model_lock:
                    ctx = _retriever.retrieve(text)
                    vec = _embed(text)
                    similar = _semantic_get(ctx, vec)
                    if similar is None:
                        for piece in llm_backend.generate_stream(_build_prompt(ctx, text)):
                            if not started:
                                piece = piece.lstrip()
                                if not piece:
                                    continue
                                started = True
                            parts.append(piece)
                            yield _ndjson({"type": "token", "text": piece})
                        hit_cap = getattr(llm_backend, "last_finish_reason", "stop") == "length"
                if similar is not None:
                    ai_text, hit_cap = similar
                    yield _ndjson({"type": "token", "text": ai_text})
                else:
                    ai_text = "".join(parts).strip()
                _cache_put(key, (ctx, ai_text, vec, hit_cap))
            ai_text = _tidy(ai_text, hit_cap)

            output_guard = guard_output(ai_text)
            if output_guard["blocked"]:
                logger.warning("Blocked disallowed output for user %d: %s", user_id, output_guard["match"])
            final_text = output_guard["text"]

            # Own session: the request's session may already be closed once
            # a streaming response is under way.
            with SessionLocal() as sdb:
                if settings.ENABLE_EVAL:
                    eval_result = evaluate_faithfulness(query, ctx, final_text, sdb)
                    if eval_result and not eval_result.get("is_faithful", True):
                        logger.warning("Faithfulness check flagged this response: %s", eval_result.get("reason"))
                sdb.add(ChatMessage(conversation_id=convo_id, user_id=user_id, role=MessageRole.user, content=query))
                sdb.add(ChatMessage(conversation_id=convo_id, user_id=user_id, role=MessageRole.assistant, content=final_text))
                c = sdb.query(Conversation).filter(Conversation.id == convo_id).first()
                c.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                if c.title == "New chat":
                    c.title = query.strip()[:50]
                sdb.commit()

            yield _ndjson({"type": "final", "response": final_text, "context": ctx, "blocked": output_guard["blocked"]})
        except Exception:
            logger.exception("Streaming chat failed for user %d", user_id)
            yield _ndjson({"type": "error", "detail": "Something went wrong on my side. Please try again."})

    return StreamingResponse(events(), media_type="application/x-ndjson", headers=headers)


@app.get("/health/db")
async def health_db():
    """Runs a trivial query so a paused database starts waking up."""
    from sqlalchemy import text
    def _ping():
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    try:
        await run_in_threadpool(_ping)
        return {"db": "ok"}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"db": "unavailable", "detail": str(exc)[:200]})


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
