"""
Standalone entrypoint for Streamlit Community Cloud.

Unlike ui/streamlit_app.py (which talks to a separate FastAPI backend over
HTTP -- the right setup for local dev with two terminals, or Docker with two
services), Streamlit Cloud only runs a single Python file with no separate
backend process. So this version imports the RAG pipeline functions
directly and calls them in-process instead of making HTTP requests.

Since this is the file actually deployed publicly (documents-mind.streamlit.app),
it needs real accounts, not just an anonymous per-visitor session:
  1. Real login (email + password, checked against the same User table the
     FastAPI backend uses) instead of a random session ID stashed in the
     URL. The old scheme meant anyone who saw/guessed a "?sid=..." URL
     could open someone else's document set - a real account fixes that.
     Isolation itself still reuses the exact same mechanism as before
     (per-session Chroma collections, app/vectorstore.py) - only the
     source of the identifier changed, from a URL param to a verified
     login.
  2. Shared Gemini API key -- the deployer's own key (set via
     GEMINI_API_KEY in secrets/.env), used for every visitor's requests.
     A per-account daily question limit (DAILY_QUERY_LIMIT) protects the
     shared quota from being drained by any one visitor.

No Google OAuth here (see app/oauth.py's docstring for why) - Streamlit
Community Cloud has no separate server for Google to redirect back to
over HTTP the way the FastAPI backend does; this stays email+password.

Trade-off worth naming: since login lives in st.session_state, not a URL
param, a hard page reload logs you out (session state doesn't survive
that) - the old URL-based scheme survived reloads but was also exactly
the security hole this fixes, so this is the right trade to make.

To deploy: point Streamlit Community Cloud's "Main file path" at
    rag_qa_documind/streamlit_app.py
Required Secrets:
    JWT_SECRET_KEY = "..."   (only used indirectly, via app.auth's password
                              hashing - no tokens are actually issued here,
                              see the login form below)
    GEMINI_API_KEY = "..."   (shared key used for every visitor's requests
                              - see DAILY_QUERY_LIMIT below for the abuse
                              guard on this)
    DATABASE_URL = "..."     (optional - defaults to a local SQLite file,
                              which will NOT persist across Streamlit Cloud
                              redeploys; use a real Azure SQL DATABASE_URL
                              for accounts that actually persist - see
                              README's "Accounts and a real database")
and optionally:
    GEMINI_MODEL = "gemini-3.1-flash-lite"
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "GEMINI_MODEL" in st.secrets:
    os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
if "DATABASE_URL" in st.secrets:
    os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
if "JWT_SECRET_KEY" in st.secrets:
    os.environ["JWT_SECRET_KEY"] = st.secrets["JWT_SECRET_KEY"]

from app.config import settings
from app.ingest import ingest_file, load_text
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection
from app.guardrails import guard_input, guard_output
from app.database import SessionLocal, init_db
from app.models import User
from app.auth import hash_password, verify_password

init_db()

DAILY_QUERY_LIMIT = 15


def get_remaining_queries(db, user):
    from datetime import date
    today = date.today().isoformat()
    if user.daily_query_date != today:
        user.daily_query_count = 0
        user.daily_query_date = today
        db.commit()
    return DAILY_QUERY_LIMIT - user.daily_query_count

st.set_page_config(page_title="DocuMind", page_icon="📚")
st.title("📚 DocuMind — Ask your documents")


# ------------------------------------------------------------ login screen

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_email = None

if not st.session_state.user_id:
    st.subheader("Log in")
    email = st.text_input("Email")
    password = st.text_input("Password (8+ characters)", type="password")

    col1, col2 = st.columns(2)
    db = SessionLocal()

    if col1.button("Log in", use_container_width=True):
        user = db.query(User).filter(User.email == email).first()
        if user is None or user.password_hash is None or not verify_password(password, user.password_hash):
            st.error("Incorrect email or password")
        elif not user.is_active:
            st.error("This account has been deactivated")
        else:
            st.session_state.user_id = user.id
            st.session_state.user_email = user.email
            st.rerun()

    if col2.button("Register", use_container_width=True):
        if len(password) < 8:
            st.error("Password must be at least 8 characters")
        elif db.query(User).filter(User.email == email).first() is not None:
            st.error("An account with this email already exists")
        else:
            user = User(email=email, password_hash=hash_password(password))
            db.add(user)
            db.commit()
            db.refresh(user)
            st.session_state.user_id = user.id
            st.session_state.user_email = user.email
            st.rerun()

    db.close()
    st.stop()  # don't render the rest of the app until logged in


# ------------------------------------------------------------ logged in

session_id = str(st.session_state.user_id)  # same identifier concept as
# before (app/vectorstore.py's session-keyed Chroma collections), just
# sourced from a verified account now instead of a URL parameter.

with st.sidebar:
    st.caption(f"Logged in as **{st.session_state.user_email}**")
    if st.button("Log out"):
        for key in ("user_id", "user_email", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    api_key = None  # shared server-side key, see app/config.py
    db_check = SessionLocal()
    user_row = db_check.query(User).filter(User.id == st.session_state.user_id).first()
    remaining = get_remaining_queries(db_check, user_row)
    db_check.close()
    st.caption(f"Questions left today: **{remaining}/{DAILY_QUERY_LIMIT}**")

    st.divider()
    st.caption("Documents you upload are private to your account.")

    st.header("Upload documents")
    uploaded = st.file_uploader(
        "Add .txt, .md, or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest files"):
        for f in uploaded:
            import tempfile
            suffix = os.path.splitext(f.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            try:
                extracted = load_text(tmp_path)
                char_count = len(extracted.strip())
                if char_count < 200:
                    st.warning(
                        f"⚠️ {f.name}: only extracted {char_count} characters of text. "
                        f"This usually means the PDF is a scanned image rather than "
                        f"real text -- try a different file, or one with a text layer "
                        f"(e.g. exported from Word/Google Docs rather than scanned)."
                    )
                else:
                    n_chunks = ingest_file(tmp_path, source_name=f.name, session_id=session_id)
                    st.success(f"{f.name}: {n_chunks} chunks indexed")
            except Exception as e:
                st.error(f"{f.name}: {e}")
            finally:
                os.remove(tmp_path)

    st.divider()
    try:
        count = get_collection(session_id).count()
        st.caption(f"Indexed chunks: {count}")
    except Exception as e:
        st.caption(f"⚠️ {e}")

    if st.button("Clear my index"):
        reset_collection(session_id)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input(
    "Ask a question about your documents...",
    disabled=remaining <= 0,
):
    if remaining <= 0:
        st.warning("Daily question limit reached. Try again tomorrow.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_guard = guard_input(question)
            if input_guard["blocked"]:
                answer_text = "I can't process that request."
                st.warning(answer_text)
            else:
                try:
                    result = answer_question(input_guard["redacted_text"], session_id=session_id, api_key=api_key)
                    output_guard = guard_output(result["answer"])
                    answer_text = output_guard["text"]
                    st.markdown(answer_text)
                    db2 = SessionLocal()
                    u = db2.query(User).filter(User.id == st.session_state.user_id).first()
                    u.daily_query_count += 1
                    db2.commit()
                    db2.close()
                    if result["sources"]:
                        with st.expander("📄 View source passages"):
                            for s in result["sources"]:
                                st.write(f"**{s['source']}** · relevance {s['score']}")
                                preview = s["text"][:300].strip()
                                st.code(preview if preview else "(empty chunk)")
                except Exception as e:
                    answer_text = f"Error: {e}"
                    st.error(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
