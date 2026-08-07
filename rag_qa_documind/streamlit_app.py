"""
Standalone entrypoint for Streamlit Community Cloud.

Unlike ui/streamlit_app.py (which talks to a separate FastAPI backend over
HTTP -- the right setup for local dev with two terminals, or Docker with two
services), Streamlit Cloud only runs a single Python file with no separate
backend process. So this version imports the RAG pipeline functions
directly and calls them in-process instead of making HTTP requests.

Since this is the file actually deployed publicly (documents-mind.streamlit.app),
it adds two things ui/streamlit_app.py doesn't need:
  1. Per-session document isolation -- each visitor gets a private Chroma
     collection, keyed by a random ID stored in their browser session, so
     concurrent strangers never see each other's uploaded documents.
  2. An optional shared-passcode gate -- set APP_PASSWORD in Secrets to
     require a code before anyone can use the app. This protects your
     Gemini free-tier quota from random traffic. Leave it unset (e.g. for
     local dev) and the app stays open, same as before.

To deploy: point Streamlit Community Cloud's "Main file path" at
    rag_qa_documind/streamlit_app.py
and set GEMINI_API_KEY / GEMINI_MODEL (and optionally APP_PASSWORD) in the
app's Secrets (TOML), e.g.:
    GEMINI_API_KEY = "your-key-here"
    GEMINI_MODEL = "gemini-3.1-flash-lite"
    APP_PASSWORD = "choose-a-passcode"
"""
import os
import sys
import tempfile
import uuid

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from app.config import settings
from app.ingest import ingest_file, load_text
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection

st.set_page_config(page_title="DocuMind", page_icon="📚")

# --- Optional shared-passcode gate ------------------------------------
# Set APP_PASSWORD in Secrets to require a code before anyone can use the
# app. Without it (e.g. local dev, or if you're fine with open access),
# the app behaves exactly as before -- no gate at all.
_app_password = st.secrets.get("APP_PASSWORD")
if _app_password and not st.session_state.get("authed"):
    st.title("📚 DocuMind")
    entered = st.text_input("Enter access code to continue", type="password")
    if entered:
        if entered == _app_password:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect code.")
    st.stop()

# --- Per-session document isolation ------------------------------------
# Each browser session gets its own private collection so concurrent
# visitors never see each other's uploaded documents. This resets if the
# tab is closed or the app restarts -- it isn't meant to be durable
# storage, just isolation between visitors while they're using the app.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
session_id = st.session_state.session_id

st.title("📚 DocuMind — Ask your documents")

with st.sidebar:
    if settings.gemini_api_key:
        st.success(f"🟢 Connected — using **{settings.gemini_model}**")
    else:
        st.error("🔴 Gemini API key not configured — add GEMINI_API_KEY in Secrets")

    st.caption(
        "Documents you upload are private to this session and aren't "
        "visible to other visitors."
    )

    st.header("Upload documents")
    uploaded = st.file_uploader(
        "Add .txt, .md, or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest files"):
        for f in uploaded:
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

    if st.button("Clear index"):
        reset_collection(session_id)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(question, session_id=session_id)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("📄 View source passages"):
                        for s in result["sources"]:
                            st.write(f"**{s['source']}** · relevance {s['score']}")
                            preview = s["text"][:300].strip()
                            st.code(preview if preview else "(empty chunk)")
                answer_text = result["answer"]
            except Exception as e:
                answer_text = f"Error: {e}"
                st.error(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
