"""
Standalone entrypoint for Streamlit Community Cloud.

Unlike ui/streamlit_app.py (which talks to a separate FastAPI backend over
HTTP -- the right setup for local dev with two terminals, or Docker with two
services), Streamlit Cloud only runs a single Python file with no separate
backend process. So this version imports the RAG pipeline functions
directly and calls them in-process instead of making HTTP requests.

Since this is the file actually deployed publicly (documents-mind.streamlit.app),
it adds things ui/streamlit_app.py doesn't need:
  1. Per-session document isolation -- each visitor gets a private Chroma
     collection, keyed by a random ID, so concurrent strangers never see
     each other's uploaded documents.
  2. Bring-your-own Gemini API key -- each visitor pastes in their own free
     Gemini key in the sidebar, and it's used only for their own requests.
     Nothing is shared across visitors, so there's no shared quota to
     protect and no need for an access gate.

To deploy: point Streamlit Community Cloud's "Main file path" at
    rag_qa_documind/streamlit_app.py
and (optionally) set GEMINI_MODEL in the app's Secrets if you want a
different default model than gemini-3.1-flash-lite:
    GEMINI_MODEL = "gemini-3.1-flash-lite"
"""
import os
import sys
import tempfile
import uuid

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "GEMINI_MODEL" in st.secrets:
    os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]

from app.config import settings
from app.ingest import ingest_file, load_text
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection

st.set_page_config(page_title="DocuMind", page_icon="📚")

params = st.query_params

# --- Per-session document isolation, persisted via the URL ---------------
# Each visitor gets their own private collection. The session ID is also
# stashed in the URL query string (not a cookie), so reloading the same
# URL keeps using the same collection instead of starting a fresh one.
# Opening the bare share link (no ?sid=... in the URL) always starts a
# brand-new, empty, isolated session -- this doesn't let visitors see or
# guess into each other's document sets.
if "session_id" not in st.session_state:
    if "sid" in params:
        st.session_state.session_id = params["sid"]
    else:
        st.session_state.session_id = uuid.uuid4().hex
        params["sid"] = st.session_state.session_id
session_id = st.session_state.session_id

st.title("📚 DocuMind — Ask your documents")

with st.sidebar:
    st.header("Your Gemini API key")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        key="gemini_api_key",
        label_visibility="collapsed",
        placeholder="Paste your Gemini API key",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    if api_key:
        st.success(f"🟢 Using your key with **{settings.gemini_model}**")
    else:
        st.info(
            "🔑 Paste a free Gemini API key to ask questions. "
            "Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) "
            "— no credit card needed."
        )
    st.caption(
        "Your key is kept only in this browser session's memory — it's "
        "never saved on the server, logged, or shared with other "
        "visitors. Each visitor uses their own key and their own free "
        "Gemini quota."
    )

    st.divider()
    st.caption(
        "Documents you upload are private to your session and aren't "
        "visible to other visitors. Bookmark this exact page URL to keep "
        "the same session next time."
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

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to start asking questions.")

if question := st.chat_input(
    "Ask a question about your documents...",
    disabled=not api_key,
):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(question, session_id=session_id, api_key=api_key)
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
