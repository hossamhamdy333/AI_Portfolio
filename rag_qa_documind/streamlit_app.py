"""
Standalone entrypoint for Streamlit Community Cloud.
"""
import os
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from app.config import settings
from app.ingest import ingest_file
from app.rag import answer_question
from app.vectorstore import reset_collection, get_collection

st.set_page_config(page_title="DocuMind", page_icon="📚")
st.title("📚 DocuMind — Ask your documents")

with st.sidebar:
    if settings.gemini_api_key:
        st.success(f"🟢 Connected — using **{settings.gemini_model}**")
    else:
        st.error("🔴 Gemini API key not configured — add GEMINI_API_KEY in Secrets")

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
                n_chunks = ingest_file(tmp_path, source_name=f.name)
                st.success(f"{f.name}: {n_chunks} chunks indexed")
            except Exception as e:
                st.error(f"{f.name}: {e}")
            finally:
                os.remove(tmp_path)

    st.divider()
    try:
        count = get_collection().count()
        st.caption(f"Indexed chunks: {count}")
    except Exception as e:
        st.caption(f"⚠️ {e}")

    if st.button("Clear index"):
        reset_collection()
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
                result = answer_question(question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.write(f"- {s['source']} (relevance {s['score']})")
                answer_text = result["answer"]
            except Exception as e:
                answer_text = f"Error: {e}"
                st.error(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})