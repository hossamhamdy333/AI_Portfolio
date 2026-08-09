"""
Streamlit front-end for DocuMind.
Run with: streamlit run ui/streamlit_app.py
Expects the FastAPI backend to be running on API_URL (default localhost:8000).

Each browser session gets its own private document index: a random
session ID is generated on first load and sent as an X-Session-Id header
on every request, so the FastAPI backend (see app/main.py) keeps this
session's uploads isolated from anyone else hitting the same backend.
"""
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

HEADERS = {"X-Session-Id": st.session_state.session_id}

st.set_page_config(page_title="DocuMind", page_icon="📚")
st.title("📚 DocuMind — Ask your documents")

with st.sidebar:
    st.caption(
        "Documents you upload are private to this browser session and "
        "aren't visible to other users of this backend."
    )

    st.header("Upload documents")
    uploaded = st.file_uploader(
        "Add .txt, .md, or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest files"):
        for f in uploaded:
            resp = requests.post(
                f"{API_URL}/ingest",
                files={"file": (f.name, f.getvalue())},
                headers=HEADERS,
            )
            if resp.ok:
                st.success(f"{f.name}: {resp.json()['chunks_indexed']} chunks indexed")
            else:
                st.error(f"{f.name}: {resp.text}")

    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", headers=HEADERS, timeout=5).json()
        st.caption(f"Indexed chunks: {health['indexed_chunks']}")
    except Exception:
        st.caption("⚠️ Backend not reachable")

    if st.button("Clear index"):
        requests.post(f"{API_URL}/reset", headers=HEADERS)
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
            resp = requests.post(
                f"{API_URL}/query", json={"question": question}, headers=HEADERS
            )
            if resp.ok:
                data = resp.json()
                st.markdown(data["answer"])
                if data["sources"]:
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.write(f"- {s['source']} (relevance {s['score']})")
                answer_text = data["answer"]
            else:
                answer_text = f"Error: {resp.text}"
                st.error(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
