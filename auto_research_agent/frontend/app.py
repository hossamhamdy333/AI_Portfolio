"""
Streamlit UI. Talks to the FastAPI backend over HTTP.
Run: streamlit run frontend/app.py
Set BACKEND_URL if the API isn't on localhost:8000 (e.g. an ngrok URL from Colab).
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Auto Research Agent", layout="centered")
st.title("Auto Research Agent")
st.caption("Planner -> Researcher (web + RAG) -> [human review if flagged] -> Analyst -> Writer")

with st.sidebar:
    st.subheader("1. Add knowledge (optional)")
    uploaded = st.file_uploader("PDF / DOCX / XLSX / MP3 / MP4 / PNG / JPG")
    if uploaded and st.button("Index file"):
        with st.spinner("Ingesting..."):
            resp = requests.post(
                f"{BACKEND_URL}/upload_knowledge",
                files={"file": (uploaded.name, uploaded.getvalue())},
            )
        if resp.ok:
            st.success(f"Indexed {resp.json()['chunks_indexed']} chunks from {uploaded.name}")
        else:
            st.error(resp.text)

st.subheader("2. Run a task")
task = st.text_area("Task", placeholder="e.g. Compare the top 3 open-source Arabic NLP models")

if "pending_thread" not in st.session_state:
    st.session_state.pending_thread = None

if st.button("Run", type="primary") and task:
    with st.spinner("Agents working..."):
        resp = requests.post(f"{BACKEND_URL}/execute_task", json={"task": task})
    if not resp.ok:
        st.error(resp.text)
    else:
        data = resp.json()
        if data["status"] == "awaiting_human_review":
            st.session_state.pending_thread = data["thread_id"]
            st.warning(f"Human review needed: {data['review_note']}")
            st.text_area("Research so far", data["research_so_far"], height=200)
        else:
            st.session_state.pending_thread = None
            st.markdown("### Report")
            st.markdown(data["report"])

if st.session_state.pending_thread:
    st.subheader("3. Resolve conflict")
    note = st.text_input("Note for the record (optional)")
    col1, col2 = st.columns(2)
    if col1.button("Approve and continue"):
        resp = requests.post(f"{BACKEND_URL}/resume/{st.session_state.pending_thread}",
                              json={"approve": True, "note": note})
        st.session_state.pending_thread = None
        st.markdown("### Report")
        st.markdown(resp.json()["report"])
    if col2.button("Cancel run"):
        requests.post(f"{BACKEND_URL}/resume/{st.session_state.pending_thread}",
                      json={"approve": False, "note": note})
        st.session_state.pending_thread = None
        st.info("Run cancelled.")
