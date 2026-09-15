"""
Streamlit front-end for DocuMind, talking to the FastAPI backend over
HTTP (app/main.py). Run with:
    streamlit run ui/streamlit_app.py
Expects the FastAPI backend to be running on API_URL (default localhost:8000).

Every request now needs a real logged-in account - the backend requires
a valid JWT access token on every route, so a random per-browser session
ID isn't what isolates users anymore (see app/main.py's docstring).
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📚")
st.title("📚 DocuMind — Ask your documents")

if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.user_email = None


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}"}


def try_refresh():
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False
    resp = requests.post(f"{API_URL}/auth/refresh", json={"refresh_token": refresh_token})
    if not resp.ok:
        return False
    st.session_state.access_token = resp.json()["access_token"]
    return True


# ------------------------------------------------------------ login screen

if not st.session_state.access_token:
    st.subheader("Log in")
    email = st.text_input("Email")
    password = st.text_input("Password (8+ characters)", type="password")

    col1, col2 = st.columns(2)
    if col1.button("Log in", use_container_width=True):
        resp = requests.post(f"{API_URL}/auth/login", json={"email": email, "password": password})
        if resp.ok:
            data = resp.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.refresh_token = data["refresh_token"]
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error(resp.json().get("detail", "Login failed"))

    if col2.button("Register", use_container_width=True):
        resp = requests.post(f"{API_URL}/auth/register", json={"email": email, "password": password})
        if resp.ok:
            data = resp.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.refresh_token = data["refresh_token"]
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error(resp.json().get("detail", "Registration failed"))

    google_login_url = f"{API_URL}/auth/google/login"
    st.caption(
        f"Google login is available directly at [{google_login_url}]({google_login_url}) "
        "if GOOGLE_CLIENT_ID is configured on the backend."
    )
    st.stop()  # don't render the rest of the app until logged in


# ------------------------------------------------------------ logged in

with st.sidebar:
    st.caption(f"Logged in as **{st.session_state.user_email}**")
    if st.button("Log out"):
        requests.post(f"{API_URL}/auth/logout", json={"refresh_token": st.session_state.get("refresh_token", "")})
        for key in ("access_token", "refresh_token", "user_email", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.caption("Documents you upload are private to your account.")

    st.header("Upload documents")
    uploaded = st.file_uploader(
        "Add .txt, .md, or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest files"):
        for f in uploaded:
            resp = requests.post(
                f"{API_URL}/ingest",
                files={"file": (f.name, f.getvalue())},
                headers=auth_headers(),
            )
            if resp.status_code == 401 and try_refresh():
                resp = requests.post(f"{API_URL}/ingest", files={"file": (f.name, f.getvalue())}, headers=auth_headers())
            if resp.ok:
                st.success(f"{f.name}: {resp.json()['chunks_indexed']} chunks indexed")
            else:
                st.error(f"{f.name}: {resp.text}")

    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", headers=auth_headers(), timeout=5).json()
        st.caption(f"Indexed chunks: {health['indexed_chunks']}")
    except Exception:
        st.caption("⚠️ Backend not reachable")

    if st.button("Clear my index"):
        requests.post(f"{API_URL}/reset", headers=auth_headers())
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
            resp = requests.post(f"{API_URL}/query", json={"question": question}, headers=auth_headers())
            if resp.status_code == 401 and try_refresh():
                resp = requests.post(f"{API_URL}/query", json={"question": question}, headers=auth_headers())

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
