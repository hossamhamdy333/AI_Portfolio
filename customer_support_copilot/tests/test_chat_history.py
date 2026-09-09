"""
Proves two things that didn't exist at all before this rewrite:

1. A chat actually gets saved (user message + assistant reply), tied to
   the user who sent it.
2. One user's chat history is never returned when a DIFFERENT logged-in
   user asks for "their" history - the real security property, not just
   that the feature works for a single user in isolation.

The model and retriever are mocked - this test is about the persistence
and access-control logic in app.py, not about whether the LLM itself
produces a good answer.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["DATABASE_URL"] = "sqlite:///./test_chat_history.db"

from fastapi.testclient import TestClient
from src.app import app
from src.database import init_db
from src import app as app_module

init_db()
client = TestClient(app)


def _register_and_login(email, password="a-real-password-123"):
    response = client.post("/auth/register", json={"email": email, "password": password})
    return response.json()["access_token"]


@patch("src.app.llm_backend.generate", return_value="Your refund will arrive in 5-7 business days.")
@patch("src.app.llm_backend.is_ready", return_value=True)
def test_chat_message_is_saved_for_the_sender(mock_ready, mock_generate):
    token = _register_and_login("chat-user-1@example.com")
    app_module._retriever = type("FakeRetriever", (), {"retrieve": staticmethod(lambda q: "Refund policy: 5-7 days.")})()

    chat_response = client.post("/chat", json={"query": "Where's my refund?"}, headers={"Authorization": f"Bearer {token}"})
    assert chat_response.status_code == 200

    history_response = client.get("/chat/history", headers={"Authorization": f"Bearer {token}"})
    history = history_response.json()
    assert len(history) == 2  # the user's question and the assistant's reply
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Where's my refund?"
    assert history[1]["role"] == "assistant"


@patch("src.app.llm_backend.generate", return_value="Some reply")
@patch("src.app.llm_backend.is_ready", return_value=True)
def test_one_users_history_is_invisible_to_a_different_user(mock_ready, mock_generate):
    app_module._retriever = type("FakeRetriever", (), {"retrieve": staticmethod(lambda q: "some context")})()

    alice_token = _register_and_login("history-alice@example.com")
    bob_token = _register_and_login("history-bob@example.com")

    client.post("/chat", json={"query": "Alice's private question"}, headers={"Authorization": f"Bearer {alice_token}"})

    bob_history = client.get("/chat/history", headers={"Authorization": f"Bearer {bob_token}"}).json()
    assert all("Alice" not in m["content"] for m in bob_history)

    alice_history = client.get("/chat/history", headers={"Authorization": f"Bearer {alice_token}"}).json()
    assert any("Alice" in m["content"] for m in alice_history)
