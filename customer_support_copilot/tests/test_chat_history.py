"""Chat persistence and access control, using the conversation API.

The model and retriever are mocked, so these tests only check what app.py
does with a message: save it, keep it private, and redact PII first.
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
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _new_conversation(headers):
    return client.post("/conversations", headers=headers).json()["id"]


def _fake_retriever(text):
    app_module._retriever = type("FakeRetriever", (), {"retrieve": staticmethod(lambda q: text)})()


@patch("src.app.llm_backend.generate", return_value="Your refund will arrive in 5-7 business days.")
@patch("src.app.llm_backend.is_ready", return_value=True)
def test_chat_message_is_saved_for_the_sender(mock_ready, mock_generate):
    headers = _register_and_login("chat-user-1@example.com")
    _fake_retriever("Refund policy: 5-7 days.")
    convo_id = _new_conversation(headers)

    chat = client.post("/chat", json={"query": "Where's my refund?", "conversation_id": convo_id}, headers=headers)
    assert chat.status_code == 200

    messages = client.get(f"/conversations/{convo_id}/messages", headers=headers).json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Where's my refund?"
    assert messages[1]["role"] == "assistant"


@patch("src.app.llm_backend.generate", return_value="Some reply")
@patch("src.app.llm_backend.is_ready", return_value=True)
def test_one_users_history_is_invisible_to_a_different_user(mock_ready, mock_generate):
    _fake_retriever("some context")
    alice = _register_and_login("history-alice@example.com")
    bob = _register_and_login("history-bob@example.com")

    alice_convo = _new_conversation(alice)
    client.post("/chat", json={"query": "Alice's private question", "conversation_id": alice_convo}, headers=alice)

    bob_view = client.get(f"/conversations/{alice_convo}/messages", headers=bob)
    assert bob_view.status_code == 404

    bob_list = client.get("/conversations", headers=bob).json()
    assert all(c["id"] != alice_convo for c in bob_list)

    alice_view = client.get(f"/conversations/{alice_convo}/messages", headers=alice).json()
    assert any("Alice" in m["content"] for m in alice_view)


@patch("src.app.llm_backend.generate", return_value="Thanks, I will check that order.")
@patch("src.app.llm_backend.is_ready", return_value=True)
def test_pii_is_redacted_before_it_is_saved(mock_ready, mock_generate):
    headers = _register_and_login("pii-user@example.com")
    _fake_retriever("Order policy.")
    convo_id = _new_conversation(headers)

    message = "My email is jane.doe@example.com and my id is 29901011234567"
    client.post("/chat", json={"query": message, "conversation_id": convo_id}, headers=headers)

    messages = client.get(f"/conversations/{convo_id}/messages", headers=headers).json()
    saved = messages[0]["content"]
    assert "jane.doe@example.com" not in saved
    assert "29901011234567" not in saved
    assert "[EMAIL]" in saved
    assert "[ID_NUMBER]" in saved
