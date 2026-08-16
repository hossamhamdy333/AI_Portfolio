import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Dummy env vars so config.Settings() validates without real secrets in CI.
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")

from fastapi.testclient import TestClient
from main import app
from agent import _extract_text

client = TestClient(app)


def test_extract_text_handles_plain_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_handles_structured_content_parts():
    # Newer Gemini models can return content as a list of parts instead of
    # a plain string. Before this was handled, the frontend would render
    # this as the literal text "[object Object]" instead of the message.
    assert _extract_text([{"type": "text", "text": "Hello there"}]) == "Hello there"


def test_extract_text_joins_multiple_parts():
    content = [{"type": "text", "text": "Part A. "}, {"type": "text", "text": "Part B."}]
    assert _extract_text(content) == "Part A. Part B."


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_query_when_auth_disabled():
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 400


def test_root_serves_html_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Azure RAG Assistant" in response.text
