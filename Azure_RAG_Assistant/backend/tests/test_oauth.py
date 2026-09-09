import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ["DATABASE_URL"] = "sqlite:///./test_oauth.db"
os.environ["GOOGLE_CLIENT_ID"] = "test-client-id"
os.environ["GOOGLE_CLIENT_SECRET"] = "test-secret"

from fastapi.testclient import TestClient
from main import app
from database import init_db

init_db()
client = TestClient(app, follow_redirects=False)


def test_google_login_redirects_to_google_with_a_state_cookie():
    response = client.get("/auth/google/login")
    assert response.status_code == 307
    assert "accounts.google.com" in response.headers["location"]
    assert "oauth_state" in response.cookies


def test_google_callback_rejects_a_mismatched_state():
    # No oauth_state cookie set - simulates a forged/expired callback URL
    response = client.get("/auth/google/callback?code=fake-code&state=whatever")
    assert response.status_code == 400
