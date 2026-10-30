import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ["DATABASE_URL"] = "sqlite:///./test_auth.db"  # a dedicated file, not shared with other test modules
# Rate limiting is off by default for the whole test suite - see
# tests/conftest.py's autouse fixture for why this isn't just an
# os.environ assignment here.

from fastapi.testclient import TestClient
from main import app
from database import SessionLocal, init_db
from models import User, Role

init_db()  # TestClient(app) without `with` doesn't run FastAPI's lifespan, so this can't be left to happen implicitly
client = TestClient(app)


def _register(email="alice@example.com", password="a-real-password-123"):
    return client.post("/auth/register", json={"email": email, "password": password})


def test_register_creates_an_account():
    response = _register("register-test@example.com")
    assert response.status_code == 201
    body = response.json()
    assert "access_token" in body
    assert "refresh_token" in body


def test_register_rejects_short_password():
    response = _register("short-pw@example.com", password="short")
    assert response.status_code == 400


def test_register_rejects_duplicate_email():
    _register("duplicate@example.com")
    response = _register("duplicate@example.com")
    assert response.status_code == 409


def test_login_with_correct_password_succeeds():
    _register("login-test@example.com", password="correct-password-123")
    response = client.post("/auth/login", json={"email": "login-test@example.com", "password": "correct-password-123"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_with_wrong_password_fails():
    _register("wrongpw-test@example.com", password="correct-password-123")
    response = client.post("/auth/login", json={"email": "wrongpw-test@example.com", "password": "wrong-password"})
    assert response.status_code == 401


def test_login_with_unknown_email_fails():
    response = client.post("/auth/login", json={"email": "nobody-here@example.com", "password": "whatever-123"})
    assert response.status_code == 401


def test_me_returns_the_logged_in_users_own_identity():
    register_response = _register("me-test@example.com")
    token = register_response.json()["access_token"]
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "me-test@example.com"


def test_me_rejects_a_garbage_token():
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_refresh_issues_a_new_access_token():
    register_response = _register("refresh-test@example.com")
    refresh_token = register_response.json()["refresh_token"]
    response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_logout_revokes_the_refresh_token():
    register_response = _register("logout-test@example.com")
    refresh_token = register_response.json()["refresh_token"]

    logout_response = client.post("/auth/logout", json={"refresh_token": refresh_token})
    assert logout_response.status_code == 200

    reuse_response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert reuse_response.status_code == 401  # the same refresh token can't be used again after logout


def test_regular_user_cannot_reach_admin_routes():
    register_response = _register("regular-user@example.com")
    token = register_response.json()["access_token"]
    response = client.get("/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_admin_can_list_users():
    register_response = _register("will-be-admin@example.com")
    token = register_response.json()["access_token"]

    # Promote directly in the database - there's no "become admin" API
    # endpoint on purpose, that would be a privilege-escalation bug, not
    # a feature. Promotion is a manual, deliberate action (see README).
    db = SessionLocal()
    user = db.query(User).filter(User.email == "will-be-admin@example.com").first()
    user.role = Role.admin
    db.commit()
    db.close()

    # the old token's role claim is now stale (it says "user"), so log in
    # again to get a fresh token with the "admin" role baked in
    login_response = client.post("/auth/login", json={"email": "will-be-admin@example.com", "password": "a-real-password-123"})
    admin_token = login_response.json()["access_token"]

    response = client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
