import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["DATABASE_URL"] = "sqlite:///./test_auth.db"

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal, init_db
from app.models import User, Role

init_db()  # TestClient(app) without `with` doesn't run FastAPI's startup event
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


def test_query_requires_login():
    response = client.post("/query", json={"question": "hello"})
    assert response.status_code in (401, 403)


def test_ingest_requires_login():
    response = client.post("/ingest", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert response.status_code in (401, 403)


def test_reset_requires_login():
    response = client.post("/reset")
    assert response.status_code in (401, 403)


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
    assert reuse_response.status_code == 401


def test_regular_user_cannot_reach_admin_routes():
    register_response = _register("regular-user@example.com")
    token = register_response.json()["access_token"]
    response = client.get("/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_admin_can_list_users():
    register_response = _register("will-be-admin@example.com")

    db = SessionLocal()
    user = db.query(User).filter(User.email == "will-be-admin@example.com").first()
    user.role = Role.admin
    db.commit()
    db.close()

    login_response = client.post("/auth/login", json={"email": "will-be-admin@example.com", "password": "a-real-password-123"})
    admin_token = login_response.json()["access_token"]

    response = client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
