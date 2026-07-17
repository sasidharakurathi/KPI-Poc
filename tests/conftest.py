"""Test setup: points the app at a throwaway SQLite DB instead of the real
Postgres one in .env, and builds a lightweight FastAPI app per phase (the
real app/main.py's lifespan does heavy ML-model/RTSP-recorder startup that
tests don't need and shouldn't depend on).

Must set env vars BEFORE any `app.*` module is imported anywhere in the
session, since app.config.settings is instantiated once at import time.
"""
import os
import tempfile
from pathlib import Path

_tmp_db = Path(tempfile.gettempdir()) / "vision_ai_phase0_test.db"
if _tmp_db.exists():
    _tmp_db.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.as_posix()}"
os.environ["JWT_SECRET_KEY"] = "test-only-secret-key-do-not-use-in-prod"
os.environ["JWT_AUTH_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session, SQLModel  # noqa: E402

from app.auth.middleware import JWTAuthMiddleware  # noqa: E402
from app.db.engine import get_engine, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def _init_test_db():
    """Fresh schema + rate-limiter state per test — auth/org data (esp. "one
    org per deployment") and the process-global rate limiters must not leak
    between tests."""
    from app.core.rate_limit import login_limiter, password_reset_limiter

    engine = get_engine()
    SQLModel.metadata.drop_all(engine)
    init_db()
    login_limiter.reset()
    password_reset_limiter.reset()
    yield


@pytest.fixture()
def db_session():
    with Session(get_engine()) as session:
        yield session


@pytest.fixture()
def client():
    """Minimal app: JWT middleware + the routers under test, no ML/RTSP
    startup. Extend the router list here as later phases add endpoints."""
    from app.api.v1.endpoints import auth as auth_ep
    from app.api.v1.endpoints import organization as org_ep
    from app.api.v1.endpoints import roles as roles_ep
    from app.api.v1.endpoints import users as users_ep

    test_app = FastAPI()
    test_app.add_middleware(JWTAuthMiddleware)
    test_app.include_router(auth_ep.router)
    test_app.include_router(org_ep.router)
    test_app.include_router(roles_ep.router)
    test_app.include_router(users_ep.router)

    with TestClient(test_app) as c:
        yield c


VALID_REGISTER_PAYLOAD = {
    "company_name": "Acme Terminal Operations",
    "tagline": "Safety first",
    "default_timezone": "Asia/Kolkata",
    "site_name": "Acme Port Terminal",
    "site_address": "1 Harbor Way",
    "latitude": 18.9548,
    "longitude": 72.9495,
    "owner_full_name": "Ada Owner",
    "owner_designation": "Operations Head",
    "owner_email": "owner@example.com",
    "owner_phone": "+15551234567",
    "username": "ada.owner",
    "password": "Str0ng!Passw0rd",
    "confirm_password": "Str0ng!Passw0rd",
}


def register_activate_login(client, payload=None) -> dict:
    """Full Phase 0 bootstrap, returning the login response body (access_token,
    refresh_token, role_id, org_id, ...). Shared by every later phase's tests
    that need an authenticated Owner."""
    from sqlmodel import select

    from app.db.models import User

    payload = payload or VALID_REGISTER_PAYLOAD
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 201, resp.text

    with Session(get_engine()) as s:
        user = s.exec(select(User).where(User.username == payload["username"])).first()
        token = user.reset_token
    activate = client.post("/api/auth/activate", json={"token": token})
    assert activate.status_code == 200, activate.text

    login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    assert login.status_code == 200, login.text
    return login.json()


@pytest.fixture()
def owner(client) -> dict:
    """Registers+activates+logs in the default Owner and returns the login
    body plus a ready-to-use `headers` dict."""
    body = register_activate_login(client)
    body["headers"] = {"Authorization": f"Bearer {body['access_token']}"}
    return body
