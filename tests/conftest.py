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

# Configured so every test's org registration auto-seeds a default
# EmailServer (see auth_service._seed_default_email_server) — otherwise every
# test that creates a user or requests a password reset would 422 on the
# "no default email server configured" check. The values are fake/
# unreachable; actual sending is mocked in _mock_email_sending below, so
# nothing here ever needs to resolve or connect for real.
os.environ["DEFAULT_SMTP_HOST"] = "smtp.test.invalid"
os.environ["DEFAULT_SMTP_PORT"] = "587"
os.environ["DEFAULT_SMTP_USERNAME"] = "test@example.com"
os.environ["DEFAULT_SMTP_PASSWORD"] = "test-password"
os.environ["DEFAULT_SMTP_FROM_ADDRESS"] = "test@example.com"
os.environ["DEFAULT_SMTP_FROM_NAME"] = "Vision AI Test"

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


@pytest.fixture(autouse=True)
def mock_email_sent(monkeypatch):
    """No test should ever make a real SMTP connection. Patches
    app.services.email_service.send_email everywhere it's used (auth_service,
    user_service, and the email_servers "send test email" endpoint all import
    it locally at call time, so patching the source module's attribute here
    is sufficient). Returns the list of {"to", "subject"} dicts "sent" during
    the test, for assertions that care what was attempted."""
    sent: list[dict] = []

    def _fake_send(server, to_addresses, subject, html, plain):
        sent.append({"to": to_addresses, "subject": subject})

    monkeypatch.setattr("app.services.email_service.send_email", _fake_send)
    return sent


@pytest.fixture()
def db_session():
    with Session(get_engine()) as session:
        yield session


@pytest.fixture()
def client():
    """Minimal app: JWT middleware + the routers under test, no ML/RTSP
    startup. Extend the router list here as later phases add endpoints."""
    from app.api.v1.endpoints import alerts as alerts_ep
    from app.api.v1.endpoints import audit as audit_ep
    from app.api.v1.endpoints import auth as auth_ep
    from app.api.v1.endpoints import cameras as cameras_ep
    from app.api.v1.endpoints import dashboard as dashboard_ep
    from app.api.v1.endpoints import email_servers as email_servers_ep
    from app.api.v1.endpoints import kpi_models as kpi_models_ep
    from app.api.v1.endpoints import organization as org_ep
    from app.api.v1.endpoints import priorities as priorities_ep
    from app.api.v1.endpoints import roles as roles_ep
    from app.api.v1.endpoints import timezones as timezones_ep
    from app.api.v1.endpoints import users as users_ep
    from app.api.v1.endpoints import ws as ws_ep
    from app.api.v1.endpoints import zones as zones_ep

    test_app = FastAPI()
    test_app.add_middleware(JWTAuthMiddleware)
    test_app.include_router(auth_ep.router)
    test_app.include_router(org_ep.router)
    test_app.include_router(roles_ep.router)
    test_app.include_router(users_ep.router)
    test_app.include_router(priorities_ep.router)
    test_app.include_router(zones_ep.router)
    test_app.include_router(email_servers_ep.router)
    test_app.include_router(kpi_models_ep.router)
    test_app.include_router(timezones_ep.router)
    test_app.include_router(cameras_ep.router)
    test_app.include_router(alerts_ep.router)
    test_app.include_router(dashboard_ep.router)
    test_app.include_router(audit_ep.router)
    test_app.include_router(ws_ep.router)

    with TestClient(test_app) as c:
        yield c


VALID_REGISTER_PAYLOAD = {
    "company_name": "Acme Terminal Operations",
    "tagline": "Safety first",
    "default_timezone_id": "1",  # fresh per-test DB always seeds timezones starting at id=1
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
