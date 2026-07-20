"""Tests for the email_servers-table-backed account email system:
- registration seeds a default EmailServer from DEFAULT_SMTP_* (or doesn't,
  gracefully, if unconfigured)
- every user-initiated action needing email (invite a user, forgot-password)
  hard-fails with 422 if the org has no usable default EmailServer
"""
from sqlmodel import select

from app.db.models import EmailServer, User

from .conftest import VALID_REGISTER_PAYLOAD, register_activate_login


def test_registration_seeds_default_email_server_when_configured(client, db_session, mock_email_sent):
    resp = client.post("/api/auth/register", json=VALID_REGISTER_PAYLOAD)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["activation_email_sent"] is True

    server = db_session.exec(select(EmailServer)).first()
    assert server is not None
    assert server.is_default is True
    assert server.enabled is True

    assert len(mock_email_sent) == 1
    assert "Activate your Vision AI account" in mock_email_sent[0]["subject"]


def test_registration_succeeds_without_default_smtp_configured(client, db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DEFAULT_SMTP_HOST", None)

    resp = client.post("/api/auth/register", json=VALID_REGISTER_PAYLOAD)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["activation_email_sent"] is False
    assert "no default email server" in body["message"].lower()

    assert db_session.exec(select(EmailServer)).first() is None

    # activation must still be reachable via the token directly (dev fallback)
    user = db_session.exec(select(User).where(User.username == VALID_REGISTER_PAYLOAD["username"])).first()
    assert user.reset_token is not None
    activate = client.post("/api/auth/activate", json={"token": user.reset_token})
    assert activate.status_code == 200


def test_create_user_fails_without_default_email_server(client, db_session):
    body = register_activate_login(client)
    headers = {"Authorization": f"Bearer {body['access_token']}"}

    # disable the seeded default server to simulate "no usable server"
    server = db_session.exec(select(EmailServer)).first()
    server.enabled = False
    db_session.add(server)
    db_session.commit()

    role_resp = client.post(
        "/api/roles", headers=headers,
        json={"name": "Viewer", "description": "", "permissions": {"dashboard": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    )
    role_id = role_resp.json()["id"]

    resp = client.post(
        "/api/users", headers=headers,
        json={"full_name": "New Guy", "username": "new.guy", "email": "newguy@example.com",
              "phone": "+15550001111", "role_id": role_id, "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 422
    assert "no default email server" in resp.json()["detail"].lower()

    # and the user must NOT have been created despite the failure
    assert db_session.exec(select(User).where(User.username == "new.guy")).first() is None


def test_create_user_sends_onboarding_email_with_default_server(client, mock_email_sent):
    body = register_activate_login(client)
    headers = {"Authorization": f"Bearer {body['access_token']}"}

    role_resp = client.post(
        "/api/roles", headers=headers,
        json={"name": "Viewer", "description": "", "permissions": {"dashboard": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    )
    role_id = role_resp.json()["id"]

    before = len(mock_email_sent)
    resp = client.post(
        "/api/users", headers=headers,
        json={"full_name": "New Guy", "username": "new.guy", "email": "newguy@example.com",
              "phone": "+15550001111", "role_id": role_id, "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 201, resp.text
    # one to the new user, one admin-confirmation to the owner
    assert len(mock_email_sent) == before + 2
    assert mock_email_sent[before]["to"] == ["newguy@example.com"]
    assert mock_email_sent[before + 1]["to"] == [VALID_REGISTER_PAYLOAD["owner_email"]]


def test_forgot_password_fails_without_default_email_server(client, db_session):
    body = register_activate_login(client)

    server = db_session.exec(select(EmailServer)).first()
    server.enabled = False
    db_session.add(server)
    db_session.commit()

    resp = client.post("/api/auth/forgot-password", json={"email": VALID_REGISTER_PAYLOAD["owner_email"]})
    assert resp.status_code == 422
    assert "no default email server" in resp.json()["detail"].lower()
