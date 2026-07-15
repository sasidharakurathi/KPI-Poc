from sqlmodel import select

from app.db.models import User

from .conftest import VALID_REGISTER_PAYLOAD


def _activated_owner_token(client, db_session) -> str:
    resp = client.post("/api/auth/register", json=VALID_REGISTER_PAYLOAD)
    assert resp.status_code == 201, resp.text
    user = db_session.exec(
        select(User).where(User.username == VALID_REGISTER_PAYLOAD["username"])
    ).first()
    client.post("/api/auth/activate", json={"token": user.reset_token})
    login = client.post(
        "/api/auth/login",
        json={
            "username": VALID_REGISTER_PAYLOAD["username"],
            "password": VALID_REGISTER_PAYLOAD["password"],
        },
    )
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


def test_get_organization_requires_auth(client, db_session):
    _activated_owner_token(client, db_session)  # ensure an org exists
    resp = client.get("/api/organization")
    assert resp.status_code == 401


def test_get_organization_returns_registered_details(client, db_session):
    token = _activated_owner_token(client, db_session)
    resp = client.get("/api/organization", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == VALID_REGISTER_PAYLOAD["company_name"]
    assert body["site_name"] == VALID_REGISTER_PAYLOAD["site_name"]
    assert body["default_timezone"] == VALID_REGISTER_PAYLOAD["default_timezone"]


def test_owner_can_update_organization(client, db_session):
    token = _activated_owner_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.put(
        "/api/organization",
        json={"tagline": "Updated tagline", "default_timezone": "UTC"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tagline"] == "Updated tagline"
    assert body["default_timezone"] == "UTC"
    # untouched fields survive a partial update
    assert body["name"] == VALID_REGISTER_PAYLOAD["company_name"]
