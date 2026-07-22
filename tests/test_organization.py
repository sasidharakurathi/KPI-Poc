from sqlmodel import select

from app.db.models import Role, User

from .conftest import VALID_REGISTER_PAYLOAD

SECOND_ORG_PAYLOAD = {
    **VALID_REGISTER_PAYLOAD,
    "company_name": "Globex Shipping",
    "site_name": "Globex Yard",
    "owner_full_name": "Bea Owner",
    "owner_email": "bea.owner@example.com",
    "username": "bea.owner",
}


def _register_activate_login(client, db_session, payload) -> dict:
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 201, resp.text
    user = db_session.exec(select(User).where(User.username == payload["username"])).first()
    client.post("/api/auth/activate", json={"token": user.reset_token})
    login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]},
    )
    assert login.status_code == 200, login.text
    return login.json()


def _activated_owner_token(client, db_session) -> str:
    return _register_activate_login(client, db_session, VALID_REGISTER_PAYLOAD)["access_token"]


def _create_limited_role(db_session, org_id, permissions=None) -> Role:
    role = Role(name="NoOrgAccess", permissions=permissions or {"dashboard": ["view"]}, org_id=org_id)
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


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
    assert body["default_timezone_id"] == VALID_REGISTER_PAYLOAD["default_timezone_id"]


def test_owner_can_update_organization(client, db_session):
    token = _activated_owner_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.put(
        "/api/organization",
        json={"tagline": "Updated tagline", "default_timezone_id": "2"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tagline"] == "Updated tagline"
    assert body["default_timezone_id"] == "2"
    # untouched fields survive a partial update
    assert body["name"] == VALID_REGISTER_PAYLOAD["company_name"]


def test_update_organization_rejects_unknown_timezone_id(client, db_session):
    token = _activated_owner_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.put(
        "/api/organization",
        json={"default_timezone_id": "999999"},
        headers=headers,
    )
    assert resp.status_code == 422, resp.text


# ── Multi-org: GET/PUT /api/organization stays scoped to the caller's own org ──

def test_organization_endpoint_is_scoped_to_callers_own_org(client, db_session):
    org1 = _register_activate_login(client, db_session, VALID_REGISTER_PAYLOAD)
    org2 = _register_activate_login(client, db_session, SECOND_ORG_PAYLOAD)

    resp1 = client.get("/api/organization", headers={"Authorization": f"Bearer {org1['access_token']}"})
    resp2 = client.get("/api/organization", headers={"Authorization": f"Bearer {org2['access_token']}"})
    assert resp1.status_code == 200 and resp2.status_code == 200
    assert resp1.json()["name"] == VALID_REGISTER_PAYLOAD["company_name"]
    assert resp2.json()["name"] == SECOND_ORG_PAYLOAD["company_name"]
    assert resp1.json()["id"] != resp2.json()["id"]

    # Updating org1 must never touch org2's row.
    client.put(
        "/api/organization", json={"tagline": "Org1 only"},
        headers={"Authorization": f"Bearer {org1['access_token']}"},
    )
    resp2_after = client.get("/api/organization", headers={"Authorization": f"Bearer {org2['access_token']}"})
    assert resp2_after.json().get("tagline") != "Org1 only"


# ── GET /api/organizations (list all) ───────────────────────────────────────

def test_list_organizations_requires_auth(client):
    assert client.get("/api/organizations").status_code == 401


def test_list_organizations_forbidden_for_non_owner_role(client, db_session):
    from app.core.security import hash_password

    org1 = _register_activate_login(client, db_session, VALID_REGISTER_PAYLOAD)
    role = _create_limited_role(db_session, int(org1["org_id"]))
    limited_user = User(
        full_name="Limited User", personal_email="limited.org@example.com",
        login_email="limited.org@example.com", username="limited.org.user",
        password_hash=hash_password("Str0ng!Passw0rd"),
        org_id=int(org1["org_id"]), role_id=role.id, status="active",
    )
    db_session.add(limited_user)
    db_session.commit()
    login = client.post("/api/auth/login", json={"username": "limited.org.user", "password": "Str0ng!Passw0rd"})
    assert login.status_code == 200, login.text

    resp = client.get("/api/organizations", headers={"Authorization": f"Bearer {login.json()['access_token']}"})
    assert resp.status_code == 403, resp.text


def test_list_organizations_returns_every_org_for_owner(client, db_session):
    org1 = _register_activate_login(client, db_session, VALID_REGISTER_PAYLOAD)
    _register_activate_login(client, db_session, SECOND_ORG_PAYLOAD)

    resp = client.get("/api/organizations", headers={"Authorization": f"Bearer {org1['access_token']}"})
    assert resp.status_code == 200, resp.text
    names = {row["name"] for row in resp.json()}
    assert names == {VALID_REGISTER_PAYLOAD["company_name"], SECOND_ORG_PAYLOAD["company_name"]}


# ── Logo upload ──────────────────────────────────────────────────────────────

def test_upload_logo_requires_permission(client):
    resp = client.post(
        "/api/organization/logo", files={"file": ("logo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert resp.status_code == 401


def test_upload_logo_succeeds_and_sets_logo_url(client, db_session):
    token = _activated_owner_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/api/organization/logo", headers=headers,
        files={"file": ("logo.png", b"\x89PNG\r\n\x1a\nfake-but-good-enough", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["logo_url"] is not None
    assert body["logo_url"].endswith(".png")

    # GET reflects the same logo_url afterwards.
    get_resp = client.get("/api/organization", headers=headers)
    assert get_resp.json()["logo_url"] == body["logo_url"]


def test_upload_logo_replaces_previous_file(client, db_session):
    from app.config import settings

    token = _activated_owner_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post(
        "/api/organization/logo", headers=headers,
        files={"file": ("logo1.png", b"first-logo-bytes", "image/png")},
    )
    first_filename = first.json()["logo_url"].rsplit("/", 1)[-1]
    assert (settings.LOGOS_DIR / first_filename).exists()

    second = client.post(
        "/api/organization/logo", headers=headers,
        files={"file": ("logo2.png", b"second-logo-bytes", "image/png")},
    )
    second_filename = second.json()["logo_url"].rsplit("/", 1)[-1]
    assert second_filename != first_filename
    assert (settings.LOGOS_DIR / second_filename).exists()
    assert not (settings.LOGOS_DIR / first_filename).exists()

    (settings.LOGOS_DIR / second_filename).unlink(missing_ok=True)  # test cleanup


def test_upload_logo_rejects_unsupported_type(client, db_session):
    token = _activated_owner_token(client, db_session)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/api/organization/logo", headers=headers,
        files={"file": ("logo.gif", b"GIF89a", "image/gif")},
    )
    assert resp.status_code == 422, resp.text
