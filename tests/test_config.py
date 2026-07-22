"""Phase 1 (Configuration: Priorities, Zones, Email Servers, KPI Models) tests.

Focused on the bugs found in the kpi-poc-v2 integration audit: missing auth,
client-controlled org_id, and org-unscoped queries — see
docs/IMPLEMENTATION_PLAN.md for the full incident writeup. CRUD happy-path
coverage is intentionally light since that part already worked.
"""
from app.core.security import hash_password
from app.db.models import Role, User


def _create_limited_role(client, headers, db_session, org_id, permissions=None):
    """A role with no 'configuration' permission at all, to prove enforcement."""
    role = Role(name="NoConfigAccess", permissions=permissions or {"dashboard": ["view"]}, org_id=org_id)
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


def _login_as_limited_user(client, db_session, org_id, role_id, username="limited.user"):
    user = User(
        full_name="Limited User", personal_email=f"{username}@example.com",
        login_email=f"{username}@example.com", username=username,
        password_hash=hash_password("Str0ng!Passw0rd"),
        org_id=org_id, role_id=role_id, status="active",
    )
    db_session.add(user)
    db_session.commit()
    login = client.post("/api/auth/login", json={"username": username, "password": "Str0ng!Passw0rd"})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def _second_org_headers(client, db_session) -> dict:
    """Registers a second, independent organization and returns Bearer
    headers for its Owner — used to prove config rows are unique per-org,
    not globally (see app/db/migrations.py's _migrate_multi_org_unique_constraints)."""
    from .conftest import VALID_REGISTER_PAYLOAD, register_activate_login

    payload = {
        **VALID_REGISTER_PAYLOAD,
        "company_name": "Globex Shipping",
        "site_name": "Globex Yard",
        "owner_full_name": "Bea Owner",
        "owner_email": "bea.owner@example.com",
        "username": "bea.owner",
    }
    body = register_activate_login(client, payload)
    return {"Authorization": f"Bearer {body['access_token']}"}


# ── Priorities ──────────────────────────────────────────────────────────────

def test_priorities_require_auth(client):
    resp = client.get("/api/config/priorities")
    assert resp.status_code == 401


def test_priorities_require_configuration_permission(client, owner, db_session):
    role = _create_limited_role(client, owner["headers"], db_session, int(owner["org_id"]))
    headers = _login_as_limited_user(client, db_session, int(owner["org_id"]), role.id)
    resp = client.get("/api/config/priorities", headers=headers)
    assert resp.status_code == 403


def test_create_priority_org_id_is_derived_not_client_supplied(client, owner):
    resp = client.post(
        "/api/config/priorities",
        headers=owner["headers"],
        json={"name": "Critical", "color": "#FF0000", "org_id": 999999},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # the bogus org_id in the request body must be ignored, not stored
    assert body["org_id"] == int(owner["org_id"])
    assert body["org_id"] != 999999


def test_priorities_list_includes_disabled_rows(client, owner):
    created = client.post(
        "/api/config/priorities", headers=owner["headers"],
        json={"name": "Low", "color": "#10B981"},
    ).json()
    client.patch(f"/api/config/priorities/{created['id']}/toggle", headers=owner["headers"])

    listed = client.get("/api/config/priorities", headers=owner["headers"]).json()
    ids = [p["id"] for p in listed]
    assert created["id"] in ids, "disabled priority must still be visible in the list"
    disabled_row = next(p for p in listed if p["id"] == created["id"])
    assert disabled_row["enabled"] is False


def test_priority_duplicate_name_within_org_409(client, owner):
    client.post("/api/config/priorities", headers=owner["headers"], json={"name": "High", "color": "#F59E0B"})
    resp = client.post("/api/config/priorities", headers=owner["headers"], json={"name": "High", "color": "#F59E0B"})
    assert resp.status_code == 409


def test_priority_same_name_across_orgs_allowed(client, owner, db_session):
    """Multi-tenant: two different organizations must each be free to use
    the same priority name — this used to be a global DB-level unique
    constraint that would 409 the second org even though the app-level
    duplicate check is (correctly) scoped to org_id."""
    other_headers = _second_org_headers(client, db_session)
    resp1 = client.post("/api/config/priorities", headers=owner["headers"], json={"name": "Critical", "color": "#FF0000"})
    resp2 = client.post("/api/config/priorities", headers=other_headers, json={"name": "Critical", "color": "#FF0000"})
    assert resp1.status_code == 201, resp1.text
    assert resp2.status_code == 201, resp2.text
    assert resp1.json()["org_id"] != resp2.json()["org_id"]


# ── Zones ───────────────────────────────────────────────────────────────────

def test_zones_require_auth(client):
    resp = client.post("/api/config/zones", json={"name": "Gate A"})
    assert resp.status_code == 401


def test_create_zone_org_id_is_derived_not_client_supplied(client, owner):
    resp = client.post(
        "/api/config/zones", headers=owner["headers"],
        json={"name": "Gate A", "org_id": 999999},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["org_id"] == int(owner["org_id"])


def test_zones_list_includes_disabled_rows(client, owner):
    created = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Yard 1"}).json()
    client.patch(f"/api/config/zones/{created['id']}/toggle", headers=owner["headers"])
    listed = client.get("/api/config/zones", headers=owner["headers"]).json()
    assert any(z["id"] == created["id"] and z["enabled"] is False for z in listed)


def test_create_zone_defaults_timezone_to_org_default(client, owner):
    resp = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Gate B"})
    assert resp.status_code == 201, resp.text
    org_resp = client.get("/api/organization", headers=owner["headers"])
    assert resp.json()["timezone_id"] == org_resp.json()["default_timezone_id"]


def test_create_zone_accepts_explicit_timezone_override(client, owner):
    resp = client.post(
        "/api/config/zones", headers=owner["headers"],
        json={"name": "Gate C", "timezone_id": "2"},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["timezone_id"] == "2"


def test_create_zone_rejects_unknown_timezone_id(client, owner):
    resp = client.post(
        "/api/config/zones", headers=owner["headers"],
        json={"name": "Gate D", "timezone_id": "999999"},
    )
    assert resp.status_code == 422, resp.text


def test_update_zone_can_change_timezone(client, owner):
    created = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Yard 2"}).json()
    resp = client.put(
        f"/api/config/zones/{created['id']}", headers=owner["headers"],
        json={"timezone_id": "2"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["timezone_id"] == "2"


def test_zone_camera_counts(client, owner):
    zone_with_cameras = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Counted Zone"}).json()
    zone_empty = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Empty Zone"}).json()
    priority = client.post(
        "/api/config/priorities", headers=owner["headers"],
        json={"name": "Count Pri", "color": "#123456"},
    ).json()

    for cam_id in ("CAM-CNT1", "CAM-CNT2"):
        resp = client.post(
            "/api/cameras", headers=owner["headers"],
            json={"camera_id": cam_id, "name": cam_id, "zone_id": str(zone_with_cameras["id"]), "priority_id": str(priority["id"])},
        )
        assert resp.status_code == 201, resp.text

    resp = client.get("/api/config/zones/camera-counts", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body[str(zone_with_cameras["id"])] == 2
    assert str(zone_empty["id"]) not in body


def test_zone_camera_counts_requires_auth(client):
    assert client.get("/api/config/zones/camera-counts").status_code == 401


# ── Email Servers ───────────────────────────────────────────────────────────

def test_email_servers_require_auth(client):
    resp = client.get("/api/config/email-servers")
    assert resp.status_code == 401


def test_create_email_server_org_id_is_derived_not_client_supplied(client, owner):
    resp = client.post(
        "/api/config/email-servers", headers=owner["headers"],
        json={
            "label": "Primary SMTP", "smtp_host": "smtp.example.com", "smtp_port": 587,
            "username": "alerts@example.com", "password": "smtp-pass-123",
            "from_address": "alerts@example.com", "from_name": "Vision AI Alerts",
            "org_id": 999999,
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["org_id"] == int(owner["org_id"])
    assert "password" not in body and "password_encrypted" not in body


def test_email_server_is_default_exclusivity(client, owner):
    first = client.post(
        "/api/config/email-servers", headers=owner["headers"],
        json={
            "label": "Server A", "smtp_host": "a.example.com", "smtp_port": 587,
            "username": "a@example.com", "password": "pw", "from_address": "a@example.com",
            "from_name": "Server A Team", "is_default": True,
        },
    ).json()
    assert first["is_default"] is True

    second = client.post(
        "/api/config/email-servers", headers=owner["headers"],
        json={
            "label": "Server B", "smtp_host": "b.example.com", "smtp_port": 587,
            "username": "b@example.com", "password": "pw", "from_address": "b@example.com",
            "from_name": "Server B Team", "is_default": True,
        },
    ).json()
    assert second["is_default"] is True

    # creating the second as default must have un-defaulted the first
    refreshed_first = client.get(f"/api/config/email-servers/{first['id']}", headers=owner["headers"]).json()
    assert refreshed_first["is_default"] is False


def test_email_server_same_label_across_orgs_allowed(client, owner, db_session):
    other_headers = _second_org_headers(client, db_session)
    body = {
        "label": "Primary SMTP", "smtp_host": "smtp.example.com", "smtp_port": 587,
        "username": "alerts@example.com", "password": "smtp-pass-123",
        "from_address": "alerts@example.com", "from_name": "Vision AI Alerts",
    }
    resp1 = client.post("/api/config/email-servers", headers=owner["headers"], json=body)
    resp2 = client.post("/api/config/email-servers", headers=other_headers, json=body)
    assert resp1.status_code == 201, resp1.text
    assert resp2.status_code == 201, resp2.text


# ── KPI Models ──────────────────────────────────────────────────────────────

def test_kpi_models_require_auth(client):
    resp = client.post("/api/config/kpi-models", json={"name": "fire_smoke_v2", "model_path": "/models/x.pt"})
    assert resp.status_code == 401


def test_create_kpi_model_org_id_is_derived_not_client_supplied(client, owner):
    resp = client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "fire_smoke_v2", "model_path": "/models/fire.pt", "org_id": 999999},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["org_id"] == int(owner["org_id"])


def test_kpi_models_list_includes_disabled_rows(client, owner):
    created = client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "ppe_v2", "model_path": "/models/ppe.pt"},
    ).json()
    client.patch(f"/api/config/kpi-models/{created['id']}/toggle", headers=owner["headers"])
    listed = client.get("/api/config/kpi-models", headers=owner["headers"]).json()
    assert any(m["id"] == created["id"] and m["enabled"] is False for m in listed)


def test_kpi_model_same_name_across_orgs_allowed(client, owner, db_session):
    other_headers = _second_org_headers(client, db_session)
    body = {"name": "fire_smoke_v2", "model_path": "/models/fire.pt"}
    resp1 = client.post("/api/config/kpi-models", headers=owner["headers"], json=body)
    resp2 = client.post("/api/config/kpi-models", headers=other_headers, json=body)
    assert resp1.status_code == 201, resp1.text
    assert resp2.status_code == 201, resp2.text


# ── Roles: same name across orgs (a fresh org's own "Owner" role) ───────────

def test_role_name_owner_across_orgs_allowed(client, owner, db_session):
    """Every organization seeds its own role literally named "Owner" at
    registration (app.services.auth_service.register_organization) — the
    second org's registration would have IntegrityError'd here if Role.name
    were still a global unique constraint instead of (org_id, name)."""
    _second_org_headers(client, db_session)  # registration succeeding at all is the assertion
    resp = client.get("/api/roles", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert any(r["name"] == "Owner" for r in resp.json())
