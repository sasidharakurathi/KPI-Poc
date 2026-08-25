"""Phase 8 (Audit Log) tests.

Covers: log_action() being called correctly from Camera/Role/User mutations
(Phase 2/6/7 retrofit), Configuration (Phase 1) writes staying un-audited
per the frontend's own scope, filtering, and the permission gate. No
zone-scoping - see app/services/audit_service.py's module docstring.
"""
from app.core.security import hash_password
from app.db.models import Role, User


def _create_limited_role(db_session, org_id, permissions=None):
    role = Role(name="NoAuditAccess", permissions=permissions or {"dashboard": ["view"]}, org_id=org_id)
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


def _login_as(client, db_session, org_id, role_id, username):
    user = User(
        full_name="Test User", personal_email=f"{username}@example.com",
        login_email=f"{username}@example.com", username=username,
        password_hash=hash_password("Str0ng!Passw0rd"),
        org_id=org_id, role_id=role_id, status="active",
    )
    db_session.add(user)
    db_session.commit()
    login = client.post("/api/auth/login", json={"username": username, "password": "Str0ng!Passw0rd"})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


def _entries_for(client, headers, **params):
    resp = client.get("/api/audit", headers=headers, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()["entries"]


# ── Auth / permission gate ───────────────────────────────────────────────────

def test_audit_requires_auth(client):
    assert client.get("/api/audit").status_code == 401


def test_audit_permission_gate(client, owner, db_session):
    role = _create_limited_role(db_session, int(owner["org_id"]))
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "no.audit.user")
    resp = client.get("/api/audit", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


# ── Camera ────────────────────────────────────────────────────────────────────

def test_camera_mutations_log_audit_entries(client, owner):
    zone_id = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Audit Zone"}).json()["id"]
    priority_id = client.post(
        "/api/config/priorities", headers=owner["headers"], json={"name": "Audit Pri", "color": "#111111"},
    ).json()["id"]

    resp = client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-AUD1", "name": "Audit Cam", "latitude": 12.9716, "longitude": 77.5946, "zone_id": str(zone_id), "priority_id": str(priority_id)},
    )
    assert resp.status_code == 201, resp.text

    client.put("/api/cameras/CAM-AUD1", headers=owner["headers"], json={"status": "inactive"})
    client.delete("/api/cameras/CAM-AUD1", headers=owner["headers"])

    entries = _entries_for(client, owner["headers"], entity="camera")
    actions = [e["action"] for e in entries if e["entity_id"] == "CAM-AUD1"]
    assert actions == ["delete", "disable", "create"]  # newest first (default sort_dir=desc)
    assert any('Audit Cam' in e["summary"] for e in entries if e["entity_id"] == "CAM-AUD1")


# ── Role ──────────────────────────────────────────────────────────────────────

def test_role_mutations_log_audit_entries(client, owner):
    created = client.post(
        "/api/roles", headers=owner["headers"],
        json={"name": "Audit Role", "description": "", "permissions": {"dashboard": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    ).json()
    client.put(
        f"/api/roles/{created['id']}", headers=owner["headers"],
        json={"name": "Audit Role Renamed", "description": "", "permissions": {"dashboard": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    )
    client.delete(f"/api/roles/{created['id']}", headers=owner["headers"])

    entries = _entries_for(client, owner["headers"], entity="role")
    actions = [e["action"] for e in entries if e["entity_id"] == created["id"]]
    assert actions == ["delete", "update", "create"]


# ── User ──────────────────────────────────────────────────────────────────────

def test_user_mutations_log_audit_entries(client, owner):
    role = client.post(
        "/api/roles", headers=owner["headers"],
        json={"name": "Audit User Role", "description": "", "permissions": {"dashboard": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    ).json()

    created = client.post(
        "/api/users", headers=owner["headers"],
        json={"full_name": "Audit Target", "username": "audit.target", "email": "audit.target@example.com",
              "phone": "+15550009999", "role_id": role["id"], "password": "Str0ng!Passw0rd"},
    )
    assert created.status_code == 201, created.text
    user_id = created.json()["id"]

    client.put(
        f"/api/users/{user_id}", headers=owner["headers"],
        json={"full_name": "Audit Target Renamed", "email": "audit.target@example.com",
              "phone": "+15550009999", "role_id": role["id"]},
    )
    client.patch(f"/api/users/{user_id}/status", headers=owner["headers"], json={"status": "inactive"})
    client.post(f"/api/users/{user_id}/reset-password", headers=owner["headers"], json={"new_password": "Br4ndNewPass!"})
    client.delete(f"/api/users/{user_id}", headers=owner["headers"])

    entries = _entries_for(client, owner["headers"], entity="user")
    actions = [e["action"] for e in entries if e["entity_id"] == user_id]
    assert actions == ["delete", "update", "disable", "update", "create"]


# ── Configuration is NOT audited ──────────────────────────────────────────────

def test_configuration_writes_are_not_audited(client, owner):
    client.post("/api/config/priorities", headers=owner["headers"], json={"name": "Not Audited", "color": "#222222"})
    client.post("/api/config/zones", headers=owner["headers"], json={"name": "Not Audited Zone"})

    entries = _entries_for(client, owner["headers"])
    assert all(e["entity"] not in ("priority", "zone", "config") for e in entries)


# ── Filters ───────────────────────────────────────────────────────────────────

def test_audit_filter_by_action(client, owner):
    zone_id = client.post("/api/config/zones", headers=owner["headers"], json={"name": "Filter Zone"}).json()["id"]
    priority_id = client.post(
        "/api/config/priorities", headers=owner["headers"], json={"name": "Filter Pri", "color": "#333333"},
    ).json()["id"]
    client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-FILT1", "name": "Filter Cam", "latitude": 12.9716, "longitude": 77.5946, "zone_id": str(zone_id), "priority_id": str(priority_id)},
    )

    entries = _entries_for(client, owner["headers"], entity="camera", action="create")
    assert all(e["action"] == "create" for e in entries)
    assert any(e["entity_id"] == "CAM-FILT1" for e in entries)
