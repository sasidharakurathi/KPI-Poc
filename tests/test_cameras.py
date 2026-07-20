"""Phase 2 (Camera Management) tests.

Covers: auth/org-scoping on a previously-open endpoint, zone_id/priority_id
validation against the caller's org, response enrichment (zone_name/
priority_name/priority_color/priority_level), and that the legacy numeric
kpi_ids field (pipeline-facing) is untouched.
"""
from app.core.security import hash_password
from app.db.models import Role, User


def _make_zone(client, headers, name="Gate A"):
    resp = client.post("/api/config/zones", headers=headers, json={"name": name})
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def _make_priority(client, headers, name="Critical", level=1, color="#FF0000"):
    resp = client.post(
        "/api/config/priorities", headers=headers,
        json={"name": name, "color": color, "level": level},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def _create_limited_role(db_session, org_id, permissions=None):
    role = Role(name="NoCameraAccess", permissions=permissions or {"dashboard": ["view"]}, org_id=org_id)
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


def _login_as_limited_user(client, db_session, org_id, role_id, username="limited.cam.user"):
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
    return login.json()["access_token"]


# ── Auth / org scoping ───────────────────────────────────────────────────────

def test_cameras_require_auth(client):
    resp = client.get("/api/cameras")
    assert resp.status_code == 401


def test_camera_crud_requires_permission(client, owner, db_session):
    role = _create_limited_role(db_session, int(owner["org_id"]))
    token = _login_as_limited_user(client, db_session, int(owner["org_id"]), role.id)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/cameras", headers=headers).status_code == 403
    assert client.post("/api/cameras", headers=headers, json={"camera_id": "CAM-1", "name": "X", "zone_id": "1", "priority_id": "1"}).status_code == 403


# ── Create / validation ──────────────────────────────────────────────────────

def test_create_camera_requires_valid_zone_and_priority(client, owner):
    resp = client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-1", "name": "Front Gate", "zone_id": "999999", "priority_id": "999999"},
    )
    assert resp.status_code == 422, resp.text


def test_create_camera_succeeds_and_enriches_response(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Main Gate")
    priority_id = _make_priority(client, owner["headers"], "Critical", 1, "#FF0000")

    resp = client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-1", "name": "Front Gate", "zone_id": zone_id, "priority_id": priority_id},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["camera_id"] == "CAM-1"
    assert body["zone_id"] == zone_id
    assert body["zone_name"] == "Main Gate"
    assert body["priority_id"] == priority_id
    assert body["priority_name"] == "Critical"
    assert body["priority_color"] == "#FF0000"
    assert body["priority_level"] == 1
    assert body["status"] == "active"
    assert body["kpi_ids"] == []


def test_create_camera_duplicate_id_409s(client, owner):
    zone_id = _make_zone(client, owner["headers"])
    priority_id = _make_priority(client, owner["headers"])
    payload = {"camera_id": "CAM-DUP", "name": "Cam", "zone_id": zone_id, "priority_id": priority_id}
    assert client.post("/api/cameras", headers=owner["headers"], json=payload).status_code == 201
    resp = client.post("/api/cameras", headers=owner["headers"], json=payload)
    assert resp.status_code == 409, resp.text


def test_create_camera_keeps_legacy_kpi_ids(client, owner):
    zone_id = _make_zone(client, owner["headers"])
    priority_id = _make_priority(client, owner["headers"])
    resp = client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-KPI", "name": "Cam", "zone_id": zone_id, "priority_id": priority_id, "kpi_ids": [7, 11]},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["kpi_ids"] == [7, 11]
    assert len(body["kpis"]) == 2
    assert {k["kpi_id"] for k in body["kpis"]} == {7, 11}


# ── List / detail ─────────────────────────────────────────────────────────────

def test_list_cameras_enriched(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Yard")
    priority_id = _make_priority(client, owner["headers"], "High", 2, "#F59E0B")
    client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-L1", "name": "Cam L1", "zone_id": zone_id, "priority_id": priority_id},
    )
    resp = client.get("/api/cameras", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] >= 1
    row = next(c for c in body["cameras"] if c["camera_id"] == "CAM-L1")
    assert row["zone_name"] == "Yard"
    assert row["priority_name"] == "High"
    assert row["priority_level"] == 2


def test_get_camera_not_found(client, owner):
    resp = client.get("/api/cameras/NOPE", headers=owner["headers"])
    assert resp.status_code == 404


# ── Update ────────────────────────────────────────────────────────────────────

def test_update_camera_partial(client, owner):
    zone_id = _make_zone(client, owner["headers"])
    priority_id = _make_priority(client, owner["headers"])
    client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-U1", "name": "Old Name", "zone_id": zone_id, "priority_id": priority_id},
    )
    resp = client.put("/api/cameras/CAM-U1", headers=owner["headers"], json={"name": "New Name"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "New Name"
    assert body["zone_id"] == zone_id  # untouched fields survive a partial update


def test_update_camera_rejects_unknown_zone(client, owner):
    zone_id = _make_zone(client, owner["headers"])
    priority_id = _make_priority(client, owner["headers"])
    client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-U2", "name": "Cam", "zone_id": zone_id, "priority_id": priority_id},
    )
    resp = client.put("/api/cameras/CAM-U2", headers=owner["headers"], json={"zone_id": "999999"})
    assert resp.status_code == 422, resp.text


def test_update_camera_status(client, owner):
    zone_id = _make_zone(client, owner["headers"])
    priority_id = _make_priority(client, owner["headers"])
    client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-U3", "name": "Cam", "zone_id": zone_id, "priority_id": priority_id},
    )
    resp = client.put("/api/cameras/CAM-U3", headers=owner["headers"], json={"status": "inactive"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "inactive"


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_camera(client, owner):
    zone_id = _make_zone(client, owner["headers"])
    priority_id = _make_priority(client, owner["headers"])
    client.post(
        "/api/cameras", headers=owner["headers"],
        json={"camera_id": "CAM-D1", "name": "Cam", "zone_id": zone_id, "priority_id": priority_id},
    )
    resp = client.delete("/api/cameras/CAM-D1", headers=owner["headers"])
    assert resp.status_code == 204
    assert client.get("/api/cameras/CAM-D1", headers=owner["headers"]).status_code == 404


def test_delete_unknown_camera_404s(client, owner):
    resp = client.delete("/api/cameras/NOPE", headers=owner["headers"])
    assert resp.status_code == 404
