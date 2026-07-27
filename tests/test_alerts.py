"""Phase 4 (Alerts) tests: auth, zone-scoped visibility, and export.

Alerts are created directly via app.db.create_alert() (the same function the
real detection pipeline calls) rather than through a full video-processing
job - job_id is optional now specifically so tests (and the camera-offline
heartbeat monitor) don't need a real Job row.
"""
from app.core.security import hash_password
from app.db import create_alert
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


def _make_camera(client, headers, camera_id, zone_id, priority_id):
    resp = client.post(
        "/api/cameras", headers=headers,
        json={"camera_id": camera_id, "name": camera_id, "zone_id": zone_id, "priority_id": priority_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_zone_restricted_role(db_session, org_id, zone_id, permissions=None):
    role = Role(
        name="ZoneGuard",
        permissions=permissions or {"alerts": ["view"]},
        org_id=org_id,
        zone_ids=[int(zone_id)],
    )
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


def _create_unrestricted_role(db_session, org_id, permissions=None):
    role = Role(name="AllZones", permissions=permissions or {"alerts": ["view"]}, org_id=org_id, zone_ids=[])
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


def _second_org_owner(client) -> dict:
    from .conftest import VALID_REGISTER_PAYLOAD, register_activate_login

    payload = {
        **VALID_REGISTER_PAYLOAD,
        "company_name": "Globex Shipping", "site_name": "Globex Yard",
        "owner_full_name": "Bea Owner", "owner_email": "bea.owner@example.com", "username": "bea.owner",
    }
    body = register_activate_login(client, payload)
    body["headers"] = {"Authorization": f"Bearer {body['access_token']}"}
    return body


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_alerts_require_auth(client):
    assert client.get("/api/alerts").status_code == 401


# ── Cross-org isolation ──────────────────────────────────────────────────────
# Regression coverage for the fix in app.db.create_alert/query_alerts/
# app.services.alert_service - org_id used to only be enforced via
# zone-scoping, which is a no-op (allowed=None) for any unrestricted role,
# including the Owner of a brand-new org that has never even seen the
# camera whose alert it's about to view.

def test_list_alerts_is_org_scoped_for_unrestricted_owner(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Org1 Zone")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-ORG1", zone_id, priority_id)
    create_alert(camera_id="CAM-ORG1", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)

    other = _second_org_owner(client)
    resp = client.get("/api/alerts", headers=other["headers"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 0
    assert resp.json()["total"] == 0


def test_get_alert_detail_is_org_scoped_for_unrestricted_owner(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Org1 Zone Detail")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-ORG1D", zone_id, priority_id)
    alert_id = create_alert(camera_id="CAM-ORG1D", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)

    other = _second_org_owner(client)
    resp = client.get(f"/api/alerts/{alert_id}", headers=other["headers"])
    assert resp.status_code == 404, resp.text

    # sanity: the owning org still sees it fine
    assert client.get(f"/api/alerts/{alert_id}", headers=owner["headers"]).status_code == 200


# ── Zone-scoped visibility ───────────────────────────────────────────────────

def test_owner_sees_alerts_from_every_zone(client, owner):
    zone_a = _make_zone(client, owner["headers"], "Zone A")
    zone_b = _make_zone(client, owner["headers"], "Zone B")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-A", zone_a, priority_id)
    _make_camera(client, owner["headers"], "CAM-B", zone_b, priority_id)

    create_alert(camera_id="CAM-A", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)
    create_alert(camera_id="CAM-B", kpi_name="fire_smoke", alert_type="fire", frame_idx=2, confidence=0.9)

    resp = client.get("/api/alerts", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    camera_ids = {a["camera_id"] for a in resp.json()["alerts"]}
    assert {"CAM-A", "CAM-B"} <= camera_ids


def test_zone_restricted_role_only_sees_its_own_zone(client, owner, db_session):
    zone_a = _make_zone(client, owner["headers"], "Zone A")
    zone_b = _make_zone(client, owner["headers"], "Zone B")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-A2", zone_a, priority_id)
    _make_camera(client, owner["headers"], "CAM-B2", zone_b, priority_id)

    alert_a = create_alert(camera_id="CAM-A2", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)
    alert_b = create_alert(camera_id="CAM-B2", kpi_name="fire_smoke", alert_type="fire", frame_idx=2, confidence=0.9)

    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "zone.guard")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/alerts", headers=headers)
    assert resp.status_code == 200, resp.text
    camera_ids = {a["camera_id"] for a in resp.json()["alerts"]}
    assert camera_ids == {"CAM-A2"}

    assert client.get(f"/api/alerts/{alert_a}", headers=headers).status_code == 200
    assert client.get(f"/api/alerts/{alert_b}", headers=headers).status_code == 404


def test_cameraless_alert_is_invisible_to_everyone(client, owner, db_session):
    """A truly camera-less alert (no job, no camera) has no resolvable
    org_id either - app.db.create_alert only ever derives org_id from a
    camera. With no owning org, it's hidden from the org-scoped REST API
    entirely: not just zone-restricted roles (the original rule), but even
    an unrestricted role in the same deployment, and the Owner who happens
    to be the only org around. This is a deliberate deny-by-default: an
    alert of unknown ownership must never leak to a caller just because
    they're "unrestricted" within their own org."""
    zone_a = _make_zone(client, owner["headers"], "Zone A3")
    priority_id = _make_priority(client, owner["headers"])
    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "zone.guard3")
    headers = {"Authorization": f"Bearer {token}"}

    unrestricted_role = _create_unrestricted_role(db_session, int(owner["org_id"]))
    unrestricted_token = _login_as(client, db_session, int(owner["org_id"]), unrestricted_role.id, "unrestricted.user")
    unrestricted_headers = {"Authorization": f"Bearer {unrestricted_token}"}

    orphan_alert = create_alert(camera_id=None, kpi_name="system", alert_type="camera_offline", frame_idx=None, confidence=1.0)

    assert client.get(f"/api/alerts/{orphan_alert}", headers=headers).status_code == 404
    assert client.get(f"/api/alerts/{orphan_alert}", headers=unrestricted_headers).status_code == 404
    assert client.get(f"/api/alerts/{orphan_alert}", headers=owner["headers"]).status_code == 404


def test_get_unknown_alert_404s(client, owner):
    assert client.get("/api/alerts/999999", headers=owner["headers"]).status_code == 404


# ── Export ────────────────────────────────────────────────────────────────────

def test_export_alert_csv(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Export Zone")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-EXP1", zone_id, priority_id)
    alert_id = create_alert(camera_id="CAM-EXP1", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)

    resp = client.get(f"/api/alerts/{alert_id}/export", headers=owner["headers"], params={"format": "csv"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/csv")
    assert b"Alert ID" in resp.content


def test_export_alert_pdf(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Export Zone 2")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-EXP2", zone_id, priority_id)
    alert_id = create_alert(camera_id="CAM-EXP2", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)

    resp = client.get(f"/api/alerts/{alert_id}/export", headers=owner["headers"], params={"format": "pdf"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")


def test_export_alert_invalid_format_422s(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Export Zone 3")
    priority_id = _make_priority(client, owner["headers"])
    _make_camera(client, owner["headers"], "CAM-EXP3", zone_id, priority_id)
    alert_id = create_alert(camera_id="CAM-EXP3", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)

    resp = client.get(f"/api/alerts/{alert_id}/export", headers=owner["headers"], params={"format": "xml"})
    assert resp.status_code == 422, resp.text
