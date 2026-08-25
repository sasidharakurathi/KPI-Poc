"""Phase 5 (Dashboard) tests.

These three endpoints (/summary, /alert-chart, /cameras) match the PRD's
spec-literal draft rather than a real frontend contract - see
app/services/dashboard_service.py's module docstring. Coverage focuses on
correctness of the aggregation + the same zone-scoping rule already proven
for Alerts (Phase 4).
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
        json={"camera_id": camera_id, "name": camera_id, "latitude": 12.9716, "longitude": 77.5946, "zone_id": zone_id, "priority_id": priority_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_zone_restricted_role(db_session, org_id, zone_id):
    role = Role(name="ZoneGuardDash", permissions={"dashboard": ["view"]}, org_id=org_id, zone_ids=[int(zone_id)])
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

def test_dashboard_requires_auth(client):
    assert client.get("/api/dashboard/summary").status_code == 401
    assert client.get("/api/dashboard/cameras").status_code == 401
    assert client.get("/api/dashboard/alert-chart", params={"camera_id": "CAM-1"}).status_code == 401


# ── Summary ───────────────────────────────────────────────────────────────────

def test_dashboard_summary_counts(client, owner):
    zone_a = _make_zone(client, owner["headers"], "Summary Zone A")
    zone_b = _make_zone(client, owner["headers"], "Summary Zone B")
    priority_id = _make_priority(client, owner["headers"], "Critical S", 1, "#FF0000")
    _make_camera(client, owner["headers"], "CAM-S1", zone_a, priority_id)
    _make_camera(client, owner["headers"], "CAM-S2", zone_b, priority_id)

    create_alert(camera_id="CAM-S1", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)
    create_alert(camera_id="CAM-S2", kpi_name="fire_smoke", alert_type="fire", frame_idx=2, confidence=0.9)

    resp = client.get("/api/dashboard/summary", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_cameras"] >= 2
    assert body["total_zones"] >= 2
    assert body["total_alerts"] >= 2
    assert body["alerts_last_24h"] >= 2
    assert body["alerts_by_priority"].get("Critical S", 0) >= 2
    # every camera starts connectivity_status="pending" until the heartbeat monitor observes it
    assert body["pending_cameras"] >= 2


def test_dashboard_summary_active_kpi_models_count(client, owner):
    m1 = client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "Dash KPI Model 1", "model_path": "/models/a.pt"},
    ).json()
    client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "Dash KPI Model 2", "model_path": "/models/b.pt"},
    )
    disabled = client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "Dash KPI Model 3", "model_path": "/models/c.pt"},
    ).json()
    client.patch(f"/api/config/kpi-models/{disabled['id']}/toggle", headers=owner["headers"])  # -> disabled

    resp = client.get("/api/dashboard/summary", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["active_kpi_models"] == 2

    client.patch(f"/api/config/kpi-models/{m1['id']}/toggle", headers=owner["headers"])  # -> disabled
    resp2 = client.get("/api/dashboard/summary", headers=owner["headers"])
    assert resp2.json()["active_kpi_models"] == 1


def test_dashboard_summary_active_kpi_models_not_zone_scoped(client, owner, db_session):
    """The KPI Models catalog isn't tied to any camera/zone, so a
    zone-restricted role still sees the org-wide count, unlike total_cameras."""
    client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "Org-wide KPI Model", "model_path": "/models/d.pt"},
    )
    zone_a = _make_zone(client, owner["headers"], "KPI Models Zone A")
    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "kpi.models.zone.guard")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["active_kpi_models"] == 1


def test_dashboard_summary_active_kpis_count(client, owner):
    """Distinct from active_kpi_models: this counts the KPI Management
    catalog (KPIConfiguration - fire_smoke/ppe/etc.), not the KPI Models/
    detection-model file catalog."""
    from app.kpis import list_registered_names

    names = list_registered_names()
    assert len(names) >= 2
    name_a, name_b = names[0], names[1]

    client.put(f"/api/kpis/catalog/{name_a}", headers=owner["headers"], json={"config": {}})
    client.put(f"/api/kpis/catalog/{name_b}", headers=owner["headers"], json={"config": {}})

    resp = client.get("/api/dashboard/summary", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["active_kpis"] == 2

    client.patch(f"/api/kpis/catalog/{name_a}/toggle", headers=owner["headers"])  # -> disabled
    resp2 = client.get("/api/dashboard/summary", headers=owner["headers"])
    assert resp2.json()["active_kpis"] == 1


def test_dashboard_summary_active_kpis_not_zone_scoped(client, owner, db_session):
    """The KPI Management catalog isn't tied to any camera/zone, so a
    zone-restricted role still sees the org-wide count."""
    from app.kpis import list_registered_names

    name = list_registered_names()[0]
    client.put(f"/api/kpis/catalog/{name}", headers=owner["headers"], json={"config": {}})

    zone_a = _make_zone(client, owner["headers"], "Active KPIs Zone A")
    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "active.kpis.zone.guard")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["active_kpis"] == 1


def test_dashboard_summary_zone_scoped(client, owner, db_session):
    zone_a = _make_zone(client, owner["headers"], "Scoped Zone A")
    zone_b = _make_zone(client, owner["headers"], "Scoped Zone B")
    priority_id = _make_priority(client, owner["headers"], "Critical Z", 1, "#FF0000")
    _make_camera(client, owner["headers"], "CAM-Z1", zone_a, priority_id)
    _make_camera(client, owner["headers"], "CAM-Z2", zone_b, priority_id)
    create_alert(camera_id="CAM-Z1", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)
    create_alert(camera_id="CAM-Z2", kpi_name="fire_smoke", alert_type="fire", frame_idx=2, confidence=0.9)

    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "dash.zone.guard")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/dashboard/summary", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_cameras"] == 1
    assert body["total_alerts"] == 1


def test_dashboard_summary_alerts_are_org_scoped_even_for_unrestricted_owner(client, owner):
    """Regression test: total_alerts/alerts_last_24h/alerts_by_priority used
    to have zero org filter at all - only zone-scoping, which is a no-op for
    an unrestricted (is_system Owner) caller - so a brand-new org's Owner
    would see every other organization's alert counts on this deployment.
    Alert.org_id (resolved from camera at creation time) now closes that."""
    zone_id = _make_zone(client, owner["headers"], "Org1 Zone")
    priority_id = _make_priority(client, owner["headers"], "Org1 Pri")
    _make_camera(client, owner["headers"], "CAM-ORG1", zone_id, priority_id)
    create_alert(camera_id="CAM-ORG1", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)
    create_alert(camera_id="CAM-ORG1", kpi_name="fire_smoke", alert_type="fire", frame_idx=2, confidence=0.9)

    other = _second_org_owner(client)
    resp = client.get("/api/dashboard/summary", headers=other["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_alerts"] == 0
    assert body["alerts_last_24h"] == 0
    assert body["alerts_by_priority"] == {}

    # sanity: the first org's Owner still sees its own two alerts
    own_resp = client.get("/api/dashboard/summary", headers=owner["headers"])
    assert own_resp.json()["total_alerts"] == 2


def test_dashboard_summary_active_kpi_models_is_global_across_orgs(client, owner):
    """active_kpi_models/active_kpis are deployment-wide, not per-org - a
    model created under one org must be counted in every other org's
    summary too (see app.db.models.domain_config.KpiModelCatalog's
    docstring: shared, not org-scoped)."""
    client.post(
        "/api/config/kpi-models", headers=owner["headers"],
        json={"name": "Cross-Org Shared Model", "model_path": "/models/shared.pt"},
    )
    other = _second_org_owner(client)
    resp = client.get("/api/dashboard/summary", headers=other["headers"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["active_kpi_models"] == 1


# ── Alert chart ───────────────────────────────────────────────────────────────

def test_alert_chart_groups_by_day_and_kpi(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Chart Zone")
    priority_id = _make_priority(client, owner["headers"], "Chart Pri")
    _make_camera(client, owner["headers"], "CAM-CHART1", zone_id, priority_id)

    create_alert(camera_id="CAM-CHART1", kpi_name="fire_smoke", alert_type="fire", frame_idx=1, confidence=0.9)
    create_alert(camera_id="CAM-CHART1", kpi_name="fire_smoke", alert_type="fire", frame_idx=2, confidence=0.9)
    create_alert(camera_id="CAM-CHART1", kpi_name="ppe", alert_type="no_helmet", frame_idx=3, confidence=0.8)

    resp = client.get("/api/dashboard/alert-chart", headers=owner["headers"], params={"camera_id": "CAM-CHART1"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["camera_id"] == "CAM-CHART1"
    by_kpi = {p["kpi_name"]: p["count"] for p in body["points"]}
    assert by_kpi["fire_smoke"] == 2
    assert by_kpi["ppe"] == 1


def test_alert_chart_unknown_camera_404s(client, owner):
    resp = client.get("/api/dashboard/alert-chart", headers=owner["headers"], params={"camera_id": "NOPE"})
    assert resp.status_code == 404


def test_alert_chart_zone_restricted_hides_camera(client, owner, db_session):
    zone_a = _make_zone(client, owner["headers"], "Chart Zone A")
    zone_b = _make_zone(client, owner["headers"], "Chart Zone B")
    priority_id = _make_priority(client, owner["headers"], "Chart Pri 2")
    _make_camera(client, owner["headers"], "CAM-CHART-A", zone_a, priority_id)
    _make_camera(client, owner["headers"], "CAM-CHART-B", zone_b, priority_id)

    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "chart.zone.guard")
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/dashboard/alert-chart", headers=headers, params={"camera_id": "CAM-CHART-A"}).status_code == 200
    assert client.get("/api/dashboard/alert-chart", headers=headers, params={"camera_id": "CAM-CHART-B"}).status_code == 404


# ── Cameras ───────────────────────────────────────────────────────────────────

def test_dashboard_cameras_enriched(client, owner):
    zone_id = _make_zone(client, owner["headers"], "Dash Cam Zone")
    priority_id = _make_priority(client, owner["headers"], "Dash Cam Pri", 2, "#00FF00")
    _make_camera(client, owner["headers"], "CAM-DASH1", zone_id, priority_id)

    resp = client.get("/api/dashboard/cameras", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    row = next(c for c in body["cameras"] if c["camera_id"] == "CAM-DASH1")
    assert row["zone_name"] == "Dash Cam Zone"
    assert row["priority_name"] == "Dash Cam Pri"
    assert row["priority_color"] == "#00FF00"
    assert row["status"] == "active"
    assert row["connectivity_status"] == "pending"
    assert row["stream_status"] == "disabled"


def test_dashboard_cameras_zone_scoped(client, owner, db_session):
    zone_a = _make_zone(client, owner["headers"], "Dash Zone A")
    zone_b = _make_zone(client, owner["headers"], "Dash Zone B")
    priority_id = _make_priority(client, owner["headers"], "Dash Pri 3")
    _make_camera(client, owner["headers"], "CAM-DASH-A", zone_a, priority_id)
    _make_camera(client, owner["headers"], "CAM-DASH-B", zone_b, priority_id)

    role = _create_zone_restricted_role(db_session, int(owner["org_id"]), zone_a)
    token = _login_as(client, db_session, int(owner["org_id"]), role.id, "dashcam.zone.guard")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/dashboard/cameras", headers=headers)
    assert resp.status_code == 200, resp.text
    ids = {c["camera_id"] for c in resp.json()["cameras"]}
    assert ids == {"CAM-DASH-A"}
