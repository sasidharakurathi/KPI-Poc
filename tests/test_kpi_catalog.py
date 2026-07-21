"""Phase 3 (KPI Management) catalog tests.

Covers the original audit fixes (auth, org scoping, unregistered-name
rejection, native JSON columns) plus "complete Phase 3": the reshaped
response matching the frontend's real KpiModelDef contract, model_ids
validation against Phase 1's KpiModelCatalog, and pipeline write-through
(config.json actually changes when the catalog changes).
"""
from app.kpis import get_registry, list_registered_names


def _a_real_kpi_name() -> str:
    names = list_registered_names()
    assert names, "expected at least one registered KPI for this test to be meaningful"
    return names[0]


def _make_detection_model(client, headers, name="Test Model"):
    resp = client.post(
        "/api/config/kpi-models", headers=headers,
        json={"name": name, "model_path": "app/models/test.pt", "confidence_threshold": 0.5},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def test_kpi_catalog_requires_auth(client):
    assert client.get("/api/kpis/catalog").status_code == 401
    assert client.put("/api/kpis/catalog/whatever", json={}).status_code == 401
    assert client.patch("/api/kpis/catalog/whatever/toggle").status_code == 401


def test_update_rejects_unregistered_kpi_name(client, owner):
    resp = client.put(
        "/api/kpis/catalog/totally_made_up_kpi_xyz", headers=owner["headers"],
        json={"config": {"x": 1}},
    )
    assert resp.status_code == 404


def test_create_and_get_catalog_entry_matches_frontend_contract(client, owner):
    name = _a_real_kpi_name()
    expected_display_name = get_registry()[name].display_name

    resp = client.put(
        f"/api/kpis/catalog/{name}", headers=owner["headers"],
        json={"config": {"confidence": 0.5}, "description": "Test description", "category": "safety"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["key"] == name
    assert body["display_name"] == expected_display_name
    assert body["description"] == "Test description"
    assert body["category"] == "safety"
    assert body["config"] == {"confidence": 0.5}
    assert body["enabled"] is True
    assert body["model_ids"] == []
    assert "added_at" in body
    # old field names must not leak through
    assert "kpi_name" not in body
    assert "enable_status" not in body
    assert "parameters" not in body
    assert "assigned_models" not in body

    get_resp = client.get(f"/api/kpis/catalog/{name}", headers=owner["headers"])
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()["config"] == {"confidence": 0.5}


def test_create_rejects_invalid_category(client, owner):
    name = _a_real_kpi_name()
    resp = client.put(
        f"/api/kpis/catalog/{name}", headers=owner["headers"],
        json={"category": "not-a-real-category"},
    )
    assert resp.status_code == 422, resp.text


def test_model_ids_validated_against_detection_model_catalog(client, owner):
    name = _a_real_kpi_name()
    resp = client.put(
        f"/api/kpis/catalog/{name}", headers=owner["headers"],
        json={"model_ids": ["999999"]},
    )
    assert resp.status_code == 422, resp.text

    real_model_id = _make_detection_model(client, owner["headers"])
    resp2 = client.put(
        f"/api/kpis/catalog/{name}", headers=owner["headers"],
        json={"model_ids": [real_model_id]},
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["model_ids"] == [real_model_id]


def test_toggle_catalog_entry(client, owner):
    name = _a_real_kpi_name()
    client.put(f"/api/kpis/catalog/{name}", headers=owner["headers"], json={"config": {}})

    resp = client.patch(f"/api/kpis/catalog/{name}/toggle", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["enabled"] is False

    resp2 = client.patch(f"/api/kpis/catalog/{name}/toggle", headers=owner["headers"])
    assert resp2.json()["enabled"] is True


def test_toggle_unknown_entry_404s(client, owner):
    resp = client.patch("/api/kpis/catalog/never-created/toggle", headers=owner["headers"])
    assert resp.status_code == 404


def test_catalog_is_org_scoped(client, owner, db_session):
    from app.db.models.kpi_configuration import KPIConfiguration

    name = _a_real_kpi_name()
    other_org_row = KPIConfiguration(kpi_name=name, org_id=999999, parameters={"foreign": True})
    db_session.add(other_org_row)
    db_session.commit()

    resp = client.get("/api/kpis/catalog", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert all(row["key"] != name for row in resp.json())  # the other org's row must not leak in

    detail = client.get(f"/api/kpis/catalog/{name}", headers=owner["headers"])
    assert detail.status_code == 404  # exists, but belongs to a different org


# ── Pipeline write-through ───────────────────────────────────────────────────

def test_config_update_writes_through_to_config_json(client, owner, monkeypatch):
    calls = []
    import app.api.v1.endpoints.kpis as kpis_ep
    monkeypatch.setattr(kpis_ep, "update_kpi_config", lambda cls_name, updates: calls.append((cls_name, updates)) or updates)

    name = _a_real_kpi_name()
    cls_name = get_registry()[name].__name__

    resp = client.put(
        f"/api/kpis/catalog/{name}", headers=owner["headers"],
        json={"config": {"confidence": 0.7}},
    )
    assert resp.status_code == 200, resp.text
    assert calls == [(cls_name, {"confidence": 0.7})]


def test_toggle_writes_through_to_config_json(client, owner, monkeypatch):
    calls = []
    import app.api.v1.endpoints.kpis as kpis_ep
    monkeypatch.setattr(kpis_ep, "update_kpi_config", lambda cls_name, updates: calls.append((cls_name, updates)) or updates)

    name = _a_real_kpi_name()
    cls_name = get_registry()[name].__name__
    client.put(f"/api/kpis/catalog/{name}", headers=owner["headers"], json={"config": {}})
    calls.clear()

    resp = client.patch(f"/api/kpis/catalog/{name}/toggle", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert calls == [(cls_name, {"enabled": False})]


def test_model_ids_do_not_write_through(client, owner, monkeypatch):
    """No sound way to collapse a list of model_ids onto config.json's single
    model_path — assigning models should not touch the pipeline config file."""
    calls = []
    import app.api.v1.endpoints.kpis as kpis_ep
    monkeypatch.setattr(kpis_ep, "update_kpi_config", lambda cls_name, updates: calls.append((cls_name, updates)) or updates)

    name = _a_real_kpi_name()
    real_model_id = _make_detection_model(client, owner["headers"], "Write-through Model")

    resp = client.put(
        f"/api/kpis/catalog/{name}", headers=owner["headers"],
        json={"model_ids": [real_model_id]},
    )
    assert resp.status_code == 200, resp.text
    assert calls == []
