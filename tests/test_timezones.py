"""Phase 1 timezone catalog tests: the static, non-editable `timezones`
reference table and its wiring into organization registration + zones."""
from .conftest import VALID_REGISTER_PAYLOAD


def test_list_timezones_is_public_and_seeded(client):
    resp = client.get("/api/timezones")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body) == 200
    assert all({"id", "abbreviation", "timezone_name"} <= row.keys() for row in body)


def test_register_rejects_unknown_timezone_id(client):
    payload = dict(VALID_REGISTER_PAYLOAD, default_timezone_id="999999", username="badtz.owner")
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 422, resp.text


def test_register_rejects_non_numeric_timezone_id(client):
    payload = dict(VALID_REGISTER_PAYLOAD, default_timezone_id="not-a-number", username="badtz2.owner")
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 422, resp.text


def test_register_succeeds_with_valid_timezone_id(client):
    resp = client.post("/api/auth/register", json=VALID_REGISTER_PAYLOAD)
    assert resp.status_code == 201, resp.text
