"""Phase 4 (Real-Time) tests: the ws_manager pub-sub core, and the
/api/ws/alerts WebSocket endpoint's auth handshake.

ws_manager's async methods are exercised directly via asyncio.run() - no
pytest-asyncio plugin needed, and no real FastAPI WebSocket required for
those (a tiny fake with an async send_json is enough). The endpoint's auth/
connect/reject behavior is tested through TestClient.websocket_connect(),
which is synchronous from the test's point of view.
"""
import asyncio

import pytest
from starlette.websockets import WebSocketDisconnect

from app.services.ws_manager import AlertsWebSocketManager

from .conftest import VALID_REGISTER_PAYLOAD, register_activate_login


class _FakeWebSocket:
    def __init__(self, fail: bool = False):
        self.sent: list[dict] = []
        self.fail = fail

    async def send_json(self, message: dict) -> None:
        if self.fail:
            raise RuntimeError("connection closed")
        self.sent.append(message)


# ── ws_manager core ──────────────────────────────────────────────────────────

def test_connection_can_see_unrestricted_within_same_org():
    manager = AlertsWebSocketManager()

    async def _run():
        ws = _FakeWebSocket()
        conn = await manager.connect(ws, org_id=1, allowed_camera_ids=None)
        assert conn.can_see("CAM-1", org_id=1)
        assert conn.can_see(None, org_id=1)  # camera-less events are visible to unrestricted connections too

    asyncio.run(_run())


def test_connection_never_sees_a_different_org_even_if_unrestricted():
    """The actual security fix: allowed_camera_ids=None means unrestricted
    *within this connection's own org*, never across every org."""
    manager = AlertsWebSocketManager()

    async def _run():
        ws = _FakeWebSocket()
        conn = await manager.connect(ws, org_id=1, allowed_camera_ids=None)
        assert not conn.can_see("CAM-1", org_id=2)
        assert not conn.can_see(None, org_id=2)
        assert not conn.can_see("CAM-1", org_id=None)  # unresolvable org never matches either

    asyncio.run(_run())


def test_connection_can_see_restricted():
    manager = AlertsWebSocketManager()

    async def _run():
        ws = _FakeWebSocket()
        conn = await manager.connect(ws, org_id=1, allowed_camera_ids={"CAM-1"})
        assert conn.can_see("CAM-1", org_id=1)
        assert not conn.can_see("CAM-2", org_id=1)
        assert not conn.can_see(None, org_id=1)
        assert not conn.can_see("CAM-1", org_id=2)  # zone-unrestricted-within-org still isn't cross-org

    asyncio.run(_run())


def test_broadcast_delivers_only_to_matching_connections():
    manager = AlertsWebSocketManager()

    async def _run():
        ws_all = _FakeWebSocket()
        ws_zone1 = _FakeWebSocket()
        await manager.connect(ws_all, org_id=1, allowed_camera_ids=None)
        await manager.connect(ws_zone1, org_id=1, allowed_camera_ids={"CAM-1"})

        await manager.broadcast("alert.created", {"id": 1}, camera_id="CAM-1", org_id=1)
        await manager.broadcast("alert.created", {"id": 2}, camera_id="CAM-2", org_id=1)

        assert [m["data"]["id"] for m in ws_all.sent] == [1, 2]
        assert [m["data"]["id"] for m in ws_zone1.sent] == [1]

    asyncio.run(_run())


def test_broadcast_never_reaches_a_different_orgs_connection():
    manager = AlertsWebSocketManager()

    async def _run():
        ws_org1 = _FakeWebSocket()
        ws_org2 = _FakeWebSocket()
        await manager.connect(ws_org1, org_id=1, allowed_camera_ids=None)
        await manager.connect(ws_org2, org_id=2, allowed_camera_ids=None)

        await manager.broadcast("alert.created", {"id": 1}, camera_id="CAM-1", org_id=1)

        assert [m["data"]["id"] for m in ws_org1.sent] == [1]
        assert ws_org2.sent == []

    asyncio.run(_run())


def test_broadcast_removes_stale_connection_on_send_failure():
    manager = AlertsWebSocketManager()

    async def _run():
        broken = _FakeWebSocket(fail=True)
        healthy = _FakeWebSocket()
        await manager.connect(broken, org_id=1, allowed_camera_ids=None)
        await manager.connect(healthy, org_id=1, allowed_camera_ids=None)

        assert manager.connection_count == 2
        await manager.broadcast("alert.created", {"id": 1}, camera_id=None, org_id=1)
        assert manager.connection_count == 1
        assert healthy.sent

    asyncio.run(_run())


def test_broadcast_threadsafe_is_a_noop_without_a_bound_loop():
    manager = AlertsWebSocketManager()
    # Should not raise even though nothing is listening / no loop is bound.
    manager.broadcast_threadsafe("alert.created", {"id": 1}, camera_id=None, org_id=1)


# ── WebSocket endpoint auth ──────────────────────────────────────────────────

def test_ws_alerts_rejects_missing_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws/alerts"):
            pass


def test_ws_alerts_rejects_invalid_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws/alerts?token=not-a-real-token"):
            pass


def test_ws_alerts_accepts_valid_owner_token(client):
    body = register_activate_login(client)
    token = body["access_token"]
    with client.websocket_connect(f"/api/ws/alerts?token={token}"):
        pass  # connecting without raising is the assertion


def test_ws_alerts_rejects_role_without_alerts_permission(client, db_session):
    from app.core.security import hash_password
    from app.db.models import Role, User

    payload = dict(VALID_REGISTER_PAYLOAD, username="ws.no.perm.owner")
    body = register_activate_login(client, payload)
    org_id = int(body["org_id"])

    role = Role(name="NoAlerts", permissions={"dashboard": ["view"]}, org_id=org_id, zone_ids=[])
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)

    user = User(
        full_name="No Alerts User", personal_email="noalerts@example.com",
        login_email="noalerts@example.com", username="no.alerts.user",
        password_hash=hash_password("Str0ng!Passw0rd"),
        org_id=org_id, role_id=role.id, status="active",
    )
    db_session.add(user)
    db_session.commit()

    login = client.post("/api/auth/login", json={"username": "no.alerts.user", "password": "Str0ng!Passw0rd"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/ws/alerts?token={token}"):
            pass
