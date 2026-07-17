from app.core.permissions import full_permission_matrix
from app.core.security import hash_password
from app.db.models import User

VIEWER_PERMISSIONS = {"dashboard": ["view"], "alerts": ["view"]}


def _create_user_directly(db_session, *, org_id, role_id, username, password="Str0ng!Passw0rd", status="active"):
    """Phase 6 must be testable without depending on Phase 7's /api/users
    endpoint, so tests that need "a user holding this role" insert one
    straight into the DB rather than going through the (not-yet-built at the
    time these were written, now built) user-creation API."""
    user = User(
        full_name=f"{username} test",
        personal_email=f"{username}@example.com",
        login_email=f"{username}@example.com",
        username=username,
        password_hash=hash_password(password),
        org_id=org_id,
        role_id=role_id,
        status=status,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_owner_lists_only_the_seeded_owner_role(client, owner):
    resp = client.get("/api/roles", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)  # plain array, not {count, roles}
    assert len(body) == 1
    assert body[0]["name"] == "Owner"
    assert body[0]["is_system"] is True
    assert body[0]["zone_ids"] == []
    assert body[0]["default_email_server_id"] is None


def test_list_roles_requires_auth(client, owner):
    resp = client.get("/api/roles")
    assert resp.status_code == 401


def test_create_role_success(client, owner):
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={
            "name": "Viewer",
            "description": "Read-only access",
            "permissions": VIEWER_PERMISSIONS,
            "default_email_server_id": None,
            "zone_ids": [],
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "Viewer"
    assert body["is_system"] is False
    assert body["permissions"] == VIEWER_PERMISSIONS
    assert isinstance(body["id"], str)


def test_create_role_duplicate_name_case_insensitive_409(client, owner):
    client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Viewer", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "VIEWER", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 409


def test_create_role_rejects_unknown_module_or_action(client, owner):
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Bad", "description": "", "permissions": {"not_a_module": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 422


def test_create_role_rejects_all_empty_permissions(client, owner):
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Empty", "description": "", "permissions": {"dashboard": []},
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 422


def test_create_role_rejects_unknown_zone(client, owner):
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Zoned", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": ["999"]},
    )
    assert resp.status_code == 422


def test_create_role_rejects_unknown_email_server(client, owner):
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Mailer", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": "999", "zone_ids": []},
    )
    assert resp.status_code == 422


def test_owner_role_cannot_be_updated_or_deleted(client, owner):
    roles = client.get("/api/roles", headers=owner["headers"]).json()
    owner_role_id = roles[0]["id"]

    update_resp = client.put(
        f"/api/roles/{owner_role_id}",
        headers=owner["headers"],
        json={"name": "Owner", "description": "hacked", "permissions": full_permission_matrix(),
              "default_email_server_id": None, "zone_ids": []},
    )
    assert update_resp.status_code == 403

    delete_resp = client.delete(f"/api/roles/{owner_role_id}", headers=owner["headers"])
    assert delete_resp.status_code == 403


def test_update_role_success_and_not_found(client, owner):
    created = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Viewer", "description": "orig", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    ).json()

    updated = client.put(
        f"/api/roles/{created['id']}",
        headers=owner["headers"],
        json={"name": "Viewer", "description": "updated desc",
              "permissions": {"dashboard": ["view", "edit"]},
              "default_email_server_id": None, "zone_ids": []},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["description"] == "updated desc"
    assert updated.json()["permissions"] == {"dashboard": ["view", "edit"]}

    missing = client.put(
        "/api/roles/999999",
        headers=owner["headers"],
        json={"name": "Ghost", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    assert missing.status_code == 404


def test_delete_role_blocked_while_users_assigned_then_succeeds(client, owner, db_session):
    role = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Viewer", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    ).json()
    user = _create_user_directly(
        db_session, org_id=int(owner["org_id"]), role_id=int(role["id"]), username="vera.viewer"
    )

    blocked = client.delete(f"/api/roles/{role['id']}", headers=owner["headers"])
    assert blocked.status_code == 409

    user.status = "soft_deleted"
    db_session.add(user)
    db_session.commit()

    ok = client.delete(f"/api/roles/{role['id']}", headers=owner["headers"])
    assert ok.status_code == 204


def test_role_user_counts(client, owner, db_session):
    role = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Viewer", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    ).json()
    _create_user_directly(
        db_session, org_id=int(owner["org_id"]), role_id=int(role["id"]), username="vera.viewer"
    )

    counts = client.get("/api/roles/user-counts", headers=owner["headers"])
    assert counts.status_code == 200, counts.text
    body = counts.json()
    assert body[role["id"]] == 1
    # Owner role has exactly the one owner user created by the `owner` fixture
    roles = client.get("/api/roles", headers=owner["headers"]).json()
    owner_role_id = next(r["id"] for r in roles if r["is_system"])
    assert body[owner_role_id] == 1


def test_role_without_roles_create_permission_is_forbidden(client, owner, db_session):
    limited_role_resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Limited", "description": "", "permissions": {"dashboard": ["view"]},
              "default_email_server_id": None, "zone_ids": []},
    )
    limited_role = limited_role_resp.json()
    _create_user_directly(
        db_session, org_id=int(owner["org_id"]), role_id=int(limited_role["id"]),
        username="larry.limited", password="Str0ng!Passw0rd",
    )

    login = client.post(
        "/api/auth/login", json={"username": "larry.limited", "password": "Str0ng!Passw0rd"}
    )
    assert login.status_code == 200, login.text
    limited_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    resp = client.post(
        "/api/roles",
        headers=limited_headers,
        json={"name": "ShouldFail", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 403


def test_create_role_rejects_whitespace_only_name(client, owner):
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "   ", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 422


def test_create_role_dedupes_duplicate_zone_ids(client, owner, db_session):
    from app.db.models import Zone

    zone = Zone(name="Gate A", org_id=int(owner["org_id"]), enabled=True)
    db_session.add(zone)
    db_session.commit()
    db_session.refresh(zone)

    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "Zoned", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": [str(zone.id), str(zone.id)]},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["zone_ids"] == [str(zone.id)]


def test_create_role_integrity_error_becomes_409_not_500(client, owner, monkeypatch):
    from sqlalchemy.exc import IntegrityError
    from sqlmodel import Session

    def _boom(self, *a, **kw):
        raise IntegrityError("insert", {}, Exception("duplicate key value violates unique constraint"))

    monkeypatch.setattr(Session, "commit", _boom)
    resp = client.post(
        "/api/roles",
        headers=owner["headers"],
        json={"name": "RaceCondition", "description": "", "permissions": VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 409, resp.text
