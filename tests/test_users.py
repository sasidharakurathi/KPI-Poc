VIEWER_PERMISSIONS = {"dashboard": ["view"]}


def _create_role(client, headers, name="Viewer", permissions=None):
    resp = client.post(
        "/api/roles",
        headers=headers,
        json={"name": name, "description": "", "permissions": permissions or VIEWER_PERMISSIONS,
              "default_email_server_id": None, "zone_ids": []},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_user(client, headers, role_id, username="vera.viewer", password="Str0ng!Passw0rd"):
    resp = client.post(
        "/api/users",
        headers=headers,
        json={
            "full_name": "Vera Viewer", "username": username,
            "email": f"{username.replace('.', '')}@example.com", "phone": "+15557654321",
            "role_id": role_id, "password": password,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_list_users_initially_has_only_the_owner(client, owner):
    resp = client.get("/api/users", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)  # plain array
    assert len(body) == 1
    assert body[0]["username"] == "ada.owner"
    assert body[0]["status"] == "active"
    assert body[0]["must_change_password"] is False


def test_list_users_requires_auth(client, owner):
    resp = client.get("/api/users")
    assert resp.status_code == 401


def test_create_user_success_forces_password_change(client, owner):
    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])
    assert user["role_id"] == role["id"]
    assert user["status"] == "active"
    assert user["must_change_password"] is True
    assert isinstance(user["id"], str)

    # the new user can log in immediately with the admin-supplied password
    login = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Str0ng!Passw0rd"}
    )
    assert login.status_code == 200, login.text
    assert login.json()["force_password_change"] is True


def test_create_user_duplicate_username_409(client, owner):
    role = _create_role(client, owner["headers"])
    _create_user(client, owner["headers"], role["id"], username="vera.viewer")
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "Another Vera", "username": "VERA.viewer", "email": "vera2@example.com",
              "phone": "+15550001111", "role_id": role["id"], "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 409


def test_create_user_duplicate_email_409(client, owner):
    role = _create_role(client, owner["headers"])
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "Dup Email", "username": "dup.email", "email": "owner@example.com",
              "phone": "+15550001111", "role_id": role["id"], "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 409


def test_create_user_invalid_role_422(client, owner):
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "No Role", "username": "no.role", "email": "norole@example.com",
              "phone": "+15550001111", "role_id": "999999", "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 422


def test_create_user_weak_password_422(client, owner):
    role = _create_role(client, owner["headers"])
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "Weak Pw", "username": "weak.pw", "email": "weakpw@example.com",
              "phone": "+15550001111", "role_id": role["id"], "password": "password"},
    )
    assert resp.status_code == 422


def test_create_user_invalid_phone_422(client, owner):
    role = _create_role(client, owner["headers"])
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "Bad Phone", "username": "bad.phone", "email": "badphone@example.com",
              "phone": "5550001111", "role_id": role["id"], "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 422


def test_get_user_detail_and_not_found(client, owner):
    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])

    resp = client.get(f"/api/users/{user['id']}", headers=owner["headers"])
    assert resp.status_code == 200
    assert resp.json()["username"] == "vera.viewer"

    missing = client.get("/api/users/999999", headers=owner["headers"])
    assert missing.status_code == 404


def test_update_user_success(client, owner):
    role = _create_role(client, owner["headers"])
    other_role = _create_role(client, owner["headers"], name="Editor", permissions={"cameras": ["view", "edit"]})
    user = _create_user(client, owner["headers"], role["id"])

    resp = client.put(
        f"/api/users/{user['id']}",
        headers=owner["headers"],
        json={"full_name": "Vera V. Viewer", "email": "vera.new@example.com",
              "phone": "+15559990000", "role_id": other_role["id"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["full_name"] == "Vera V. Viewer"
    assert body["email"] == "vera.new@example.com"
    assert body["role_id"] == other_role["id"]


def test_cannot_demote_the_last_active_admin(client, owner):
    role = _create_role(client, owner["headers"])
    roles = client.get("/api/roles", headers=owner["headers"]).json()
    owner_role_id = next(r["id"] for r in roles if r["is_system"])
    owner_user_id = client.get("/api/users", headers=owner["headers"]).json()[0]["id"]

    resp = client.put(
        f"/api/users/{owner_user_id}",
        headers=owner["headers"],
        json={"full_name": "Ada Owner", "email": "owner@example.com",
              "phone": "+15551234567", "role_id": role["id"]},
    )
    assert resp.status_code == 409
    # still admin afterwards
    roles_after = client.get("/api/users", headers=owner["headers"]).json()
    assert roles_after[0]["role_id"] == owner_role_id


def test_disabling_a_user_blocks_login_and_kills_live_session(client, owner):
    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])
    login = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Str0ng!Passw0rd"}
    )
    access_token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    resp = client.patch(f"/api/users/{user['id']}/status", headers=owner["headers"], json={"status": "inactive"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "inactive"

    # live session dies immediately
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    # and login is blocked
    relogin = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Str0ng!Passw0rd"}
    )
    assert relogin.status_code == 401

    # reactivating restores login
    reactivate = client.patch(f"/api/users/{user['id']}/status", headers=owner["headers"], json={"status": "active"})
    assert reactivate.status_code == 200
    relogin2 = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Str0ng!Passw0rd"}
    )
    assert relogin2.status_code == 200


def test_cannot_disable_or_delete_the_last_active_admin(client, owner):
    owner_user_id = client.get("/api/users", headers=owner["headers"]).json()[0]["id"]

    disable = client.patch(
        f"/api/users/{owner_user_id}/status", headers=owner["headers"], json={"status": "inactive"}
    )
    assert disable.status_code == 409

    delete = client.delete(f"/api/users/{owner_user_id}", headers=owner["headers"])
    assert delete.status_code == 409


def test_soft_delete_hides_user_but_retains_record(client, owner, db_session):
    from sqlmodel import select as sa_select

    from app.db.models import User as UserModel

    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])

    resp = client.delete(f"/api/users/{user['id']}", headers=owner["headers"])
    assert resp.status_code == 204

    listed = client.get("/api/users", headers=owner["headers"]).json()
    assert all(u["id"] != user["id"] for u in listed)

    detail = client.get(f"/api/users/{user['id']}", headers=owner["headers"])
    assert detail.status_code == 404  # hidden, per PRD "hidden from active lists"

    db_row = db_session.exec(sa_select(UserModel).where(UserModel.id == int(user["id"]))).first()
    assert db_row is not None  # but the record itself is retained
    assert db_row.status == "soft_deleted"


def test_reset_password_forces_change_and_kills_live_session(client, owner):
    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])
    login = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Str0ng!Passw0rd"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    resp = client.post(
        f"/api/users/{user['id']}/reset-password",
        headers=owner["headers"],
        json={"new_password": "Br4ndNewPass!"},
    )
    assert resp.status_code == 204

    assert client.get("/api/auth/me", headers=headers).status_code == 401

    old_login = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Str0ng!Passw0rd"}
    )
    assert old_login.status_code == 401

    new_login = client.post(
        "/api/auth/login", json={"username": "vera.viewer", "password": "Br4ndNewPass!"}
    )
    assert new_login.status_code == 200
    assert new_login.json()["force_password_change"] is True


def test_reset_password_weak_password_422(client, owner):
    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])
    resp = client.post(
        f"/api/users/{user['id']}/reset-password",
        headers=owner["headers"],
        json={"new_password": "weak"},
    )
    assert resp.status_code == 422


def test_user_management_forbidden_without_users_permission(client, owner, db_session):
    from app.core.security import hash_password
    from app.db.models import User as UserModel

    limited_role = _create_role(client, owner["headers"], name="NoUsersAccess",
                                 permissions={"dashboard": ["view"]})
    limited_user = UserModel(
        full_name="No Access", personal_email="noaccess@example.com",
        login_email="noaccess@example.com", username="no.access",
        password_hash=hash_password("Str0ng!Passw0rd"),
        org_id=int(owner["org_id"]), role_id=int(limited_role["id"]), status="active",
    )
    db_session.add(limited_user)
    db_session.commit()

    login = client.post(
        "/api/auth/login", json={"username": "no.access", "password": "Str0ng!Passw0rd"}
    )
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    resp = client.get("/api/users", headers=headers)
    assert resp.status_code == 403


def test_create_user_rejects_whitespace_only_full_name(client, owner):
    role = _create_role(client, owner["headers"])
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "   ", "username": "blank.name", "email": "blankname@example.com",
              "phone": "+15550001111", "role_id": role["id"], "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 422


def test_update_user_rejects_whitespace_only_full_name(client, owner):
    role = _create_role(client, owner["headers"])
    user = _create_user(client, owner["headers"], role["id"])
    resp = client.put(
        f"/api/users/{user['id']}",
        headers=owner["headers"],
        json={"full_name": "   ", "email": "vera.viewer@example.com",
              "phone": "+15557654321", "role_id": role["id"]},
    )
    assert resp.status_code == 422


def test_create_user_integrity_error_becomes_409_not_500(client, owner, monkeypatch):
    from sqlalchemy.exc import IntegrityError
    from sqlmodel import Session

    role = _create_role(client, owner["headers"])

    def _boom(self, *a, **kw):
        raise IntegrityError("insert", {}, Exception("duplicate key value violates unique constraint"))

    monkeypatch.setattr(Session, "commit", _boom)
    resp = client.post(
        "/api/users",
        headers=owner["headers"],
        json={"full_name": "Race Condition", "username": "race.condition",
              "email": "race@example.com", "phone": "+15550001111",
              "role_id": role["id"], "password": "Str0ng!Passw0rd"},
    )
    assert resp.status_code == 409, resp.text
