from sqlmodel import select

from app.db.models import User

from .conftest import VALID_REGISTER_PAYLOAD


def _register(client, payload=None):
    return client.post("/api/auth/register", json=payload or VALID_REGISTER_PAYLOAD)


def _activation_token(db_session, username: str) -> str:
    user = db_session.exec(select(User).where(User.username == username)).first()
    assert user is not None
    assert user.reset_token, "expected an activation token to be set on register"
    return user.reset_token


def _register_and_activate(client, db_session, payload=None):
    payload = payload or VALID_REGISTER_PAYLOAD
    resp = _register(client, payload)
    assert resp.status_code == 201, resp.text
    token = _activation_token(db_session, payload["username"])
    activate_resp = client.post("/api/auth/activate", json={"token": token})
    assert activate_resp.status_code == 200, activate_resp.text
    return payload


def test_register_creates_pending_org_and_owner(client, db_session):
    resp = _register(client)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["username"] == VALID_REGISTER_PAYLOAD["username"]
    assert body["organization_name"] == VALID_REGISTER_PAYLOAD["company_name"]

    user = db_session.exec(
        select(User).where(User.username == VALID_REGISTER_PAYLOAD["username"])
    ).first()
    assert user is not None
    assert user.status == "pending_verification"
    assert user.reset_token is not None


def test_register_second_org_rejected(client):
    first = _register(client)
    assert first.status_code == 201, first.text
    second = _register(client, {**VALID_REGISTER_PAYLOAD, "username": "someone.else"})
    assert second.status_code == 409


def test_register_password_mismatch_rejected(client):
    payload = {**VALID_REGISTER_PAYLOAD, "confirm_password": "Different!123"}
    resp = _register(client, payload)
    assert resp.status_code == 422


def test_register_weak_password_rejected(client):
    payload = {**VALID_REGISTER_PAYLOAD, "password": "weakpass", "confirm_password": "weakpass"}
    resp = _register(client, payload)
    assert resp.status_code == 422


def test_login_before_activation_fails_generic(client, db_session):
    _register(client)
    resp = client.post(
        "/api/auth/login",
        json={"username": VALID_REGISTER_PAYLOAD["username"], "password": VALID_REGISTER_PAYLOAD["password"]},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password."


def test_activate_with_wrong_token_rejected(client):
    _register(client)
    resp = client.post("/api/auth/activate", json={"token": "not-a-real-token"})
    assert resp.status_code == 400


def test_login_lockout_after_five_failed_attempts(client, db_session):
    payload = _register_and_activate(client, db_session)
    for _ in range(5):
        resp = client.post(
            "/api/auth/login", json={"username": payload["username"], "password": "WrongPassword!1"}
        )
        assert resp.status_code == 401
    # 6th attempt, even with the CORRECT password, is now locked out.
    resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password."


def test_login_success_returns_tokens(client, db_session):
    payload = _register_and_activate(client, db_session)
    resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["role_name"] == "Owner"
    assert body["force_password_change"] is False


def test_me_requires_auth_and_returns_full_owner_permissions(client, db_session):
    payload = _register_and_activate(client, db_session)
    login_resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    access_token = login_resp.json()["access_token"]

    unauth = client.get("/api/auth/me")
    assert unauth.status_code == 401

    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["username"] == payload["username"]
    assert body["role_name"] == "Owner"
    assert "cameras" in body["permissions"]
    assert "delete" in body["permissions"]["cameras"]


def test_refresh_rotates_and_invalidates_old_token(client, db_session):
    payload = _register_and_activate(client, db_session)
    login_resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    old_refresh = login_resp.json()["refresh_token"]

    refreshed = client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["refresh_token"] != old_refresh

    reuse = client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
    assert reuse.status_code == 401


def test_logout_revokes_refresh_token_and_kills_the_live_access_token(client, db_session):
    payload = _register_and_activate(client, db_session)
    login_resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    access_token = login_resp.json()["access_token"]
    refresh_token = login_resp.json()["refresh_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # access token works before logout
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    logout_resp = client.post("/api/auth/logout", json={"refresh_token": refresh_token})
    assert logout_resp.status_code == 204

    # the refresh token is dead...
    reuse = client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
    assert reuse.status_code == 401

    # ...and so is the still-unexpired access token issued before logout —
    # this is the whole point: revocation doesn't wait for JWT expiry.
    after_logout = client.get("/api/auth/me", headers=headers)
    assert after_logout.status_code == 401


def test_forgot_password_unknown_email_is_silent_200(client):
    resp = client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})
    assert resp.status_code == 200


def test_reset_password_flow(client, db_session):
    payload = _register_and_activate(client, db_session)

    pre_reset_login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    pre_reset_access_token = pre_reset_login.json()["access_token"]

    forgot_resp = client.post("/api/auth/forgot-password", json={"email": payload["owner_email"]})
    assert forgot_resp.status_code == 200

    user = db_session.exec(select(User).where(User.username == payload["username"])).first()
    db_session.refresh(user)
    reset_token = user.reset_token
    assert reset_token

    weak = client.post(
        "/api/auth/reset-password", json={"token": reset_token, "new_password": "weak"}
    )
    assert weak.status_code == 422

    new_password = "NewStr0ng!Pass"
    ok = client.post(
        "/api/auth/reset-password", json={"token": reset_token, "new_password": new_password}
    )
    assert ok.status_code == 200, ok.text

    old_login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    assert old_login.status_code == 401

    new_login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": new_password}
    )
    assert new_login.status_code == 200, new_login.text

    # a password reset invalidates every session that existed before it,
    # including still-unexpired access tokens issued pre-reset.
    stale = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {pre_reset_access_token}"}
    )
    assert stale.status_code == 401


def test_change_password_requires_current_password(client, db_session):
    payload = _register_and_activate(client, db_session)
    login_resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    wrong = client.post(
        "/api/auth/change-password",
        json={"current_password": "WrongOldPassword!1", "new_password": "AnotherStr0ng!Pass"},
        headers=headers,
    )
    assert wrong.status_code == 401

    ok = client.post(
        "/api/auth/change-password",
        json={"current_password": payload["password"], "new_password": "AnotherStr0ng!Pass"},
        headers=headers,
    )
    assert ok.status_code == 200, ok.text

    relogin = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": "AnotherStr0ng!Pass"}
    )
    assert relogin.status_code == 200


def test_disabling_a_user_kills_their_live_access_token(client, db_session):
    """Phase 7's disable/delete endpoints don't exist yet, but the mechanism
    they'll rely on (require_auth re-checking User.status on every request)
    is already live — flip the status directly to prove it."""
    payload = _register_and_activate(client, db_session)
    login_resp = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    user = db_session.exec(select(User).where(User.username == payload["username"])).first()
    user.status = "disabled"
    db_session.add(user)
    db_session.commit()

    resp = client.get("/api/auth/me", headers=headers)
    assert resp.status_code == 401


def _reset_token_for(db_session, username: str) -> str:
    user = db_session.exec(select(User).where(User.username == username)).first()
    db_session.refresh(user)
    assert user.reset_token
    return user.reset_token


def test_reset_password_form_renders_and_escapes_token(client, db_session):
    payload = _register_and_activate(client, db_session)
    client.post("/api/auth/forgot-password", json={"email": payload["owner_email"]})
    token = _reset_token_for(db_session, payload["username"])

    resp = client.get("/api/auth/reset-password", params={"token": token})
    assert resp.status_code == 200
    assert "<form" in resp.text
    assert token in resp.text  # our own tokens are urlsafe-b64, safe to embed raw

    # a hostile token containing HTML must come back escaped, not executable
    hostile = "\"><script>alert(1)</script>"
    resp2 = client.get("/api/auth/reset-password", params={"token": hostile})
    assert resp2.status_code == 200
    assert "<script>alert(1)</script>" not in resp2.text


def test_reset_password_form_submit_mismatched_passwords(client, db_session):
    payload = _register_and_activate(client, db_session)
    client.post("/api/auth/forgot-password", json={"email": payload["owner_email"]})
    token = _reset_token_for(db_session, payload["username"])

    resp = client.post(
        "/api/auth/reset-password/complete",
        data={"token": token, "new_password": "NewStr0ng!Pass", "confirm_password": "Different!123"},
    )
    assert resp.status_code == 422
    assert "match" in resp.text.lower()


def test_reset_password_form_submit_weak_password(client, db_session):
    payload = _register_and_activate(client, db_session)
    client.post("/api/auth/forgot-password", json={"email": payload["owner_email"]})
    token = _reset_token_for(db_session, payload["username"])

    resp = client.post(
        "/api/auth/reset-password/complete",
        data={"token": token, "new_password": "weak", "confirm_password": "weak"},
    )
    assert resp.status_code == 422


def test_reset_password_form_submit_success_then_login(client, db_session):
    payload = _register_and_activate(client, db_session)
    client.post("/api/auth/forgot-password", json={"email": payload["owner_email"]})
    token = _reset_token_for(db_session, payload["username"])

    new_password = "FormReset!9Pass"
    resp = client.post(
        "/api/auth/reset-password/complete",
        data={"token": token, "new_password": new_password, "confirm_password": new_password},
    )
    assert resp.status_code == 200, resp.text
    assert "Password reset" in resp.text

    old_login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    assert old_login.status_code == 401

    new_login = client.post(
        "/api/auth/login", json={"username": payload["username"], "password": new_password}
    )
    assert new_login.status_code == 200, new_login.text
