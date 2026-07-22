"""Authentication endpoints — Phase 0.

Implements:
  POST /api/auth/register          Organization sign-up (multi-tenant — any number of orgs may register)
  POST /api/auth/activate          Consume the activation token (JSON)
  GET  /api/auth/activate          Same, for clicking the emailed link directly
  POST /api/auth/login             Username + password -> access + refresh tokens
  POST /api/auth/refresh           Rotate refresh token -> new access + refresh tokens
  POST /api/auth/logout            Revoke the given refresh token
  GET  /api/auth/me                Current user profile + resolved permissions
  POST /api/auth/forgot-password    Send reset link (generic response either way)
  POST /api/auth/reset-password     Consume reset token, set new password (JSON)
  GET  /api/auth/reset-password     Same, as a browser-friendly form for the emailed link
  POST /api/auth/reset-password/complete   Form submission target for the above
  POST /api/auth/change-password    Authenticated password change (also clears
                                     force_password_change, for Phase 7's forced
                                     first-login flow)

Business logic lives in app.services.auth_service; this module is a thin
request/response layer.
"""
import html as _html

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app.core.dependencies import CurrentUser, DbSession
from app.db.models import Role, User
from app.schemas.auth import (
    ActivateRequest, ChangePasswordRequest, LoginRequest, LoginResponse,
    MeResponse, OrgRegisterRequest, PasswordResetConfirm, PasswordResetRequest,
    RefreshRequest, RegisterResponse, TokenResponse,
)
from app.services import auth_service

_PAGE = "<html><body style='font-family:sans-serif;max-width:420px;margin:60px auto'>{body}</body></html>"

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(payload: OrgRegisterRequest, db: DbSession) -> RegisterResponse:
    org, owner, _role, email_sent = auth_service.register_organization(db, payload)
    message = (
        "Organization created. Check the owner's email for an activation link."
        if email_sent else
        "Organization created, but no default email server is configured for this "
        "deployment — the activation link could not be emailed. Add a default email "
        "server under Configuration, or fetch the activation token directly."
    )
    return RegisterResponse(
        organization_name=org.name,
        username=owner.username,
        message=message,
        activation_email_sent=email_sent,
    )


@router.post("/activate")
def activate(payload: ActivateRequest, db: DbSession) -> dict:
    auth_service.activate_account(db, payload.token)
    return {"message": "Account activated. You can now sign in."}


@router.get("/activate", response_class=HTMLResponse)
def activate_via_link(token: str, db: DbSession) -> HTMLResponse:
    """Convenience endpoint for the emailed activation link — no frontend page
    exists to consume this yet, so it renders a minimal confirmation itself."""
    try:
        auth_service.activate_account(db, token)
        body = "<h1>Account activated</h1><p>You can now sign in.</p>"
    except HTTPException as exc:
        body = f"<h1>Activation failed</h1><p>{_html.escape(str(exc.detail))}</p>"
    return HTMLResponse(_PAGE.format(body=body))


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, db: DbSession) -> LoginResponse:
    user, role = auth_service.authenticate(db, payload.username, payload.password, request)
    tokens = auth_service.issue_tokens(db, user, role)
    return LoginResponse(**tokens)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, db: DbSession) -> TokenResponse:
    tokens = auth_service.refresh_tokens(db, payload.refresh_token)
    return TokenResponse(**tokens)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: RefreshRequest, db: DbSession) -> None:
    """Revokes the whole session: every access token already issued to this
    user stops working immediately (not just the refresh token), see
    auth_service.revoke_all_sessions()."""
    auth_service.logout(db, payload.refresh_token)


@router.get("/me", response_model=MeResponse)
def me(user: CurrentUser, db: DbSession) -> MeResponse:
    db_user = db.get(User, int(user["sub"]))
    if db_user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    role = db.get(Role, db_user.role_id) if db_user.role_id else None
    return MeResponse(
        id=db_user.id,
        username=db_user.username,
        full_name=db_user.full_name,
        login_email=db_user.login_email,
        personal_email=db_user.personal_email,
        role_id=role.id if role else None,
        role_name=role.name if role else None,
        org_id=db_user.org_id,
        permissions=(role.permissions if role else None) or {},
        force_password_change=db_user.force_password_change,
        status=db_user.status,
    )


@router.post("/forgot-password")
def forgot_password(payload: PasswordResetRequest, request: Request, db: DbSession) -> dict:
    auth_service.request_password_reset(db, payload.email, request)
    return {"message": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(payload: PasswordResetConfirm, db: DbSession) -> dict:
    auth_service.reset_password(db, payload.token, payload.new_password)
    return {"message": "Password has been reset. You can now sign in."}


@router.get("/reset-password", response_class=HTMLResponse)
def reset_password_form(token: str) -> HTMLResponse:
    """Convenience page for the emailed reset link — no frontend page exists
    to consume this yet, so it renders a minimal form itself. `token` is
    escaped before being echoed into the page (it's attacker-reachable via
    the query string, unlike the tokens we generate ourselves)."""
    safe_token = _html.escape(token, quote=True)
    body = f"""
    <h1>Reset your password</h1>
    <form method="post" action="/api/auth/reset-password/complete">
      <input type="hidden" name="token" value="{safe_token}">
      <label>New password<br>
        <input type="password" name="new_password" required minlength="8" style="width:100%">
      </label><br><br>
      <label>Confirm new password<br>
        <input type="password" name="confirm_password" required minlength="8" style="width:100%">
      </label><br><br>
      <button type="submit">Reset password</button>
    </form>
    <p style="color:#666;font-size:13px">At least 8 characters, with an uppercase letter,
    a lowercase letter, a number, and a symbol.</p>
    """
    return HTMLResponse(_PAGE.format(body=body))


@router.post("/reset-password/complete", response_class=HTMLResponse)
def reset_password_form_submit(
    db: DbSession,
    token: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
) -> HTMLResponse:
    if new_password != confirm_password:
        return HTMLResponse(
            _PAGE.format(body="<h1>Passwords don't match</h1><p>Go back and try again.</p>"),
            status_code=422,
        )
    try:
        validated = PasswordResetConfirm(token=token, new_password=new_password)
    except ValidationError as exc:
        message = "; ".join(e["msg"] for e in exc.errors())
        return HTMLResponse(
            _PAGE.format(body=f"<h1>Could not reset password</h1><p>{_html.escape(message)}</p>"),
            status_code=422,
        )
    try:
        auth_service.reset_password(db, validated.token, validated.new_password)
    except HTTPException as exc:
        return HTMLResponse(
            _PAGE.format(body=f"<h1>Could not reset password</h1><p>{_html.escape(str(exc.detail))}</p>"),
            status_code=exc.status_code,
        )
    return HTMLResponse(_PAGE.format(
        body="<h1>Password reset</h1><p>You can now sign in with your new password.</p>"
    ))


@router.post("/change-password")
def change_password(payload: ChangePasswordRequest, user: CurrentUser, db: DbSession) -> dict:
    auth_service.change_password(db, int(user["sub"]), payload.current_password, payload.new_password)
    return {"message": "Password changed."}
