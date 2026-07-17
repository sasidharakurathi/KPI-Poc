"""Business logic for Phase 0: organization bootstrap + authentication.

Kept out of app/api/v1/endpoints/auth.py so the router stays a thin
request/response layer, per app/services/camera_service.py's precedent.
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.permissions import full_permission_matrix
from app.core.rate_limit import login_limiter, password_reset_limiter
from app.core.security import (
    create_access_token, generate_secure_token, hash_password, hash_token, verify_password,
)
from app.db.models import Organization, RefreshToken, Role, User

logger = logging.getLogger(__name__)

LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_MINUTES = 15
ACTIVATION_TOKEN_TTL_HOURS = 48
RESET_TOKEN_TTL_HOURS = 2

_GENERIC_LOGIN_ERROR = "Invalid username or password."


def _client_ip(request: Optional[Request]) -> str:
    if request is None or request.client is None:
        return "unknown"
    return request.client.host


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "org"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Registration & activation ────────────────────────────────────────────────

def register_organization(db: Session, payload) -> tuple[Organization, User, Role]:
    """PRD §2.1: one organization per deployment. Creates the Organization,
    its built-in Owner role (is_system=True, every permission), and the first
    User (status=pending_verification until the activation link is used)."""
    if db.exec(select(Organization)).first() is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "An organization already exists on this deployment. This platform "
            "supports one organization per deployment.",
        )
    if db.exec(select(User).where(User.username == payload.username)).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username is already taken.")

    org = Organization(
        # "One organization per deployment" is enforced above by a
        # check-then-insert that has a TOCTOU race window under concurrent
        # requests — two requests can both see "no org exists" before either
        # commits. Pinning id=1 turns this into a real, DB-enforced
        # singleton-row constraint: a second concurrent insert collides on
        # the primary key and raises IntegrityError, caught below.
        id=1,
        org_id=_slugify(payload.company_name),
        name=payload.company_name,
        tagline=payload.tagline,
        default_timezone=payload.default_timezone,
        site_name=payload.site_name,
        site_address=payload.site_address,
        latitude=payload.latitude,
        longitude=payload.longitude,
    )
    activation_token = generate_secure_token()
    try:
        db.add(org)
        db.flush()

        owner_role = Role(
            name="Owner",
            description=(
                "Full access to every module, including organization and billing "
                "settings. Cannot be edited or deleted."
            ),
            permissions=full_permission_matrix(),
            is_system=True,
            org_id=org.id,
        )
        db.add(owner_role)
        db.flush()

        owner = User(
            full_name=payload.owner_full_name,
            designation=payload.owner_designation,
            personal_email=payload.owner_email,
            phone=payload.owner_phone,
            username=payload.username,
            login_email=payload.owner_email,
            password_hash=hash_password(payload.password),
            org_id=org.id,
            role_id=owner_role.id,
            status="pending_verification",
            reset_token=activation_token,
            reset_token_expires=datetime.utcnow() + timedelta(hours=ACTIVATION_TOKEN_TTL_HOURS),
        )
        db.add(owner)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "An organization already exists on this deployment, or the username is "
            "already taken. This platform supports one organization per deployment.",
        )
    db.refresh(org)
    db.refresh(owner)
    db.refresh(owner_role)

    _send_activation_email(owner, activation_token)
    return org, owner, owner_role


def _send_activation_email(user: User, token: str) -> None:
    from app.config import settings
    from app.notifications import send_transactional_email

    link = f"{settings.PUBLIC_BASE_URL}/api/auth/activate?token={token}"
    subject = "Activate your Vision AI account"
    plain = (
        f"Hi {user.full_name},\n\n"
        f"Your Vision AI Safety & Compliance Platform account has been created.\n"
        f"Activate it within {ACTIVATION_TOKEN_TTL_HOURS} hours: {link}\n\n"
        f"Or activate via the API directly:\n"
        f'POST /api/auth/activate  {{"token": "{token}"}}\n'
    )
    html = (
        f"<p>Hi {user.full_name},</p>"
        f"<p>Your Vision AI Safety &amp; Compliance Platform account has been created.</p>"
        f'<p><a href="{link}">Activate your account</a> '
        f"(valid for {ACTIVATION_TOKEN_TTL_HOURS} hours).</p>"
    )
    try:
        send_transactional_email([user.login_email], subject, html, plain)
    except Exception:
        logger.warning(
            "[auth] could not send activation email to %s — SMTP may not be "
            "configured yet. Activation token: %s", user.login_email, token,
        )


def activate_account(db: Session, token: str) -> User:
    user = db.exec(select(User).where(User.reset_token == token)).first()
    if user is None or user.status != "pending_verification":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or already-used activation token.")
    if user.reset_token_expires is None or _now() > _aware(user.reset_token_expires):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Activation link has expired.")

    user.status = "active"
    user.verify_status = True
    user.reset_token = None
    user.reset_token_expires = None
    user.updated_at = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ── Login / tokens ───────────────────────────────────────────────────────────

def authenticate(db: Session, username: str, password: str, request: Optional[Request]) -> tuple[User, Role]:
    """PRD §2.2/§2.3: generic failure message (never reveals whether the
    username exists), 5 failed attempts within 15 minutes locks the account
    for 15 minutes, plus IP/username rate limiting on top."""
    ip = _client_ip(request)
    if login_limiter.hit(f"ip:{ip}", 900) > 30:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many login attempts from this network. Try again later.")
    if login_limiter.hit(f"user:{username.lower()}", 900) > 10:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many login attempts for this account. Try again later.")

    user = db.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_LOGIN_ERROR)

    if user.locked_until and _now() < _aware(user.locked_until):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_LOGIN_ERROR)

    if user.status != "active" or not verify_password(password, user.password_hash):
        if user.status == "active":
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= LOGIN_MAX_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
            db.add(user)
            db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_LOGIN_ERROR)

    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)

    role = db.get(Role, user.role_id) if user.role_id else None
    if role is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "User has no role assigned.")
    return user, role


def issue_tokens(db: Session, user: User, role: Role) -> dict:
    """PRD §2.3: short-lived JWT access token + server-tracked refresh token."""
    from app.config import settings

    access_token = create_access_token(
        subject=str(user.id),
        roles=[role.name],
        extra_claims={
            "username": user.username,
            "org_id": user.org_id,
            "role_id": role.id,
            "is_system": role.is_system,
            "token_version": user.token_version,
        },
    )
    raw_refresh = generate_secure_token()
    db.add(RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_refresh),
        expires_at=datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_EXPIRE_DAYS),
    ))
    db.commit()

    return {
        "access_token": access_token,
        "refresh_token": raw_refresh,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role_id": role.id,
        "role_name": role.name,
        "org_id": user.org_id,
        "force_password_change": user.force_password_change,
    }


def refresh_tokens(db: Session, raw_refresh_token: str) -> dict:
    """Rotate on every use: the presented token is revoked and a new pair issued."""
    token_hash = hash_token(raw_refresh_token)
    row = db.exec(select(RefreshToken).where(RefreshToken.token_hash == token_hash)).first()
    if row is None or row.revoked or _now() > _aware(row.expires_at):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token is invalid or expired.")

    user = db.get(User, row.user_id)
    if user is None or user.status != "active":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token is invalid or expired.")
    role = db.get(Role, user.role_id) if user.role_id else None
    if role is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "User has no role assigned.")

    row.revoked = True
    db.add(row)
    db.commit()

    return issue_tokens(db, user, role)


def revoke_all_sessions(db: Session, user: User) -> None:
    """Immediately invalidates every access token already issued to this user
    (bumping token_version, which require_auth compares against the JWT's
    claim on every request) and revokes every outstanding refresh token.

    A still-unexpired access token from before this call stops working on its
    very next use — it doesn't have to wait out its `exp`. Used by logout and
    password reset; Phase 7's disable/delete should call this too.
    """
    user.token_version += 1
    db.add(user)
    for row in db.exec(
        select(RefreshToken).where(RefreshToken.user_id == user.id, RefreshToken.revoked.is_(False))
    ).all():
        row.revoked = True
        db.add(row)
    db.commit()


def logout(db: Session, raw_refresh_token: str) -> None:
    """Logout revokes the whole session, not just the one refresh token: every
    access token already handed out to this user is invalidated too (see
    revoke_all_sessions). If the token isn't found or already revoked, this is
    a silent no-op — the response is identical either way, so it can't be used
    to probe whether a token is valid."""
    token_hash = hash_token(raw_refresh_token)
    row = db.exec(select(RefreshToken).where(RefreshToken.token_hash == token_hash)).first()
    if row is None:
        return
    user = db.get(User, row.user_id)
    if user is None:
        return
    revoke_all_sessions(db, user)


# ── Password reset / change ─────────────────────────────────────────────────

def request_password_reset(db: Session, email: str, request: Optional[Request]) -> None:
    ip = _client_ip(request)
    if password_reset_limiter.hit(f"ip:{ip}", 900) > 10:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many reset requests. Try again later.")

    user = db.exec(select(User).where(User.login_email == email)).first()
    if user is None or user.status == "soft_deleted":
        return  # never reveal whether the email exists

    token = generate_secure_token()
    user.reset_token = token
    user.reset_token_expires = datetime.utcnow() + timedelta(hours=RESET_TOKEN_TTL_HOURS)
    db.add(user)
    db.commit()

    from app.config import settings

    link = f"{settings.PUBLIC_BASE_URL}/api/auth/reset-password?token={token}"
    subject = "Reset your Vision AI password"
    plain = (
        f"Hi {user.full_name},\n\n"
        f"Reset your password within {RESET_TOKEN_TTL_HOURS} hours: {link}\n\n"
        f"Or use the token directly via the API:\n"
        f'POST /api/auth/reset-password  {{"token": "{token}", "new_password": "..."}}\n\n'
        f"If you didn't request this, you can ignore this email.\n"
    )
    html = (
        f"<p>Hi {user.full_name},</p>"
        f'<p><a href="{link}">Reset your password</a> '
        f"(valid for {RESET_TOKEN_TTL_HOURS} hours).</p>"
        f"<p>If you didn't request this, you can safely ignore this email.</p>"
    )
    try:
        from app.notifications import send_transactional_email
        send_transactional_email([user.login_email], subject, html, plain)
    except Exception:
        logger.warning(
            "[auth] could not send password reset email to %s. Reset token: %s",
            user.login_email, token,
        )


def reset_password(db: Session, token: str, new_password: str) -> User:
    user = db.exec(select(User).where(User.reset_token == token)).first()
    if user is None or user.reset_token_expires is None or _now() > _aware(user.reset_token_expires):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Reset link is invalid or has expired.")

    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_token_expires = None
    user.failed_login_attempts = 0
    user.locked_until = None
    user.force_password_change = False
    db.add(user)
    db.commit()

    revoke_all_sessions(db, user)  # kills every other live session too
    db.refresh(user)
    return user


def change_password(db: Session, user_id: int, current_password: str, new_password: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if not verify_password(current_password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Current password is incorrect.")

    user.password_hash = hash_password(new_password)
    user.force_password_change = False
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
