"""Business logic for Phase 7: User Management.

Kept out of app/api/v1/endpoints/users.py so the router stays a thin
request/response layer, per app/services/camera_service.py's precedent.
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.security import hash_password
from app.db.models import Role, User
from app.schemas.user import UserCreateInput, UserResponse, UserUpdateInput
from app.services.auth_service import revoke_all_sessions
from app.services.role_service import _to_int_id

logger = logging.getLogger(__name__)

# Backend status -> frontend status (AppUser.status: 'active'|'inactive'|'deleted').
# "pending_verification" only ever applies to the Phase 0 org-owner sign-up
# flow, not Phase-7-created users, but is mapped defensively since the Owner
# themselves shows up in this same user list.
_STATUS_TO_FRONTEND = {
    "active": "active",
    "disabled": "inactive",
    "soft_deleted": "deleted",
    "pending_verification": "inactive",
}
_FRONTEND_TO_BACKEND_STATUS = {"active": "active", "inactive": "disabled"}


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        full_name=user.full_name,
        username=user.username,
        email=user.login_email,
        phone=user.phone or "",
        role_id=str(user.role_id) if user.role_id is not None else None,
        status=_STATUS_TO_FRONTEND.get(user.status, "inactive"),
        must_change_password=user.force_password_change,
        mfa_enabled=user.mfa_enabled,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
        created_at=user.created_at.isoformat(),
    )


def list_users(db: Session, org_id: Optional[int]) -> list[UserResponse]:
    users = db.exec(
        select(User).where(User.org_id == org_id, User.status != "soft_deleted")
    ).all()
    return [_to_user_response(u) for u in users]


def get_user(db: Session, org_id: Optional[int], user_id: int) -> UserResponse:
    user = db.get(User, user_id)
    if user is None or user.org_id != org_id or user.status == "soft_deleted":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    return _to_user_response(user)


def _active_admin_count(db: Session, org_id: Optional[int], excluding_user_id: Optional[int] = None) -> int:
    admin_role = db.exec(select(Role).where(Role.org_id == org_id, Role.is_system.is_(True))).first()
    if admin_role is None:
        return 0
    users = db.exec(
        select(User).where(
            User.org_id == org_id, User.role_id == admin_role.id, User.status == "active",
        )
    ).all()
    return len([u for u in users if u.id != excluding_user_id])


def _get_role_or_422(db: Session, org_id: Optional[int], role_id_raw: str) -> Role:
    role_id = _to_int_id(role_id_raw, "role_id")
    role = db.get(Role, role_id) if role_id is not None else None
    if role is None or role.org_id != org_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Select a valid role.")
    return role


def _check_email_unique(
    db: Session, org_id: Optional[int], email: str, excluding_user_id: Optional[int] = None
) -> None:
    """Scoped to org_id: emails only need to be unique within an organization,
    not globally. Note this is currently only enforced at the application
    level — User.personal_email/login_email still carry a bare (non-composite)
    unique=True DB constraint, which is harmless while "one org per
    deployment" holds but would need a real (org_id, email) composite
    constraint if multi-tenancy is ever turned on."""
    clash = db.exec(
        select(User).where(
            User.org_id == org_id,
            (User.personal_email == email) | (User.login_email == email),
        )
    ).all()
    if any(u.id != excluding_user_id for u in clash):
        raise HTTPException(status.HTTP_409_CONFLICT, f'Email "{email}" is already in use.')


def create_user(db: Session, org_id: Optional[int], payload: UserCreateInput) -> UserResponse:
    username = payload.username.strip()
    if db.exec(
        select(User).where(User.org_id == org_id, User.username.ilike(username))
    ).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f'Username "{username}" is already taken.')
    _check_email_unique(db, org_id, payload.email)
    role = _get_role_or_422(db, org_id, payload.role_id)

    user = User(
        full_name=payload.full_name.strip(),
        personal_email=payload.email,
        login_email=payload.email,
        phone=payload.phone,
        username=username,
        password_hash=hash_password(payload.password),
        org_id=org_id,
        role_id=role.id,
        status="active",
        force_password_change=True,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Backstop for the TOCTOU race in the username/email-uniqueness
        # checks above — two concurrent creates can both pass them.
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f'Username "{username}" or email "{payload.email}" is already in use.',
        )
    db.refresh(user)

    _send_onboarding_emails(db, org_id, user, payload.password)
    return _to_user_response(user)


def _send_onboarding_emails(db: Session, org_id: Optional[int], user: User, temp_password: str) -> None:
    from app.notifications import send_transactional_email

    subject = "Your Vision AI account is ready"
    plain = (
        f"Hi {user.full_name},\n\n"
        f"An account has been created for you on the Vision AI Safety & "
        f"Compliance Platform.\n\n"
        f"Username: {user.username}\nTemporary password: {temp_password}\n\n"
        f"You'll be asked to set a new password the first time you sign in.\n"
    )
    html = (
        f"<p>Hi {user.full_name},</p>"
        f"<p>An account has been created for you on the Vision AI Safety &amp; "
        f"Compliance Platform.</p>"
        f"<p>Username: <b>{user.username}</b><br>Temporary password: <b>{temp_password}</b></p>"
        f"<p>You'll be asked to set a new password the first time you sign in.</p>"
    )
    try:
        send_transactional_email([user.login_email], subject, html, plain)
    except Exception:
        logger.warning("[users] could not send onboarding email to %s", user.login_email)

    admins = db.exec(
        select(User).join(Role, User.role_id == Role.id).where(
            Role.org_id == org_id, Role.is_system.is_(True), User.status == "active", User.id != user.id,
        )
    ).all()
    if not admins:
        return
    admin_subject = f'New user onboarded: "{user.full_name}"'
    admin_plain = (
        f'A new user has been onboarded to your organization:\n\n'
        f"Name: {user.full_name}\nUsername: {user.username}\nEmail: {user.login_email}\n"
    )
    admin_html = (
        f"<p>A new user has been onboarded to your organization:</p>"
        f"<p>Name: {user.full_name}<br>Username: {user.username}<br>Email: {user.login_email}</p>"
    )
    try:
        send_transactional_email([a.login_email for a in admins], admin_subject, admin_html, admin_plain)
    except Exception:
        logger.warning("[users] could not send onboarding confirmation to admins")


def update_user(db: Session, org_id: Optional[int], user_id: int, payload: UserUpdateInput) -> UserResponse:
    user = db.get(User, user_id)
    if user is None or user.org_id != org_id or user.status == "soft_deleted":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    _check_email_unique(db, org_id, payload.email, excluding_user_id=user.id)
    new_role = _get_role_or_422(db, org_id, payload.role_id)

    was_admin = user.role_id is not None and db.get(Role, user.role_id) is not None and db.get(Role, user.role_id).is_system
    if was_admin and not new_role.is_system and _active_admin_count(db, org_id, excluding_user_id=user.id) == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "At least one active Administrator must remain — assign another admin before changing this role.",
        )

    user.full_name = payload.full_name.strip()
    user.personal_email = payload.email
    user.login_email = payload.email
    user.phone = payload.phone
    user.role_id = new_role.id
    user.updated_at = datetime.utcnow()
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, f'Email "{payload.email}" is already in use.')
    db.refresh(user)
    return _to_user_response(user)


def set_status(db: Session, org_id: Optional[int], user_id: int, frontend_status: str) -> UserResponse:
    user = db.get(User, user_id)
    if user is None or user.org_id != org_id or user.status == "soft_deleted":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    backend_status = _FRONTEND_TO_BACKEND_STATUS[frontend_status]
    role = db.get(Role, user.role_id) if user.role_id else None
    if backend_status == "disabled" and role is not None and role.is_system:
        if _active_admin_count(db, org_id, excluding_user_id=user.id) == 0:
            raise HTTPException(status.HTTP_409_CONFLICT, "At least one active Administrator must remain.")

    user.status = backend_status
    db.add(user)
    db.commit()
    db.refresh(user)

    if backend_status == "disabled":
        # PRD: "Disable: blocks login immediately and revokes any active session."
        revoke_all_sessions(db, user)
    return _to_user_response(user)


def soft_delete_user(db: Session, org_id: Optional[int], user_id: int) -> None:
    user = db.get(User, user_id)
    if user is None or user.org_id != org_id or user.status == "soft_deleted":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    role = db.get(Role, user.role_id) if user.role_id else None
    if role is not None and role.is_system and _active_admin_count(db, org_id, excluding_user_id=user.id) == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "At least one active Administrator must remain — cannot delete the last admin.",
        )

    user.status = "soft_deleted"
    db.add(user)
    db.commit()
    revoke_all_sessions(db, user)  # PRD: "permanently loses access"


def reset_password(db: Session, org_id: Optional[int], user_id: int, new_password: str) -> None:
    user = db.get(User, user_id)
    if user is None or user.org_id != org_id or user.status == "soft_deleted":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    user.password_hash = hash_password(new_password)
    user.force_password_change = True
    db.add(user)
    db.commit()
    revoke_all_sessions(db, user)  # a password reset should kill any live sessions too
