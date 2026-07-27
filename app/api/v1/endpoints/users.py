"""User management endpoints - Phase 7.

Implements (paths/shapes match vision-ai-frontend/src/api/users.ts exactly):
  GET    /api/users                 List (plain array, excludes soft-deleted)
  GET    /api/users/{user_id}       Detail (not in the frontend mock, added
                                     for API completeness)
  POST   /api/users                 Create; sends onboarding email to the new
                                     user + a confirmation to active admins
  PUT    /api/users/{user_id}       Edit full_name, email, phone, role_id
  PATCH  /api/users/{user_id}/status   {status: "active"|"inactive"} - NOT
                                     "/disable" as an earlier stub assumed
  DELETE /api/users/{user_id}       Soft-delete (hidden from lists, history
                                     retained, status="deleted")
  POST   /api/users/{user_id}/reset-password   {new_password}

Disabling, deleting, or resetting a user's password all immediately kill
their live sessions via auth_service.revoke_all_sessions() - not just future
logins (see app/services/user_service.py).

Business logic lives in app.services.user_service.
"""
from fastapi import APIRouter, Depends, status

from app.core.dependencies import DbSession, require_permission
from app.schemas.user import (
    UserCreateInput, UserPasswordReset, UserResponse, UserStatusUpdate, UserUpdateInput,
)
from app.services import user_service

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=list[UserResponse])
def list_users(
    db: DbSession,
    user: dict = Depends(require_permission("users", "view")),
) -> list[UserResponse]:
    return user_service.list_users(db, user.get("org_id"))


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("users", "view")),
) -> UserResponse:
    return user_service.get_user(db, user.get("org_id"), user_id)


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreateInput,
    db: DbSession,
    user: dict = Depends(require_permission("users", "create")),
) -> UserResponse:
    return user_service.create_user(db, user, payload)


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    payload: UserUpdateInput,
    db: DbSession,
    user: dict = Depends(require_permission("users", "edit")),
) -> UserResponse:
    return user_service.update_user(db, user, user_id, payload)


@router.patch("/{user_id}/status", response_model=UserResponse)
def set_user_status(
    user_id: int,
    payload: UserStatusUpdate,
    db: DbSession,
    user: dict = Depends(require_permission("users", "edit")),
) -> UserResponse:
    return user_service.set_status(db, user, user_id, payload.status)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("users", "delete")),
) -> None:
    user_service.soft_delete_user(db, user, user_id)


@router.post("/{user_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
def reset_user_password(
    user_id: int,
    payload: UserPasswordReset,
    db: DbSession,
    user: dict = Depends(require_permission("users", "edit")),
) -> None:
    user_service.reset_password(db, user, user_id, payload.new_password)
