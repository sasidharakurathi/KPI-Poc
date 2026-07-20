"""Role & permissions endpoints — Phase 6.

Implements (paths/shapes match vision-ai-frontend/src/api/roles.ts exactly —
plain arrays, not {count, roles} wrappers):
  GET    /api/roles                List (plain array)
  GET    /api/roles/user-counts    {role_id: active-user-count}, registered
                                    before /{role_id} so it isn't shadowed
  GET    /api/roles/{role_id}      Detail (not in the frontend mock, added
                                    for API completeness)
  POST   /api/roles                Create (409 on duplicate name, case-insensitive)
  PUT    /api/roles/{role_id}      Update (403 if is_system)
  DELETE /api/roles/{role_id}      Delete (403 if is_system, 409 if referenced
                                    by any non-soft-deleted user)

The Owner role (is_system=True) can never be edited or deleted.
Business logic lives in app.services.role_service.
"""
from fastapi import APIRouter, Depends, status

from app.core.dependencies import DbSession, require_permission
from app.schemas.role import RoleInput, RoleResponse
from app.services import role_service

router = APIRouter(prefix="/api/roles", tags=["roles"])


@router.get("", response_model=list[RoleResponse])
def list_roles(
    db: DbSession,
    user: dict = Depends(require_permission("roles", "view")),
) -> list[RoleResponse]:
    return role_service.list_roles(db, user.get("org_id"))


@router.get("/user-counts", response_model=dict[str, int])
def role_user_counts(
    db: DbSession,
    user: dict = Depends(require_permission("roles", "view")),
) -> dict[str, int]:
    return role_service.user_counts(db, user.get("org_id"))


@router.get("/{role_id}", response_model=RoleResponse)
def get_role(
    role_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("roles", "view")),
) -> RoleResponse:
    return role_service.get_role(db, user.get("org_id"), role_id)


@router.post("", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
def create_role(
    payload: RoleInput,
    db: DbSession,
    user: dict = Depends(require_permission("roles", "create")),
) -> RoleResponse:
    return role_service.create_role(db, user, payload)


@router.put("/{role_id}", response_model=RoleResponse)
def update_role(
    role_id: int,
    payload: RoleInput,
    db: DbSession,
    user: dict = Depends(require_permission("roles", "edit")),
) -> RoleResponse:
    return role_service.update_role(db, user, role_id, payload)


@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_role(
    role_id: int,
    db: DbSession,
    user: dict = Depends(require_permission("roles", "delete")),
) -> None:
    role_service.delete_role(db, user, role_id)
