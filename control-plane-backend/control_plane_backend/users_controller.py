from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Path, status
from fastapi.responses import JSONResponse
from fred_core import (
    BaseUserStore,
    GcuVersionsType,
    KeycloakUser,
    TeamPermission,
    get_current_user,
    get_current_user_without_gcu,
)
from fred_core.common import PERSONAL_TEAM_ID
from fred_core.users.store.postgres_user_store import get_user_store
from pydantic import BaseModel

from control_plane_backend.teams_structures import (
    TeamWithPermissions,
)
from control_plane_backend.users_service import (
    create_user as create_user_from_service,
)
from control_plane_backend.users_service import (
    delete_user as delete_user_from_service,
)
from control_plane_backend.users_service import (
    find_user_details_by_id,
    update_gcu_validation,
)
from control_plane_backend.users_service import (
    list_users as list_users_from_service,
)
from control_plane_backend.users_structures import (
    CreateUserRequest,
    KeycloakM2MUserOperationDisabledError,
    UserAlreadyExistsError,
    UserNotFoundError,
    UserSummary,
)

router = APIRouter(tags=["Users"])


def _parse_persisted_user_id(user: KeycloakUser) -> UUID | None:
    """Return a persisted user UUID when the authenticated principal has one.

    Why this exists:
    Control Plane supports a local development mode where authentication is
    disabled and fred-core returns a mock admin user whose uid is not a UUID.
    Endpoints that read or update SQL-backed user details should treat that
    dev-only identity as "no persisted profile" instead of raising a 500.

    How to use it:
    Call this helper before reading or writing user records that are keyed by a
    database UUID. When it returns ``None``, skip the persistence lookup/update
    and continue with the dev-mode fallback response.

    Example:
        user_id = _parse_persisted_user_id(user)
        if user_id is None:
            return None
    """
    try:
        return UUID(user.uid)
    except ValueError:
        return None


def register_exception_handlers(app: FastAPI) -> None:
    """Register user-domain exception handlers."""

    @app.exception_handler(KeycloakM2MUserOperationDisabledError)
    async def keycloak_disabled_for_users_handler(
        _request,
        exc: KeycloakM2MUserOperationDisabledError,
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(UserAlreadyExistsError)
    async def user_already_exists_handler(
        _request,
        exc: UserAlreadyExistsError,
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(UserNotFoundError)
    async def user_not_found_handler(_request, exc: UserNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})


@router.get(
    "/users",
    response_model=list[UserSummary],
    response_model_exclude_none=True,
    summary="List users registered in Keycloak.",
)
async def list_users(
    user: KeycloakUser = Depends(get_current_user),
) -> list[UserSummary]:
    return await list_users_from_service(user)


@router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    response_model=UserSummary,
    response_model_exclude_none=True,
    summary="Temporary bootstrap endpoint to create a user.",
)
async def create_user(
    request: CreateUserRequest,
    user: KeycloakUser = Depends(get_current_user),
) -> UserSummary:
    """Create a user in Keycloak for temporary bootstrap and testing flows."""
    return await create_user_from_service(user, request)


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Temporary bootstrap endpoint to delete a user.",
)
async def delete_user(
    user_id: Annotated[str, Path(min_length=1)],
    user: KeycloakUser = Depends(get_current_user),
) -> None:
    """Delete a user in Keycloak for temporary bootstrap and testing flows."""
    await delete_user_from_service(user, user_id)


class UserDetails(BaseModel):
    cguValidated: GcuVersionsType | None
    personalTeam: TeamWithPermissions


@router.get(
    "/user",
    summary="Return user informations.",
)
async def get_user_details(
    user: KeycloakUser = Depends(get_current_user_without_gcu),
    user_store: BaseUserStore = Depends(get_user_store),
) -> UserDetails:
    """Return the current user profile without failing in auth-disabled dev mode.

    Why this exists:
    The frontend needs a single endpoint that returns GCU acceptance state plus
    the personal team scaffold used by the Control Plane UI.

    How to use it:
    Call ``GET /control-plane/v1/user`` with the current authenticated user. In
    local auth-disabled mode, the endpoint still responds successfully even when
    there is no persisted SQL user row for the mock admin identity.
    """
    persisted_user_id = _parse_persisted_user_id(user)
    user_details = (
        await find_user_details_by_id(persisted_user_id, user_store)
        if persisted_user_id is not None
        else None
    )

    return UserDetails(
        cguValidated=user_details.gcuVersionAccepted if user_details else None,
        personalTeam=TeamWithPermissions(
            id=PERSONAL_TEAM_ID,
            name="Equipe personnelle",
            member_count=1,
            is_private=True,
            owners=[],
            permissions=[
                TeamPermission("can_read"),
                TeamPermission("can_update_resources"),
                TeamPermission("can_update_agents"),
            ],
        ),
    )


@router.post("/gcu")
async def validate_gcu(
    user: KeycloakUser = Depends(get_current_user_without_gcu),
    user_store: BaseUserStore = Depends(get_user_store),
) -> None:
    """Persist GCU acceptance for real users and no-op for dev mock identities.

    Why this exists:
    The UI needs a write endpoint to record the configured GCU version for the
    current user when a persisted SQL-backed user identity exists.

    How to use it:
    Call ``POST /control-plane/v1/gcu`` for an authenticated user. In local
    auth-disabled mode, the request is accepted and skipped because the mock
    admin identity does not map to a persisted UUID row.
    """
    persisted_user_id = _parse_persisted_user_id(user)
    if persisted_user_id is None:
        return

    await update_gcu_validation(persisted_user_id, user_store)
