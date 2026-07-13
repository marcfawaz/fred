# Copyright Thales 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AUTHZ-05 item 8b: `_get_team_permissions_for_user` must depend only on
persisted OpenFGA tuples. `KeycloakUser` no longer carries a `groups` field at
all (AUTHZ-05 final sweep), so there is no longer a Keycloak-groups fallback
path to reach `has_permission` — these tests keep the persisted-tuple-only
proof (negative: no tuple, no permission; positive: tuple, permission)."""

from __future__ import annotations

from typing import cast

import pytest
from control_plane_backend.teams.service import _get_team_permissions_for_user
from fred_core import KeycloakUser, RebacEngine, TeamPermission
from fred_core.common import TeamId


AI_WIKI_PERMISSIONS = {
    TeamPermission.CAN_READ_WIKIS,
    TeamPermission.CAN_CONTRIBUTE_WIKIS,
    TeamPermission.CAN_REVIEW_WIKI_CHANGES,
    TeamPermission.CAN_MANAGE_WIKI_SCHEMA,
    TeamPermission.CAN_MANAGE_WIKI_LIFECYCLE,
    TeamPermission.CAN_MANAGE_WIKI_GOVERNANCE,
    TeamPermission.CAN_USE_WIKI_REVIEW_ASSISTANT,
    TeamPermission.CAN_RUN_WIKI_GUARDED_AUTO_APPLY,
    TeamPermission.CAN_RUN_WIKI_AUTONOMOUS_APPLY,
}


class _FakeRebac:
    """Grants permissions purely from an explicit in-memory set — mirrors a
    persisted-tuple-only OpenFGA engine. Records every `contextual_relations`
    it receives so tests can assert no contextual fallback ever reaches
    `has_permission`."""

    def __init__(self, granted: set[TeamPermission]) -> None:
        self.granted = granted
        self.received_contextual_relations: list[object] = []

    async def has_permission(
        self,
        subject,
        permission,
        resource,
        *,
        contextual_relations=None,
        consistency_token=None,
    ) -> bool:
        self.received_contextual_relations.append(contextual_relations)
        return permission in self.granted


def _user() -> KeycloakUser:
    return KeycloakUser(
        uid="alice",
        username="alice",
        roles=[],
        email="alice@example.com",
    )


@pytest.mark.asyncio
async def test_team_permissions_empty_without_persisted_tuple() -> None:
    """No persisted OpenFGA tuple grants no permission."""
    rebac = _FakeRebac(granted=set())
    user = _user()

    permissions = await _get_team_permissions_for_user(
        cast(RebacEngine, rebac), user, TeamId("team-x")
    )

    assert permissions == []
    assert all(cr is None for cr in rebac.received_contextual_relations)


@pytest.mark.asyncio
async def test_team_permissions_come_from_persisted_tuples() -> None:
    """Permissions reflect persisted tuples."""
    rebac = _FakeRebac(
        granted={TeamPermission.CAN_READ, TeamPermission.CAN_USE_TEAM_AGENTS}
    )
    user = _user()

    permissions = await _get_team_permissions_for_user(
        cast(RebacEngine, rebac), user, TeamId("team-x")
    )

    assert set(permissions) == {
        TeamPermission.CAN_READ,
        TeamPermission.CAN_USE_TEAM_AGENTS,
    }
    assert all(cr is None for cr in rebac.received_contextual_relations)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role_name", "granted", "expected_ai_wiki_permissions"),
    [
        (
            "team_member",
            {TeamPermission.CAN_READ_WIKIS},
            {TeamPermission.CAN_READ_WIKIS},
        ),
        (
            "team_analyst",
            {TeamPermission.CAN_READ_WIKIS},
            {TeamPermission.CAN_READ_WIKIS},
        ),
        (
            "team_editor",
            {
                TeamPermission.CAN_READ_WIKIS,
                TeamPermission.CAN_CONTRIBUTE_WIKIS,
            },
            {
                TeamPermission.CAN_READ_WIKIS,
                TeamPermission.CAN_CONTRIBUTE_WIKIS,
            },
        ),
        (
            "team_admin",
            {
                TeamPermission.CAN_READ_WIKIS,
                TeamPermission.CAN_REVIEW_WIKI_CHANGES,
                TeamPermission.CAN_MANAGE_WIKI_SCHEMA,
                TeamPermission.CAN_MANAGE_WIKI_LIFECYCLE,
                TeamPermission.CAN_MANAGE_WIKI_GOVERNANCE,
                TeamPermission.CAN_USE_WIKI_REVIEW_ASSISTANT,
                TeamPermission.CAN_RUN_WIKI_GUARDED_AUTO_APPLY,
                TeamPermission.CAN_RUN_WIKI_AUTONOMOUS_APPLY,
            },
            {
                TeamPermission.CAN_READ_WIKIS,
                TeamPermission.CAN_REVIEW_WIKI_CHANGES,
                TeamPermission.CAN_MANAGE_WIKI_SCHEMA,
                TeamPermission.CAN_MANAGE_WIKI_LIFECYCLE,
                TeamPermission.CAN_MANAGE_WIKI_GOVERNANCE,
                TeamPermission.CAN_USE_WIKI_REVIEW_ASSISTANT,
                TeamPermission.CAN_RUN_WIKI_GUARDED_AUTO_APPLY,
                TeamPermission.CAN_RUN_WIKI_AUTONOMOUS_APPLY,
            },
        ),
        (
            "team_admin_and_editor",
            AI_WIKI_PERMISSIONS,
            AI_WIKI_PERMISSIONS,
        ),
        (
            "public_only_visibility",
            {TeamPermission.CAN_READ},
            set(),
        ),
        (
            "platform_admin_without_team_relation",
            set(),
            set(),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_ai_wiki_team_permissions_are_exposed_from_rebac_only(
    role_name: str,
    granted: set[TeamPermission],
    expected_ai_wiki_permissions: set[TeamPermission],
) -> None:
    """TeamWithPermissions exposes the AI Wiki capabilities ReBAC grants.

    Role semantics are guarded in the compiled schema tests; this keeps the
    control-plane adapter honest: no public/platform fallback and no dropped
    new enum values when `_get_team_permissions_for_user` builds
    `TeamWithPermissions.permissions`.
    """
    rebac = _FakeRebac(granted=granted)
    user = _user()

    permissions = await _get_team_permissions_for_user(
        cast(RebacEngine, rebac), user, TeamId(f"{role_name}-team")
    )

    assert set(permissions) & AI_WIKI_PERMISSIONS == expected_ai_wiki_permissions
    assert all(cr is None for cr in rebac.received_contextual_relations)
