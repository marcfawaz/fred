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

"""Guard AUTHZ-WIKI-01 in Fred's compiled canonical OpenFGA model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from fred_core.security.rebac.openfga_schema import DEFAULT_SCHEMA
from fred_core.security.rebac.rebac_engine import TeamPermission


AI_WIKI_CAPABILITY_RELATIONS = {
    "can_read_wikis": "team_member",
    "can_contribute_wikis": "team_editor",
    "can_review_wiki_changes": "team_admin",
    "can_manage_wiki_schema": "team_admin",
    "can_manage_wiki_lifecycle": "team_admin",
    "can_manage_wiki_governance": "team_admin",
    "can_use_wiki_review_assistant": "team_admin",
    "can_run_wiki_guarded_auto_apply": "team_admin",
    "can_run_wiki_autonomous_apply": "team_admin",
}

AI_WIKI_TEAM_PERMISSION_VALUES = {
    TeamPermission.CAN_READ_WIKIS: "can_read_wikis",
    TeamPermission.CAN_CONTRIBUTE_WIKIS: "can_contribute_wikis",
    TeamPermission.CAN_REVIEW_WIKI_CHANGES: "can_review_wiki_changes",
    TeamPermission.CAN_MANAGE_WIKI_SCHEMA: "can_manage_wiki_schema",
    TeamPermission.CAN_MANAGE_WIKI_LIFECYCLE: "can_manage_wiki_lifecycle",
    TeamPermission.CAN_MANAGE_WIKI_GOVERNANCE: "can_manage_wiki_governance",
    TeamPermission.CAN_USE_WIKI_REVIEW_ASSISTANT: "can_use_wiki_review_assistant",
    TeamPermission.CAN_RUN_WIKI_GUARDED_AUTO_APPLY: "can_run_wiki_guarded_auto_apply",
    TeamPermission.CAN_RUN_WIKI_AUTONOMOUS_APPLY: "can_run_wiki_autonomous_apply",
}

FORBIDDEN_AI_WIKI_RELATION_REFERENCES = {
    "public",
    "organization",
    "platform_admin",
    "platform_observer",
    "can_read",
}


def _type_definition(type_name: str) -> dict[str, Any]:
    model = json.loads(DEFAULT_SCHEMA)
    return next(t for t in model["type_definitions"] if t["type"] == type_name)


def _relation_references(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        if "relation" in node and isinstance(node["relation"], str):
            yield node["relation"]
        for value in node.values():
            yield from _relation_references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _relation_references(value)


def test_ai_wiki_capabilities_exist_with_exact_team_role_roots() -> None:
    team = _type_definition("team")

    for capability, expected_relation in AI_WIKI_CAPABILITY_RELATIONS.items():
        assert team["relations"][capability] == {
            "computedUserset": {"relation": expected_relation}
        }


def test_ai_wiki_capabilities_do_not_use_public_or_platform_relations() -> None:
    team = _type_definition("team")

    for capability in AI_WIKI_CAPABILITY_RELATIONS:
        references = set(_relation_references(team["relations"][capability]))
        assert references.isdisjoint(FORBIDDEN_AI_WIKI_RELATION_REFERENCES), (
            f"{capability} must stay team-role-only; found forbidden "
            f"references: {references & FORBIDDEN_AI_WIKI_RELATION_REFERENCES}"
        )


def test_ai_wiki_model_does_not_add_wiki_object_type() -> None:
    model = json.loads(DEFAULT_SCHEMA)

    assert "wiki" not in {t["type"] for t in model["type_definitions"]}


def test_ai_wiki_additions_preserve_existing_team_role_relations() -> None:
    team = _type_definition("team")

    assert team["relations"]["team_admin"] == {"this": {}}
    assert team["relations"]["team_editor"] == {"this": {}}
    assert team["relations"]["team_analyst"] == {"this": {}}
    assert team["relations"]["team_member"] == {
        "union": {
            "child": [
                {"this": {}},
                {"computedUserset": {"relation": "team_admin"}},
                {"computedUserset": {"relation": "team_editor"}},
                {"computedUserset": {"relation": "team_analyst"}},
            ]
        }
    }
    assert team["relations"]["can_read"] == {
        "union": {
            "child": [
                {"computedUserset": {"relation": "team_member"}},
                {"computedUserset": {"relation": "public"}},
            ]
        }
    }


def test_team_permission_enum_contains_exact_ai_wiki_values() -> None:
    assert {permission.value for permission in AI_WIKI_TEAM_PERMISSION_VALUES} == set(
        AI_WIKI_CAPABILITY_RELATIONS
    )
    for permission, expected_value in AI_WIKI_TEAM_PERMISSION_VALUES.items():
        assert permission.value == expected_value
