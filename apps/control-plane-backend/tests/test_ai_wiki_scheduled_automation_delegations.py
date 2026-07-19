from __future__ import annotations

from types import SimpleNamespace

import pytest
from control_plane_backend.teams.dependencies import TeamServiceDependencies
from control_plane_backend.teams.schemas import (
    ScheduledAutomationDelegationRelation,
    ScheduledAutomationDelegationRequest,
    TeamNotFoundError,
)
from control_plane_backend.teams.service import (
    assign_scheduled_automation_delegation,
    list_scheduled_automation_delegations,
    revoke_scheduled_automation_delegation,
)
from fred_core import (
    KeycloakUser,
    RebacReference,
    Relation,
    RelationType,
    Resource,
    TeamPermission,
)
from fred_core.common import TeamId
from fred_core.scheduler import SchedulerBackend
from fred_core.teams.metadata_store import TeamMetadata
from pydantic import ValidationError


SERVICE_AGENT_ROLE = "service_agent"


class _FakeRebac:
    def __init__(self) -> None:
        self.relations: set[tuple[str, str, str]] = set()
        self.permission_checks: list[tuple[str, tuple[TeamPermission, ...]]] = []

    async def check_user_team_permissions_or_raise(self, *, user, team_id, permissions):
        self.permission_checks.append((str(team_id), tuple(permissions)))
        return "token"

    async def add_relation(self, relation: Relation) -> None:
        self.relations.add(
            (str(relation.resource.id), relation.relation.value, relation.subject.id)
        )

    async def delete_relations(self, relations: list[Relation]) -> None:
        for relation in relations:
            self.relations.discard(
                (
                    str(relation.resource.id),
                    relation.relation.value,
                    relation.subject.id,
                )
            )

    async def lookup_subjects(
        self, resource, relation: RelationType, subject_type, **kwargs
    ):
        return [
            RebacReference(Resource.USER, subject)
            for team_id, rel, subject in sorted(self.relations)
            if team_id == str(resource.id) and rel == relation.value
        ]


class _FakeMetadataStore:
    async def get_by_team_id(self, team_id, session=None):
        if str(team_id) == "fredlab":
            return TeamMetadata(id=TeamId("fredlab"), name="Fredlab")
        if str(team_id) == "other-team":
            return TeamMetadata(id=TeamId("other-team"), name="Other")
        return None


def _deps(
    rebac: _FakeRebac,
    *,
    service_subject: str | None = "service-account-sub",
    audit_records: list | None = None,
) -> TeamServiceDependencies:
    metadata = _FakeMetadataStore()

    async def _resolve_service_subject(_client_id: str) -> str | None:
        return service_subject

    async def _append_audit(record) -> None:
        if audit_records is not None:
            audit_records.append(record)

    return TeamServiceDependencies(
        configuration=SimpleNamespace(app=SimpleNamespace()),  # type: ignore[arg-type]
        rebac=rebac,  # type: ignore[arg-type]
        scheduler_backend=SchedulerBackend.MEMORY,
        get_team_metadata_store=lambda: metadata,  # type: ignore[return-value]
        get_content_store=lambda: None,  # type: ignore[return-value]
        get_session_store=lambda: None,  # type: ignore[return-value]
        get_purge_queue_store=lambda: None,  # type: ignore[return-value]
        get_policy_catalog=lambda: None,  # type: ignore[return-value]
        get_users_by_ids=lambda _ids: {},  # type: ignore[arg-type]
        run_lifecycle_manager_once_in_memory=lambda _input: None,  # type: ignore[arg-type]
        resolve_service_account_subject=_resolve_service_subject,
        append_scheduled_automation_audit=_append_audit,
    )


def _admin() -> KeycloakUser:
    return KeycloakUser(uid="human-admin-sub", username="admin", email=None, roles=[])


def _service_agent() -> KeycloakUser:
    return KeycloakUser(uid="service-agent-sub", username="service-account-fred-ai-wiki-worker", email=None, roles=[SERVICE_AGENT_ROLE])


@pytest.mark.asyncio
async def test_assign_list_and_revoke_scheduled_automation_delegation_is_idempotent() -> (
    None
):
    rebac = _FakeRebac()
    audit_records = []
    request = ScheduledAutomationDelegationRequest(
        service_client_id="fred-ai-wiki-worker",
        relation=ScheduledAutomationDelegationRelation.WIKI_REVIEW_ASSISTANT_RUNNER,
    )

    await assign_scheduled_automation_delegation(
        _admin(),
        TeamId("fredlab"),
        request,
        _deps(rebac, audit_records=audit_records),
    )
    await assign_scheduled_automation_delegation(
        _admin(),
        TeamId("fredlab"),
        request,
        _deps(rebac, audit_records=audit_records),
    )

    listed = await list_scheduled_automation_delegations(
        _admin(), TeamId("fredlab"), _deps(rebac)
    )
    assert [(item.service_subject, item.relation) for item in listed] == [
        (
            "service-account-sub",
            ScheduledAutomationDelegationRelation.WIKI_REVIEW_ASSISTANT_RUNNER,
        )
    ]
    assert listed[0].service_client_id == "fred-ai-wiki-worker"
    assert rebac.permission_checks[-1] == (
        "fredlab",
        (TeamPermission.CAN_ADMINISTER_ADMINS,),
    )

    await revoke_scheduled_automation_delegation(
        _admin(),
        TeamId("fredlab"),
        request,
        _deps(rebac, audit_records=audit_records),
    )
    await revoke_scheduled_automation_delegation(
        _admin(),
        TeamId("fredlab"),
        request,
        _deps(rebac, audit_records=audit_records),
    )

    assert (
        await list_scheduled_automation_delegations(
            _admin(), TeamId("fredlab"), _deps(rebac)
        )
        == []
    )
    assert [record.action for record in audit_records] == ["assign", "assign", "revoke", "revoke"]
    assert {record.human_admin_subject for record in audit_records} == {"human-admin-sub"}
    assert {record.service_client_id for record in audit_records} == {"fred-ai-wiki-worker"}
    assert {record.service_subject for record in audit_records} == {"service-account-sub"}
    assert {record.team_id for record in audit_records} == {"fredlab"}
    assert {record.result for record in audit_records} == {"succeeded"}


@pytest.mark.asyncio
async def test_scheduled_automation_delegation_is_team_scoped() -> None:
    rebac = _FakeRebac()
    request = ScheduledAutomationDelegationRequest(
        service_client_id="fred-ai-wiki-worker",
        relation=ScheduledAutomationDelegationRelation.WIKI_GUARDED_AUTO_APPLY_RUNNER,
    )

    await assign_scheduled_automation_delegation(
        _admin(), TeamId("fredlab"), request, _deps(rebac)
    )

    assert (
        await list_scheduled_automation_delegations(
            _admin(), TeamId("other-team"), _deps(rebac)
        )
        == []
    )


@pytest.mark.asyncio
async def test_scheduled_automation_delegation_requires_existing_team() -> None:
    request = ScheduledAutomationDelegationRequest(
        service_client_id="fred-ai-wiki-worker",
        relation=ScheduledAutomationDelegationRelation.WIKI_AUTONOMOUS_APPLY_RUNNER,
    )

    with pytest.raises(TeamNotFoundError):
        await assign_scheduled_automation_delegation(
            _admin(), TeamId("missing"), request, _deps(_FakeRebac())
        )


def test_scheduled_automation_delegation_rejects_unapproved_service_clients() -> None:
    with pytest.raises(ValidationError):
        ScheduledAutomationDelegationRequest(
            service_client_id="agentic",
            relation=ScheduledAutomationDelegationRelation.WIKI_REVIEW_ASSISTANT_RUNNER,
        )
    with pytest.raises(ValidationError):
        ScheduledAutomationDelegationRequest(
            service_client_id="human-user-sub",
            relation=ScheduledAutomationDelegationRelation.WIKI_REVIEW_ASSISTANT_RUNNER,
        )


@pytest.mark.asyncio
async def test_scheduled_automation_delegation_requires_resolved_service_account() -> (
    None
):
    request = ScheduledAutomationDelegationRequest(
        service_client_id="fred-ai-wiki-worker",
        relation=ScheduledAutomationDelegationRelation.WIKI_REVIEW_ASSISTANT_RUNNER,
    )

    with pytest.raises(ValueError):
        await assign_scheduled_automation_delegation(
            _admin(),
            TeamId("fredlab"),
            request,
            _deps(_FakeRebac(), service_subject=None),
        )


@pytest.mark.asyncio
async def test_scheduled_automation_service_identity_cannot_manage_delegations() -> None:
    request = ScheduledAutomationDelegationRequest(
        service_client_id="fred-ai-wiki-worker",
        relation=ScheduledAutomationDelegationRelation.WIKI_REVIEW_ASSISTANT_RUNNER,
    )

    with pytest.raises(PermissionError):
        await assign_scheduled_automation_delegation(
            _service_agent(), TeamId("fredlab"), request, _deps(_FakeRebac())
        )
    with pytest.raises(PermissionError):
        await revoke_scheduled_automation_delegation(
            _service_agent(), TeamId("fredlab"), request, _deps(_FakeRebac())
        )
