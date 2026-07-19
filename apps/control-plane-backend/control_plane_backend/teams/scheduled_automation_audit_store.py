from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from fred_core.sql import make_session_factory, use_session
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from control_plane_backend.models.scheduled_automation_audit_models import ScheduledAutomationDelegationAuditRow


@dataclass(frozen=True)
class ScheduledAutomationDelegationAuditRecord:
    human_admin_subject: str
    action: str
    service_client_id: str
    service_subject: str
    team_id: str
    relation: str
    result: str


class ScheduledAutomationDelegationAuditStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = make_session_factory(engine)

    async def append(
        self,
        record: ScheduledAutomationDelegationAuditRecord,
        *,
        session: AsyncSession | None = None,
    ) -> str:
        async with use_session(self._sessions, session) as s:
            row = ScheduledAutomationDelegationAuditRow(
                id=str(uuid4()),
                human_admin_subject=record.human_admin_subject,
                action=record.action,
                service_client_id=record.service_client_id,
                service_subject=record.service_subject,
                team_id=record.team_id,
                relation=record.relation,
                result=record.result,
            )
            s.add(row)
            return row.id
