from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from control_plane_backend.models.base import Base, utcnow


class ScheduledAutomationDelegationAuditRow(Base):
    """Durable audit row for AI Wiki scheduled-automation delegation changes."""

    __tablename__ = "scheduled_automation_delegation_audit"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    human_admin_subject: Mapped[str] = mapped_column(String, nullable=False, index=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    service_client_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    service_subject: Mapped[str] = mapped_column(String, nullable=False, index=True)
    team_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String, nullable=False, index=True)
    result: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True, default=utcnow)
