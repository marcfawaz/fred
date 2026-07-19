"""add scheduled automation delegation audit

Revision ID: f5a6b7c8d9e0
Revises: 6e4149d46705
Create Date: 2026-07-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f5a6b7c8d9e0"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "6e4149d46705"  # pragma: allowlist secret
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scheduled_automation_delegation_audit",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("human_admin_subject", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("service_client_id", sa.String(), nullable=False),
        sa.Column("service_subject", sa.String(), nullable=False),
        sa.Column("team_id", sa.String(), nullable=False),
        sa.Column("relation", sa.String(), nullable=False),
        sa.Column("result", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scheduled_automation_delegation_audit_human_admin_subject", "scheduled_automation_delegation_audit", ["human_admin_subject"])
    op.create_index("ix_scheduled_automation_delegation_audit_action", "scheduled_automation_delegation_audit", ["action"])
    op.create_index("ix_scheduled_automation_delegation_audit_service_client_id", "scheduled_automation_delegation_audit", ["service_client_id"])
    op.create_index("ix_scheduled_automation_delegation_audit_service_subject", "scheduled_automation_delegation_audit", ["service_subject"])
    op.create_index("ix_scheduled_automation_delegation_audit_team_id", "scheduled_automation_delegation_audit", ["team_id"])
    op.create_index("ix_scheduled_automation_delegation_audit_relation", "scheduled_automation_delegation_audit", ["relation"])
    op.create_index("ix_scheduled_automation_delegation_audit_created_at", "scheduled_automation_delegation_audit", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_scheduled_automation_delegation_audit_created_at", table_name="scheduled_automation_delegation_audit")
    op.drop_index("ix_scheduled_automation_delegation_audit_relation", table_name="scheduled_automation_delegation_audit")
    op.drop_index("ix_scheduled_automation_delegation_audit_team_id", table_name="scheduled_automation_delegation_audit")
    op.drop_index("ix_scheduled_automation_delegation_audit_service_subject", table_name="scheduled_automation_delegation_audit")
    op.drop_index("ix_scheduled_automation_delegation_audit_service_client_id", table_name="scheduled_automation_delegation_audit")
    op.drop_index("ix_scheduled_automation_delegation_audit_action", table_name="scheduled_automation_delegation_audit")
    op.drop_index("ix_scheduled_automation_delegation_audit_human_admin_subject", table_name="scheduled_automation_delegation_audit")
    op.drop_table("scheduled_automation_delegation_audit")
