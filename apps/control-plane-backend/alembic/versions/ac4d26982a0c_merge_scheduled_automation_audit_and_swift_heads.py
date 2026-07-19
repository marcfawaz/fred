"""merge scheduled automation audit and swift heads

Revision ID: ac4d26982a0c
Revises: f5a6b7c8d9e0, f824bb94e60d
Create Date: 2026-07-19 16:06:19.909271

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "ac4d26982a0c"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = ("f5a6b7c8d9e0", "f824bb94e60d")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
