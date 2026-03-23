"""Add odds_sourced JSON column to predictions.

Revision ID: 003_odds_sourced
Revises: 002_model_registry
Create Date: 2026-03-23
"""

import sqlalchemy as sa
from alembic import op

revision = "003_odds_sourced"
down_revision = "002_model_registry"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    cols = [c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)]
    return column in cols


def upgrade() -> None:
    if not _has_column("predictions", "odds_sourced"):
        op.add_column(
            "predictions",
            sa.Column("odds_sourced", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if _has_column("predictions", "odds_sourced"):
        op.drop_column("predictions", "odds_sourced")
