"""Widen odds_snapshots.market column and add description for player props.

Revision ID: 004_player_props
Revises: 003_odds_sourced
Create Date: 2025-06-25
"""

import sqlalchemy as sa
from alembic import op

revision = "004_player_props"
down_revision = "003_odds_sourced"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    cols = [c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)]
    return column in cols


def upgrade() -> None:
    # Widen market column to accommodate player prop market keys
    # e.g. "player_points_rebounds_assists_alternate" (41 chars)
    op.alter_column(
        "odds_snapshots",
        "market",
        type_=sa.String(60),
        existing_type=sa.String(30),
        existing_nullable=False,
    )

    # Add description column for player name on prop bets
    if not _has_column("odds_snapshots", "description"):
        op.add_column(
            "odds_snapshots",
            sa.Column("description", sa.String(120), nullable=True),
        )


def downgrade() -> None:
    if _has_column("odds_snapshots", "description"):
        op.drop_column("odds_snapshots", "description")

    op.alter_column(
        "odds_snapshots",
        "market",
        type_=sa.String(30),
        existing_type=sa.String(60),
        existing_nullable=False,
    )
