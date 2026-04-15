"""Add permanent game_odds_archive table for training data.

Unlike odds_snapshots (pruned weekly), this table is never deleted.
Stores the first snapshot per (game, bookmaker, market, outcome, day),
giving every historical game real market feature values at train time.

Revision ID: 008_game_odds_archive
Revises: 007_ingestion_failures
Create Date: 2026-04-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008_game_odds_archive"
down_revision: str | None = "007_ingestion_failures"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "game_odds_archive",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=30), nullable=False),
        sa.Column("bookmaker", sa.String(length=60), nullable=False),
        sa.Column("market", sa.String(length=60), nullable=False),
        sa.Column("outcome_name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=120), nullable=True),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("point", sa.Float(), nullable=True),
        sa.Column("capture_date", sa.Date(), nullable=False),
        sa.Column("captured_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "game_id",
            "bookmaker",
            "market",
            "outcome_name",
            "capture_date",
            name="uq_game_odds_archive_daily",
        ),
    )
    op.create_index("ix_game_odds_archive_game_id", "game_odds_archive", ["game_id"])
    op.create_index("ix_game_odds_archive_capture_date", "game_odds_archive", ["capture_date"])
    op.create_index(
        "ix_game_odds_archive_game_date", "game_odds_archive", ["game_id", "capture_date"]
    )

    # Back-fill archive from existing odds_snapshots so historical data is
    # immediately available for the next training run.  One row per
    # (game, bookmaker, market, outcome, day) — the earliest captured_at wins.
    op.execute(
        """
        INSERT INTO game_odds_archive
            (game_id, source, bookmaker, market, outcome_name, description,
             price, point, capture_date, captured_at)
        SELECT DISTINCT ON (game_id, bookmaker, market, outcome_name,
                            DATE(captured_at))
            game_id, source, bookmaker, market, outcome_name, description,
            price, point, DATE(captured_at) AS capture_date, captured_at
        FROM odds_snapshots
        ORDER BY game_id, bookmaker, market, outcome_name,
                 DATE(captured_at), captured_at ASC
        ON CONFLICT DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("ix_game_odds_archive_game_date", table_name="game_odds_archive")
    op.drop_index("ix_game_odds_archive_capture_date", table_name="game_odds_archive")
    op.drop_index("ix_game_odds_archive_game_id", table_name="game_odds_archive")
    op.drop_table("game_odds_archive")
