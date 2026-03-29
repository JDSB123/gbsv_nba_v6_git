"""Data-integrity hardening: CASCADE FKs, NOT NULL, CHECK constraints, wider outcome_name.

Revision ID: 005_data_integrity
Revises: 004_player_props
Create Date: 2025-07-12
"""

import contextlib

import sqlalchemy as sa
from alembic import op

revision = "005_data_integrity"
down_revision = "004_player_props"
branch_labels = None
depends_on = None


def _has_constraint(table: str, constraint_name: str) -> bool:
    """Check if a named constraint already exists (prevents duplicate-add errors)."""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return any(ck.get("name") == constraint_name for ck in insp.get_check_constraints(table))


def upgrade() -> None:
    # ── 1. Game: make status & season NOT NULL with defaults ────────
    op.execute("UPDATE games SET status = 'NS' WHERE status IS NULL")
    op.execute("UPDATE games SET season = '' WHERE season IS NULL")
    op.alter_column(
        "games",
        "status",
        existing_type=sa.String(10),
        nullable=False,
        server_default="NS",
    )
    op.alter_column(
        "games",
        "season",
        existing_type=sa.String(10),
        nullable=False,
        server_default="",
    )

    # ── 2. Player: make is_active NOT NULL ─────────────────────────
    op.execute("UPDATE players SET is_active = true WHERE is_active IS NULL")
    op.alter_column(
        "players",
        "is_active",
        existing_type=sa.Boolean(),
        nullable=False,
        server_default="true",
    )

    # ── 3. CHECK constraints on quarter scores (non-negative) ──────
    for col in (
        "home_q1",
        "home_q2",
        "home_q3",
        "home_q4",
        "away_q1",
        "away_q2",
        "away_q3",
        "away_q4",
    ):
        name = f"ck_games_{col}_non_neg"
        if not _has_constraint("games", name):
            op.create_check_constraint(name, "games", f"{col} IS NULL OR {col} >= 0")

    # ── 4. Widen outcome_name to 128 chars ─────────────────────────
    op.alter_column(
        "odds_snapshots",
        "outcome_name",
        existing_type=sa.String(60),
        type_=sa.String(128),
        existing_nullable=False,
    )

    # ── 5. Recreate FKs with ON DELETE CASCADE ─────────────────────
    _recreate_fk("players", "players_team_id_fkey", "team_id", "teams", "id")
    _recreate_fk("games", "games_home_team_id_fkey", "home_team_id", "teams", "id")
    _recreate_fk("games", "games_away_team_id_fkey", "away_team_id", "teams", "id")
    _recreate_fk("team_season_stats", "team_season_stats_team_id_fkey", "team_id", "teams", "id")
    _recreate_fk(
        "player_game_stats", "player_game_stats_player_id_fkey", "player_id", "players", "id"
    )
    _recreate_fk("player_game_stats", "player_game_stats_game_id_fkey", "game_id", "games", "id")
    _recreate_fk("odds_snapshots", "odds_snapshots_game_id_fkey", "game_id", "games", "id")
    _recreate_fk("predictions", "predictions_game_id_fkey", "game_id", "games", "id")
    _recreate_fk("injuries", "injuries_player_id_fkey", "player_id", "players", "id")
    _recreate_fk("injuries", "injuries_team_id_fkey", "team_id", "teams", "id")


def _recreate_fk(table: str, fk_name: str, local_col: str, ref_table: str, ref_col: str) -> None:
    """Drop FK and re-add with ON DELETE CASCADE."""
    with contextlib.suppress(Exception):
        op.drop_constraint(fk_name, table, type_="foreignkey")
    op.create_foreign_key(fk_name, table, ref_table, [local_col], [ref_col], ondelete="CASCADE")


def downgrade() -> None:
    # Remove CHECK constraints
    for col in (
        "home_q1",
        "home_q2",
        "home_q3",
        "home_q4",
        "away_q1",
        "away_q2",
        "away_q3",
        "away_q4",
    ):
        name = f"ck_games_{col}_non_neg"
        with contextlib.suppress(Exception):
            op.drop_constraint(name, "games", type_="check")

    # Revert outcome_name width
    op.alter_column(
        "odds_snapshots",
        "outcome_name",
        existing_type=sa.String(128),
        type_=sa.String(60),
        existing_nullable=False,
    )

    # Revert NOT NULL changes
    op.alter_column(
        "games", "status", existing_type=sa.String(10), nullable=True, server_default=None
    )
    op.alter_column(
        "games", "season", existing_type=sa.String(10), nullable=True, server_default=None
    )
    op.alter_column(
        "players", "is_active", existing_type=sa.Boolean(), nullable=True, server_default=None
    )

    # Revert CASCADE FKs back to NO ACTION (default)
    _recreate_fk_no_action("players", "players_team_id_fkey", "team_id", "teams", "id")
    _recreate_fk_no_action("games", "games_home_team_id_fkey", "home_team_id", "teams", "id")
    _recreate_fk_no_action("games", "games_away_team_id_fkey", "away_team_id", "teams", "id")
    _recreate_fk_no_action(
        "team_season_stats", "team_season_stats_team_id_fkey", "team_id", "teams", "id"
    )
    _recreate_fk_no_action(
        "player_game_stats", "player_game_stats_player_id_fkey", "player_id", "players", "id"
    )
    _recreate_fk_no_action(
        "player_game_stats", "player_game_stats_game_id_fkey", "game_id", "games", "id"
    )
    _recreate_fk_no_action(
        "odds_snapshots", "odds_snapshots_game_id_fkey", "game_id", "games", "id"
    )
    _recreate_fk_no_action("predictions", "predictions_game_id_fkey", "game_id", "games", "id")
    _recreate_fk_no_action("injuries", "injuries_player_id_fkey", "player_id", "players", "id")
    _recreate_fk_no_action("injuries", "injuries_team_id_fkey", "team_id", "teams", "id")


def _recreate_fk_no_action(
    table: str, fk_name: str, local_col: str, ref_table: str, ref_col: str
) -> None:
    with contextlib.suppress(Exception):
        op.drop_constraint(fk_name, table, type_="foreignkey")
    op.create_foreign_key(fk_name, table, ref_table, [local_col], [ref_col])
