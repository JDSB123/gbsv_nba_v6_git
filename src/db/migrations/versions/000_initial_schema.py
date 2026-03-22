"""Create the initial application schema.

Revision ID: 000_initial
Revises:
Create Date: 2026-03-22
"""

import sqlalchemy as sa
from alembic import op

revision = "000_initial"
down_revision = None
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    return any(
        index["name"] == index_name
        for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    )


def _has_unique_constraint(table_name: str, constraint_name: str) -> bool:
    return any(
        constraint["name"] == constraint_name
        for constraint in sa.inspect(op.get_bind()).get_unique_constraints(table_name)
    )


def upgrade() -> None:
    if not _has_table("teams"):
        op.create_table(
            "teams",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("abbreviation", sa.String(length=10), nullable=False),
            sa.Column("conference", sa.String(length=20), nullable=True),
            sa.Column("division", sa.String(length=30), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("players"):
        op.create_table(
            "players",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("team_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("position", sa.String(length=20), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=True),
            sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("games"):
        op.create_table(
            "games",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("odds_api_id", sa.String(length=64), nullable=True),
            sa.Column("home_team_id", sa.Integer(), nullable=False),
            sa.Column("away_team_id", sa.Integer(), nullable=False),
            sa.Column("commence_time", sa.DateTime(), nullable=False),
            sa.Column("status", sa.String(length=10), nullable=True),
            sa.Column("season", sa.String(length=10), nullable=True),
            sa.Column("home_q1", sa.Integer(), nullable=True),
            sa.Column("home_q2", sa.Integer(), nullable=True),
            sa.Column("home_q3", sa.Integer(), nullable=True),
            sa.Column("home_q4", sa.Integer(), nullable=True),
            sa.Column("home_ot", sa.Integer(), nullable=True),
            sa.Column("away_q1", sa.Integer(), nullable=True),
            sa.Column("away_q2", sa.Integer(), nullable=True),
            sa.Column("away_q3", sa.Integer(), nullable=True),
            sa.Column("away_q4", sa.Integer(), nullable=True),
            sa.Column("away_ot", sa.Integer(), nullable=True),
            sa.Column("home_score_1h", sa.Integer(), nullable=True),
            sa.Column("away_score_1h", sa.Integer(), nullable=True),
            sa.Column("home_score_fg", sa.Integer(), nullable=True),
            sa.Column("away_score_fg", sa.Integer(), nullable=True),
            sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"]),
            sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("team_season_stats"):
        op.create_table(
            "team_season_stats",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("team_id", sa.Integer(), nullable=False),
            sa.Column("season", sa.String(length=10), nullable=False),
            sa.Column("games_played", sa.Integer(), nullable=True),
            sa.Column("wins", sa.Integer(), nullable=True),
            sa.Column("losses", sa.Integer(), nullable=True),
            sa.Column("ppg", sa.Float(), nullable=True),
            sa.Column("oppg", sa.Float(), nullable=True),
            sa.Column("pace", sa.Float(), nullable=True),
            sa.Column("off_rating", sa.Float(), nullable=True),
            sa.Column("def_rating", sa.Float(), nullable=True),
            sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("player_game_stats"):
        op.create_table(
            "player_game_stats",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("player_id", sa.Integer(), nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("minutes", sa.Integer(), nullable=True),
            sa.Column("points", sa.Integer(), nullable=True),
            sa.Column("rebounds", sa.Integer(), nullable=True),
            sa.Column("assists", sa.Integer(), nullable=True),
            sa.Column("steals", sa.Integer(), nullable=True),
            sa.Column("blocks", sa.Integer(), nullable=True),
            sa.Column("turnovers", sa.Integer(), nullable=True),
            sa.Column("fg_pct", sa.Float(), nullable=True),
            sa.Column("three_pct", sa.Float(), nullable=True),
            sa.Column("ft_pct", sa.Float(), nullable=True),
            sa.Column("plus_minus", sa.Float(), nullable=True),
            sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
            sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("odds_snapshots"):
        op.create_table(
            "odds_snapshots",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("source", sa.String(length=30), nullable=False),
            sa.Column("bookmaker", sa.String(length=60), nullable=False),
            sa.Column("market", sa.String(length=30), nullable=False),
            sa.Column("outcome_name", sa.String(length=60), nullable=False),
            sa.Column("price", sa.Float(), nullable=False),
            sa.Column("point", sa.Float(), nullable=True),
            sa.Column("captured_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("predictions"):
        op.create_table(
            "predictions",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("game_id", sa.Integer(), nullable=False),
            sa.Column("model_version", sa.String(length=20), nullable=False),
            sa.Column("predicted_home_fg", sa.Float(), nullable=False),
            sa.Column("predicted_away_fg", sa.Float(), nullable=False),
            sa.Column("predicted_home_1h", sa.Float(), nullable=False),
            sa.Column("predicted_away_1h", sa.Float(), nullable=False),
            sa.Column("fg_spread", sa.Float(), nullable=True),
            sa.Column("fg_total", sa.Float(), nullable=True),
            sa.Column("fg_home_ml_prob", sa.Float(), nullable=True),
            sa.Column("h1_spread", sa.Float(), nullable=True),
            sa.Column("h1_total", sa.Float(), nullable=True),
            sa.Column("h1_home_ml_prob", sa.Float(), nullable=True),
            sa.Column("predicted_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_table("injuries"):
        op.create_table(
            "injuries",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("player_id", sa.Integer(), nullable=False),
            sa.Column("team_id", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("description", sa.String(length=255), nullable=True),
            sa.Column("reported_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
            sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_index("players", "ix_players_team_id"):
        op.create_index("ix_players_team_id", "players", ["team_id"], unique=False)

    if not _has_index("games", "ix_games_odds_api_id"):
        op.create_index("ix_games_odds_api_id", "games", ["odds_api_id"], unique=True)
    if not _has_index("games", "ix_games_home_team_id"):
        op.create_index("ix_games_home_team_id", "games", ["home_team_id"], unique=False)
    if not _has_index("games", "ix_games_away_team_id"):
        op.create_index("ix_games_away_team_id", "games", ["away_team_id"], unique=False)
    if not _has_index("games", "ix_games_commence_time"):
        op.create_index("ix_games_commence_time", "games", ["commence_time"], unique=False)
    if not _has_index("games", "ix_games_status_commence"):
        op.create_index(
            "ix_games_status_commence",
            "games",
            ["status", "commence_time"],
            unique=False,
        )

    if not _has_index("team_season_stats", "ix_team_season_stats_team_id"):
        op.create_index(
            "ix_team_season_stats_team_id",
            "team_season_stats",
            ["team_id"],
            unique=False,
        )
    if not _has_unique_constraint("team_season_stats", "uq_team_season"):
        op.create_unique_constraint(
            "uq_team_season",
            "team_season_stats",
            ["team_id", "season"],
        )

    if not _has_index("player_game_stats", "ix_player_game_stats_player_id"):
        op.create_index(
            "ix_player_game_stats_player_id",
            "player_game_stats",
            ["player_id"],
            unique=False,
        )
    if not _has_index("player_game_stats", "ix_player_game_stats_game_id"):
        op.create_index(
            "ix_player_game_stats_game_id",
            "player_game_stats",
            ["game_id"],
            unique=False,
        )
    if not _has_unique_constraint("player_game_stats", "uq_player_game"):
        op.create_unique_constraint(
            "uq_player_game",
            "player_game_stats",
            ["player_id", "game_id"],
        )

    if not _has_index("odds_snapshots", "ix_odds_snapshots_game_id"):
        op.create_index(
            "ix_odds_snapshots_game_id",
            "odds_snapshots",
            ["game_id"],
            unique=False,
        )
    if not _has_index("odds_snapshots", "ix_odds_snapshots_captured_at"):
        op.create_index(
            "ix_odds_snapshots_captured_at",
            "odds_snapshots",
            ["captured_at"],
            unique=False,
        )
    if not _has_index("odds_snapshots", "ix_odds_game_market_captured"):
        op.create_index(
            "ix_odds_game_market_captured",
            "odds_snapshots",
            ["game_id", "market", "captured_at"],
            unique=False,
        )

    if not _has_index("predictions", "ix_predictions_game_id"):
        op.create_index(
            "ix_predictions_game_id",
            "predictions",
            ["game_id"],
            unique=False,
        )
    if not _has_index("predictions", "ix_predictions_predicted_at"):
        op.create_index(
            "ix_predictions_predicted_at",
            "predictions",
            ["predicted_at"],
            unique=False,
        )
    if not _has_index("predictions", "ix_pred_game_version"):
        op.create_index(
            "ix_pred_game_version",
            "predictions",
            ["game_id", "model_version"],
            unique=False,
        )

    if not _has_index("injuries", "ix_injuries_player_id"):
        op.create_index(
            "ix_injuries_player_id",
            "injuries",
            ["player_id"],
            unique=False,
        )
    if not _has_index("injuries", "ix_injuries_team_id"):
        op.create_index(
            "ix_injuries_team_id",
            "injuries",
            ["team_id"],
            unique=False,
        )


def downgrade() -> None:
    for table_name in (
        "injuries",
        "predictions",
        "odds_snapshots",
        "player_game_stats",
        "team_season_stats",
        "games",
        "players",
        "teams",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
