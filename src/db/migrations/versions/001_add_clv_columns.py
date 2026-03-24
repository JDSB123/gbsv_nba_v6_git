"""Add CLV columns to predictions table.

Revision ID: 001_add_clv
Revises:
Create Date: 2026-03-22

"""

import sqlalchemy as sa
from alembic import op

revision = "001_add_clv"
down_revision = "000_initial"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    return any(
        index["name"] == index_name for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    )


def _has_unique_constraint(table_name: str, constraint_name: str) -> bool:
    return any(
        constraint["name"] == constraint_name
        for constraint in sa.inspect(op.get_bind()).get_unique_constraints(table_name)
    )


def _ensure_predictions_table() -> None:
    if _has_table("predictions"):
        return

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
        sa.Column("opening_spread", sa.Float(), nullable=True),
        sa.Column("opening_total", sa.Float(), nullable=True),
        sa.Column("closing_spread", sa.Float(), nullable=True),
        sa.Column("closing_total", sa.Float(), nullable=True),
        sa.Column("clv_spread", sa.Float(), nullable=True),
        sa.Column("clv_total", sa.Float(), nullable=True),
        sa.Column("predicted_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    if not _has_index("predictions", "ix_predictions_game_id"):
        op.create_index("ix_predictions_game_id", "predictions", ["game_id"], unique=False)
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
    if not _has_unique_constraint("predictions", "uq_predictions_game_model_version"):
        op.create_unique_constraint(
            "uq_predictions_game_model_version",
            "predictions",
            ["game_id", "model_version"],
        )


def upgrade() -> None:
    _ensure_predictions_table()

    existing_columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("predictions")
    }
    columns_to_add = (
        ("opening_spread", sa.Float()),
        ("opening_total", sa.Float()),
        ("closing_spread", sa.Float()),
        ("closing_total", sa.Float()),
        ("clv_spread", sa.Float()),
        ("clv_total", sa.Float()),
    )
    for column_name, column_type in columns_to_add:
        if column_name not in existing_columns:
            op.add_column(
                "predictions",
                sa.Column(column_name, column_type, nullable=True),
            )


def downgrade() -> None:
    if not _has_table("predictions"):
        return

    existing_columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("predictions")
    }
    for column_name in (
        "clv_total",
        "clv_spread",
        "closing_total",
        "closing_spread",
        "opening_total",
        "opening_spread",
    ):
        if column_name in existing_columns:
            op.drop_column("predictions", column_name)
