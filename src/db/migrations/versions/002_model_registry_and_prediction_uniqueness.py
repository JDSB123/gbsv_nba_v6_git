"""Add model registry and enforce prediction idempotency.

Revision ID: 002_model_registry
Revises: 001_add_clv
Create Date: 2026-03-22
"""

import sqlalchemy as sa
from alembic import op

revision = "002_model_registry"
down_revision = "001_add_clv"
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


def _dedupe_predictions_for_unique_constraint() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            DELETE FROM predictions p
            USING (
              SELECT ctid
              FROM (
                SELECT ctid,
                       row_number() OVER (
                         PARTITION BY game_id, model_version
                         ORDER BY predicted_at DESC, id DESC
                       ) AS rn
                FROM predictions
              ) ranked
              WHERE ranked.rn > 1
            ) dupes
            WHERE p.ctid = dupes.ctid
            """
        )


def upgrade() -> None:
    if not _has_table("model_registry"):
        op.create_table(
            "model_registry",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("model_version", sa.String(length=64), nullable=False),
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("promoted_at", sa.DateTime(), nullable=True),
            sa.Column("retired_at", sa.DateTime(), nullable=True),
            sa.Column("promotion_reason", sa.String(length=255), nullable=True),
            sa.Column("metrics_json", sa.Text(), nullable=True),
            sa.Column("params_json", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("model_version", name="uq_model_registry_version"),
        )

    if not _has_index("model_registry", "ix_model_registry_is_active"):
        op.create_index(
            "ix_model_registry_is_active",
            "model_registry",
            ["is_active"],
            unique=False,
        )

    if _has_table("predictions") and not _has_unique_constraint(
        "predictions", "uq_predictions_game_model_version"
    ):
        _dedupe_predictions_for_unique_constraint()
        op.create_unique_constraint(
            "uq_predictions_game_model_version",
            "predictions",
            ["game_id", "model_version"],
        )


def downgrade() -> None:
    if _has_table("predictions") and _has_unique_constraint(
        "predictions", "uq_predictions_game_model_version"
    ):
        op.drop_constraint(
            "uq_predictions_game_model_version",
            "predictions",
            type_="unique",
        )

    if _has_table("model_registry"):
        if _has_index("model_registry", "ix_model_registry_is_active"):
            op.drop_index("ix_model_registry_is_active", table_name="model_registry")
        op.drop_table("model_registry")
