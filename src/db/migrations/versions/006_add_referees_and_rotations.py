"""Add referees and rotation changes.

Revision ID: 006_referees_rotations
Revises: 005_data_integrity
Create Date: 2026-03-29
"""

import sqlalchemy as sa
from alembic import op

revision = "006_referees_rotations"
down_revision = "005_data_integrity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create game_referees table
    op.create_table(
        "game_referees",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("referee_name", sa.String(100), nullable=False),
        sa.Column("role", sa.String(50), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_game_referees_game_id", "game_referees", ["game_id"])

    # Create rotation_changes table
    op.create_table(
        "rotation_changes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column(
            "is_replacement_starter", sa.Boolean(), server_default=sa.text("false"), nullable=True
        ),
        sa.Column("minutes_projection", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rotation_changes_game_id", "rotation_changes", ["game_id"])


def downgrade() -> None:
    op.drop_index("ix_rotation_changes_game_id", table_name="rotation_changes")
    op.drop_table("rotation_changes")

    op.drop_index("ix_game_referees_game_id", table_name="game_referees")
    op.drop_table("game_referees")
