"""Add CLV columns to predictions table.

Revision ID: 001_add_clv
Revises:
Create Date: 2026-03-22

"""

import sqlalchemy as sa
from alembic import op

revision = "001_add_clv"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("opening_spread", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("opening_total", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("closing_spread", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("closing_total", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("clv_spread", sa.Float(), nullable=True))
    op.add_column("predictions", sa.Column("clv_total", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("predictions", "clv_total")
    op.drop_column("predictions", "clv_spread")
    op.drop_column("predictions", "closing_total")
    op.drop_column("predictions", "closing_spread")
    op.drop_column("predictions", "opening_total")
    op.drop_column("predictions", "opening_spread")
