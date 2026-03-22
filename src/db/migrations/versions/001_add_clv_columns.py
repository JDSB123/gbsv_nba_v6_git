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


def upgrade() -> None:
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
