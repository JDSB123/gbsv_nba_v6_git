import asyncio
import uuid

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from src.config import get_settings


def _asyncpg_kwargs(database: str) -> dict[str, object]:
    url = make_url(get_settings().database_url)
    kwargs: dict[str, object] = {
        "user": url.username,
        "password": url.password,
        "host": url.host or "localhost",
        "port": url.port or 5432,
        "database": database,
    }
    if url.query.get("ssl") == "require":
        kwargs["ssl"] = "require"
    return kwargs


async def _can_connect(database: str) -> bool:
    try:
        conn = await asyncpg.connect(**_asyncpg_kwargs(database))
    except Exception:
        return False
    await conn.close()
    return True


async def _create_database(database: str) -> None:
    conn = await asyncpg.connect(**_asyncpg_kwargs("postgres"))
    try:
        await conn.execute(f'CREATE DATABASE "{database}"')
    finally:
        await conn.close()


async def _drop_database(database: str) -> None:
    conn = await asyncpg.connect(**_asyncpg_kwargs("postgres"))
    try:
        await conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            database,
        )
        await conn.execute(f'DROP DATABASE IF EXISTS "{database}"')
    finally:
        await conn.close()


async def _schema_snapshot(database: str) -> tuple[set[str], set[str], str]:
    conn = await asyncpg.connect(**_asyncpg_kwargs(database))
    try:
        tables = {
            row["table_name"]
            for row in await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            )
        }
        prediction_columns = {
            row["column_name"]
            for row in await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'predictions'
                """
            )
        }
        revision = await conn.fetchval("SELECT version_num FROM alembic_version")
    finally:
        await conn.close()
    return tables, prediction_columns, revision


@pytest.mark.integration
def test_alembic_upgrade_bootstraps_fresh_database(monkeypatch: pytest.MonkeyPatch):
    if not asyncio.run(_can_connect("postgres")):
        pytest.fail("PostgreSQL is not reachable — check DATABASE_URL and server firewall")

    database = f"nba_gbsv_mig_{uuid.uuid4().hex[:8]}"
    target_url = (
        make_url(get_settings().database_url)
        .set(database=database)
        .render_as_string(hide_password=False)
    )

    asyncio.run(_create_database(database))
    monkeypatch.setenv("DATABASE_URL", target_url)
    get_settings.cache_clear()

    try:
        command.upgrade(Config("alembic.ini"), "head")
        tables, prediction_columns, revision = asyncio.run(_schema_snapshot(database))
    finally:
        get_settings.cache_clear()
        asyncio.run(_drop_database(database))

    assert {
        "teams",
        "players",
        "games",
        "team_season_stats",
        "player_game_stats",
        "odds_snapshots",
        "predictions",
        "injuries",
        "alembic_version",
    }.issubset(tables)
    assert {
        "opening_spread",
        "opening_total",
        "closing_spread",
        "closing_total",
        "clv_spread",
        "clv_total",
    }.issubset(prediction_columns)
    assert revision == "002_model_registry"
