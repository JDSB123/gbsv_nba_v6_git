from datetime import date, datetime
from unittest.mock import AsyncMock

import pytest

from src.data.basketball_client import BasketballClient, normalize_team_stats
from src.data.seasons import (
    current_nba_season,
    parse_api_datetime,
    resolve_backfill_window,
    season_for_date,
)


def test_normalize_team_stats_accepts_dict_payload():
    payload = {"games": {"played": {"all": 82}}}
    assert normalize_team_stats(payload) == payload


def test_normalize_team_stats_accepts_list_payload():
    payload = [{"games": {"played": {"all": 82}}}]
    assert normalize_team_stats(payload) == payload[0]


def test_current_nba_season_tracks_calendar():
    assert current_nba_season(date(2026, 3, 22)) == "2025-2026"
    assert current_nba_season(date(2026, 10, 5)) == "2026-2027"


def test_season_for_date_uses_previous_year_before_october():
    assert season_for_date(date(2026, 3, 22)) == "2025-2026"


def test_resolve_backfill_window_caps_to_historical_season_end():
    season, start_date, end_date = resolve_backfill_window(
        "2024-2025",
        120,
        today=date(2026, 3, 22),
    )

    assert season == "2024-2025"
    assert start_date == date(2025, 3, 2)
    assert end_date == date(2025, 6, 30)


def test_resolve_backfill_window_caps_to_season_start():
    season, start_date, end_date = resolve_backfill_window(
        "2024-2025",
        400,
        today=date(2025, 2, 1),
    )

    assert season == "2024-2025"
    assert start_date == date(2024, 10, 1)
    assert end_date == date(2025, 2, 1)


def test_parse_api_datetime_returns_naive_utc():
    parsed = parse_api_datetime("2025-11-22T00:00:00Z")

    assert parsed == datetime(2025, 11, 22, 0, 0)
    assert parsed.tzinfo is None


@pytest.mark.asyncio
async def test_persist_team_season_stats_accepts_dict_payload():
    client = BasketballClient()
    db = AsyncMock()
    stats = {
        "games": {
            "played": {"all": "82"},
            "wins": {"all": {"total": "50"}},
            "loses": {"all": {"total": "32"}},
        },
        "points": {
            "for": {"average": {"all": "118.2"}},
            "against": {"average": {"all": "111.1"}},
        },
    }

    await client.persist_team_season_stats(133, stats, "2024-2025", db)

    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()
