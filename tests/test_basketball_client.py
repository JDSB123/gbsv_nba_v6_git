from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.basketball_client import (
    INJURY_STATUS_MAP,
    BasketballClient,
    _compute_advanced_stats,
    normalize_team_stats,
)
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


def test_compute_advanced_stats_from_box_score_aggregates():
    """When the API provides FG/FT/REB/TOV data, pace & ratings are computed
    via the Dean Oliver possession formula."""
    stats = {
        "field_goals": {"total": {"all": "3200"}, "percentage": {"all": "0.46"}},
        "free_throws": {"total": {"all": "1400"}, "percentage": {"all": "0.78"}},
        "rebounds": {"offReb": {"all": "900"}},
        "turnovers": {"total": {"all": "1100"}},
        "points": {
            "for": {"total": {"all": "9700"}, "average": {"all": "118.2"}},
            "against": {"total": {"all": "9100"}, "average": {"all": "111.1"}},
        },
    }
    pace, off_rtg, def_rtg = _compute_advanced_stats(stats, 82, 118.2, 111.1)
    assert pace is not None and pace > 80
    assert off_rtg is not None and off_rtg > 90
    assert def_rtg is not None and def_rtg > 90
    # Off rating should exceed def rating for this winning team
    assert off_rtg > def_rtg


def test_compute_advanced_stats_fallback_from_ppg():
    """When box-score fields are absent, fall back to PPG/OPPG estimate."""
    stats = {
        "points": {
            "for": {"average": {"all": "115.0"}},
            "against": {"average": {"all": "110.0"}},
        },
    }
    pace, off_rtg, def_rtg = _compute_advanced_stats(stats, 82, 115.0, 110.0)
    assert pace is not None and pace > 80
    assert off_rtg is not None and off_rtg > 90
    assert def_rtg is not None and def_rtg > 90


def test_compute_advanced_stats_no_games():
    """Returns None tuple when games_played is zero."""
    pace, off_rtg, def_rtg = _compute_advanced_stats({}, 0, None, None)
    assert pace is None
    assert off_rtg is None
    assert def_rtg is None


def test_compute_advanced_stats_whole_number_percentages():
    """Percentage values >1 (e.g. 46.5 meaning 46.5%) are handled."""
    stats = {
        "field_goals": {"total": {"all": "3200"}, "percentage": {"all": "46"}},
        "free_throws": {"total": {"all": "1400"}, "percentage": {"all": "78"}},
        "rebounds": {"offReb": {"all": "900"}},
        "turnovers": {"total": {"all": "1100"}},
        "points": {
            "for": {"total": {"all": "9700"}, "average": {"all": "118.2"}},
            "against": {"total": {"all": "9100"}, "average": {"all": "111.1"}},
        },
    }
    pace, off_rtg, def_rtg = _compute_advanced_stats(stats, 82, 118.2, 111.1)
    assert pace is not None and pace > 80
    assert off_rtg is not None and off_rtg > 90


# ── Injury pipeline tests ─────────────────────────────────────────


def test_injury_status_map_covers_expected_statuses():
    """All common NBA API injury statuses are mapped."""
    assert INJURY_STATUS_MAP["out"] == "out"
    assert INJURY_STATUS_MAP["doubtful"] == "doubtful"
    assert INJURY_STATUS_MAP["questionable"] == "questionable"
    assert INJURY_STATUS_MAP["probable"] == "probable"
    assert INJURY_STATUS_MAP["day-to-day"] == "questionable"
    assert INJURY_STATUS_MAP["out for season"] == "out"
    assert INJURY_STATUS_MAP["out indefinitely"] == "out"


@pytest.mark.asyncio
async def test_persist_injuries_clears_and_inserts():
    """persist_injuries clears existing injuries and inserts new ones."""
    client = BasketballClient()
    db = AsyncMock()
    db.add = MagicMock()

    # Simulate: delete returns nothing, team lookup → id=1, player lookup → id=10
    db.execute = AsyncMock(
        side_effect=[
            AsyncMock(scalar_one_or_none=lambda: None),  # delete
            AsyncMock(scalar_one_or_none=lambda: 1),  # team lookup
            AsyncMock(scalar_one_or_none=lambda: 10),  # player lookup
        ]
    )

    injuries_data = [
        {
            "player": {"firstname": "LeBron", "lastname": "James"},
            "team": {"name": "Los Angeles Lakers"},
            "status": {"type": "Day-To-Day", "description": "Ankle"},
        }
    ]

    await client.persist_injuries(injuries_data, db)

    # Should have called delete, two selects, db.add, and commit
    assert db.execute.await_count == 3
    db.add.assert_called_once()
    db.commit.assert_awaited_once()

    injury_obj = db.add.call_args[0][0]
    assert injury_obj.player_id == 10
    assert injury_obj.team_id == 1
    assert injury_obj.status == "questionable"  # day-to-day → questionable
    assert injury_obj.description == "Ankle"


@pytest.mark.asyncio
async def test_persist_injuries_skips_unknown_team():
    """Injuries for teams not in our DB are skipped."""
    client = BasketballClient()
    db = AsyncMock()
    db.add = MagicMock()

    db.execute = AsyncMock(
        side_effect=[
            AsyncMock(scalar_one_or_none=lambda: None),  # delete
            AsyncMock(scalar_one_or_none=lambda: None),  # team not found
        ]
    )

    injuries_data = [
        {
            "player": {"firstname": "Test", "lastname": "Player"},
            "team": {"name": "Unknown Team"},
            "status": {"type": "Out", "description": "Knee"},
        }
    ]

    await client.persist_injuries(injuries_data, db)

    db.add.assert_not_called()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_injuries_skips_unknown_player():
    """Injuries for players not in our DB are skipped."""
    client = BasketballClient()
    db = AsyncMock()
    db.add = MagicMock()

    db.execute = AsyncMock(
        side_effect=[
            AsyncMock(scalar_one_or_none=lambda: None),  # delete
            AsyncMock(scalar_one_or_none=lambda: 1),  # team found
            AsyncMock(scalar_one_or_none=lambda: None),  # player not found
        ]
    )

    injuries_data = [
        {
            "player": {"firstname": "Unknown", "lastname": "Player"},
            "team": {"name": "Boston Celtics"},
            "status": {"type": "Out", "description": "Shoulder"},
        }
    ]

    await client.persist_injuries(injuries_data, db)

    db.add.assert_not_called()
    db.commit.assert_awaited_once()
