from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.basketball_client import (
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
