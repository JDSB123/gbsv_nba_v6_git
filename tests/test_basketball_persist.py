"""Tests for BasketballClient persistence helpers and fetch methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.basketball_client import (
    BasketballClient,
    _box_score_percentage,
    _compute_advanced_stats,
    _pct_to_decimal,
)

_MOD = "src.data.basketball_client"


@pytest.fixture
def client():
    with patch(f"{_MOD}.get_settings") as mock_settings:
        s = MagicMock()
        s.basketball_api_base = "https://api.example.com"
        s.basketball_api_key = "test-key"
        s.basketball_api_league_id = 12
        s.nba_api_base = "https://nba.example.com"
        mock_settings.return_value = s
        yield BasketballClient()


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    return db


# ── _pct_to_decimal edge cases ──────────────────────────────────


class TestPctToDecimalEdge:
    def test_out_of_range_after_division(self):
        """Values that produce > 1.0 after /100 warn and return None."""
        # A value like 150 → 150/100 = 1.5 which is > 1.0
        assert _pct_to_decimal(150) is None

    def test_value_exactly_one(self):
        """100% → 1.0 is valid."""
        assert _pct_to_decimal(100) == 1.0

    def test_decimal_passthrough(self):
        assert _pct_to_decimal(0.465) == 0.465


class TestBoxScorePercentage:
    def test_uses_percentage_when_present(self):
        assert _box_score_percentage({"percentage": "38.5", "total": 4, "attempts": 11}) == 38.5

    def test_derives_percentage_from_totals_when_missing(self):
        assert _box_score_percentage({"total": 4, "attempts": 11, "percentage": None}) == pytest.approx(
            36.36,
            abs=0.01,
        )

    def test_returns_none_when_attempts_missing(self):
        assert _box_score_percentage({"total": 4, "attempts": 0, "percentage": None}) is None


# ── _compute_advanced_stats edge cases ──────────────────────────


class TestComputeAdvancedStatsEdge:
    def test_zero_fg_pct_falls_through(self):
        """When fg_pct is 0, primary path is skipped → fallback used."""
        stats = {
            "field_goals": {
                "total": {"all": "100"},
                "percentage": {"all": "0"},  # 0% → 0.0
            },
            "free_throws": {
                "total": {"all": "50"},
                "percentage": {"all": "80"},  # 80% → 0.8
            },
            "turnovers": {"total": {"all": "10"}},
            "rebounds": {},
            "points": {
                "for": {"total": {"all": "8000"}, "average": {"all": "100"}},
                "against": {"total": {"all": "7800"}, "average": {"all": "97.5"}},
            },
        }
        pace, off_r, def_r = _compute_advanced_stats(stats, 80, 100.0, 97.5)
        # Falls through to PPG/OPPG fallback
        assert pace is not None
        assert off_r is not None

    def test_zero_ft_pct_falls_through(self):
        """When ft_pct is 0, primary path is skipped → fallback."""
        stats = {
            "field_goals": {
                "total": {"all": "100"},
                "percentage": {"all": "46.5"},
            },
            "free_throws": {
                "total": {"all": "50"},
                "percentage": {"all": "0"},
            },
            "turnovers": {"total": {"all": "10"}},
            "rebounds": {},
            "points": {
                "for": {"total": {"all": "8000"}, "average": {"all": "100"}},
                "against": {"total": {"all": "7800"}, "average": {"all": "97.5"}},
            },
        }
        pace, off_r, def_r = _compute_advanced_stats(stats, 80, 100.0, 97.5)
        assert pace is not None

    def test_no_ppg_oppg_returns_none(self):
        """When both primary and fallback fail, returns all None."""
        stats = {"field_goals": {}, "free_throws": {}, "turnovers": {}, "rebounds": {}}
        pace, off_r, def_r = _compute_advanced_stats(stats, 80, None, None)
        assert pace is None
        assert off_r is None
        assert def_r is None


# ── Fetch method happy paths ────────────────────────────────────


class TestFetchMethods:
    @pytest.mark.anyio
    async def test_fetch_games(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [{"id": 1}]
            result = await client.fetch_games()
            assert result == [{"id": 1}]
            mock_get.assert_awaited_once()

    @pytest.mark.anyio
    async def test_fetch_games_with_date(self, client):
        from datetime import date

        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            await client.fetch_games(game_date=date(2024, 12, 1))
            call_args = mock_get.call_args
            assert call_args[0][0] == "games"
            assert "date" in call_args[0][1]

    @pytest.mark.anyio
    async def test_fetch_team_stats(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"games": {}}
            result = await client.fetch_team_stats(team_id=1)
            assert result == {"games": {}}

    @pytest.mark.anyio
    async def test_fetch_player_stats(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [{"player": {"id": 5}}]
            result = await client.fetch_player_stats(game_id=100)
            assert result == [{"player": {"id": 5}}]

    @pytest.mark.anyio
    async def test_fetch_team_game_stats(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            result = await client.fetch_team_game_stats(game_id=100)
            assert result == []

    @pytest.mark.anyio
    async def test_fetch_standings(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [{"team": {"id": 1}}]
            result = await client.fetch_standings()
            assert len(result) == 1

    @pytest.mark.anyio
    async def test_fetch_h2h(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            result = await client.fetch_h2h(1, 2)
            mock_get.assert_awaited_once_with("games/h2h", {"h2h": "1-2"})
            assert result == []

    @pytest.mark.anyio
    async def test_fetch_players(self, client):
        with patch(f"{_MOD}.BasketballClient._get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [{"id": 10}]
            result = await client.fetch_players(team_id=5)
            assert result == [{"id": 10}]

    @pytest.mark.anyio
    async def test_fetch_injuries(self, client):
        with patch("httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": [{"player": {"id": 1}}]}
            mock_resp.raise_for_status = MagicMock()
            ctx = AsyncMock()
            ctx.__aenter__.return_value = MagicMock(
                get=AsyncMock(return_value=mock_resp)
            )
            mock_cls.return_value = ctx
            result = await client.fetch_injuries(season="2024-2025")
            assert len(result) == 1


# ── persist_teams ───────────────────────────────────────────────


class TestPersistTeams:
    @pytest.mark.anyio
    async def test_upserts_from_standings(self, client, mock_db):
        standings = [
            [
                {
                    "team": {"id": 1, "name": "Lakers", "code": "LAL"},
                    "group": {"name": "Western"},
                }
            ]
        ]
        await client.persist_teams(standings, mock_db)
        mock_db.execute.assert_awaited()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_non_list_group(self, client, mock_db):
        """Standings entries that aren't nested lists are handled."""
        standings = [
            {
                "team": {"id": 2, "name": "Celtics"},
                "group": {"name": "Eastern"},
            }
        ]
        await client.persist_teams(standings, mock_db)
        mock_db.execute.assert_awaited()


# ── persist_games ───────────────────────────────────────────────


class TestPersistGames:
    @pytest.mark.anyio
    async def test_basic_game_upsert(self, client, mock_db):
        games = [
            {
                "id": 1001,
                "scores": {
                    "home": {
                        "quarter_1": 28,
                        "quarter_2": 30,
                        "quarter_3": 25,
                        "quarter_4": 22,
                        "over_time": 0,
                        "total": 105,
                    },
                    "away": {
                        "quarter_1": 22,
                        "quarter_2": 26,
                        "quarter_3": 30,
                        "quarter_4": 24,
                        "over_time": 0,
                        "total": 102,
                    },
                },
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "status": {"short": "FT"},
                "date": "2024-12-01T00:00:00Z",
                "league": {"season": "2024-2025"},
            }
        ]
        count = await client.persist_games(games, mock_db)
        assert count == 1
        mock_db.execute.assert_awaited()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_game_with_no_quarter_scores(self, client, mock_db):
        """Games with None quarter scores (unplayed) are handled."""
        games = [
            {
                "id": 2002,
                "scores": {"home": {}, "away": {}},
                "teams": {"home": {"id": 3}, "away": {"id": 4}},
                "status": {"short": "NS"},
                "date": "2025-01-01T00:00:00Z",
                "league": {"season": "2024-2025"},
            }
        ]
        count = await client.persist_games(games, mock_db)
        assert count == 1


# ── persist_team_season_stats ───────────────────────────────────


class TestPersistTeamSeasonStats:
    @pytest.mark.anyio
    async def test_full_stats_persist(self, client, mock_db):
        stats = {
            "games": {"played": {"all": 82}},
            "points": {
                "for": {
                    "total": {"all": "9020"},
                    "average": {"all": "110.0"},
                },
                "against": {
                    "total": {"all": "8856"},
                    "average": {"all": "108.0"},
                },
            },
            "field_goals": {
                "total": {"all": "3200"},
                "percentage": {"all": "46.5"},
            },
            "free_throws": {
                "total": {"all": "1500"},
                "percentage": {"all": "78.0"},
            },
            "rebounds": {"offReb": {"all": "900"}},
            "turnovers": {"total": {"all": "1100"}},
        }
        await client.persist_team_season_stats(1, stats, "2024-2025", mock_db)
        mock_db.execute.assert_awaited()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_empty_stats_returns_early(self, client, mock_db):
        await client.persist_team_season_stats(1, {}, "2024-2025", mock_db)
        mock_db.execute.assert_not_awaited()

    @pytest.mark.anyio
    async def test_none_stats_returns_early(self, client, mock_db):
        await client.persist_team_season_stats(1, None, "2024-2025", mock_db)
        mock_db.execute.assert_not_awaited()

    @pytest.mark.anyio
    async def test_list_stats_normalized(self, client, mock_db):
        """List payload is normalized to first element."""
        stats = [
            {
                "games": {"played": {"all": 10}},
                "points": {
                    "for": {"total": {"all": "1100"}, "average": {"all": "110.0"}},
                    "against": {"total": {"all": "1050"}, "average": {"all": "105.0"}},
                },
            }
        ]
        await client.persist_team_season_stats(1, stats, "2024-2025", mock_db)
        mock_db.execute.assert_awaited()

    @pytest.mark.anyio
    async def test_unexpected_type_warns(self, client, mock_db):
        """Non-dict, non-list stats log warning and return."""
        await client.persist_team_season_stats(1, "bad", "2024-2025", mock_db)
        mock_db.execute.assert_not_awaited()


# ── persist_player_game_stats ───────────────────────────────────


class TestPersistPlayerGameStats:
    @pytest.mark.anyio
    async def test_upserts_player_stats(self, client, mock_db):
        # Player exists
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 100
        mock_db.execute = AsyncMock(return_value=mock_result)

        stats_data = [
            {
                "player": {"id": 100, "name": "LeBron James"},
                "team": {"id": 1},
                "points": 30,
                "assists": 8,
                "steals": 2,
                "blocks": 1,
                "turnovers": 3,
                "minutes": "36:22",
                "plusMinus": "12.5",
                "rebounds": {"total": 10},
                "field_goals": {"percentage": "52.4"},
                "threepoint_goals": {"percentage": "38.5"},
                "freethrows_goals": {"percentage": "85.0"},
            }
        ]
        count = await client.persist_player_game_stats(1001, stats_data, mock_db)
        assert count == 1
        mock_db.commit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_creates_new_player(self, client, mock_db):
        """If player doesn't exist in DB, creates player row first."""
        # First execute = player check → None; second = upsert stats
        call_count = 0

        async def side_effect(stmt, *a, **kw):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 1:
                r.scalar_one_or_none.return_value = None  # Player not found
            return r

        mock_db.execute = side_effect
        mock_db.flush = AsyncMock()

        stats_data = [
            {
                "player": {"id": 999, "name": "New Player"},
                "team": {"id": 2},
                "points": 5,
                "rebounds": {"total": 2},
            }
        ]
        count = await client.persist_player_game_stats(2001, stats_data, mock_db)
        assert count == 1
        mock_db.add.assert_called_once()  # Player was created
        mock_db.flush.assert_awaited()

    @pytest.mark.anyio
    async def test_skips_entry_without_player_id(self, client, mock_db):
        stats_data = [
            {"player": {}, "team": {"id": 1}, "points": 10}
        ]
        count = await client.persist_player_game_stats(3001, stats_data, mock_db)
        assert count == 0

    @pytest.mark.anyio
    async def test_handles_none_stat_values(self, client, mock_db):
        """None/empty stat values are safely handled."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 50
        mock_db.execute = AsyncMock(return_value=mock_result)

        stats_data = [
            {
                "player": {"id": 50, "name": "P"},
                "team": {"id": 1},
                "points": None,
                "assists": "",
                "minutes": None,
                "rebounds": {},
                "field_goals": None,
                "threepoint_goals": None,
                "freethrows_goals": None,
            }
        ]
        count = await client.persist_player_game_stats(4001, stats_data, mock_db)
        assert count == 1

    @pytest.mark.anyio
    async def test_invalid_stat_strings_are_coerced_to_none(self, client, mock_db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 51
        mock_db.execute = AsyncMock(return_value=mock_result)

        stats_data = [
            {
                "player": {"id": 51, "name": "P"},
                "team": {"id": 1},
                "points": "bad-int",
                "assists": "oops",
                "minutes": "bad:minutes",
                "plusMinus": "bad-float",
                "rebounds": {"total": "nan-but-text"},
                "field_goals": {"percentage": "not-a-pct"},
                "threepoint_goals": {"percentage": "??"},
                "freethrows_goals": {"percentage": object()},
            }
        ]

        count = await client.persist_player_game_stats(4002, stats_data, mock_db)
        assert count == 1

    @pytest.mark.anyio
    async def test_persist_injuries_empty_name_skipped(self, client, mock_db):
        """Injuries with empty team/player names are skipped."""
        mock_db.begin_nested = MagicMock(return_value=AsyncMock())
        mock_db.execute = AsyncMock()  # delete only
        injuries_data = [
            {
                "player": {"firstname": "", "lastname": ""},
                "team": {"name": "Lakers"},
                "status": {"type": "Out"},
            },
            {
                "player": {"firstname": "John", "lastname": "Doe"},
                "team": {"name": ""},
                "status": {"type": "Out"},
            },
        ]
        await client.persist_injuries(injuries_data, mock_db)
        mock_db.add.assert_not_called()
