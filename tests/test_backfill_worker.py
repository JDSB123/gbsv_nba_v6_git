"""Tests for backfill.py (run_backfill)."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

# ── backfill.py ──────────────────────────────────────────────────


class TestRunBackfill:
    @patch("src.data.backfill.async_session_factory")
    @patch("src.data.backfill.resolve_backfill_window")
    async def test_full_pipeline_runs(self, mock_window, mock_session):
        """run_backfill orchestrates all 7 steps without error."""
        from src.data.backfill import run_backfill

        mock_window.return_value = ("2024-2025", date(2024, 10, 1), date(2024, 10, 3))

        mock_bball = MagicMock()
        mock_bball.fetch_standings = AsyncMock(return_value=[])
        mock_bball.persist_teams = AsyncMock()
        mock_bball.fetch_games = AsyncMock(return_value=[])
        mock_bball.persist_games = AsyncMock(return_value=0)
        mock_bball.fetch_team_stats = AsyncMock(return_value={})
        mock_bball.persist_team_season_stats = AsyncMock()
        mock_bball.fetch_player_stats = AsyncMock(return_value=[])
        mock_bball.persist_player_game_stats = AsyncMock()
        mock_bball.fetch_injuries = AsyncMock(return_value=[])
        mock_bball.persist_injuries = AsyncMock(return_value=0)

        mock_odds = MagicMock()
        mock_odds.fetch_events = AsyncMock(return_value=[])
        mock_odds.fetch_odds = AsyncMock(return_value=[])
        mock_odds.persist_odds = AsyncMock(return_value=0)
        mock_odds.fetch_event_odds = AsyncMock(return_value=None)

        mock_db = AsyncMock()
        team_result = MagicMock()
        team_result.fetchall.return_value = [(1,), (2,)]
        game_result = MagicMock()
        game_result.fetchall.return_value = []
        game_match = MagicMock()
        game_match.scalar_one_or_none.return_value = None

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            sql = str(stmt).lower()
            if "teams" in sql and "select" in sql:
                result.fetchall.return_value = [(1,), (2,)]
                return result
            if "games" in sql and "select" in sql:
                result.fetchall.return_value = []
                result.scalar_one_or_none.return_value = None
                return result
            result.fetchall.return_value = []
            result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.data.backfill.BasketballClient", return_value=mock_bball),
            patch("src.data.backfill.OddsClient", return_value=mock_odds),
        ):
            await run_backfill(season="2024-2025", days_back=3)

        mock_bball.fetch_standings.assert_awaited_once()

    @patch("src.data.backfill.async_session_factory")
    @patch("src.data.backfill.resolve_backfill_window")
    async def test_handles_odds_failure(self, mock_window, mock_session):
        """Backfill continues even if odds sync fails."""
        from src.data.backfill import run_backfill

        mock_window.return_value = ("2024-2025", date(2024, 10, 1), date(2024, 10, 1))

        mock_bball = MagicMock()
        mock_bball.fetch_standings = AsyncMock(return_value=[])
        mock_bball.persist_teams = AsyncMock()
        mock_bball.fetch_games = AsyncMock(return_value=[])
        mock_bball.persist_games = AsyncMock(return_value=0)
        mock_bball.fetch_team_stats = AsyncMock(return_value={})
        mock_bball.persist_team_season_stats = AsyncMock()
        mock_bball.fetch_player_stats = AsyncMock(return_value=[])
        mock_bball.persist_player_game_stats = AsyncMock()
        mock_bball.fetch_injuries = AsyncMock(return_value=[])
        mock_bball.persist_injuries = AsyncMock(return_value=0)

        mock_odds = MagicMock()
        mock_odds.fetch_events = AsyncMock(side_effect=RuntimeError("API down"))
        mock_odds.fetch_odds = AsyncMock(side_effect=RuntimeError("API down"))

        mock_db = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = []
        result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result)
        mock_db.commit = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.data.backfill.BasketballClient", return_value=mock_bball),
            patch("src.data.backfill.OddsClient", return_value=mock_odds),
        ):
            # Should not raise — backfill catches odds errors
            await run_backfill(season="2024-2025", days_back=1)

    @patch("src.data.backfill.async_session_factory")
    @patch("src.data.backfill.resolve_backfill_window")
    async def test_with_standings_data(self, mock_window, mock_session):
        """run_backfill persists teams when standings return data."""
        from src.data.backfill import run_backfill

        mock_window.return_value = ("2024-2025", date(2024, 10, 1), date(2024, 10, 1))

        mock_bball = MagicMock()
        mock_bball.fetch_standings = AsyncMock(return_value=[{"team": {"id": 1, "name": "BOS"}}])
        mock_bball.persist_teams = AsyncMock()
        mock_bball.fetch_games = AsyncMock(return_value=[])
        mock_bball.persist_games = AsyncMock(return_value=0)
        mock_bball.fetch_team_stats = AsyncMock(return_value={})
        mock_bball.persist_team_season_stats = AsyncMock()
        mock_bball.fetch_player_stats = AsyncMock(return_value=[])
        mock_bball.persist_player_game_stats = AsyncMock()
        mock_bball.fetch_injuries = AsyncMock(return_value=[])
        mock_bball.persist_injuries = AsyncMock(return_value=0)

        mock_odds = MagicMock()
        mock_odds.fetch_events = AsyncMock(return_value=[])
        mock_odds.fetch_odds = AsyncMock(return_value=[])
        mock_odds.persist_odds = AsyncMock(return_value=0)

        mock_db = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = []
        result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result)
        mock_db.commit = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.data.backfill.BasketballClient", return_value=mock_bball),
            patch("src.data.backfill.OddsClient", return_value=mock_odds),
        ):
            await run_backfill(season="2024-2025", days_back=1)

        mock_bball.persist_teams.assert_awaited_once()

    @patch("src.data.backfill.async_session_factory")
    @patch("src.data.backfill.resolve_backfill_window")
    async def test_happy_path_all_data(self, mock_window, mock_session):
        """Covers steps 2-7 happy paths: games, stats, box scores, odds sync, odds persist, injuries."""
        from src.data.backfill import run_backfill

        mock_window.return_value = ("2024-2025", date(2024, 10, 1), date(2024, 10, 1))

        mock_bball = MagicMock()
        mock_bball.fetch_standings = AsyncMock(return_value=[{"team": {"id": 1}}])
        mock_bball.persist_teams = AsyncMock()
        mock_bball.fetch_games = AsyncMock(return_value=[{"id": 99}])
        mock_bball.persist_games = AsyncMock(return_value=2)
        mock_bball.fetch_team_stats = AsyncMock(return_value={"ppg": 110})
        mock_bball.persist_team_season_stats = AsyncMock()
        mock_bball.fetch_player_stats = AsyncMock(return_value=[{"player": "A"}])
        mock_bball.persist_player_game_stats = AsyncMock()
        mock_bball.fetch_injuries = AsyncMock(return_value=[{"player": "X"}])
        mock_bball.persist_injuries = AsyncMock(return_value=3)

        mock_odds = MagicMock()
        mock_odds.fetch_events = AsyncMock(
            return_value=[
                {"id": "evt1", "commence_time": "2024-10-01T00:00:00Z"},
                {},  # no commence_time — skipped
            ]
        )
        mock_odds.fetch_odds = AsyncMock(return_value=[{"id": "evt1"}])
        mock_odds.persist_odds = AsyncMock(return_value=5)
        mock_odds.fetch_event_odds = AsyncMock(
            return_value={"bookmakers": [{"key": "bk"}]}
        )

        # Build a mock game with odds_api_id = None so backfill maps it
        from types import SimpleNamespace
        mock_game = SimpleNamespace(odds_api_id=None)

        call_count = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_count["n"] += 1
            sql = str(stmt).lower()
            result = MagicMock()
            if "teams" in sql:
                result.fetchall.return_value = [(1,)]
                return result
            if "where games.commence_time" in sql:
                # Step 5: game lookup by commence_time (full Game select)
                result.scalar_one_or_none.return_value = mock_game
                return result
            if "status" in sql:
                # Step 4: finished game ids (Game.id where status='FT')
                result.fetchall.return_value = [(100,)]
                return result
            result.fetchall.return_value = []
            result.scalar_one_or_none.return_value = None
            return result

        mock_db = AsyncMock()
        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.data.backfill.BasketballClient", return_value=mock_bball),
            patch("src.data.backfill.OddsClient", return_value=mock_odds),
            patch("src.data.backfill.parse_api_datetime") as mock_parse,
        ):
            mock_parse.return_value = "2024-10-01T00:00:00"
            await run_backfill(season="2024-2025", days_back=1)

        # Step 2: games persisted
        mock_bball.persist_games.assert_awaited()
        # Step 3: team stats persisted
        mock_bball.persist_team_season_stats.assert_awaited()
        # Step 4: player box scores
        mock_bball.persist_player_game_stats.assert_awaited()
        # Step 5: odds_api_id was mapped
        assert mock_game.odds_api_id == "evt1"
        # Step 6: odds persisted
        mock_odds.persist_odds.assert_awaited()
        # Step 6: 1H odds fetched
        mock_odds.fetch_event_odds.assert_awaited()
        # Step 7: injuries persisted
        mock_bball.persist_injuries.assert_awaited_once()

    @patch("src.data.backfill.async_session_factory")
    @patch("src.data.backfill.resolve_backfill_window")
    async def test_injury_fetch_failure(self, mock_window, mock_session):
        """Backfill continues when injury fetch raises."""
        from src.data.backfill import run_backfill

        mock_window.return_value = ("2024-2025", date(2024, 10, 1), date(2024, 10, 1))

        mock_bball = MagicMock()
        mock_bball.fetch_standings = AsyncMock(return_value=[])
        mock_bball.persist_teams = AsyncMock()
        mock_bball.fetch_games = AsyncMock(return_value=[])
        mock_bball.persist_games = AsyncMock(return_value=0)
        mock_bball.fetch_team_stats = AsyncMock(return_value={})
        mock_bball.persist_team_season_stats = AsyncMock()
        mock_bball.fetch_player_stats = AsyncMock(return_value=[])
        mock_bball.persist_player_game_stats = AsyncMock()
        mock_bball.fetch_injuries = AsyncMock(side_effect=RuntimeError("API fail"))
        mock_bball.persist_injuries = AsyncMock()

        mock_odds = MagicMock()
        mock_odds.fetch_events = AsyncMock(return_value=[])
        mock_odds.fetch_odds = AsyncMock(return_value=[])
        mock_odds.persist_odds = AsyncMock(return_value=0)
        mock_odds.fetch_event_odds = AsyncMock(return_value=None)

        mock_db = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = []
        result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result)
        mock_db.commit = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.data.backfill.BasketballClient", return_value=mock_bball),
            patch("src.data.backfill.OddsClient", return_value=mock_odds),
        ):
            await run_backfill(season="2024-2025", days_back=1)  # should not raise
