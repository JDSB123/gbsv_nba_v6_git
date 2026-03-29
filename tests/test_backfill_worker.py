"""Tests for backfill.py (run_backfill) and worker.py shim."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

# ── worker.py shim ───────────────────────────────────────────────


class TestWorkerShim:
    def test_main_calls_cmd_work(self):
        with patch("src.worker.cmd_work") as mock_cmd:
            from src.worker import main

            main()
            mock_cmd.assert_called_once()


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
