"""Tests for __main__.py CLI routing and setup functions."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.__main__ import _setup_logging, main

# ── _setup_logging ───────────────────────────────────────────────


class TestSetupLogging:
    def test_dev_env_uses_basic_config(self):
        original_level = logging.root.level
        original_handlers = logging.root.handlers[:]
        try:
            logging.root.handlers.clear()
            with patch.dict(os.environ, {"APP_ENV": "development"}, clear=False):
                _setup_logging("DEBUG")
            assert logging.root.level == logging.DEBUG
        finally:
            logging.root.setLevel(original_level)
            logging.root.handlers = original_handlers

    def test_production_env_uses_json_formatter(self):
        original_level = logging.root.level
        original_handlers = logging.root.handlers[:]
        try:
            with patch.dict(os.environ, {"APP_ENV": "production"}, clear=False):
                _setup_logging("WARNING")
            assert logging.root.level == logging.WARNING
        finally:
            logging.root.setLevel(original_level)
            logging.root.handlers = original_handlers

    def test_default_level_from_settings(self):
        original_level = logging.root.level
        original_handlers = logging.root.handlers[:]
        try:
            logging.root.handlers.clear()
            with patch("src.__main__.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(log_level="ERROR")
                with patch.dict(os.environ, {"APP_ENV": "test"}, clear=False):
                    _setup_logging()
                assert logging.root.level == logging.ERROR
        finally:
            logging.root.setLevel(original_level)
            logging.root.handlers = original_handlers


# ── CLI arg parsing ──────────────────────────────────────────────


class TestMainArgParsing:
    @patch("src.__main__._setup_logging")
    @patch("src.__main__.get_settings")
    def test_serve_command(self, mock_settings, mock_log):
        mock_settings.return_value = MagicMock(app_env="development")
        with (
            patch("sys.argv", ["src", "serve", "--port", "9000"]),
            patch("uvicorn.run") as mock_uvicorn,
        ):
            main()
            mock_uvicorn.assert_called_once()
            call_kwargs = mock_uvicorn.call_args
            assert call_kwargs[1]["port"] == 9000 or call_kwargs[0][0] == "src.api.main:app"

    @patch("src.__main__._setup_logging")
    def test_train_command(self, mock_log):
        with patch("sys.argv", ["src", "train"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_predict_command(self, mock_log):
        with patch("sys.argv", ["src", "predict"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_backfill_command_with_season(self, mock_log):
        with (
            patch("sys.argv", ["src", "backfill", "--season", "2024-2025", "--days", "30"]),
            patch("asyncio.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_migrate_command(self, mock_log):
        with (
            patch("sys.argv", ["src", "migrate"]),
            patch("alembic.command.upgrade") as mock_upgrade,
        ):
            main()
            mock_upgrade.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_sync_command(self, mock_log):
        with patch("sys.argv", ["src", "sync"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_odds_command(self, mock_log):
        with patch("sys.argv", ["src", "odds"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_perf_command(self, mock_log):
        with patch("sys.argv", ["src", "perf"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_audit_command(self, mock_log):
        with patch("sys.argv", ["src", "audit"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_publish_teams_command(self, mock_log):
        with patch("sys.argv", ["src", "publish-teams"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    def test_missing_command_exits(self):
        with patch("sys.argv", ["src"]), pytest.raises(SystemExit):
            main()


# ── Async runners ────────────────────────────────────────────────


class TestRunTrain:
    @pytest.mark.anyio
    async def test_train_complete(self):
        from src.__main__ import _run_train

        mock_trainer = MagicMock()
        mock_trainer.train = AsyncMock(return_value={"rmse": 5.0, "accuracy": 0.65})

        with patch("src.db.session.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.models.trainer.ModelTrainer", return_value=mock_trainer):
                await _run_train()
                mock_trainer.train.assert_called_once()

    @pytest.mark.anyio
    async def test_train_skipped(self):
        from src.__main__ import _run_train

        mock_trainer = MagicMock()
        mock_trainer.train = AsyncMock(return_value=None)

        with patch("src.db.session.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.models.trainer.ModelTrainer", return_value=mock_trainer):
                await _run_train()


class TestRunPredict:
    @pytest.mark.anyio
    async def test_predict_not_ready(self, capsys):
        from src.__main__ import _run_predict

        mock_predictor = MagicMock()
        mock_predictor.is_ready = False
        mock_db = AsyncMock()

        with (
            patch("src.data.scheduler.poll_stats", new_callable=AsyncMock),
            patch("src.data.scheduler.poll_scores_and_box", new_callable=AsyncMock),
            patch("src.data.scheduler.sync_events_to_games", new_callable=AsyncMock),
            patch("src.data.scheduler.poll_fg_odds", new_callable=AsyncMock),
            patch("src.data.scheduler.poll_1h_odds", new_callable=AsyncMock),
            patch("src.data.scheduler.poll_player_props", new_callable=AsyncMock),
            patch(
                "src.data.scheduler.purge_invalid_upcoming_predictions",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.db.session.async_session_factory") as mock_sf,
        ):
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await _run_predict()
            output = capsys.readouterr().out
            assert "not loaded" in output.lower() or "not ready" in output.lower()

    @pytest.mark.anyio
    async def test_predict_success(self, capsys):
        from src.__main__ import _run_predict

        mock_predictor = MagicMock()
        mock_predictor.is_ready = True
        mock_predictor.predict_upcoming = AsyncMock(return_value=[1, 2, 3])

        with (
            patch("src.data.scheduler.poll_stats", new_callable=AsyncMock) as mock_stats,
            patch("src.data.scheduler.poll_scores_and_box", new_callable=AsyncMock) as mock_box,
            patch("src.data.scheduler.sync_events_to_games", new_callable=AsyncMock) as mock_sync,
            patch("src.data.scheduler.poll_fg_odds", new_callable=AsyncMock) as mock_fg,
            patch("src.data.scheduler.poll_1h_odds", new_callable=AsyncMock) as mock_h1,
            patch("src.data.scheduler.poll_player_props", new_callable=AsyncMock) as mock_props,
            patch(
                "src.data.scheduler.purge_invalid_upcoming_predictions",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.db.session.async_session_factory") as mock_sf,
            patch(
                "src.__main__._summarize_upcoming_coverage",
                new_callable=AsyncMock,
                return_value={
                    "ns_game_count": 5,
                    "linked_ns_game_count": 3,
                    "awaiting_odds_games": ["Phoenix Suns @ Orlando Magic (2026-03-31T23:00:00)"],
                },
            ),
        ):
            mock_db = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await _run_predict()
            output = capsys.readouterr().out
            assert "3" in output
            assert "waiting on odds coverage" in output.lower()
        mock_stats.assert_awaited_once()
        mock_box.assert_awaited_once()
        mock_sync.assert_awaited_once()
        mock_fg.assert_awaited_once()
        mock_h1.assert_awaited_once()
        mock_props.assert_awaited_once()


class TestRunPublishTeams:
    @pytest.mark.anyio
    async def test_no_config(self, capsys):
        from src.__main__ import _run_publish_teams

        with patch("src.config.get_settings") as mock_s:
            mock_s.return_value = MagicMock(
                teams_team_id=None,
                teams_channel_id=None,
                teams_webhook_url=None,
            )
            await _run_publish_teams()
            output = capsys.readouterr().out
            assert "not configured" in output.lower()

    @pytest.mark.anyio
    async def test_with_webhook(self, capsys):
        from src.__main__ import _run_publish_teams

        with patch("src.config.get_settings") as mock_s:
            mock_s.return_value = MagicMock(
                teams_team_id=None,
                teams_channel_id=None,
                teams_webhook_url="https://webhook.example.com",
            )
            with patch(
                "src.data.scheduler.generate_predictions_and_publish",
                new_callable=AsyncMock,
                return_value=8,
            ):
                await _run_publish_teams()
                output = capsys.readouterr().out
                assert "executed" in output.lower()
                assert "8 predictions" in output.lower()


# ── Additional async runner tests ────────────────────────────────


class TestRunBackfill:
    @pytest.mark.anyio
    async def test_backfill_delegates(self):
        from src.__main__ import _run_backfill

        with patch("src.data.backfill.run_backfill", new_callable=AsyncMock) as mock_bf:
            await _run_backfill("2024-2025", 30)
            mock_bf.assert_called_once_with(season="2024-2025", days_back=30)


class TestRunSync:
    @pytest.mark.anyio
    async def test_sync_delegates(self):
        from src.__main__ import _run_sync

        with patch("src.data.scheduler.sync_events_to_games", new_callable=AsyncMock) as mock_sync:
            await _run_sync()
            mock_sync.assert_called_once()


class TestRunOdds:
    @pytest.mark.anyio
    async def test_odds_delegates(self):
        from src.__main__ import _run_odds

        with patch("src.data.scheduler.poll_fg_odds", new_callable=AsyncMock) as mock_odds:
            await _run_odds()
            mock_odds.assert_called_once()


class TestRunPerf:
    @pytest.mark.anyio
    async def test_perf_no_rows(self, capsys):
        from src.__main__ import _run_perf

        with patch("src.db.session.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.all.return_value = []
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await _run_perf()
            output = capsys.readouterr().out
            assert "no completed" in output.lower()

    @pytest.mark.anyio
    async def test_perf_with_rows(self, capsys):
        from datetime import UTC, datetime

        from src.__main__ import _run_perf

        pred = MagicMock()
        pred.predicted_at = datetime(2024, 3, 15, tzinfo=UTC)
        game = MagicMock()
        game.id = 1

        with patch("src.db.session.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.all.return_value = [(pred, game)]
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            with (
                patch("src.api.routes.performance._grade_game", return_value=[]),
                patch("src.api.routes.performance._score_accuracy", return_value={"fg_ats": 0}),
                patch("src.api.routes.performance._build_stats", return_value={}),
                patch("src.api.routes.performance._clv_summary", return_value={}),
            ):
                await _run_perf()
            output = capsys.readouterr().out
            assert "games_graded" in output


class TestRunAudit:
    @pytest.mark.anyio
    async def test_audit_runs_with_data(self, capsys):
        from datetime import UTC, datetime

        from src.__main__ import _run_audit

        # Build a mock that works as both model registry entry and prediction
        mock_pred = MagicMock()
        # Model registry attributes
        mock_pred.model_version = "v6.0"
        mock_pred.is_active = True
        mock_pred.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        # Prediction attributes
        mock_pred.opening_spread = -3.5
        mock_pred.opening_total = 220.0
        mock_pred.clv_spread = 1.5
        mock_pred.predicted_home_fg = 110.0
        mock_pred.predicted_away_fg = 105.0
        mock_pred.predicted_at = datetime(2024, 3, 15, tzinfo=UTC)
        mock_pred.game = MagicMock()
        mock_pred.game.home_team = MagicMock(name="Celtics")
        mock_pred.game.away_team = MagicMock(name="Heat")
        mock_pred.game.status = "FT"
        mock_pred.game.home_score_fg = 112
        mock_pred.game.away_score_fg = 108
        mock_pred.game.commence_time = datetime(2024, 3, 15, tzinfo=UTC)
        mock_pred.game.odds_api_id = "abc123"
        # Upcoming game attributes (also reuse)
        mock_pred.home_team = MagicMock(name="Lakers")
        mock_pred.away_team = MagicMock(name="Warriors")
        mock_pred.commence_time = datetime(2024, 3, 20, tzinfo=UTC)
        mock_pred.odds_api_id = "xyz789"

        async def mock_execute(query):
            result = MagicMock()
            result.scalar.return_value = 42
            result.scalar_one_or_none.return_value = datetime(2024, 1, 1, tzinfo=UTC)
            result.all.return_value = [("FT", 100), ("NS", 10)]
            # For scalars().all() — use mock_model which has proper attrs.
            # Even if it appears in "recent predictions" loop, its opening_spread
            # and other attrs are MagicMock. To avoid format string issues,
            # make sure it behaves like a prediction too.
            result.scalars.return_value = MagicMock(all=MagicMock(return_value=[mock_pred]))
            return result

        with patch("src.db.session.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_db.execute = mock_execute
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await _run_audit()
            output = capsys.readouterr().out
            assert "DATA AUDIT" in output
            assert "AUDIT COMPLETE" in output
            assert "v6.0" in output


# ── New commands: retrain, backtest, perf --format ────────────────


class TestNewCLICommands:
    @patch("src.__main__._setup_logging")
    def test_retrain_command(self, mock_log):
        with patch("sys.argv", ["src", "retrain"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_retrain_with_season(self, mock_log):
        with (
            patch("sys.argv", ["src", "retrain", "--season", "2024-2025"]),
            patch("asyncio.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_train_with_season(self, mock_log):
        with (
            patch("sys.argv", ["src", "train", "--season", "2024-2025"]),
            patch("asyncio.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_backtest_command(self, mock_log):
        with patch("sys.argv", ["src", "backtest"]), patch("asyncio.run") as mock_run:
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_backtest_with_csv_format(self, mock_log):
        with (
            patch("sys.argv", ["src", "backtest", "--format", "csv", "--output", "report.csv"]),
            patch("asyncio.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()

    @patch("src.__main__._setup_logging")
    def test_perf_with_model_version(self, mock_log):
        with (
            patch("sys.argv", ["src", "perf", "--model-version", "v6.5.0"]),
            patch("asyncio.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()
