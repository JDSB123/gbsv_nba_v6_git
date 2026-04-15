"""Tests for __main__.py: cmd_work, cmd_train, cmd_audit, main() parser,
worker.py shim, and config.py edge cases."""

from __future__ import annotations

import argparse
import asyncio
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MAIN_MOD = "src.__main__"


class TestCmdWork:
    def test_cmd_work_starts_scheduler(self):
        """cmd_work should create scheduler, start it, and run event loop."""
        with (
            patch(f"{_MAIN_MOD}._setup_logging"),
            patch(f"{_MAIN_MOD}.asyncio.run") as mock_run,
        ):
            from src.__main__ import cmd_work

            cmd_work(argparse.Namespace())
            mock_run.assert_called_once()

    def test_cmd_work_runs_scheduler_shutdown_path(self):
        real_run = asyncio.run
        scheduler = MagicMock()
        scheduler.get_jobs.return_value = [1, 2]

        stop_event = MagicMock()
        stop_event.wait = AsyncMock(side_effect=KeyboardInterrupt)
        stop_event.set = MagicMock()

        loop = MagicMock()

        def _add_signal_handler(sig, callback):
            if loop.add_signal_handler.call_count == 0:
                raise NotImplementedError()
            callback()

        loop.add_signal_handler = MagicMock(side_effect=_add_signal_handler)

        with (
            patch(f"{_MAIN_MOD}._setup_logging"),
            patch("src.data.scheduler.create_scheduler", return_value=scheduler),
            patch("src.data.health_check.run_startup_health_check", new_callable=AsyncMock),
            patch(f"{_MAIN_MOD}.asyncio.Event", return_value=stop_event),
            patch(f"{_MAIN_MOD}.asyncio.get_running_loop", return_value=loop),
            patch(f"{_MAIN_MOD}.asyncio.run", side_effect=lambda coro: real_run(coro)),
        ):
            from src.__main__ import cmd_work

            cmd_work(argparse.Namespace())

        scheduler.start.assert_called_once()
        scheduler.shutdown.assert_called_once_with(wait=True)
        assert loop.add_signal_handler.call_count == 2
        assert stop_event.set.call_count >= 1


class TestCmdTrain:
    def test_cmd_train_invokes_asyncio_run(self):
        with (
            patch(f"{_MAIN_MOD}._setup_logging"),
            patch(f"{_MAIN_MOD}.asyncio.run") as mock_run,
        ):
            from src.__main__ import cmd_train

            cmd_train(argparse.Namespace())
            mock_run.assert_called_once()


class TestRunTrain:
    @pytest.mark.anyio
    async def test_run_train_with_metrics(self, capsys):
        mock_trainer = MagicMock()
        mock_trainer.train = AsyncMock(return_value={"mae": 5.0, "rmse": 7.0})

        mock_db = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_ctx)

        with (
            patch("src.db.session.async_session_factory", mock_factory),
            patch("src.models.trainer.ModelTrainer", return_value=mock_trainer),
        ):
            from src.__main__ import _run_train

            await _run_train()

        captured = capsys.readouterr()
        assert "Training complete" in captured.out

    @pytest.mark.anyio
    async def test_run_train_no_metrics(self, capsys):
        mock_trainer = MagicMock()
        mock_trainer.train = AsyncMock(return_value=None)

        mock_db = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_ctx)

        with (
            patch("src.db.session.async_session_factory", mock_factory),
            patch("src.models.trainer.ModelTrainer", return_value=mock_trainer),
        ):
            from src.__main__ import _run_train

            await _run_train()

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()


class TestRunPredict:
    @pytest.mark.anyio
    async def test_run_predict_not_ready(self, capsys):
        mock_predictor = MagicMock()
        mock_predictor.is_ready = False
        mock_db = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_ctx)

        with (
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.db.session.async_session_factory", mock_factory),
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
        ):
            from src.__main__ import _run_predict

            await _run_predict()

        captured = capsys.readouterr()
        assert "not loaded" in captured.out.lower()

    @pytest.mark.anyio
    async def test_run_predict_with_predictions(self, capsys):
        mock_predictor = MagicMock()
        mock_predictor.is_ready = True
        mock_predictor.predict_upcoming = AsyncMock(return_value=[1, 2, 3])

        mock_db = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_ctx)

        with (
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.db.session.async_session_factory", mock_factory),
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
            from src.__main__ import _run_predict

            await _run_predict()

        captured = capsys.readouterr()
        assert "3 predictions" in captured.out.lower()
        assert "waiting on odds coverage" in captured.out.lower()


class TestRunPublishTeams:
    @pytest.mark.anyio
    async def test_run_publish_not_configured(self, capsys):
        mock_settings = MagicMock()
        mock_settings.teams_team_id = ""
        mock_settings.teams_channel_id = ""
        mock_settings.teams_webhook_url = ""

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.__main__ import _run_publish_teams

            await _run_publish_teams()

        captured = capsys.readouterr()
        assert "not configured" in captured.out.lower()

    @pytest.mark.anyio
    async def test_run_publish_with_webhook(self, capsys):
        mock_settings = MagicMock()
        mock_settings.teams_team_id = ""
        mock_settings.teams_channel_id = ""
        mock_settings.teams_webhook_url = "https://hook.example.com"

        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch(
                "src.data.scheduler.generate_predictions_and_publish",
                new_callable=AsyncMock,
                return_value=8,
            ),
        ):
            from src.__main__ import _run_publish_teams

            await _run_publish_teams()

        captured = capsys.readouterr()
        assert "executed" in captured.out.lower()
        assert "8 predictions" in captured.out.lower()


class TestRunAuditEmpty:
    @pytest.mark.anyio
    async def test_audit_empty_model_registry(self, capsys):
        """Cover the '(empty)' print branch in _run_audit."""
        mock_db = AsyncMock()

        async def _exec(stmt, *a, **kw):
            result = MagicMock()
            result.scalar = MagicMock(return_value=0)
            result.scalar_one_or_none = MagicMock(return_value=None)
            result.all = MagicMock(return_value=[])
            result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
            return result

        mock_db.execute = AsyncMock(side_effect=_exec)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_ctx)

        with patch("src.db.session.async_session_factory", mock_factory):
            from src.__main__ import _run_audit

            await _run_audit()

        captured = capsys.readouterr()
        assert "(empty)" in captured.out


class TestMainParser:
    def test_main_calls_func(self):
        """Cover main() which parses args and calls func."""
        with (
            patch("sys.argv", ["src", "work"]),
            patch(f"{_MAIN_MOD}.cmd_work") as mock_work,
        ):
            from src.__main__ import main

            main()
            mock_work.assert_called_once()


class TestMainEntrypoint:
    def test_module_main_guard_invokes_main(self):
        mock_func = MagicMock()

        with patch(
            "argparse.ArgumentParser.parse_args", return_value=SimpleNamespace(func=mock_func)
        ):
            runpy.run_module("src.__main__", run_name="__main__")

        mock_func.assert_called_once()


class TestConfigEnvFiles:
    def test_settings_use_single_repo_env_file(self):
        from src.config import Settings

        # .env is the single runtime env file. settings_customise_sources still
        # excludes os.environ and dotenv sources from field loading; this just
        # declares the canonical path for the repo contract.
        assert Settings.model_config["env_file"] == ".env"

    def test_settings_ignore_app_env_specific_env_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        (tmp_path / ".env").write_text("ODDS_API_KEY=base\nBASKETBALL_API_KEY=base\n")
        (tmp_path / ".env.staging").write_text("ODDS_API_KEY=staging\n")
        monkeypatch.chdir(tmp_path)

        from src.config import Settings

        settings = Settings(app_env="test")
        assert settings.odds_api_key == ""


class TestConfigRequiredSecrets:
    def test_missing_required_secrets_raises(self, monkeypatch):
        """Cover line 101: validation error for missing secrets."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ODDS_API_KEY", "")
        monkeypatch.setenv("BASKETBALL_API_KEY", "")
        monkeypatch.setenv("DATABASE_URL", "")

        from src.config import Settings

        with pytest.raises(Exception, match="Missing required env vars"):
            Settings(
                app_env="production",
                odds_api_key="",
                basketball_api_key="",
                database_url="",
            )


class TestResolveDatabaseUrl:
    def test_resolve_database_url_from_env(self, monkeypatch):
        """DATABASE_URL from os.environ must not override .env/defaults."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://custom/db")
        monkeypatch.setattr("src.config.load_selected_env_values", lambda: {})
        from src.config import resolve_database_url

        result = resolve_database_url()
        assert result != "postgresql+asyncpg://custom/db"

    def test_resolve_database_url_from_settings(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.config import resolve_database_url

        result = resolve_database_url()
        assert "postgresql" in result
