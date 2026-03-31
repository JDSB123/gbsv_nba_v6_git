"""Extended scheduler tests for additional job coverage."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.scheduler import (
    check_prediction_drift,
    daily_retrain,
    db_maintenance,
    generate_predictions_and_publish,
    poll_injuries,
    pregame_check,
    prune_old_odds,
)

_POLL = "src.data.jobs.polling"
_PRED = "src.data.jobs.predictions"
_MAINT = "src.data.jobs.maintenance"


# -- daily_retrain --


class TestDailyRetrain:
    @pytest.mark.anyio
    async def test_retrain_success(self):
        with patch(f"{_POLL}.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.models.trainer.ModelTrainer") as mock_cls:
                mock_trainer = MagicMock()
                mock_trainer.train = AsyncMock(return_value={"rmse": 5.0})
                mock_cls.return_value = mock_trainer
                await daily_retrain()
                mock_trainer.train.assert_called_once()

    @pytest.mark.anyio
    async def test_retrain_failure_records(self):
        with patch(f"{_POLL}.async_session_factory") as mock_sf:
            mock_sf.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("db error")
            )
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.models.trainer.ModelTrainer") as mock_cls:
                mock_cls.side_effect = RuntimeError("boom")
                with (
                    patch(f"{_POLL}._record_failure", new_callable=AsyncMock) as mock_rec,
                    patch("src.notifications.teams.send_alert", new_callable=AsyncMock),
                ):
                    await daily_retrain()
                    mock_rec.assert_called_once()


# -- poll_injuries --


class TestPollInjuries:
    @pytest.mark.anyio
    async def test_injuries_success(self):
        with patch("src.data.basketball_client.BasketballClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.fetch_injuries = AsyncMock(return_value=[{"player": "test"}])
            mock_client.persist_injuries = AsyncMock(return_value=5)
            mock_cls.return_value = mock_client
            with patch(f"{_POLL}.async_session_factory") as mock_sf:
                mock_db = AsyncMock()
                mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
                await poll_injuries()
                mock_client.fetch_injuries.assert_called_once()

    @pytest.mark.anyio
    async def test_injuries_none(self):
        with patch("src.data.basketball_client.BasketballClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.fetch_injuries = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client
            await poll_injuries()


# -- pregame_check --


class TestPregameCheck:
    @pytest.mark.anyio
    async def test_already_published_today(self):
        import src.data.jobs.predictions as pmod

        today = datetime.now().date()
        original = pmod._pregame_published_date
        pmod._pregame_published_date = today
        try:
            await pregame_check()  # Should return immediately
        finally:
            pmod._pregame_published_date = original

    @pytest.mark.anyio
    async def test_no_upcoming_games(self):
        import src.data.jobs.predictions as pmod

        original = pmod._pregame_published_date
        pmod._pregame_published_date = None
        try:
            with patch(f"{_PRED}.async_session_factory") as mock_sf:
                mock_db = AsyncMock()
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_db.execute = AsyncMock(return_value=mock_result)
                mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
                await pregame_check()
        finally:
            pmod._pregame_published_date = original


# -- generate_predictions_and_publish --


class TestGeneratePredictionsAndPublish:
    @pytest.mark.anyio
    async def test_predictor_not_ready(self):
        with (
            patch(f"{_POLL}.poll_stats", new_callable=AsyncMock),
            patch(f"{_POLL}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_POLL}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_POLL}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_POLL}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_POLL}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_POLL}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_PRED}.async_session_factory") as mock_sf,
            patch(f"{_PRED}.purge_invalid_upcoming_predictions", new_callable=AsyncMock, return_value=0),
            patch("src.models.predictor.Predictor") as mock_cls,
            patch("src.models.features.reset_elo_cache"),
            patch(f"{_PRED}.get_settings") as mock_s,
        ):
            mock_db = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = MagicMock(is_ready=False)
            mock_s.return_value = MagicMock()
            await generate_predictions_and_publish()
            # Should return without publishing


# -- check_prediction_drift --


class TestCheckPredictionDrift:
    @pytest.mark.anyio
    async def test_not_enough_data(self):
        with patch(f"{_PRED}.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.all.return_value = []
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await check_prediction_drift()

    @pytest.mark.anyio
    async def test_no_drift(self):
        rows_30d = [(110.0, 108.0)] * 30
        rows_7d = [(110.0, 108.0)] * 10

        with patch(f"{_PRED}.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            call_count = 0

            async def mock_execute(query):
                nonlocal call_count
                call_count += 1
                mock_result = MagicMock()
                if call_count == 1:
                    mock_result.all.return_value = rows_30d
                else:
                    mock_result.all.return_value = rows_7d
                return mock_result

            mock_db.execute = mock_execute
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await check_prediction_drift()

    @pytest.mark.anyio
    async def test_drift_detected(self):
        rows_30d = [(110.0, 108.0)] * 30  # total ~218
        rows_7d = [(120.0, 115.0)] * 10   # total ~235, drift > 5

        with patch(f"{_PRED}.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            call_count = 0

            async def mock_execute(query):
                nonlocal call_count
                call_count += 1
                mock_result = MagicMock()
                if call_count == 1:
                    mock_result.all.return_value = rows_30d
                else:
                    mock_result.all.return_value = rows_7d
                return mock_result

            mock_db.execute = mock_execute
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.notifications.teams.send_alert", new_callable=AsyncMock):
                await check_prediction_drift()


# -- prune_old_odds --


class TestPruneOldOdds:
    @pytest.mark.anyio
    async def test_prune_runs(self):
        with patch(f"{_MAINT}.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 42
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_db.commit = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await prune_old_odds()


# -- db_maintenance --


class TestDbMaintenance:
    @pytest.mark.anyio
    async def test_maintenance_runs(self):
        with patch(f"{_MAINT}.async_session_factory") as mock_sf:
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock()
            mock_db.commit = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await db_maintenance()
            # 4 ANALYZE calls + 1 commit
            assert mock_db.execute.call_count == 4

    @pytest.mark.anyio
    async def test_maintenance_error_handled(self):
        with patch(f"{_MAINT}.async_session_factory") as mock_sf:
            mock_sf.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            # Should not raise
            await db_maintenance()
