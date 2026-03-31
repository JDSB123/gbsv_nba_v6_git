"""Tests targeting scheduler.py exception handler / error paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MOD = "src.data.scheduler"


def _mock_session_factory(mock_db):
    """Create a mock that behaves like async_session_factory() → async context manager."""
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=mock_ctx)


_CB_MOD = "src.data.circuit_breaker"


class TestPollFgOddsException:
    @pytest.mark.anyio
    async def test_poll_fg_odds_exception_records_failure(self):
        mock_breaker = MagicMock()
        mock_breaker.should_skip.return_value = False

        with (
            patch(f"{_CB_MOD}.odds_api_breaker", mock_breaker),
            patch(f"{_MOD}.sync_events_to_games", new_callable=AsyncMock, side_effect=Exception("boom")),
            patch(f"{_MOD}._record_failure", new_callable=AsyncMock) as mock_record,
        ):
            from src.data.scheduler import poll_fg_odds

            await poll_fg_odds()
            mock_breaker.record_failure.assert_called_once()
            mock_record.assert_awaited_once()


class TestPoll1hOddsException:
    @pytest.mark.anyio
    async def test_poll_1h_odds_exception_records_failure(self):
        mock_breaker = MagicMock()
        mock_breaker.should_skip.return_value = False

        with (
            patch(f"{_CB_MOD}.odds_api_breaker", mock_breaker),
            patch(f"{_MOD}.sync_events_to_games", new_callable=AsyncMock, side_effect=Exception("boom")),
            patch(f"{_MOD}._record_failure", new_callable=AsyncMock) as mock_record,
        ):
            from src.data.scheduler import poll_1h_odds

            await poll_1h_odds()
            mock_breaker.record_failure.assert_called_once()
            mock_record.assert_awaited_once()


class TestPollStatsException:
    @pytest.mark.anyio
    async def test_poll_stats_exception_records_failure(self):
        mock_breaker = MagicMock()
        mock_breaker.should_skip.return_value = False
        mock_db = AsyncMock()
        team_result = MagicMock()
        team_result.fetchall.return_value = [(1,)]
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        mock_db.execute = AsyncMock(side_effect=[team_result, count_result])

        with (
            patch(f"{_CB_MOD}.basketball_api_breaker", mock_breaker),
            patch("src.data.basketball_client.BasketballClient") as MockClient,
            patch(f"{_MOD}._record_failure", new_callable=AsyncMock) as mock_record,
            patch(f"{_MOD}.async_session_factory", _mock_session_factory(mock_db)),
            patch(f"{_MOD}.reconcile_duplicate_games", new_callable=AsyncMock, return_value=0),
        ):
            client = AsyncMock()
            client.fetch_games = AsyncMock(return_value=[])
            client.persist_games = AsyncMock(return_value=0)
            client.fetch_team_stats = AsyncMock(side_effect=Exception("stats error"))
            MockClient.return_value = client

            from src.data.scheduler import poll_stats

            await poll_stats()
            mock_breaker.record_failure.assert_called_once()
            mock_record.assert_awaited_once()


class TestPollInjuriesException:
    @pytest.mark.anyio
    async def test_poll_injuries_exception_logged(self):
        with patch("src.data.basketball_client.BasketballClient") as MockClient:
            client = AsyncMock()
            client.fetch_injuries = AsyncMock(side_effect=Exception("injuries error"))
            MockClient.return_value = client

            from src.data.scheduler import poll_injuries

            await poll_injuries()
            # Should not raise — just log error


class TestPregameCheckException:
    @pytest.mark.anyio
    async def test_pregame_check_exception_logged(self):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("db down"))

        with patch(f"{_MOD}.async_session_factory", _mock_session_factory(mock_db)):
            from src.data.scheduler import pregame_check

            await pregame_check()
            # Should not raise — just logs error


class TestCheckPredictionDriftException:
    @pytest.mark.anyio
    async def test_check_prediction_drift_exception_logged(self):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("drift error"))

        with patch(f"{_MOD}.async_session_factory", _mock_session_factory(mock_db)):
            from src.data.scheduler import check_prediction_drift

            await check_prediction_drift()


class TestPruneOldOddsException:
    @pytest.mark.anyio
    async def test_prune_old_odds_exception_logged(self):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("prune error"))

        with patch(f"{_MOD}.async_session_factory", _mock_session_factory(mock_db)):
            from src.data.scheduler import prune_old_odds

            await prune_old_odds()


class TestGeneratePredictionsException:
    @pytest.mark.anyio
    async def test_generate_predictions_exception_sends_alert(self):
        # generate_predictions_and_publish refreshes fresh inputs first, then
        # Predictor(). We no-op the sub-calls and make Predictor() raise to
        # trigger the outer except block.
        with (
            patch(f"{_MOD}.poll_stats", new_callable=AsyncMock),
            patch(f"{_MOD}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_MOD}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_MOD}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_MOD}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_MOD}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_MOD}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_MOD}.async_session_factory") as mock_sf,
            patch(f"{_MOD}.purge_invalid_upcoming_predictions", new_callable=AsyncMock, return_value=0),
            patch(f"{_MOD}.get_settings"),
            patch("src.models.features.reset_elo_cache"),
            patch("src.models.predictor.Predictor", side_effect=RuntimeError("model load boom")),
            patch("src.notifications.teams.send_alert", new_callable=AsyncMock) as mock_alert,
        ):
            mock_db = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            from src.data.scheduler import generate_predictions_and_publish

            await generate_predictions_and_publish()
            mock_alert.assert_awaited_once()


class TestScoresAndBoxException:
    @pytest.mark.anyio
    async def test_poll_scores_exception_logged(self):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("scores error"))

        with patch(f"{_MOD}.async_session_factory", _mock_session_factory(mock_db)):
            from src.data.scheduler import poll_scores_and_box

            await poll_scores_and_box()


class TestDbMaintenanceException:
    @pytest.mark.anyio
    async def test_db_maintenance_exception_logged(self):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("analyze error"))

        with patch(f"{_MOD}.async_session_factory", _mock_session_factory(mock_db)):
            from src.data.scheduler import db_maintenance

            await db_maintenance()
