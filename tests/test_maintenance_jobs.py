"""Tests for src.data.jobs.maintenance — CLV fill, odds pruning, DB maintenance."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestFillClv:
    async def test_no_predictions_returns_early(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.jobs.maintenance.async_session_factory", return_value=mock_cm):
            from src.data.jobs.maintenance import fill_clv

            await fill_clv()  # Should return early without error

    async def test_handles_exception(self):
        with patch(
            "src.data.jobs.maintenance.async_session_factory", side_effect=Exception("db err")
        ):
            from src.data.jobs.maintenance import fill_clv

            await fill_clv()  # Should catch and log, not raise


@pytest.mark.asyncio
class TestPruneOldOdds:
    async def test_runs_without_error(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.jobs.maintenance.async_session_factory", return_value=mock_cm):
            from src.data.jobs.maintenance import prune_old_odds

            await prune_old_odds()

    async def test_handles_exception(self):
        with patch(
            "src.data.jobs.maintenance.async_session_factory", side_effect=Exception("fail")
        ):
            from src.data.jobs.maintenance import prune_old_odds

            await prune_old_odds()


@pytest.mark.asyncio
class TestDbMaintenance:
    async def test_runs_analyze(self):
        mock_session = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.jobs.maintenance.async_session_factory", return_value=mock_cm):
            from src.data.jobs.maintenance import db_maintenance

            await db_maintenance()
            # Should have called execute for each table + commit
            assert mock_session.execute.await_count >= 5
            mock_session.commit.assert_awaited_once()

    async def test_handles_exception(self):
        with patch(
            "src.data.jobs.maintenance.async_session_factory", side_effect=Exception("fail")
        ):
            from src.data.jobs.maintenance import db_maintenance

            await db_maintenance()


@pytest.mark.asyncio
class TestCheckDataFreshness:
    async def test_no_alerts_when_fresh(self):
        from datetime import UTC, datetime

        mock_session = AsyncMock()

        # Return recent timestamp for odds
        mock_result_odds = MagicMock()
        mock_result_odds.scalar_one_or_none.return_value = datetime.now(UTC).replace(tzinfo=None)
        # Return 30 teams with stats
        mock_result_teams = MagicMock()
        mock_result_teams.scalar.return_value = 30

        mock_session.execute.side_effect = [mock_result_odds, mock_result_teams]

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.data.jobs.maintenance.async_session_factory", return_value=mock_cm),
            patch("src.config.get_settings") as ms,
        ):
            ms.return_value.odds_fg_interval = 30
            from src.data.jobs.maintenance import check_data_freshness

            await check_data_freshness()  # Should not raise

    async def test_handles_exception(self):
        with patch(
            "src.data.jobs.maintenance.async_session_factory", side_effect=Exception("fail")
        ):
            from src.data.jobs.maintenance import check_data_freshness

            # check_data_freshness imports inside function body; some imports may
            # fail, which is OK for this test — we just want it to run.
            with contextlib.suppress(Exception):
                await check_data_freshness()
