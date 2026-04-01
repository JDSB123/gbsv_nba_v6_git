"""Tests for the scheduler – job registration and _record_failure."""

from unittest.mock import AsyncMock, patch

import pytest

from src.data.scheduler import _record_failure, create_scheduler

# ── create_scheduler registers all expected jobs ───────────────


def test_create_scheduler_registers_all_jobs():
    scheduler = create_scheduler()

    job_ids = {j.id for j in scheduler.get_jobs()}
    expected = {
        "poll_fg_odds",
        "poll_1h_odds",
        "poll_player_props",
        "poll_stats",
        "poll_scores",
        "sync_events",
        "morning_slate",
        "refresh_prediction_cache",
        "pregame_check",
        "fill_clv",
        "daily_retrain",
        "poll_injuries",
        "check_prediction_drift",
        "prune_old_odds",
        "db_maintenance",
        "process_dead_letter_queue",
    }
    assert expected == job_ids, f"Missing: {expected - job_ids}, Extra: {job_ids - expected}"


def test_create_scheduler_job_count():
    scheduler = create_scheduler()
    assert len(scheduler.get_jobs()) == 16


def test_create_scheduler_timezone():
    scheduler = create_scheduler()
    assert str(scheduler.timezone) == "US/Eastern"


# ── _record_failure ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_failure_logs_to_db():
    mock_db = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("src.data.jobs.polling.async_session_factory", return_value=mock_ctx):
        await _record_failure("test_job", ValueError("test error"))

    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()

    # Check the added object
    failure_obj = mock_db.add.call_args[0][0]
    assert failure_obj.job_name == "test_job"
    assert "test error" in failure_obj.error_message


@pytest.mark.asyncio
async def test_record_failure_truncates_long_error():
    mock_db = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    long_error = "x" * 5000
    with patch("src.data.jobs.polling.async_session_factory", return_value=mock_ctx):
        await _record_failure("test_job", ValueError(long_error))

    failure_obj = mock_db.add.call_args[0][0]
    assert len(failure_obj.error_message) <= 2000


@pytest.mark.asyncio
async def test_record_failure_swallows_db_errors():
    """If the DB call itself fails, _record_failure should not raise."""
    with patch("src.data.jobs.polling.async_session_factory", side_effect=Exception("db down")):
        # Should not raise
        await _record_failure("test_job", ValueError("original"))
