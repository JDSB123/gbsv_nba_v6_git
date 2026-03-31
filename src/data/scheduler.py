"""Scheduler -- job registration and backward-compatible re-exports.

The actual job implementations live in:
  - src.data.jobs.polling      (poll_*, sync_events_to_games, daily_retrain)
  - src.data.jobs.predictions  (generate_predictions_and_publish, pregame_check, ...)
  - src.data.jobs.maintenance  (fill_clv, prune_old_odds, db_maintenance)
  - src.data.reconciliation    (_find_matching_game, _merge_game_records, ...)
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import get_settings

# -- Re-exports (all existing `from src.data.scheduler import X` keep working) --
from src.data.jobs.maintenance import (  # noqa: F401
    db_maintenance,
    fill_clv,
    prune_old_odds,
)
from src.data.jobs.polling import (  # noqa: F401
    _record_failure,
    daily_retrain,
    poll_1h_odds,
    poll_fg_odds,
    poll_injuries,
    poll_player_props,
    poll_scores_and_box,
    poll_stats,
    sync_events_to_games,
)
from src.data.jobs.predictions import (  # noqa: F401
    check_prediction_drift,
    generate_predictions_and_publish,
    pregame_check,
    purge_invalid_upcoming_predictions,
    refresh_prediction_cache,
)
from src.data.reconciliation import (  # noqa: F401
    _copy_prediction_payload,
    _find_matching_game,
    _merge_game_records,
    reconcile_duplicate_games,
)

logger = logging.getLogger(__name__)


# -- Scheduler setup --


def create_scheduler() -> AsyncIOScheduler:
    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    scheduler.add_job(
        poll_fg_odds, "interval", minutes=settings.odds_fg_interval, id="poll_fg_odds"
    )
    scheduler.add_job(
        poll_1h_odds, "interval", minutes=settings.odds_1h_interval, id="poll_1h_odds"
    )
    scheduler.add_job(poll_player_props, "interval", minutes=30, id="poll_player_props")
    scheduler.add_job(poll_stats, "interval", minutes=settings.stats_interval, id="poll_stats")
    scheduler.add_job(poll_scores_and_box, "interval", minutes=60, id="poll_scores")
    scheduler.add_job(sync_events_to_games, "interval", minutes=60, id="sync_events")
    scheduler.add_job(
        generate_predictions_and_publish,
        "cron",
        hour=settings.morning_slate_hour,
        minute=0,
        id="morning_slate",
    )
    scheduler.add_job(
        refresh_prediction_cache,
        "interval",
        minutes=settings.prediction_cache_refresh_interval_minutes,
        id="refresh_prediction_cache",
    )
    scheduler.add_job(pregame_check, "interval", minutes=5, id="pregame_check")
    scheduler.add_job(fill_clv, "interval", minutes=90, id="fill_clv")
    scheduler.add_job(
        daily_retrain, "cron", hour=settings.retrain_hour, minute=0, id="daily_retrain"
    )
    scheduler.add_job(
        poll_injuries,
        "interval",
        minutes=settings.injuries_interval,
        id="poll_injuries",
    )
    scheduler.add_job(
        check_prediction_drift,
        "cron",
        hour=7,
        minute=0,
        id="check_prediction_drift",
    )
    scheduler.add_job(
        prune_old_odds,
        "cron",
        day_of_week="sun",
        hour=4,
        minute=0,
        id="prune_old_odds",
    )
    scheduler.add_job(
        db_maintenance,
        "cron",
        day_of_week="sun",
        hour=4,
        minute=30,
        id="db_maintenance",
    )

    return scheduler
