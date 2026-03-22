import logging
from datetime import date

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from src.config import get_settings
from src.db.models import Game, Team
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Scheduled jobs ─────────────────────────────────────────────────


async def poll_fg_odds() -> None:
    """Fetch full-game odds every 15 minutes."""
    logger.info("Polling full-game odds...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        odds_data = await client.fetch_odds()
        if odds_data:
            async with async_session_factory() as db:
                await client.persist_odds(odds_data, db)
    except Exception:
        logger.exception("Error polling FG odds")


async def poll_1h_odds() -> None:
    """Fetch 1st-half odds for upcoming games every 30 minutes."""
    logger.info("Polling 1st-half odds...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        events = await client.fetch_events()
        async with async_session_factory() as db:
            for event in events[:10]:
                event_id = event.get("id")
                if event_id:
                    data = await client.fetch_event_odds(event_id)
                    if data and data.get("bookmakers"):
                        await client.persist_odds([data], db)
    except Exception:
        logger.exception("Error polling 1H odds")


async def poll_stats() -> None:
    """Fetch team stats and recent games every 2 hours."""
    logger.info("Polling stats from Basketball API...")
    try:
        from src.data.basketball_client import BasketballClient

        client = BasketballClient()
        async with async_session_factory() as db:
            games = await client.fetch_games(game_date=date.today())
            if games:
                await client.persist_games(games, db)

            result = await db.execute(select(Team.id))
            team_ids = [row[0] for row in result.fetchall()]
            for team_id in team_ids:
                stats = await client.fetch_team_stats(team_id)
                if stats:
                    await client.persist_team_season_stats(
                        team_id, stats, "2024-2025", db
                    )
    except Exception:
        logger.exception("Error polling stats")


async def poll_scores_and_box() -> None:
    """Fetch completed game scores and box scores."""
    logger.info("Polling scores and box scores...")
    try:
        from src.data.basketball_client import BasketballClient

        client = BasketballClient()
        async with async_session_factory() as db:
            result = await db.execute(
                select(Game.id).where(Game.status == "FT").limit(20)
            )
            finished_ids = [row[0] for row in result.fetchall()]
            for game_id in finished_ids:
                stats = await client.fetch_player_stats(game_id)
                if stats:
                    await client.persist_player_game_stats(game_id, stats, db)
    except Exception:
        logger.exception("Error polling scores/box scores")


async def daily_retrain() -> None:
    """Trigger model retrain at 6am ET daily."""
    logger.info("Starting daily retrain...")
    try:
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer()
        async with async_session_factory() as db:
            await trainer.train(db)
        logger.info("Daily retrain completed")
    except Exception:
        logger.exception("Error during daily retrain")


async def sync_events_to_games() -> None:
    """Sync Odds API events to games table (map odds_api_id)."""
    logger.info("Syncing events to games...")
    try:
        from datetime import datetime

        from src.data.odds_client import OddsClient

        client = OddsClient()
        events = await client.fetch_events()
        async with async_session_factory() as db:
            for event in events:
                odds_id = event["id"]
                commence = event.get("commence_time")
                if commence:
                    ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    result = await db.execute(
                        select(Game).where(Game.commence_time == ct)
                    )
                    game = result.scalar_one_or_none()
                    if game and not game.odds_api_id:
                        game.odds_api_id = odds_id
            await db.commit()
    except Exception:
        logger.exception("Error syncing events")


# ── Scheduler setup ────────────────────────────────────────────────


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    scheduler.add_job(
        poll_fg_odds, "interval", minutes=settings.odds_fg_interval, id="poll_fg_odds"
    )
    scheduler.add_job(
        poll_1h_odds, "interval", minutes=settings.odds_1h_interval, id="poll_1h_odds"
    )
    scheduler.add_job(
        poll_stats, "interval", minutes=settings.stats_interval, id="poll_stats"
    )
    scheduler.add_job(poll_scores_and_box, "interval", minutes=60, id="poll_scores")
    scheduler.add_job(sync_events_to_games, "interval", minutes=60, id="sync_events")
    scheduler.add_job(
        daily_retrain, "cron", hour=settings.retrain_hour, minute=0, id="daily_retrain"
    )

    return scheduler
