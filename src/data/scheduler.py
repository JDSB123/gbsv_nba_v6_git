import logging
from datetime import UTC, date, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.data.seasons import current_nba_season, parse_api_datetime
from src.db.models import Game, Team
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)

# Dedup flag: tracks the date for which pregame publish already fired
_pregame_published_date: date | None = None


async def _record_failure(job_name: str, error: Exception) -> None:
    """Log a failed ingestion job to the dead-letter table (best-effort)."""
    try:
        from src.db.models import IngestionFailure

        async with async_session_factory() as db:
            db.add(
                IngestionFailure(
                    job_name=job_name,
                    error_message=str(error)[:2000],
                )
            )
            await db.commit()
    except Exception:
        logger.debug("Failed to record ingestion failure for %s", job_name, exc_info=True)


# ── Scheduled jobs ─────────────────────────────────────────────────


async def poll_fg_odds() -> None:
    """Fetch full-game odds every 15 minutes."""
    from src.data.circuit_breaker import odds_api_breaker

    if odds_api_breaker.should_skip():
        return
    logger.info("Polling full-game odds...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        odds_data = await client.fetch_odds()
        if odds_data:
            async with async_session_factory() as db:
                await client.persist_odds(odds_data, db)
        odds_api_breaker.record_success()
    except Exception as exc:
        logger.exception("Error polling FG odds")
        odds_api_breaker.record_failure()
        await _record_failure("poll_fg_odds", exc)


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


async def poll_player_props() -> None:
    """Fetch player prop odds for upcoming games every 30 minutes."""
    logger.info("Polling player prop odds...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        events = await client.fetch_events()
        async with async_session_factory() as db:
            for event in events[:10]:
                event_id = event.get("id")
                if event_id:
                    data = await client.fetch_player_props(event_id)
                    if data and data.get("bookmakers"):
                        await client.persist_odds([data], db)
    except Exception:
        logger.exception("Error polling player props")


async def poll_stats() -> None:
    """Fetch team stats and recent games every 2 hours."""
    from src.data.circuit_breaker import basketball_api_breaker

    if basketball_api_breaker.should_skip():
        return
    logger.info("Polling stats from Basketball API...")
    try:
        from sqlalchemy import func as sa_func

        from src.data.basketball_client import BasketballClient
        from src.db.models import TeamSeasonStats

        client = BasketballClient()
        season = current_nba_season()
        async with async_session_factory() as db:
            games = await client.fetch_games(game_date=date.today(), season=season)
            if games:
                await client.persist_games(games, db)

            result = await db.execute(select(Team.id))
            team_ids = [row[0] for row in result.fetchall()]
            for team_id in team_ids:
                stats = await client.fetch_team_stats(team_id, season=season)
                if stats:
                    await client.persist_team_season_stats(team_id, stats, season, db)

            # Data completeness check
            team_count = (
                await db.execute(
                    select(sa_func.count()).select_from(TeamSeasonStats).where(
                        TeamSeasonStats.season == season
                    )
                )
            ).scalar() or 0
            if team_count < 30:
                logger.warning(
                    "Data completeness: only %d/30 teams have season stats for %s",
                    team_count,
                    season,
                )
        basketball_api_breaker.record_success()
    except Exception as exc:
        logger.exception("Error polling stats")
        basketball_api_breaker.record_failure()
        await _record_failure("poll_stats", exc)


async def poll_scores_and_box() -> None:
    """Fetch box scores for finished games that don't have player stats yet."""
    logger.info("Polling scores and box scores...")
    try:
        from sqlalchemy import and_

        from src.data.basketball_client import BasketballClient
        from src.db.models import PlayerGameStats

        client = BasketballClient()
        async with async_session_factory() as db:
            # Find finished games that have NO player_game_stats rows yet
            subq = select(PlayerGameStats.game_id).distinct().scalar_subquery()
            result = await db.execute(
                select(Game.id)
                .where(
                    and_(
                        Game.status.in_(["FT", "AOT"]),
                        Game.id.notin_(subq),
                    )
                )
                .order_by(Game.commence_time.desc())
                .limit(25)
            )
            missing_ids = [row[0] for row in result.fetchall()]
            if not missing_ids:
                logger.info("All finished games already have box scores")
                return
            logger.info(
                "Fetching box scores for %d games missing player stats",
                len(missing_ids),
            )
            for game_id in missing_ids:
                try:
                    stats = await client.fetch_player_stats(game_id)
                    if stats:
                        await client.persist_player_game_stats(game_id, stats, db)
                except Exception:
                    logger.warning("Failed to fetch box score for game %d", game_id, exc_info=True)
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
    except Exception as exc:
        logger.exception("Error during daily retrain")
        await _record_failure("daily_retrain", exc)
        from src.notifications.teams import send_alert

        await send_alert("Daily Retrain Failed", "Model retraining raised an exception. Check worker logs.", "error")


async def poll_injuries() -> None:
    """Fetch current injury report (NBA API) every 2 hours."""
    logger.info("Polling injury report...")
    try:
        from src.data.basketball_client import BasketballClient

        client = BasketballClient()
        injuries = await client.fetch_injuries()
        if injuries:
            async with async_session_factory() as db:
                count = await client.persist_injuries(injuries, db)
                logger.info("Refreshed %d injuries", count)
        else:
            logger.info("No injury data returned")
    except Exception:
        logger.exception("Error polling injuries")


async def sync_events_to_games() -> None:
    """Sync Odds API events to games table (map odds_api_id).

    Matches by exact commence_time first, then falls back to matching
    by home-team name within the same calendar day (UTC).  This handles
    the common case where the Basketball API and Odds API report
    slightly different tip-off times for the same game.
    """
    from datetime import timedelta

    logger.info("Syncing events to games...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        events = await client.fetch_events()
        matched = 0
        async with async_session_factory() as db:
            for event in events:
                odds_id = event["id"]
                commence = event.get("commence_time")
                home_team_name = event.get("home_team", "")
                if not commence:
                    continue

                ct = parse_api_datetime(commence)

                # 1) Exact commence_time match
                result = await db.execute(select(Game).where(Game.commence_time == ct))
                game = result.scalar_one_or_none()

                # 2) Fallback: same home team within ±12 hours
                if game is None and home_team_name:
                    window_start = ct - timedelta(hours=12)
                    window_end = ct + timedelta(hours=12)
                    result = await db.execute(
                        select(Game)
                        .join(Team, Game.home_team_id == Team.id)
                        .where(
                            Game.commence_time.between(window_start, window_end),
                            Team.name == home_team_name,
                        )
                    )
                    game = result.scalar_one_or_none()

                if game is not None:
                    game_odds_api_id = cast(Any, game.odds_api_id)
                    if game_odds_api_id is None:
                        game.odds_api_id = odds_id
                        matched += 1
            await db.commit()
        logger.info("Synced %d events to games", matched)
    except Exception:
        logger.exception("Error syncing events")


async def fill_clv() -> None:
    """Fill closing-line value for predictions whose games have finished."""
    logger.info("Filling CLV for finished games...")
    try:
        from sqlalchemy import and_

        from src.db.models import OddsSnapshot, Prediction

        async with async_session_factory() as db:
            # Predictions that lack a closing spread and belong to finished games
            result = await db.execute(
                select(Prediction)
                .join(Game, Prediction.game_id == Game.id)
                .options(selectinload(Prediction.game).selectinload(Game.home_team))
                .where(
                    and_(
                        Game.status == "FT",
                        Prediction.closing_spread.is_(None),
                    )
                )
            )
            preds = result.scalars().all()
            if not preds:
                return

            import numpy as np

            for pred in preds:
                pred_any = cast(Any, pred)
                game_obj = cast(Any, pred).game
                home_name = game_obj.home_team.name if game_obj and game_obj.home_team else ""
                # Latest odds snapshot for this game
                odds_q = await db.execute(
                    select(OddsSnapshot)
                    .where(OddsSnapshot.game_id == pred.game_id)
                    .order_by(OddsSnapshot.captured_at.desc())
                    .limit(100)
                )
                snapshots = odds_q.scalars().all()
                if not snapshots:
                    continue

                # Spreads: betting convention (negative = home favorite)
                spreads = [
                    float(cast(Any, s.point))
                    for s in snapshots
                    if cast(Any, s.market) == "spreads"
                    and s.point is not None
                    and cast(Any, s.outcome_name) == home_name
                ]
                totals = [
                    float(cast(Any, s.point))
                    for s in snapshots
                    if cast(Any, s.market) == "totals" and s.point is not None
                ]
                if spreads:
                    closing_spread = round(float(np.mean(spreads)), 1)
                    pred_any.closing_spread = closing_spread
                    opening_spread = cast(Any, pred.opening_spread)
                    if opening_spread is not None:
                        pred_any.clv_spread = round(
                            float(closing_spread) - float(opening_spread), 1
                        )
                if totals:
                    closing_total = round(float(np.mean(totals)), 1)
                    pred_any.closing_total = closing_total
                    opening_total = cast(Any, pred.opening_total)
                    if opening_total is not None:
                        pred_any.clv_total = round(float(closing_total) - float(opening_total), 1)

            await db.commit()
            logger.info("CLV filled for %d predictions", len(preds))
    except Exception:
        logger.exception("Error filling CLV")


async def pregame_check() -> None:
    """Publish predictions once, ~1 hour before the first game of the day."""
    global _pregame_published_date

    et = ZoneInfo("US/Eastern")
    now = datetime.now(et)
    today = now.date()

    if _pregame_published_date == today:
        return  # already published pregame today

    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(Game.commence_time)
                .where(Game.status == "NS")
                .order_by(Game.commence_time)
                .limit(1)
            )
            first_ct = result.scalar_one_or_none()

        if first_ct is None:
            return

        # commence_time is stored as naive UTC
        first_et = first_ct.replace(tzinfo=ZoneInfo("UTC")).astimezone(et)
        if first_et.date() != today:
            return

        minutes_until = (first_et - now).total_seconds() / 60
        if 0 < minutes_until <= get_settings().pregame_lead_minutes:
            logger.info(
                "Pregame window: %d min until first game, publishing...",
                int(minutes_until),
            )
            await generate_predictions_and_publish()
            _pregame_published_date = today
    except Exception:
        logger.exception("Error in pregame check")


async def generate_predictions_and_publish() -> None:
    """Generate upcoming predictions and publish formatted output to Teams."""
    logger.info("Generating predictions and publishing to Teams...")
    try:
        from sqlalchemy import func as sa_func

        from src.db.models import OddsSnapshot
        from src.models.features import reset_elo_cache
        from src.models.predictor import Predictor
        from src.notifications.teams import (
            build_teams_card,
            send_card_to_teams,
            send_card_via_graph,
        )

        # Invalidate Elo cache so fresh ratings are computed from latest results
        reset_elo_cache()

        # Check data freshness — refresh odds/stats if stale
        async with async_session_factory() as _check_db:
            latest_odds_ts = (
                await _check_db.execute(
                    select(sa_func.max(OddsSnapshot.captured_at))
                )
            ).scalar_one_or_none()

        if latest_odds_ts is not None:
            odds_age_min = (datetime.now(UTC) - latest_odds_ts.replace(tzinfo=UTC)).total_seconds() / 60
            _s = get_settings()
            if odds_age_min > _s.odds_freshness_max_age_minutes:
                logger.warning(
                    "Odds data is %.0f min stale (threshold: %d min), refreshing before predictions...",
                    odds_age_min,
                    _s.odds_freshness_max_age_minutes,
                )
                await poll_fg_odds()
        else:
            logger.warning("No odds data found, refreshing before predictions...")
            await poll_fg_odds()

        predictor = Predictor()
        if not predictor.is_ready:
            logger.warning("Models not ready; skipping prediction publish")
            return

        async with async_session_factory() as db:
            predictions = await predictor.predict_upcoming(db)
            if not predictions:
                logger.info("No upcoming games to publish")
                return

            game_ids = [int(cast(Any, p.game_id)) for p in predictions]
            game_result = await db.execute(
                select(Game)
                .options(
                    selectinload(Game.home_team).selectinload(Team.season_stats),
                    selectinload(Game.away_team).selectinload(Team.season_stats),
                )
                .where(Game.id.in_(game_ids))
                .order_by(Game.commence_time)
            )
            games = game_result.scalars().all()
            game_by_id = {int(cast(Any, g.id)): g for g in games}

            rows: list[tuple[Any, Game]] = []
            for pred in predictions:
                game = game_by_id.get(int(cast(Any, pred.game_id)))
                if game is not None:
                    rows.append((pred, game))

            # Latest odds pull timestamp
            odds_ts_result = await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
            odds_pulled_at = odds_ts_result.scalar_one_or_none()

            if rows:
                download_url: str | None = None

                # Link to the HTML slate if the API is reachable
                _s = get_settings()
                if _s.api_base_url:
                    download_url = f"{_s.api_base_url}/predictions/slate.html"

                payload = build_teams_card(
                    rows,
                    _s.teams_max_games_per_message,
                    odds_pulled_at=odds_pulled_at,
                    download_url=download_url,
                )
                if _s.teams_team_id and _s.teams_channel_id:
                    await send_card_via_graph(
                        _s.teams_team_id,
                        _s.teams_channel_id,
                        payload,
                    )
                    logger.info("Published %d predictions to Teams (Graph API)", len(rows))
                elif _s.teams_webhook_url:
                    await send_card_to_teams(_s.teams_webhook_url, payload)
                    logger.info("Published %d predictions to Teams (webhook)", len(rows))
                else:
                    logger.info("No Teams delivery configured; skipping publish")
    except Exception:
        logger.exception("Error generating/publishing predictions")
        from src.notifications.teams import send_alert

        await send_alert(
            "Prediction Publish Failed",
            "generate_predictions_and_publish raised an exception. Check worker logs.",
            "error",
        )


async def check_prediction_drift() -> None:
    """Compare recent prediction distributions against 30-day trailing window."""
    logger.info("Checking prediction drift...")
    try:
        from datetime import timedelta

        import numpy as np

        from src.db.models import Prediction

        async with async_session_factory() as db:
            now = datetime.now(UTC)
            cutoff_30d = now - timedelta(days=30)
            cutoff_7d = now - timedelta(days=7)

            # 30-day baseline
            result_30d = await db.execute(
                select(Prediction.predicted_home_fg, Prediction.predicted_away_fg)
                .where(Prediction.predicted_at > cutoff_30d)
            )
            rows_30d = result_30d.all()

            # Recent 7-day window
            result_7d = await db.execute(
                select(Prediction.predicted_home_fg, Prediction.predicted_away_fg)
                .where(Prediction.predicted_at > cutoff_7d)
            )
            rows_7d = result_7d.all()

            if len(rows_30d) < 20 or len(rows_7d) < 5:
                logger.info("Not enough predictions for drift analysis")
                return

            totals_30d = [float(r[0] or 0) + float(r[1] or 0) for r in rows_30d]
            totals_7d = [float(r[0] or 0) + float(r[1] or 0) for r in rows_7d]

            mean_30d = float(np.mean(totals_30d))
            mean_7d = float(np.mean(totals_7d))
            drift = abs(mean_7d - mean_30d)

            if drift > 5.0:
                logger.warning(
                    "Prediction drift detected: 7d mean total=%.1f vs 30d mean=%.1f (delta=%.1f)",
                    mean_7d,
                    mean_30d,
                    drift,
                )
                from src.notifications.teams import send_alert

                await send_alert(
                    "Prediction Drift Warning",
                    f"7-day mean predicted total ({mean_7d:.1f}) drifted {drift:.1f} points "
                    f"from 30-day baseline ({mean_30d:.1f}). Review model performance.",
                    "warning",
                )
            else:
                logger.info(
                    "Prediction drift OK: 7d=%.1f, 30d=%.1f, delta=%.1f",
                    mean_7d,
                    mean_30d,
                    drift,
                )
    except Exception:
        logger.exception("Error checking prediction drift")


async def prune_old_odds() -> None:
    """Delete odds snapshots older than 30 days for finished games."""
    logger.info("Pruning old odds snapshots...")
    try:
        from datetime import timedelta

        from sqlalchemy import and_, delete

        from src.db.models import OddsSnapshot

        cutoff = datetime.now(UTC) - timedelta(days=30)
        async with async_session_factory() as db:
            # Only prune for games that are finished
            result = await db.execute(
                delete(OddsSnapshot).where(
                    and_(
                        OddsSnapshot.captured_at < cutoff,
                        OddsSnapshot.game_id.in_(
                            select(Game.id).where(Game.status.in_(["FT", "AOT"]))
                        ),
                    )
                )
            )
            await db.commit()
            logger.info("Pruned %d old odds snapshots", result.rowcount)
    except Exception:
        logger.exception("Error pruning old odds")


async def db_maintenance() -> None:
    """Run ANALYZE on key tables to keep query planner stats current."""
    logger.info("Running database maintenance (ANALYZE)...")
    try:
        from sqlalchemy import text

        tables = ["games", "odds_snapshots", "predictions", "player_game_stats"]
        async with async_session_factory() as db:
            for table in tables:
                await db.execute(text(f"ANALYZE {table}"))
            await db.commit()
        logger.info("Database ANALYZE complete")
    except Exception:
        logger.exception("Error during database maintenance")


# ── Scheduler setup ────────────────────────────────────────────────


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
