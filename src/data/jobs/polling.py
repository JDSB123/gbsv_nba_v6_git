"""Polling jobs — fetch data from external APIs on a schedule."""

import logging
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from typing import Any, cast

from sqlalchemy import delete, func, select

from src.config import get_settings
from src.data.reconciliation import _find_matching_game, _GAME_MATCH_WINDOW
from src.data.seasons import current_nba_season, parse_api_datetime
from src.db.models import Game, GameReferee, OddsSnapshot, PlayerGameStats, Team
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)


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


async def poll_fg_odds() -> None:
    """Fetch full-game odds every 15 minutes."""
    from src.data.circuit_breaker import odds_api_breaker

    if odds_api_breaker.should_skip():
        logger.warning("Odds API circuit breaker open — skipping FG odds poll")
        return
    logger.info("Polling full-game odds...")
    try:
        from src.data.odds_client import OddsClient

        await sync_events_to_games()

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
    """Fetch 1st-half odds for ALL upcoming games every 30 minutes."""
    from src.data.circuit_breaker import odds_api_breaker

    if odds_api_breaker.should_skip():
        logger.warning("Odds API circuit breaker open — skipping 1H odds poll")
        return
    logger.info("Polling 1st-half odds...")
    try:
        from src.data.odds_client import OddsClient

        await sync_events_to_games()

        client = OddsClient()
        events = await client.fetch_events()
        fetched = 0
        async with async_session_factory() as db:
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue
                if client._should_skip():
                    logger.warning(
                        "Quota low — stopped 1H poll after %d/%d events",
                        fetched,
                        len(events),
                    )
                    break
                data = await client.fetch_event_odds(event_id)
                if data and data.get("bookmakers"):
                    await client.persist_odds([data], db)
                    fetched += 1
        logger.info("1H odds: fetched %d / %d events", fetched, len(events))
        odds_api_breaker.record_success()
    except Exception as exc:
        logger.exception("Error polling 1H odds")
        odds_api_breaker.record_failure()
        await _record_failure("poll_1h_odds", exc)


async def poll_player_props() -> None:
    """Fetch player prop odds for ALL upcoming games every 30 minutes."""
    from src.data.circuit_breaker import odds_api_breaker

    if odds_api_breaker.should_skip():
        logger.warning("Odds API circuit breaker open — skipping player props poll")
        return
    logger.info("Polling player prop odds...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        events = await client.fetch_events()
        fetched = 0
        async with async_session_factory() as db:
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue
                if client._should_skip():
                    logger.warning(
                        "Quota low — stopped props poll after %d/%d events",
                        fetched,
                        len(events),
                    )
                    break
                data = await client.fetch_player_props(event_id)
                if data and data.get("bookmakers"):
                    await client.persist_odds([data], db)
                    fetched += 1
        logger.info("Player props: fetched %d / %d events", fetched, len(events))
        odds_api_breaker.record_success()
    except Exception as exc:
        logger.exception("Error polling player props")
        odds_api_breaker.record_failure()
        await _record_failure("poll_player_props", exc)


async def poll_stats() -> None:
    """Fetch team stats and recent games every 2 hours."""
    from src.data.circuit_breaker import basketball_api_breaker

    if basketball_api_breaker.should_skip():
        logger.warning("Basketball API circuit breaker open — skipping poll_stats")
        return
    logger.info("Polling stats from Basketball API...")
    try:
        from sqlalchemy import func as sa_func

        from src.data.basketball_client import BasketballClient
        from src.data.reconciliation import reconcile_duplicate_games
        from src.db.models import TeamSeasonStats

        client = BasketballClient()
        season = current_nba_season()
        async with async_session_factory() as db:
            today = datetime.now(UTC).date()
            yesterday = today - timedelta(days=1)
            tomorrow = today + timedelta(days=1)
            total_games = 0
            for game_date in (yesterday, today, tomorrow):
                games = await client.fetch_games(game_date=game_date, season=season)
                if games:
                    persisted = await client.persist_games(games, db)
                    total_games += persisted
            reconciled = await reconcile_duplicate_games(db)
            logger.info(
                "poll_stats: ingested %d games for %s / %s / %s",
                total_games,
                yesterday,
                today,
                tomorrow,
            )
            if reconciled:
                logger.warning("poll_stats: reconciled %d synthetic/official duplicates", reconciled)

            result = await db.execute(select(Team.id))
            team_ids = [row[0] for row in result.fetchall()]
            for team_id in team_ids:
                stats = await client.fetch_team_stats(team_id, season=season)
                if stats:
                    await client.persist_team_season_stats(team_id, stats, season, db)

            team_count = (
                await db.execute(
                    select(sa_func.count())
                    .select_from(TeamSeasonStats)
                    .where(TeamSeasonStats.season == season)
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
    """Fetch box scores (referees and player stats) for finished games."""
    logger.info("Polling scores and box scores...")
    try:
        from sqlalchemy import and_

        from src.data.basketball_client import BasketballClient

        client = BasketballClient()
        async with async_session_factory() as db:
            # 1. Fetch Referees for ANY game that doesn't have them yet
            ref_subq = select(GameReferee.game_id).distinct().scalar_subquery()
            now = datetime.now(UTC).replace(tzinfo=None)
            ref_game_result = await db.execute(
                select(Game.id)
                .where(
                    and_(
                        Game.id.notin_(ref_subq),
                        Game.commence_time.between(now - timedelta(days=1), now + timedelta(days=2)),
                    )
                )
                .limit(20)
            )
            missing_ref_ids = [row[0] for row in ref_game_result.fetchall()]
            if missing_ref_ids:
                logger.info("Fetching referees for %d games", len(missing_ref_ids))
                for gid in missing_ref_ids:
                    refs_result = getattr(client, "fetch_nba_officials", None)
                    if refs_result is None:
                        refs: list[str] = []
                    else:
                        maybe_refs = refs_result(gid)
                        if isawaitable(maybe_refs):
                            refs = await maybe_refs
                        elif isinstance(maybe_refs, list):
                            refs = maybe_refs
                        else:
                            logger.debug(
                                "Skipping referee sync for game %s because fetch_nba_officials "
                                "did not return an awaitable or list",
                                gid,
                            )
                            refs = []
                    if refs:
                        await db.execute(delete(GameReferee).where(GameReferee.game_id == gid))
                        for rname in refs:
                            db.add(GameReferee(game_id=gid, referee_name=rname))
                await db.commit()

            # 2. Find finished games that have NO player_game_stats rows yet
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

        # Bust the cached Predictor so the API picks up the new model
        from src.api.dependencies import reload_predictor

        reload_predictor()
        logger.info("Daily retrain completed")
    except Exception as exc:
        logger.exception("Error during daily retrain")
        await _record_failure("daily_retrain", exc)
        from src.notifications.teams import send_alert

        await send_alert(
            "Daily Retrain Failed",
            "Model retraining raised an exception. Check worker logs.",
            "error",
        )


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
    by home-team name within the same calendar day (UTC).
    """
    logger.info("Syncing events to games...")
    try:
        from src.data.odds_client import OddsClient
        from src.data.reconciliation import reconcile_duplicate_games

        client = OddsClient()
        events = await client.fetch_events()
        matched = 0
        skipped = 0
        reconciled = 0
        async with async_session_factory() as db:
            team_rows = (await db.execute(select(Team.id, Team.name))).all()
            team_by_name: dict[str, int] = {name: tid for tid, name in team_rows}

            for event in events:
                odds_id = event["id"]
                commence = event.get("commence_time")
                home_team_name = event.get("home_team", "")
                away_team_name = event.get("away_team", "")
                if not commence:
                    continue

                ct = parse_api_datetime(commence)
                home_id = team_by_name.get(home_team_name)
                away_id = team_by_name.get(away_team_name)

                existing = await db.execute(select(Game).where(Game.odds_api_id == odds_id))
                existing_game = existing.scalar_one_or_none()
                if existing_game is not None:
                    existing_game_id = int(cast(Any, getattr(existing_game, "id", 0)))
                    if existing_game_id < 0:
                        from src.data.reconciliation import _merge_game_records

                        real_match = await _find_matching_game(
                            db,
                            home_id,
                            away_id,
                            ct,
                            real_only=True,
                            exclude_game_id=existing_game_id,
                        )
                        if real_match is not None and await _merge_game_records(
                            db,
                            existing_game,
                            real_match,
                        ):
                            reconciled += 1
                    continue

                game = await _find_matching_game(db, home_id, away_id, ct)

                # 2) Fallback: exact commence_time match
                if game is None:
                    result = await db.execute(select(Game).where(Game.commence_time == ct))
                    game = result.scalar_one_or_none()

                # 3) Fallback: same home team within ±12 hours (pick closest)
                if game is None and home_team_name:
                    window_start = ct - _GAME_MATCH_WINDOW
                    window_end = ct + _GAME_MATCH_WINDOW
                    result = await db.execute(
                        select(Game)
                        .join(Team, Game.home_team_id == Team.id)
                        .where(
                            Game.commence_time.between(window_start, window_end),
                            Team.name == home_team_name,
                        )
                        .order_by(func.abs(func.extract("epoch", Game.commence_time - ct)))
                        .limit(1)
                    )
                    game = result.scalar_one_or_none()

                if game is not None:
                    game_odds_api_id = cast(Any, game.odds_api_id)
                    if game_odds_api_id is None:
                        game.odds_api_id = odds_id
                        matched += 1
                    elif game_odds_api_id != odds_id:
                        logger.warning(
                            "Odds event %s matched game %s already linked to %s",
                            odds_id,
                            game.id,
                            game_odds_api_id,
                        )
                else:
                    skipped += 1
                    if home_id is not None and away_id is not None:
                        logger.warning(
                            "Skipping odds event %s: no matching official game found for %s @ %s at %s",
                            odds_id,
                            away_team_name,
                            home_team_name,
                            ct.isoformat() if isinstance(ct, datetime) else ct,
                        )
                    else:
                        logger.warning(
                            "Skipping odds event %s: "
                            "team lookup failed (home=%r→%s, away=%r→%s)",
                            odds_id,
                            home_team_name,
                            home_id,
                            away_team_name,
                            away_id,
                        )
            await db.commit()
        logger.info(
            "Synced events: %d matched, %d skipped, %d reconciled",
            matched,
            skipped,
            reconciled,
        )
    except Exception:
        logger.exception("Error syncing events")
