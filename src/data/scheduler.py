import logging
from datetime import UTC, date, datetime, timedelta
from inspect import isawaitable
from typing import Any, cast
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import case, delete, func, select, update
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.data.seasons import current_nba_season, parse_api_datetime
from src.db.models import (
    Game,
    GameReferee,
    OddsSnapshot,
    PlayerGameStats,
    Prediction,
    RotationChange,
    Team,
)
from src.db.session import async_session_factory
from src.services.prediction_integrity import (
    prediction_payload_has_integrity_issues,
    prediction_rank,
)

logger = logging.getLogger(__name__)

# Dedup flag: tracks the date for which pregame publish already fired
_pregame_published_date: date | None = None
_GAME_MATCH_WINDOW = timedelta(hours=12)


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


def _copy_prediction_payload(
    target: Prediction,
    preferred: Prediction,
    fallback: Prediction | None = None,
) -> None:
    fields = [
        "predicted_home_fg",
        "predicted_away_fg",
        "predicted_home_1h",
        "predicted_away_1h",
        "fg_spread",
        "fg_total",
        "fg_home_ml_prob",
        "h1_spread",
        "h1_total",
        "h1_home_ml_prob",
        "opening_spread",
        "opening_total",
        "closing_spread",
        "closing_total",
        "clv_spread",
        "clv_total",
        "odds_sourced",
        "predicted_at",
    ]
    for field in fields:
        preferred_value = getattr(preferred, field)
        if preferred_value is not None:
            setattr(target, field, preferred_value)
            continue
        if fallback is not None:
            setattr(target, field, getattr(fallback, field))


async def purge_invalid_upcoming_predictions(db: Any) -> int:
    """Remove malformed upcoming predictions that cannot be safely published.

    Upcoming predictions should always be tied to an odds-linked game and
    carry a parseable odds capture timestamp plus internally consistent
    score/spread/total fields. Legacy or corrupt rows are deleted so they
    cannot outrank fresh rows or confuse operators.
    """
    result = await db.execute(
        select(Prediction, Game)
        .join(Game, Prediction.game_id == Game.id)
        .where(Game.status == "NS")
    )
    rows = result.all()

    removed = 0
    for prediction, game in rows:
        game_odds_api_id = cast(Any, getattr(game, "odds_api_id", None))
        if game_odds_api_id is None or prediction_payload_has_integrity_issues(prediction):
            await db.delete(prediction)
            removed += 1

    if removed:
        await db.commit()
    return removed


async def _find_matching_game(
    db: Any,
    home_team_id: int | None,
    away_team_id: int | None,
    commence_time: datetime,
    *,
    real_only: bool = False,
    exclude_game_id: int | None = None,
) -> Game | None:
    if home_team_id is None or away_team_id is None:
        return None

    filters = [
        Game.home_team_id == home_team_id,
        Game.away_team_id == away_team_id,
        Game.commence_time.between(
            commence_time - _GAME_MATCH_WINDOW,
            commence_time + _GAME_MATCH_WINDOW,
        ),
    ]
    if real_only:
        filters.append(Game.id > 0)
    if exclude_game_id is not None:
        filters.append(Game.id != exclude_game_id)

    result = await db.execute(
        select(Game)
        .where(*filters)
        .order_by(
            case((Game.id > 0, 0), else_=1),
            func.abs(func.extract("epoch", Game.commence_time - commence_time)),
            Game.commence_time,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _merge_game_records(db: Any, source_game: Game, target_game: Game) -> bool:
    if source_game.id == target_game.id:
        return False

    if source_game.id > 0 and target_game.id < 0:
        source_game, target_game = target_game, source_game

    source_id = int(cast(Any, source_game.id))
    target_id = int(cast(Any, target_game.id))
    if source_id > 0 or target_id < 0:
        logger.warning(
            "Refusing non-canonical game merge: source=%s target=%s",
            source_id,
            target_id,
        )
        return False

    source_odds_id = cast(Any, source_game.odds_api_id)
    target_odds_id = cast(Any, target_game.odds_api_id)
    if target_odds_id is None and source_odds_id is not None:
        source_game.odds_api_id = None
        target_game.odds_api_id = source_odds_id
    elif target_odds_id not in (None, source_odds_id) and source_odds_id is not None:
        logger.warning(
            "Conflicting odds_api_id during merge: keeping real game %s=%s and dropping synthetic %s=%s",
            target_id,
            target_odds_id,
            source_id,
            source_odds_id,
        )

    target_preds = (
        await db.execute(select(Prediction).where(Prediction.game_id == target_id))
    ).scalars().all()
    source_preds = (
        await db.execute(select(Prediction).where(Prediction.game_id == source_id))
    ).scalars().all()
    target_by_version = {
        str(cast(Any, pred.model_version)): pred for pred in target_preds if pred.model_version is not None
    }

    for source_pred in source_preds:
        model_version = str(cast(Any, source_pred.model_version))
        existing = target_by_version.get(model_version)
        if existing is None:
            source_pred.game_id = target_id
            continue

        preferred = max((existing, source_pred), key=prediction_rank)
        fallback = source_pred if preferred is existing else existing
        _copy_prediction_payload(existing, preferred, fallback)
        await db.delete(source_pred)

    target_player_ids = set(
        (
            await db.execute(
                select(PlayerGameStats.player_id).where(PlayerGameStats.game_id == target_id)
            )
        ).scalars().all()
    )
    if target_player_ids:
        await db.execute(
            delete(PlayerGameStats).where(
                PlayerGameStats.game_id == source_id,
                PlayerGameStats.player_id.in_(target_player_ids),
            )
        )

    await db.execute(
        update(OddsSnapshot)
        .where(OddsSnapshot.game_id == source_id)
        .values(game_id=target_id)
    )
    await db.execute(
        update(PlayerGameStats)
        .where(PlayerGameStats.game_id == source_id)
        .values(game_id=target_id)
    )
    await db.execute(
        update(GameReferee)
        .where(GameReferee.game_id == source_id)
        .values(game_id=target_id)
    )
    await db.execute(
        update(RotationChange)
        .where(RotationChange.game_id == source_id)
        .values(game_id=target_id)
    )

    await db.flush()
    await db.delete(source_game)
    logger.info(
        "Reconciled synthetic game %s into official game %s",
        source_id,
        target_id,
    )
    return True


async def reconcile_duplicate_games(db: Any, lookback_days: int | None = 7) -> int:
    query = select(Game).where(Game.id < 0).order_by(Game.commence_time)
    if lookback_days is not None:
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=lookback_days)
        query = query.where(Game.commence_time >= cutoff)

    synthetic_games = (await db.execute(query)).scalars().all()
    reconciled = 0
    for synthetic_game in synthetic_games:
        real_match = await _find_matching_game(
            db,
            int(cast(Any, synthetic_game.home_team_id)),
            int(cast(Any, synthetic_game.away_team_id)),
            cast(Any, synthetic_game.commence_time),
            real_only=True,
            exclude_game_id=int(cast(Any, synthetic_game.id)),
        )
        if real_match is None:
            continue
        if await _merge_game_records(db, synthetic_game, real_match):
            reconciled += 1
    return reconciled


# ── Scheduled jobs ─────────────────────────────────────────────────


async def poll_fg_odds() -> None:
    """Fetch full-game odds every 15 minutes."""
    from src.data.circuit_breaker import odds_api_breaker

    if odds_api_breaker.should_skip():
        logger.warning("Odds API circuit breaker open — skipping FG odds poll")
        return
    logger.info("Polling full-game odds...")
    try:
        from src.data.odds_client import OddsClient

        # Ensure events are linked to Game rows BEFORE persisting odds,
        # otherwise persist_odds silently skips unlinked events.
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

        # Link events to Game rows before persisting per-event 1H odds.
        # Otherwise these snapshots can be skipped due to missing odds_api_id.
        await sync_events_to_games()

        client = OddsClient()
        events = await client.fetch_events()
        fetched = 0
        async with async_session_factory() as db:
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue
                # Respect quota check per call (6 credits each)
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
        from src.db.models import TeamSeasonStats

        client = BasketballClient()
        season = current_nba_season()
        async with async_session_factory() as db:
            # Fetch yesterday + today + tomorrow (UTC) to cover all
            # timezone boundaries.  NBA games at 10 PM ET = 03:00 UTC
            # next day, so a single date misses late tips.  Including
            # yesterday catches games that finished after midnight UTC.
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

            # Data completeness check
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
        from src.db.models import GameReferee, PlayerGameStats

        client = BasketballClient()
        async with async_session_factory() as db:
            # 1. Fetch Referees for ANY game that doesn't have them yet (upcoming or finished)
            ref_subq = select(GameReferee.game_id).distinct().scalar_subquery()
            # Fetch for games in the next 48 hours or last 24 hours
            from datetime import timedelta
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
                        # Clear any existing refs (savepoint handling)
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
    by home-team name within the same calendar day (UTC).  This handles
    the common case where the Basketball API and Odds API report
    slightly different tip-off times for the same game.
    """
    logger.info("Syncing events to games...")
    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        events = await client.fetch_events()
        matched = 0
        skipped = 0
        reconciled = 0
        async with async_session_factory() as db:
            # Pre-load team name -> id map so Odds events can be matched to official games.
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

                # Skip if already linked by odds_api_id
                existing = await db.execute(select(Game).where(Game.odds_api_id == odds_id))
                existing_game = existing.scalar_one_or_none()
                if existing_game is not None:
                    existing_game_id = int(cast(Any, getattr(existing_game, "id", 0)))
                    if existing_game_id < 0:
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


async def generate_predictions_and_publish() -> int:
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

        # 0) Refresh all upstream inputs the model depends on right before inference.
        #    That keeps prediction runs deterministic and avoids reusing stale local state.
        await poll_stats()
        await poll_scores_and_box()
        await poll_injuries()

        # 1) Sync Odds-API events → internal games so odds_api_id is set
        #    before persist_odds tries to link snapshots.
        await sync_events_to_games()

        # 2) Always pull fresh FG + 1H odds plus player props before generating predictions.
        #    This function runs infrequently (morning cron / manual refresh),
        #    so the API cost is justified vs. the risk of stale or missing inputs.
        _s = get_settings()
        await poll_fg_odds()
        await poll_1h_odds()
        await poll_player_props()

        async with async_session_factory() as db:
            purged_count = await purge_invalid_upcoming_predictions(db)
            if purged_count:
                logger.warning(
                    "Purged %d malformed upcoming predictions before publish",
                    purged_count,
                )

            predictor = Predictor()
            if not predictor.is_ready:
                logger.warning("Models not ready; skipping prediction publish")
                return 0

            # Pre-check: how many NS games do we have, and how many are
            # already linked to Odds API events and therefore eligible
            # for fresh prediction generation right now?
            ns_count_result = await db.execute(
                select(sa_func.count(Game.id)).where(Game.status == "NS")
            )
            ns_game_count = ns_count_result.scalar() or 0
            linked_ns_count_result = await db.execute(
                select(sa_func.count(Game.id)).where(
                    Game.status == "NS",
                    Game.odds_api_id.is_not(None),
                )
            )
            linked_ns_game_count = linked_ns_count_result.scalar() or 0
            unlinked_ns_game_count = max(ns_game_count - linked_ns_game_count, 0)
            logger.info(
                "Found %d NS games in database (%d odds-linked, %d awaiting odds coverage)",
                ns_game_count,
                linked_ns_game_count,
                unlinked_ns_game_count,
            )

            predictions = await predictor.predict_upcoming(db)
            if not predictions:
                if ns_game_count == 0:
                    logger.info("No NS games in database \u2014 nothing to predict")
                elif linked_ns_game_count == 0:
                    logger.info(
                        "No odds-linked NS games yet — waiting on odds coverage for %d games",
                        ns_game_count,
                    )
                else:
                    logger.error(
                        "DATA LOSS: 0 predictions generated for %d odds-linked NS games "
                        "(%d total NS games in DB)",
                        linked_ns_game_count,
                        ns_game_count,
                    )
                    from src.notifications.teams import send_alert

                    await send_alert(
                        "DATA LOSS — 0 Eligible Predictions",
                        f"{linked_ns_game_count} odds-linked NS games were eligible, but "
                        f"the prediction pipeline produced 0 predictions. Total NS games "
                        f"in DB: {ns_game_count}. Check model readiness, stored odds "
                        f"freshness, and feature availability.",
                        "error",
                    )
                return 0

            pred_count = len(predictions)
            if unlinked_ns_game_count:
                logger.info(
                    "Waiting on odds coverage for %d / %d NS games",
                    unlinked_ns_game_count,
                    ns_game_count,
                )
            if pred_count < linked_ns_game_count:
                logger.warning(
                    "INCOMPLETE ELIGIBLE COVERAGE: predicted %d / %d odds-linked NS games "
                    "(%d missing, %d awaiting odds coverage)",
                    pred_count,
                    linked_ns_game_count,
                    linked_ns_game_count - pred_count,
                    unlinked_ns_game_count,
                )
            else:
                logger.info(
                    "Full eligible coverage: predicted %d / %d odds-linked NS games",
                    pred_count,
                    linked_ns_game_count,
                )

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
                _s = get_settings()
                download_url: str | None = None

                # Link to the HTML slate if the API is reachable
                if _s.api_base_url:
                    download_url = f"{_s.api_base_url.rstrip('/')}/predictions/slate.html"

                if _s.teams_team_id and _s.teams_channel_id:
                    # ── Graph API path: card + CSV file + HTML slate ──
                    from src.notifications.teams import (
                        build_html_slate,
                        build_slate_csv,
                        send_html_via_graph,
                        upload_csv_to_channel,
                    )

                    # 1) Upload CSV to channel Files tab
                    csv_url: str | None = None
                    try:
                        csv_content = build_slate_csv(rows)
                        csv_filename = f"nba_slate_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
                        csv_url = await upload_csv_to_channel(
                            _s.teams_team_id,
                            _s.teams_channel_id,
                            csv_filename,
                            csv_content,
                        )
                        logger.info("CSV slate uploaded: %s", csv_url)
                    except Exception:
                        logger.warning("Failed to upload CSV slate", exc_info=True)

                    # 2) Send Adaptive Card (with CSV download link)
                    payload = build_teams_card(
                        rows,
                        _s.teams_max_games_per_message,
                        odds_pulled_at=odds_pulled_at,
                        download_url=download_url,
                        csv_download_url=csv_url,
                    )
                    await send_card_via_graph(
                        _s.teams_team_id,
                        _s.teams_channel_id,
                        payload,
                    )

                    # 3) Post full HTML slate as a follow-up message
                    try:
                        html = build_html_slate(rows, odds_pulled_at=odds_pulled_at)
                        await send_html_via_graph(
                            _s.teams_team_id,
                            _s.teams_channel_id,
                            html,
                        )
                        logger.info("HTML slate posted to Teams channel")
                    except Exception:
                        logger.warning("Failed to post HTML slate", exc_info=True)

                    logger.info("Published %d predictions to Teams (Graph API)", len(rows))

                elif _s.teams_webhook_url:
                    # ALWAYS dump the HTML file locally for the user too!
                    try:
                        from src.notifications.teams import build_html_slate
                        html = build_html_slate(rows, odds_pulled_at=odds_pulled_at)
                        with open("nba_picks_slate_livesync.html", "w", encoding="utf-8") as f:
                            f.write(html)
                        logger.info("Saved local HTML slate copy to nba_picks_slate_livesync.html")
                    except Exception as e:
                        logger.warning(f"Failed to dump html locally: {e}")

                    csv_dl = (
                        f"{_s.api_base_url.rstrip('/')}/predictions/slate.csv"
                        if _s.api_base_url
                        else None
                    )
                    payload = build_teams_card(
                        rows,
                        _s.teams_max_games_per_message,
                        odds_pulled_at=odds_pulled_at,
                        download_url=download_url,
                        csv_download_url=csv_dl,
                    )
                    await send_card_to_teams(_s.teams_webhook_url, payload)
                    logger.info("Published %d predictions to Teams (webhook)", len(rows))
                else:
                    logger.info("No Teams delivery configured; skipping publish")
            return len(rows)
    except Exception:
        logger.exception("Error generating/publishing predictions")
        from src.notifications.teams import send_alert

        await send_alert(
            "Prediction Publish Failed",
            "generate_predictions_and_publish raised an exception. Check worker logs.",
            "error",
        )
        return 0


async def check_prediction_drift() -> None:
    """Compare recent prediction distributions against 30-day trailing window."""
    logger.info("Checking prediction drift...")
    try:
        from datetime import timedelta

        import numpy as np

        from src.db.models import Prediction

        async with async_session_factory() as db:
            now = datetime.now(UTC).replace(tzinfo=None)
            cutoff_30d = now - timedelta(days=30)
            cutoff_7d = now - timedelta(days=7)

            # 30-day baseline
            result_30d = await db.execute(
                select(Prediction.predicted_home_fg, Prediction.predicted_away_fg).where(
                    Prediction.predicted_at > cutoff_30d
                )
            )
            rows_30d = result_30d.all()

            # Recent 7-day window
            result_7d = await db.execute(
                select(Prediction.predicted_home_fg, Prediction.predicted_away_fg).where(
                    Prediction.predicted_at > cutoff_7d
                )
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

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=30)
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
