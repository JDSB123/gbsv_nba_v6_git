"""Maintenance jobs — CLV fill, odds pruning, DB upkeep."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.db.models import Game, OddsSnapshot, Prediction
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)


async def fill_clv() -> None:
    """Fill closing-line value for predictions whose games have finished."""
    logger.info("Filling CLV for finished games...")
    try:
        from sqlalchemy import and_

        async with async_session_factory() as db:
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

            for pred in preds:
                pred_any = cast(Any, pred)
                game_obj = cast(Any, pred).game
                home_name = game_obj.home_team.name if game_obj and game_obj.home_team else ""
                game_start = getattr(game_obj, "commence_time", None)

                # Fetch last pre-game snapshots (true closing line)
                snap_query = (
                    select(OddsSnapshot)
                    .where(OddsSnapshot.game_id == pred.game_id)
                )
                if game_start is not None:
                    snap_query = snap_query.where(
                        OddsSnapshot.captured_at <= game_start
                    )
                snap_query = (
                    snap_query
                    .order_by(OddsSnapshot.captured_at.desc())
                    .limit(100)
                )
                odds_q = await db.execute(snap_query)
                snapshots = odds_q.scalars().all()
                if not snapshots:
                    continue

                # Use the most recent pre-game snapshot per market (true close)
                last_spread: float | None = None
                last_total: float | None = None
                for s in snapshots:
                    if last_spread is None and cast(Any, s.market) == "spreads" \
                            and s.point is not None \
                            and cast(Any, s.outcome_name) == home_name:
                        last_spread = float(cast(Any, s.point))
                    if last_total is None and cast(Any, s.market) == "totals" \
                            and s.point is not None:
                        last_total = float(cast(Any, s.point))
                    if last_spread is not None and last_total is not None:
                        break

                if last_spread is not None:
                    closing_spread = round(last_spread, 1)
                    pred_any.closing_spread = closing_spread
                    opening_spread = cast(Any, pred.opening_spread)
                    if opening_spread is not None:
                        pred_any.clv_spread = round(
                            float(closing_spread) - float(opening_spread), 1
                        )
                if last_total is not None:
                    closing_total = round(last_total, 1)
                    pred_any.closing_total = closing_total
                    opening_total = cast(Any, pred.opening_total)
                    if opening_total is not None:
                        pred_any.clv_total = round(float(closing_total) - float(opening_total), 1)

            await db.commit()
            logger.info("CLV filled for %d predictions", len(preds))
    except Exception:
        logger.exception("Error filling CLV")


async def prune_old_odds() -> None:
    """Delete odds snapshots older than 30 days for finished games."""
    logger.info("Pruning old odds snapshots...")
    try:
        from sqlalchemy import and_, delete

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=30)
        async with async_session_factory() as db:
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

        tables = [
            "games", "odds_snapshots", "game_odds_archive",
            "predictions", "player_game_stats",
        ]
        async with async_session_factory() as db:
            for table in tables:
                await db.execute(text(f"ANALYZE {table}"))
            await db.commit()
        logger.info("Database ANALYZE complete")
    except Exception:
        logger.exception("Error during database maintenance")


async def check_data_freshness() -> None:
    """Alert when any critical data source goes stale.

    Runs every 30 minutes.  Sends a Teams alert if full-game odds
    haven't been refreshed within 2× the configured interval.
    """
    from sqlalchemy import func as sa_func

    from src.config import get_settings

    settings = get_settings()
    logger.info("Checking data freshness...")
    try:
        async with async_session_factory() as db:
            now = datetime.now(UTC)
            alerts: list[str] = []

            # -- Odds freshness (threshold = 2× interval) --
            latest_odds = (
                await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
            ).scalar_one_or_none()
            if latest_odds:
                age_min = (now - latest_odds.replace(tzinfo=UTC)).total_seconds() / 60
                threshold = settings.odds_fg_interval * 2
                if age_min > threshold:
                    alerts.append(
                        f"Odds data STALE — last update {age_min:.0f} min ago "
                        f"(threshold {threshold} min)"
                    )
            else:
                alerts.append("Odds data MISSING — no snapshots in DB")

            # -- Games coverage --
            from src.data.seasons import current_nba_season as _cur_season
            from src.db.models import TeamSeasonStats

            team_count = (
                await db.execute(
                    select(sa_func.count())
                    .select_from(TeamSeasonStats)
                    .where(TeamSeasonStats.season == _cur_season())
                )
            ).scalar() or 0
            if team_count < 25:
                alerts.append(
                    f"Team stats incomplete — only {team_count}/30 teams have season stats"
                )

            if alerts:
                logger.warning("DATA FRESHNESS ISSUES:\n  %s", "\n  ".join(alerts))
                try:
                    from src.notifications.teams import send_alert

                    await send_alert(
                        "Data Pipeline Alert",
                        "\n".join(alerts),
                        "warning",
                    )
                except Exception:
                    logger.warning("Could not send freshness alert to Teams")
            else:
                logger.info("Data freshness OK — all sources within thresholds")
    except Exception:
        logger.exception("Error checking data freshness")
