"""Prediction jobs — generate, publish, cache, and validate predictions."""

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.db.models import Game, OddsSnapshot, Prediction, Team
from src.db.session import async_session_factory
from src.services.prediction_integrity import prediction_payload_has_integrity_issues

logger = logging.getLogger(__name__)

# Dedup flag: tracks the date for which pregame publish already fired
_pregame_published_date: date | None = None


async def purge_invalid_upcoming_predictions(db: Any) -> int:
    """Remove malformed upcoming predictions that cannot be safely published."""
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


async def pregame_check() -> None:
    """Publish predictions once, ~1 hour before the first game of the day."""
    global _pregame_published_date

    et = ZoneInfo("US/Eastern")
    now = datetime.now(et)
    today = now.date()

    if _pregame_published_date == today:
        return

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


async def generate_predictions_and_publish(
    *,
    publish: bool = True,
    refresh_inputs: bool = True,
    send_alerts: bool = True,
) -> int:
    """Generate upcoming predictions and publish formatted output to Teams."""
    logger.info("Generating predictions and publishing to Teams...")
    try:
        from sqlalchemy import func as sa_func

        from src.data.jobs.polling import (
            poll_1h_odds,
            poll_fg_odds,
            poll_injuries,
            poll_player_props,
            poll_scores_and_box,
            poll_stats,
            sync_events_to_games,
        )
        from src.models.features import reset_elo_cache
        from src.models.predictor import Predictor
        from src.notifications.teams import (
            build_teams_card,
            send_card_to_teams,
            send_card_via_graph,
        )

        reset_elo_cache()

        if refresh_inputs:
            await poll_stats()
            await poll_scores_and_box()
            await poll_injuries()
            await sync_events_to_games()
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
                    logger.info("No NS games in database — nothing to predict")
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
                    if send_alerts:
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

            odds_ts_result = await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
            odds_pulled_at = odds_ts_result.scalar_one_or_none()

            if rows and publish:
                _s = get_settings()
                download_url: str | None = None

                if _s.api_base_url:
                    download_url = f"{_s.api_base_url.rstrip('/')}/predictions/slate.html"

                if _s.teams_team_id and _s.teams_channel_id:
                    from src.notifications.teams import (
                        build_html_slate,
                        build_slate_csv,
                        send_html_via_graph,
                        upload_csv_to_channel,
                        upload_html_to_channel,
                    )

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

                    # Build single consolidated HTML (picks + odds + data sources)
                    html = build_html_slate(rows, odds_pulled_at=odds_pulled_at)

                    # Upload HTML to channel Files tab (OneDrive) for download
                    html_dl_url: str | None = None
                    try:
                        html_filename = f"GBSV_NBA_Slate_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.html"
                        html_dl_url = await upload_html_to_channel(
                            _s.teams_team_id,
                            _s.teams_channel_id,
                            html_filename,
                            html,
                        )
                        logger.info("HTML slate uploaded to OneDrive: %s", html_dl_url)
                    except Exception:
                        logger.warning("Failed to upload HTML slate to OneDrive", exc_info=True)

                    payload = build_teams_card(
                        rows,
                        _s.teams_max_games_per_message,
                        odds_pulled_at=odds_pulled_at,
                        download_url=html_dl_url or download_url,
                        csv_download_url=csv_url,
                    )
                    await send_card_via_graph(
                        _s.teams_team_id,
                        _s.teams_channel_id,
                        payload,
                    )

                    try:
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
                    try:
                        from src.notifications.teams import build_html_slate
                        html = build_html_slate(rows, odds_pulled_at=odds_pulled_at)
                        with open("nba_picks_slate_livesync.html", "w", encoding="utf-8") as f:
                            f.write(html)
                        logger.info("Saved local HTML slate copy to nba_picks_slate_livesync.html")
                    except Exception as e:
                        logger.warning("Failed to dump html locally: %s", e)

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
        if send_alerts:
            from src.notifications.teams import send_alert

            await send_alert(
                "Prediction Publish Failed",
                "generate_predictions_and_publish raised an exception. Check worker logs.",
                "error",
            )
        return 0


async def refresh_prediction_cache() -> int:
    """Refresh cached predictions without re-polling inputs or publishing side effects."""
    logger.info("Refreshing cached predictions from current persisted inputs...")
    return await generate_predictions_and_publish(
        publish=False,
        refresh_inputs=False,
        send_alerts=False,
    )


async def check_prediction_drift() -> None:
    """Compare recent prediction distributions against 30-day trailing window."""
    logger.info("Checking prediction drift...")
    try:
        import numpy as np

        async with async_session_factory() as db:
            now = datetime.now(UTC).replace(tzinfo=None)
            cutoff_30d = now - timedelta(days=30)
            cutoff_7d = now - timedelta(days=7)

            result_30d = await db.execute(
                select(Prediction.predicted_home_fg, Prediction.predicted_away_fg).where(
                    Prediction.predicted_at > cutoff_30d
                )
            )
            rows_30d = result_30d.all()

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
