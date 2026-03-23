"""Historical data backfill — single-shot import of games, stats, odds.

Usage:
    python -m src backfill --season 2024-2025 --days 90

This is the ONLY way to seed the database with historical data.
It uses the same client classes and persist helpers as the scheduler,
ensuring a single source of truth for data ingestion logic.
"""

import logging
from datetime import timedelta
from typing import Any, cast

from sqlalchemy import select

from src.data.basketball_client import BasketballClient
from src.data.odds_client import OddsClient
from src.data.seasons import parse_api_datetime, resolve_backfill_window
from src.db.models import Game, Team
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)


async def run_backfill(season: str | None = None, days_back: int = 90) -> None:
    """Run the full backfill pipeline.

    Steps (in order):
    1. Standings → upsert teams
    2. Games for each day in the date range
    3. Team season stats for every team
    4. Player box scores for finished games
    5. Odds API event sync (map odds_api_id)
    6. Current odds snapshot for upcoming games
    7. Injury report
    """
    bball = BasketballClient()
    odds = OddsClient()

    resolved_season, start_date, end_date = resolve_backfill_window(season, days_back)

    async with async_session_factory() as db:
        # ── 1. Teams from standings ────────────────────────────
        logger.info(
            "Step 1/6: Fetching standings for season %s ...", resolved_season
        )
        standings = await bball.fetch_standings(season=resolved_season)
        if standings:
            await bball.persist_teams(standings, db)
            logger.info("  Upserted teams from standings")
        else:
            logger.warning("  No standings data returned")

        # ── 2. Games day-by-day ────────────────────────────────
        logger.info(
            "Step 2/6: Fetching games from %s through %s for season %s ...",
            start_date.isoformat(),
            end_date.isoformat(),
            resolved_season,
        )
        total_games = 0
        for offset in range((end_date - start_date).days + 1):
            game_date = start_date + timedelta(days=offset)
            games = await bball.fetch_games(game_date=game_date, season=resolved_season)
            if games:
                count = await bball.persist_games(games, db)
                total_games += count
        logger.info("  Total games upserted: %d", total_games)

        # ── 3. Team season stats ───────────────────────────────
        logger.info("Step 3/6: Fetching team season stats ...")
        result = await db.execute(select(Team.id))
        team_ids = [row[0] for row in result.fetchall()]
        for team_id in team_ids:
            stats = await bball.fetch_team_stats(team_id, season=resolved_season)
            if stats:
                await bball.persist_team_season_stats(
                    team_id,
                    stats,
                    resolved_season,
                    db,
                )
        logger.info("  Stats fetched for %d teams", len(team_ids))

        # ── 4. Player box scores for finished games ────────────
        logger.info("Step 4/6: Fetching player box scores ...")
        result = await db.execute(
            select(Game.id).where(Game.status == "FT").order_by(Game.commence_time)
        )
        finished_ids = [row[0] for row in result.fetchall()]
        box_count = 0
        for game_id in finished_ids:
            stats = await bball.fetch_player_stats(game_id)
            if stats:
                await bball.persist_player_game_stats(game_id, stats, db)
                box_count += 1
        logger.info("  Box scores fetched for %d games", box_count)

        # ── 5. Sync Odds API events to games ───────────────────
        logger.info("Step 5/6: Syncing Odds API events ...")
        events: list[dict[str, Any]] = []
        try:
            events = await odds.fetch_events()
            mapped = 0
            for event in events:
                commence = event.get("commence_time")
                if not commence:
                    continue
                ct = parse_api_datetime(commence)
                result = await db.execute(select(Game).where(Game.commence_time == ct))
                game = result.scalar_one_or_none()
                game_odds_api_id = (
                    cast(Any, game.odds_api_id) if game is not None else None
                )
                if game is not None and game_odds_api_id is None:
                    game.odds_api_id = event["id"]
                    mapped += 1
            await db.commit()
            logger.info("  Mapped %d events to games", mapped)
        except Exception:
            logger.exception(
                "Odds event sync failed during backfill; continuing without odds bootstrap"
            )

        # ── 6. Current odds for upcoming games ─────────────────
        logger.info("Step 6/6: Fetching current odds ...")
        try:
            fg_odds = await odds.fetch_odds()
            if fg_odds:
                count = await odds.persist_odds(fg_odds, db)
                logger.info("  Persisted %d full-game odds snapshots", count)

            # 1H odds for the next few events
            h1_count = 0
            for event in events[:10]:
                event_id = event.get("id")
                if event_id:
                    data = await odds.fetch_event_odds(event_id)
                    if data and data.get("bookmakers"):
                        h1_count += await odds.persist_odds([data], db)
            logger.info("  Persisted %d 1H odds snapshots", h1_count)
        except Exception:
            logger.exception(
                "Odds fetch failed during backfill; continuing without odds snapshots"
            )

        # ── 7. Injury report ──────────────────────────────────────
        logger.info("Step 7/7: Fetching injury report ...")
        try:
            injuries = await bball.fetch_injuries(season=resolved_season)
            if injuries:
                count = await bball.persist_injuries(injuries, db)
                logger.info("  Loaded %d injuries", count)
            else:
                logger.info("  No injury data available")
        except Exception:
            logger.exception(
                "Injury fetch failed during backfill; continuing without injury data"
            )

    logger.info("Backfill complete.")
