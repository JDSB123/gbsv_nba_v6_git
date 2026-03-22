"""Historical data backfill — single-shot import of games, stats, odds.

Usage:
    python -m src backfill --season 2024-2025 --days 90

This is the ONLY way to seed the database with historical data.
It uses the same client classes and persist helpers as the scheduler,
ensuring a single source of truth for data ingestion logic.
"""

import logging
from datetime import date, timedelta
from typing import Any, cast

from sqlalchemy import select

from src.data.basketball_client import BasketballClient
from src.data.odds_client import OddsClient
from src.db.models import Game, Team
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)


async def run_backfill(season: str = "2024-2025", days_back: int = 90) -> None:
    """Run the full backfill pipeline.

    Steps (in order):
    1. Standings → upsert teams
    2. Games for each day in the date range
    3. Team season stats for every team
    4. Player box scores for finished games
    5. Odds API event sync (map odds_api_id)
    6. Current odds snapshot for upcoming games
    """
    bball = BasketballClient()
    odds = OddsClient()

    async with async_session_factory() as db:
        # ── 1. Teams from standings ────────────────────────────
        logger.info("Step 1/6: Fetching standings for season %s ...", season)
        standings = await bball.fetch_standings(season=season)
        if standings:
            await bball.persist_teams(standings, db)
            logger.info("  Upserted teams from standings")
        else:
            logger.warning("  No standings data returned")

        # ── 2. Games day-by-day ────────────────────────────────
        logger.info("Step 2/6: Fetching games for last %d days ...", days_back)
        today = date.today()
        total_games = 0
        for offset in range(days_back, -1, -1):
            game_date = today - timedelta(days=offset)
            games = await bball.fetch_games(game_date=game_date, season=season)
            if games:
                count = await bball.persist_games(games, db)
                total_games += count
        logger.info("  Total games upserted: %d", total_games)

        # ── 3. Team season stats ───────────────────────────────
        logger.info("Step 3/6: Fetching team season stats ...")
        result = await db.execute(select(Team.id))
        team_ids = [row[0] for row in result.fetchall()]
        for team_id in team_ids:
            stats = await bball.fetch_team_stats(team_id, season=season)
            if stats:
                await bball.persist_team_season_stats(team_id, stats, season, db)
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
        events = await odds.fetch_events()
        mapped = 0
        for event in events:
            from datetime import datetime

            commence = event.get("commence_time")
            if not commence:
                continue
            ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            result = await db.execute(select(Game).where(Game.commence_time == ct))
            game = result.scalar_one_or_none()
            game_odds_api_id = cast(Any, game.odds_api_id) if game is not None else None
            if game is not None and game_odds_api_id is None:
                game.odds_api_id = event["id"]
                mapped += 1
        await db.commit()
        logger.info("  Mapped %d events to games", mapped)

        # ── 6. Current odds for upcoming games ─────────────────
        logger.info("Step 6/6: Fetching current odds ...")
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

    logger.info("Backfill complete.")
