import logging
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.models import Game, OddsSnapshot

logger = logging.getLogger(__name__)
settings = get_settings()


class OddsClient:
    """Client for The Odds API v4."""

    def __init__(self) -> None:
        self.base_url = settings.odds_api_base
        self.api_key = settings.odds_api_key
        self.sport = settings.odds_api_sport_key
        self._remaining_quota: int | None = None

    def _track_quota(self, response: httpx.Response) -> None:
        remaining = response.headers.get("x-requests-remaining")
        if remaining is not None:
            self._remaining_quota = int(remaining)
            logger.info("Odds API quota remaining: %s", self._remaining_quota)

    @property
    def quota_remaining(self) -> int | None:
        return self._remaining_quota

    def _should_skip(self) -> bool:
        if self._remaining_quota is not None and self._remaining_quota < settings.odds_api_quota_min:
            logger.warning("Odds API quota low (%s), skipping request", self._remaining_quota)
            return True
        return False

    async def fetch_events(self) -> list[dict[str, Any]]:
        """Fetch upcoming NBA events (free, no quota cost)."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/events",
                params={"apiKey": self.api_key},
                timeout=30,
            )
            resp.raise_for_status()
            self._track_quota(resp)
            return resp.json()

    async def fetch_odds(
        self,
        markets: str = "h2h,spreads,totals",
        regions: str = "us,us2",
        odds_format: str = "american",
    ) -> list[dict[str, Any]]:
        """Fetch full-game odds for all upcoming NBA games.

        Uses us+us2 regions to capture both retail (square) and
        offshore/professional (sharp, e.g. Pinnacle) book lines.
        Cost: ~6 credits (3 per region).
        """
        if self._should_skip():
            return []
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._track_quota(resp)
            return resp.json()

    async def fetch_event_odds(
        self,
        event_id: str,
        markets: str = "h2h_1st_half,spreads_1st_half,totals_1st_half",
        regions: str = "us,us2",
        odds_format: str = "american",
    ) -> dict[str, Any]:
        """Fetch 1st-half odds for a specific event.

        Uses us+us2 regions for sharp/square coverage.
        Cost: ~6 credits per event.
        """
        if self._should_skip():
            return {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._track_quota(resp)
            return resp.json()

    async def fetch_scores(self, days_from: int = 1) -> list[dict[str, Any]]:
        """Fetch recent scores. Cost: 2 credits."""
        if self._should_skip():
            return []
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/scores",
                params={"apiKey": self.api_key, "daysFrom": days_from},
                timeout=30,
            )
            resp.raise_for_status()
            self._track_quota(resp)
            return resp.json()

    async def persist_odds(self, odds_data: list[dict], db: AsyncSession) -> int:
        """Parse odds response and insert OddsSnapshot rows. Returns count of inserted rows."""
        now = datetime.utcnow()
        count = 0
        for event in odds_data:
            odds_api_id = event.get("id")
            # Look up internal game_id by odds_api_id
            result = await db.execute(
                select(Game.id).where(Game.odds_api_id == odds_api_id)
            )
            game_id = result.scalar_one_or_none()
            if game_id is None:
                continue

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker["key"]
                for market in bookmaker.get("markets", []):
                    market_key = market["key"]
                    for outcome in market.get("outcomes", []):
                        snapshot = OddsSnapshot(
                            game_id=game_id,
                            source="odds_api",
                            bookmaker=bk_name,
                            market=market_key,
                            outcome_name=outcome["name"],
                            price=outcome["price"],
                            point=outcome.get("point"),
                            captured_at=now,
                        )
                        db.add(snapshot)
                        count += 1
        await db.commit()
        logger.info("Persisted %d odds snapshots", count)
        return count
