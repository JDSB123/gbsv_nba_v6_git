import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.db.models import Game, OddsSnapshot

logger = logging.getLogger(__name__)

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
    ),
    before_sleep=lambda rs: logger.warning("Retry #%d for %s", rs.attempt_number, rs.fn.__name__),
    reraise=True,
)


class OddsClient:
    """Client for The Odds API v4."""

    def __init__(self) -> None:
        _settings = get_settings()
        self.base_url = _settings.odds_api_base
        self.api_key = _settings.odds_api_key
        self.sport = _settings.odds_api_sport_key
        self._quota_min = _settings.odds_api_quota_min
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
        if self._remaining_quota is not None and self._remaining_quota < self._quota_min:
            logger.warning("Odds API quota low (%s), skipping request", self._remaining_quota)
            return True
        return False

    @_RETRY
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

    @_RETRY
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

    @_RETRY
    async def fetch_event_odds(
        self,
        event_id: str,
        markets: str = "h2h_h1,spreads_h1,totals_h1",
        regions: str = "us,us2",
        odds_format: str = "american",
    ) -> dict[str, Any]:
        """Fetch 1st-half odds for a specific event.

        Uses the per-event endpoint required for game-period markets.
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

    @_RETRY
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

    # ── Player props ───────────────────────────────────────────────

    PLAYER_PROP_MARKETS = (
        "player_points,player_rebounds,player_assists,player_threes,"
        "player_blocks,player_steals,player_turnovers,"
        "player_points_rebounds_assists,player_points_rebounds,"
        "player_points_assists,player_rebounds_assists,"
        "player_double_double,player_triple_double"
    )

    @_RETRY
    async def fetch_player_props(
        self,
        event_id: str,
        markets: str | None = None,
        regions: str = "us,us2",
        odds_format: str = "american",
    ) -> dict[str, Any]:
        """Fetch player prop odds for a specific event.

        Uses the per-event endpoint (required for non-featured markets).
        Cost: ~6 credits per event per region.
        """
        if self._should_skip():
            return {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets or self.PLAYER_PROP_MARKETS,
                    "oddsFormat": odds_format,
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._track_quota(resp)
            return resp.json()

    async def persist_odds(self, odds_data: list[dict], db: AsyncSession) -> int:
        """Parse odds response and insert OddsSnapshot rows. Returns count of inserted rows."""
        now = datetime.now(UTC).replace(tzinfo=None)
        count = 0
        skipped_no_game = 0
        for event in odds_data:
            odds_api_id = event.get("id")
            if not odds_api_id:
                continue
            # Look up internal game_id by odds_api_id
            result = await db.execute(select(Game.id).where(Game.odds_api_id == odds_api_id))
            game_id = result.scalar_one_or_none()
            if game_id is None:
                skipped_no_game += 1
                continue

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker["key"]
                for market in bookmaker.get("markets", []):
                    market_key = market["key"]
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price")
                        # Skip outcomes with missing or non-numeric price
                        if price is None:
                            continue
                        try:
                            price = float(price)
                        except (ValueError, TypeError):
                            logger.debug("Skipping outcome with invalid price: %s", price)
                            continue

                        point = outcome.get("point")
                        if point is not None:
                            try:
                                point = float(point)
                            except (ValueError, TypeError):
                                point = None

                        name = outcome.get("name")
                        if not name:
                            continue

                        snapshot = OddsSnapshot(
                            game_id=game_id,
                            source="odds_api",
                            bookmaker=bk_name,
                            market=market_key,
                            outcome_name=name[:128],
                            description=outcome.get("description"),
                            price=price,
                            point=point,
                            captured_at=now,
                        )
                        db.add(snapshot)
                        count += 1
        await db.commit()
        if skipped_no_game:
            logger.warning(
                "Persisted %d odds snapshots (%d events skipped — no matching game_id)",
                count,
                skipped_no_game,
            )
        else:
            logger.info("Persisted %d odds snapshots (all events matched)", count)
        return count
