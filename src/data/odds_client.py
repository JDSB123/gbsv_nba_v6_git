import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.db.models import Game, GameOddsArchive, OddsSnapshot

logger = logging.getLogger(__name__)

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
    ),
    before_sleep=lambda rs: logger.warning(
        "Retry #%d for %s",
        rs.attempt_number,
        getattr(rs.fn, "__name__", "<unknown>"),
    ),
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
        # Centralised from Settings — single source of truth
        self._regions = _settings.odds_api_regions
        self._markets_fg = _settings.odds_api_markets_fg
        self._markets_1h = _settings.odds_api_markets_1h

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
        markets: str | None = None,
        regions: str | None = None,
        odds_format: str = "american",
    ) -> list[dict[str, Any]]:
        """Fetch full-game odds for all upcoming NBA games.

        Regions and markets default to the centralized Settings values
        (us,us2,eu) to capture retail, offshore, AND sharp (Pinnacle) books.
        """
        if self._should_skip():
            return []
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions or self._regions,
                    "markets": markets or self._markets_fg,
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
        markets: str | None = None,
        regions: str | None = None,
        odds_format: str = "american",
    ) -> dict[str, Any]:
        """Fetch 1st-half odds for a specific event.

        Regions default to the centralized Settings value so all
        configured books are always polled.
        Cost: ~6 credits per event.
        """
        if self._should_skip():
            return {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions or self._regions,
                    "markets": markets or self._markets_1h,
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
        regions: str | None = None,
        odds_format: str = "american",
    ) -> dict[str, Any]:
        """Fetch player prop odds for a specific event.

        Regions default to the centralized Settings value.
        Cost: ~6 credits per event per region.
        """
        if self._should_skip():
            return {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions or self._regions,
                    "markets": markets or self.PLAYER_PROP_MARKETS,
                    "oddsFormat": odds_format,
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._track_quota(resp)
            return resp.json()

    async def persist_odds(self, odds_data: list[dict], db: AsyncSession) -> int:
        """Parse odds response and persist odds records.

        Inserts `OddsSnapshot` rows for each parsed outcome and also writes
        first-seen-per-day records to `GameOddsArchive`. Returns the count of
        inserted `OddsSnapshot` rows.
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        capture_date = now.date()
        count = 0
        skipped_no_game = 0
        skipped_no_id = 0
        archive_rows: list[dict] = []
        for event in odds_data:
            odds_api_id = event.get("id")
            if not odds_api_id:
                skipped_no_id += 1
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
                        # Collect for permanent archive (first-seen per day)
                        archive_rows.append({
                            "game_id": game_id,
                            "source": "odds_api",
                            "bookmaker": bk_name,
                            "market": market_key,
                            "outcome_name": name[:128],
                            "description": outcome.get("description"),
                            "price": price,
                            "point": point,
                            "capture_date": capture_date,
                            "captured_at": now,
                        })
        # Batch-insert into permanent archive using ON CONFLICT DO NOTHING so we
        # keep the FIRST snapshot of each day for every game/bookmaker/market/outcome.
        if archive_rows:
            await db.execute(
                pg_insert(GameOddsArchive)
                .values(archive_rows)
                .on_conflict_do_nothing(
                    index_elements=[
                        "game_id", "bookmaker", "market", "outcome_name", "capture_date",
                    ]
                )
            )

        await db.commit()

        if skipped_no_id:
            logger.warning(
                "persist_odds: %d events had no 'id' field and were skipped",
                skipped_no_id,
            )
        total_events = len(odds_data)
        if skipped_no_game:
            skip_ratio = skipped_no_game / max(total_events, 1)
            log_fn = logger.error if skip_ratio > 0.5 else logger.warning
            log_fn(
                "Persisted %d odds snapshots (%d/%d events skipped — no matching game_id, %.0f%%)",
                count,
                skipped_no_game,
                total_events,
                skip_ratio * 100,
            )
        else:
            logger.info("Persisted %d odds snapshots (all events matched)", count)
        return count

