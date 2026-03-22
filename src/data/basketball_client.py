import logging
from datetime import date, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.models import (
    Game,
    Injury,
    Player,
    PlayerGameStats,
    Team,
    TeamSeasonStats,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class BasketballClient:
    """Client for api-sports.io Basketball API v1."""

    def __init__(self) -> None:
        self.base_url = settings.basketball_api_base
        self.api_key = settings.basketball_api_key
        self.league_id = settings.basketball_api_league_id

    def _headers(self) -> dict[str, str]:
        return {"x-apisports-key": self.api_key}

    async def _get(
        self, endpoint: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/{endpoint}",
                headers=self._headers(),
                params=params or {},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", [])

    # ── Raw API calls ──────────────────────────────────────────────

    async def fetch_games(
        self, game_date: date | None = None, season: str = "2024-2025"
    ) -> list[dict]:
        params: dict[str, Any] = {"league": self.league_id, "season": season}
        if game_date:
            params["date"] = game_date.isoformat()
        return await self._get("games", params)

    async def fetch_team_stats(
        self, team_id: int, season: str = "2024-2025"
    ) -> list[dict]:
        return await self._get(
            "statistics",
            {"team": team_id, "league": self.league_id, "season": season},
        )

    async def fetch_player_stats(self, game_id: int) -> list[dict]:
        return await self._get("games/statistics/players", {"id": game_id})

    async def fetch_team_game_stats(self, game_id: int) -> list[dict]:
        return await self._get("games/statistics/teams", {"id": game_id})

    async def fetch_standings(self, season: str = "2024-2025") -> list[dict]:
        return await self._get(
            "standings", {"league": self.league_id, "season": season}
        )

    async def fetch_h2h(self, team1_id: int, team2_id: int) -> list[dict]:
        return await self._get("games/h2h", {"h2h": f"{team1_id}-{team2_id}"})

    async def fetch_players(
        self, team_id: int, season: str = "2024-2025"
    ) -> list[dict]:
        return await self._get(
            "players", {"team": team_id, "league": self.league_id, "season": season}
        )

    # ── Persistence helpers ────────────────────────────────────────

    async def persist_teams(self, standings: list[dict], db: AsyncSession) -> None:
        """Upsert teams from standings data."""
        for group in standings:
            if not isinstance(group, list):
                group = [group]
            for entry in group:
                team_data = entry.get("team", {})
                stmt = (
                    pg_insert(Team)
                    .values(
                        id=team_data["id"],
                        name=team_data["name"],
                        abbreviation=team_data.get("name", "")[:10],
                        conference=entry.get("group", {}).get("name"),
                    )
                    .on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "name": team_data["name"],
                            "conference": entry.get("group", {}).get("name"),
                        },
                    )
                )
                await db.execute(stmt)
        await db.commit()
        logger.info("Teams upserted from standings")

    async def persist_games(self, games: list[dict], db: AsyncSession) -> int:
        """Upsert games from Basketball API response. Returns count."""
        count = 0
        for g in games:
            game_id = g["id"]
            scores = g.get("scores", {})
            home = scores.get("home", {})
            away = scores.get("away", {})

            home_q1 = home.get("quarter_1")
            home_q2 = home.get("quarter_2")
            home_q3 = home.get("quarter_3")
            home_q4 = home.get("quarter_4")
            away_q1 = away.get("quarter_1")
            away_q2 = away.get("quarter_2")
            away_q3 = away.get("quarter_3")
            away_q4 = away.get("quarter_4")

            home_1h = (home_q1 or 0) + (home_q2 or 0) if home_q1 is not None else None
            away_1h = (away_q1 or 0) + (away_q2 or 0) if away_q1 is not None else None

            status_info = g.get("status", {})
            commence = g.get("date")
            if isinstance(commence, str):
                commence = datetime.fromisoformat(commence.replace("Z", "+00:00"))

            stmt = (
                pg_insert(Game)
                .values(
                    id=game_id,
                    home_team_id=g["teams"]["home"]["id"],
                    away_team_id=g["teams"]["away"]["id"],
                    commence_time=commence,
                    status=status_info.get("short", "NS"),
                    season=g.get("league", {}).get("season"),
                    home_q1=home_q1,
                    home_q2=home_q2,
                    home_q3=home_q3,
                    home_q4=home_q4,
                    home_ot=home.get("over_time", 0),
                    away_q1=away_q1,
                    away_q2=away_q2,
                    away_q3=away_q3,
                    away_q4=away_q4,
                    away_ot=away.get("over_time", 0),
                    home_score_1h=home_1h,
                    away_score_1h=away_1h,
                    home_score_fg=home.get("total"),
                    away_score_fg=away.get("total"),
                )
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "status": status_info.get("short", "NS"),
                        "home_q1": home_q1,
                        "home_q2": home_q2,
                        "home_q3": home_q3,
                        "home_q4": home_q4,
                        "away_q1": away_q1,
                        "away_q2": away_q2,
                        "away_q3": away_q3,
                        "away_q4": away_q4,
                        "home_score_1h": home_1h,
                        "away_score_1h": away_1h,
                        "home_score_fg": home.get("total"),
                        "away_score_fg": away.get("total"),
                    },
                )
            )
            await db.execute(stmt)
            count += 1
        await db.commit()
        logger.info("Upserted %d games", count)
        return count

    async def persist_team_season_stats(
        self, team_id: int, stats: list[dict], season: str, db: AsyncSession
    ) -> None:
        """Upsert team season stats."""
        if not stats:
            return
        s = stats[0]
        games_data = s.get("games", {})
        points = s.get("points", {})

        stmt = (
            pg_insert(TeamSeasonStats)
            .values(
                team_id=team_id,
                season=season,
                games_played=games_data.get("played", {}).get("all", 0),
                wins=games_data.get("wins", {}).get("all", {}).get("total", 0),
                losses=games_data.get("loses", {}).get("all", {}).get("total", 0),
                ppg=points.get("for", {}).get("average", {}).get("all"),
                oppg=points.get("against", {}).get("average", {}).get("all"),
            )
            .on_conflict_do_update(
                constraint="uq_team_season",
                set_={
                    "games_played": games_data.get("played", {}).get("all", 0),
                    "wins": games_data.get("wins", {}).get("all", {}).get("total", 0),
                    "losses": games_data.get("loses", {})
                    .get("all", {})
                    .get("total", 0),
                    "ppg": points.get("for", {}).get("average", {}).get("all"),
                    "oppg": points.get("against", {}).get("average", {}).get("all"),
                },
            )
        )
        await db.execute(stmt)
        await db.commit()

    async def persist_player_game_stats(
        self, game_id: int, stats_data: list[dict], db: AsyncSession
    ) -> None:
        """Upsert player box score stats for a game."""
        for team_stats in stats_data:
            players = team_stats.get("players", [])
            for p in players:
                player_id = p.get("player", {}).get("id")
                if not player_id:
                    continue

                # Ensure player exists
                existing = await db.execute(
                    select(Player.id).where(Player.id == player_id)
                )
                if existing.scalar_one_or_none() is None:
                    player_obj = Player(
                        id=player_id,
                        team_id=team_stats.get("team", {}).get("id", 0),
                        name=p.get("player", {}).get("name", "Unknown"),
                        position=p.get("pos"),
                    )
                    db.add(player_obj)
                    await db.flush()

                def _safe_float(val: Any) -> float | None:
                    if val is None or val == "":
                        return None
                    try:
                        return float(str(val).replace("%", ""))
                    except (ValueError, TypeError):
                        return None

                def _safe_int(val: Any) -> int | None:
                    if val is None or val == "":
                        return None
                    try:
                        return (
                            int(str(val).split(":")[0]) if ":" in str(val) else int(val)
                        )
                    except (ValueError, TypeError):
                        return None

                stmt = (
                    pg_insert(PlayerGameStats)
                    .values(
                        player_id=player_id,
                        game_id=game_id,
                        minutes=_safe_int(p.get("min")),
                        points=_safe_int(p.get("points")),
                        rebounds=_safe_int(p.get("totReb")),
                        assists=_safe_int(p.get("assists")),
                        steals=_safe_int(p.get("steals")),
                        blocks=_safe_int(p.get("blocks")),
                        turnovers=_safe_int(p.get("turnovers")),
                        fg_pct=_safe_float(p.get("fgp")),
                        three_pct=_safe_float(p.get("tpp")),
                        ft_pct=_safe_float(p.get("ftp")),
                        plus_minus=_safe_float(p.get("plusMinus")),
                    )
                    .on_conflict_do_update(
                        constraint="uq_player_game",
                        set_={
                            "minutes": _safe_int(p.get("min")),
                            "points": _safe_int(p.get("points")),
                            "rebounds": _safe_int(p.get("totReb")),
                            "assists": _safe_int(p.get("assists")),
                        },
                    )
                )
                await db.execute(stmt)
        await db.commit()
        logger.info("Persisted player stats for game %d", game_id)
