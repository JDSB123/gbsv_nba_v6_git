import logging
from datetime import date
from typing import Any

import httpx
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.data.seasons import current_nba_season, parse_api_datetime
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

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
    ),
    before_sleep=lambda rs: logger.warning(
        "Retry #%d for %s", rs.attempt_number, rs.fn.__name__
    ),
    reraise=True,
)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def normalize_team_stats(stats: Any) -> dict[str, Any] | None:
    if isinstance(stats, dict):
        return stats
    if isinstance(stats, list) and stats and isinstance(stats[0], dict):
        return stats[0]
    return None


# NBA API injury status types → our schema status values
INJURY_STATUS_MAP: dict[str, str] = {
    "out": "out",
    "out for season": "out",
    "out indefinitely": "out",
    "doubtful": "doubtful",
    "day-to-day": "questionable",
    "questionable": "questionable",
    "probable": "probable",
}


def _pct_to_decimal(value: Any) -> float | None:
    """Convert a percentage value from the API to a 0-1 decimal.

    The Basketball API may return percentages as strings like ``"46.5"``
    (meaning 46.5%) or as decimals like ``"0.465"``.  Values > 1 are
    treated as whole-number percentages and divided by 100.
    """
    f = _as_float(value)
    if f is None or f <= 0:
        return None
    return f / 100 if f > 1 else f


def _compute_advanced_stats(
    s: dict[str, Any],
    games_played: int,
    ppg: float | None,
    oppg: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Derive pace, off_rating, and def_rating from API stats.

    Uses the Dean Oliver possession estimate when the API provides
    field-goal, free-throw, rebound, and turnover aggregates.  Falls
    back to a PPG/OPPG-based approximation otherwise.

    Returns (pace, off_rating, def_rating).
    """
    if not games_played or games_played <= 0:
        return None, None, None

    field_goals = s.get("field_goals", {}) or {}
    free_throws = s.get("free_throws", {}) or {}
    rebounds = s.get("rebounds", {}) or {}
    turnovers = s.get("turnovers", {}) or {}

    fg_made = _as_float(field_goals.get("total", {}).get("all"))
    fg_pct = _pct_to_decimal(field_goals.get("percentage", {}).get("all"))
    ft_made = _as_float(free_throws.get("total", {}).get("all"))
    ft_pct = _pct_to_decimal(free_throws.get("percentage", {}).get("all"))
    # Offensive rebounds may appear under various keys
    off_reb = (
        _as_float(rebounds.get("offReb", {}).get("all"))
        or _as_float(rebounds.get("off", {}).get("all"))
        or _as_float(rebounds.get("offensive", {}).get("all"))
    )
    tov = _as_float(turnovers.get("total", {}).get("all"))

    total_pts_for = _as_float(s.get("points", {}).get("for", {}).get("total", {}).get("all"))
    total_pts_against = _as_float(
        s.get("points", {}).get("against", {}).get("total", {}).get("all")
    )

    # ── Primary path: Dean Oliver possession estimate ──────────
    if fg_made and fg_pct and ft_made and ft_pct and tov is not None:
        fga = fg_made / fg_pct  # field goals attempted
        fta = ft_made / ft_pct  # free throws attempted
        orb = off_reb or 0.0
        total_poss = fga + 0.44 * fta - orb + tov
        if total_poss > 0:
            pace = total_poss / games_played
            off_rating = 100.0 * total_pts_for / total_poss if total_pts_for else None
            def_rating = 100.0 * total_pts_against / total_poss if total_pts_against else None
            return pace, off_rating, def_rating

    # ── Fallback: estimate from PPG / OPPG ─────────────────────
    if ppg and oppg and ppg > 0 and oppg > 0:
        # Average scoring ≈ proxy for pace (higher scoring → faster game)
        avg_scoring = (ppg + oppg) / 2.0
        # NBA league-average is ~112 ppg scoring and ~100 possessions;
        # scale proportionally for a stable per-100-possession rating.
        est_poss_per_game = avg_scoring / 1.12
        pace = est_poss_per_game
        off_rating = 100.0 * ppg / est_poss_per_game
        def_rating = 100.0 * oppg / est_poss_per_game
        return pace, off_rating, def_rating

    return None, None, None


class BasketballClient:
    """Client for api-sports.io Basketball API v1."""

    def __init__(self) -> None:
        self.base_url = settings.basketball_api_base
        self.api_key = settings.basketball_api_key
        self.league_id = settings.basketball_api_league_id

    def _headers(self) -> dict[str, str]:
        return {"x-apisports-key": self.api_key}

    def _resolve_season(self, season: str | None) -> str:
        return season or current_nba_season()

    @_RETRY
    async def _get(self, endpoint: str, params: dict | None = None) -> Any:
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
        self, game_date: date | None = None, season: str | None = None
    ) -> list[dict]:
        params: dict[str, Any] = {
            "league": self.league_id,
            "season": self._resolve_season(season),
        }
        if game_date:
            params["date"] = game_date.isoformat()
        return await self._get("games", params)

    async def fetch_team_stats(
        self, team_id: int, season: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return await self._get(
            "statistics",
            {
                "team": team_id,
                "league": self.league_id,
                "season": self._resolve_season(season),
            },
        )

    async def fetch_player_stats(self, game_id: int) -> list[dict]:
        return await self._get("games/statistics/players", {"id": game_id})

    async def fetch_team_game_stats(self, game_id: int) -> list[dict]:
        return await self._get("games/statistics/teams", {"id": game_id})

    async def fetch_standings(self, season: str | None = None) -> list[dict]:
        return await self._get(
            "standings",
            {"league": self.league_id, "season": self._resolve_season(season)},
        )

    async def fetch_h2h(self, team1_id: int, team2_id: int) -> list[dict]:
        return await self._get("games/h2h", {"h2h": f"{team1_id}-{team2_id}"})

    async def fetch_players(self, team_id: int, season: str | None = None) -> list[dict]:
        return await self._get(
            "players",
            {
                "team": team_id,
                "league": self.league_id,
                "season": self._resolve_season(season),
            },
        )

    # ── Injury data (NBA API v2, shares same api-sports key) ─────

    @_RETRY
    async def fetch_injuries(self, season: str | None = None) -> list[dict]:
        """Fetch current injury report from the NBA API.

        Uses ``v2.nba.api-sports.io`` which shares the same api-sports
        API key and provides a dedicated ``/players/injuries`` endpoint
        not available in the Basketball API v1.
        """
        resolved = self._resolve_season(season)
        # NBA API uses just the start year, e.g. "2024" not "2024-2025"
        nba_season = resolved.split("-")[0]
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.nba_api_base}/players/injuries",
                headers=self._headers(),
                params={"league": "standard", "season": nba_season},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", [])

    async def persist_injuries(self, injuries_data: list[dict], db: AsyncSession) -> int:
        """Replace current injury data with a fresh report.

        Clears the entire ``injuries`` table then re-populates from the
        latest API response.  Only players already tracked in the
        ``players`` table are linked; unknown players are skipped so the
        downstream feature code can look up their ``PlayerGameStats``.
        """
        await db.execute(delete(Injury))

        count = 0
        for entry in injuries_data:
            team_info = entry.get("team") or {}
            player_info = entry.get("player") or {}
            status_info = entry.get("status") or {}

            team_name = team_info.get("name", "")
            first = player_info.get("firstname", "")
            last = player_info.get("lastname", "")
            full_name = f"{first} {last}".strip()
            if not team_name or not full_name:
                continue

            raw_status = (status_info.get("type") or "").lower()
            mapped_status = INJURY_STATUS_MAP.get(raw_status, "questionable")
            description = status_info.get("description", "")

            # Look up team by name
            team_result = await db.execute(select(Team.id).where(Team.name == team_name))
            team_id_row = team_result.scalar_one_or_none()
            if team_id_row is None:
                continue

            # Look up player by name (case-insensitive) within team
            player_result = await db.execute(
                select(Player.id).where(
                    Player.team_id == team_id_row,
                    func.lower(Player.name) == full_name.lower(),
                )
            )
            player_id_row = player_result.scalar_one_or_none()
            if player_id_row is None:
                continue

            injury = Injury(
                player_id=player_id_row,
                team_id=team_id_row,
                status=mapped_status,
                description=description[:255] if description else None,
            )
            db.add(injury)
            count += 1

        await db.commit()
        logger.info("Refreshed %d injuries", count)
        return count

    # ── Persistence helpers ────────────────────────────────────────

    async def persist_teams(self, standings: list[dict], db: AsyncSession) -> None:
        """Upsert teams from standings data."""
        for group in standings:
            items = group if isinstance(group, list) else [group]
            for entry in items:
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
                commence = parse_api_datetime(commence)

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
        self,
        team_id: int,
        stats: dict[str, Any] | list[dict[str, Any]],
        season: str,
        db: AsyncSession,
    ) -> None:
        """Upsert team season stats."""
        if not stats:
            return
        s = normalize_team_stats(stats)
        if s is None:
            logger.warning(
                "Skipping unexpected team stats payload for team %s: %s",
                team_id,
                type(stats).__name__,
            )
            return
        games_data = s.get("games", {})
        points = s.get("points", {})

        games_played = _as_int(games_data.get("played", {}).get("all"))
        ppg = _as_float(points.get("for", {}).get("average", {}).get("all"))
        oppg = _as_float(points.get("against", {}).get("average", {}).get("all"))

        # Compute pace, off_rating, def_rating from box-score aggregates
        pace, off_rating, def_rating = _compute_advanced_stats(s, games_played, ppg, oppg)

        values = dict(
            team_id=team_id,
            season=season,
            games_played=games_played,
            wins=_as_int(games_data.get("wins", {}).get("all", {}).get("total")),
            losses=_as_int(games_data.get("loses", {}).get("all", {}).get("total")),
            ppg=ppg,
            oppg=oppg,
            pace=pace,
            off_rating=off_rating,
            def_rating=def_rating,
        )

        update_fields = {k: v for k, v in values.items() if k not in ("team_id", "season")}

        stmt = (
            pg_insert(TeamSeasonStats)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_team_season",
                set_=update_fields,
            )
        )
        await db.execute(stmt)
        await db.commit()

    async def persist_player_game_stats(
        self, game_id: int, stats_data: list[dict], db: AsyncSession
    ) -> int:
        """Upsert player box score stats for a game.

        The Basketball API v1 ``/games/statistics/players`` endpoint returns a
        **flat list** of per-player objects, each containing top-level keys:
        ``player``, ``team``, ``game``, ``points``, ``assists``, ``rebounds``,
        ``field_goals``, ``threepoint_goals``, ``freethrows_goals``, ``minutes``,
        etc.
        """
        count = 0
        for p in stats_data:
            player_info = p.get("player", {})
            player_id = player_info.get("id")
            if not player_id:
                continue

            team_id = p.get("team", {}).get("id", 0)

            # Ensure player exists
            existing = await db.execute(select(Player.id).where(Player.id == player_id))
            if existing.scalar_one_or_none() is None:
                player_obj = Player(
                    id=player_id,
                    team_id=team_id,
                    name=player_info.get("name", "Unknown"),
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
                    return int(str(val).split(":")[0]) if ":" in str(val) else int(val)
                except (ValueError, TypeError):
                    return None

            # Parse minutes from "MM:SS" format
            minutes = _safe_int(p.get("minutes"))

            # Extract nested stat objects
            fg = p.get("field_goals", {}) or {}
            tp = p.get("threepoint_goals", {}) or {}
            ft = p.get("freethrows_goals", {}) or {}
            reb = p.get("rebounds", {}) or {}

            stmt = (
                pg_insert(PlayerGameStats)
                .values(
                    player_id=player_id,
                    game_id=game_id,
                    minutes=minutes,
                    points=_safe_int(p.get("points")),
                    rebounds=_safe_int(reb.get("total")),
                    assists=_safe_int(p.get("assists")),
                    steals=_safe_int(p.get("steals")),
                    blocks=_safe_int(p.get("blocks")),
                    turnovers=_safe_int(p.get("turnovers")),
                    fg_pct=_safe_float(fg.get("percentage")),
                    three_pct=_safe_float(tp.get("percentage")),
                    ft_pct=_safe_float(ft.get("percentage")),
                    plus_minus=_safe_float(p.get("plusMinus")),
                )
                .on_conflict_do_update(
                    constraint="uq_player_game",
                    set_={
                        "minutes": minutes,
                        "points": _safe_int(p.get("points")),
                        "rebounds": _safe_int(reb.get("total")),
                        "assists": _safe_int(p.get("assists")),
                        "steals": _safe_int(p.get("steals")),
                        "blocks": _safe_int(p.get("blocks")),
                        "turnovers": _safe_int(p.get("turnovers")),
                        "fg_pct": _safe_float(fg.get("percentage")),
                        "three_pct": _safe_float(tp.get("percentage")),
                        "ft_pct": _safe_float(ft.get("percentage")),
                        "plus_minus": _safe_float(p.get("plusMinus")),
                    },
                )
            )
            await db.execute(stmt)
            count += 1
        await db.commit()
        logger.info("Persisted %d player stats for game %d", count, game_id)
        return count
