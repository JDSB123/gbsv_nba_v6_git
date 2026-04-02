"""Team-level feature builders: season stats, recent form, schedule, injuries."""

from datetime import timedelta
from typing import Any, cast

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game, Injury, PlayerGameStats, TeamSeasonStats
from src.models.features._helpers import (
    INJURY_WEIGHTS,
    NaN,
    _as_float,
    _as_str,
)


async def _team_season_stats(
    db: AsyncSession, home_id: int, away_id: int, season: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        ts_result = await db.execute(
            select(TeamSeasonStats).where(
                TeamSeasonStats.team_id == team_id,
                TeamSeasonStats.season == season,
            )
        )
        stats = ts_result.scalar_one_or_none()
        if stats:
            stats_any = cast(Any, stats)
            features[f"{prefix}_ppg"] = _as_float(stats_any.ppg)
            features[f"{prefix}_oppg"] = _as_float(stats_any.oppg)
            features[f"{prefix}_wins"] = _as_float(stats_any.wins)
            features[f"{prefix}_losses"] = _as_float(stats_any.losses)
            features[f"{prefix}_pace"] = _as_float(stats_any.pace)
            features[f"{prefix}_off_rating"] = _as_float(stats_any.off_rating)
            features[f"{prefix}_def_rating"] = _as_float(stats_any.def_rating)
            games_played = int(_as_float(stats_any.games_played))
            wins = _as_float(stats_any.wins)
            win_pct = wins / max(games_played, 1) if games_played else NaN
            features[f"{prefix}_win_pct"] = win_pct
        else:
            for key in [
                "ppg", "oppg", "wins", "losses", "pace",
                "off_rating", "def_rating", "win_pct",
            ]:
                features[f"{prefix}_{key}"] = NaN
    return features


async def _recent_form(
    db: AsyncSession, home_id: int, away_id: int, game: Any,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        for n, label in [(5, "l5"), (10, "l10")]:
            rg_result = await db.execute(
                select(Game)
                .where(
                    Game.status == "FT",
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                    Game.commence_time < game.commence_time,
                )
                .order_by(Game.commence_time.desc())
                .limit(n)
            )
            recent = rg_result.scalars().all()
            if recent:
                pts_scored = []
                pts_allowed = []
                h1_scored = []
                h1_allowed = []
                for g in recent:
                    if int(cast(Any, g.home_team_id)) == team_id:
                        pts_scored.append(_as_float(cast(Any, g.home_score_fg)))
                        pts_allowed.append(_as_float(cast(Any, g.away_score_fg)))
                        h1_scored.append(_as_float(cast(Any, g.home_score_1h)))
                        h1_allowed.append(_as_float(cast(Any, g.away_score_1h)))
                    else:
                        pts_scored.append(_as_float(cast(Any, g.away_score_fg)))
                        pts_allowed.append(_as_float(cast(Any, g.home_score_fg)))
                        h1_scored.append(_as_float(cast(Any, g.away_score_1h)))
                        h1_allowed.append(_as_float(cast(Any, g.home_score_1h)))
                features[f"{prefix}_{label}_pts_avg"] = float(np.mean(pts_scored))
                features[f"{prefix}_{label}_pts_allowed_avg"] = float(np.mean(pts_allowed))
                features[f"{prefix}_{label}_1h_pts_avg"] = float(np.mean(h1_scored))
                features[f"{prefix}_{label}_1h_allowed_avg"] = float(np.mean(h1_allowed))
            else:
                for k in ["pts_avg", "pts_allowed_avg", "1h_pts_avg", "1h_allowed_avg"]:
                    features[f"{prefix}_{label}_{k}"] = NaN
    return features


async def _schedule_features(
    db: AsyncSession, home_id: int, away_id: int, game: Any,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        lgt_result = await db.execute(
            select(Game.commence_time)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(1)
        )
        last_game_time = lgt_result.scalar_one_or_none()
        if last_game_time is not None:
            rest_days = (game.commence_time - last_game_time).days
            features[f"{prefix}_rest_days"] = float(rest_days)
            features[f"{prefix}_b2b"] = 1.0 if rest_days <= 1 else 0.0
        else:
            features[f"{prefix}_rest_days"] = 3.0
            features[f"{prefix}_b2b"] = 0.0

        week_ago = game.commence_time - timedelta(days=7)
        g7d_result = await db.execute(
            select(func.count(Game.id)).where(
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time.between(week_ago, game.commence_time),
                Game.status == "FT",
            )
        )
        features[f"{prefix}_games_7d"] = float(g7d_result.scalar() or 0)
    return features


async def _injury_features(
    db: AsyncSession, home_id: int, away_id: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        inj_result = await db.execute(select(Injury).where(Injury.team_id == team_id))
        injuries = inj_result.scalars().all()
        injury_impact = 0.0
        injured_count = 0
        for inj in injuries:
            weight = INJURY_WEIGHTS.get(_as_str(cast(Any, inj.status)).lower(), 0.0)
            if weight > 0:
                p_stats_result = await db.execute(
                    select(
                        func.avg(PlayerGameStats.points),
                        func.avg(PlayerGameStats.minutes),
                    ).where(PlayerGameStats.player_id == inj.player_id)
                )
                row = p_stats_result.one_or_none()
                avg_pts = float(row[0] or 0) if row else 0.0
                avg_min = float(row[1] or 0) if row else 0.0
                injury_impact += weight * avg_pts * (avg_min / 30.0)
                injured_count += 1
        features[f"{prefix}_injury_impact"] = injury_impact
        features[f"{prefix}_injured_count"] = float(injured_count)
    return features
