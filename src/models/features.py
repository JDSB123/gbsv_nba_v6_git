import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Game,
    Injury,
    OddsSnapshot,
    Player,
    PlayerGameStats,
    TeamSeasonStats,
)

logger = logging.getLogger(__name__)

# Injury status weights
INJURY_WEIGHTS = {"out": 1.0, "doubtful": 0.75, "questionable": 0.25, "probable": 0.05}


async def build_feature_vector(
    game: Game, db: AsyncSession
) -> dict[str, float] | None:
    """Build a feature dict for a single upcoming game."""
    home_id = game.home_team_id
    away_id = game.away_team_id
    season = game.season or "2024-2025"

    features: dict[str, float] = {}

    # ── Team season stats ───────────────────────────────────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        result = await db.execute(
            select(TeamSeasonStats).where(
                TeamSeasonStats.team_id == team_id,
                TeamSeasonStats.season == season,
            )
        )
        stats = result.scalar_one_or_none()
        if stats:
            features[f"{prefix}_ppg"] = stats.ppg or 0.0
            features[f"{prefix}_oppg"] = stats.oppg or 0.0
            features[f"{prefix}_wins"] = float(stats.wins or 0)
            features[f"{prefix}_losses"] = float(stats.losses or 0)
            features[f"{prefix}_pace"] = stats.pace or 0.0
            features[f"{prefix}_off_rating"] = stats.off_rating or 0.0
            features[f"{prefix}_def_rating"] = stats.def_rating or 0.0
            win_pct = stats.wins / max(stats.games_played, 1) if stats.games_played else 0.0
            features[f"{prefix}_win_pct"] = win_pct
        else:
            for key in ["ppg", "oppg", "wins", "losses", "pace", "off_rating", "def_rating", "win_pct"]:
                features[f"{prefix}_{key}"] = 0.0

    # ── Recent game averages (last 5 & 10) ──────────────────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        for n, label in [(5, "l5"), (10, "l10")]:
            result = await db.execute(
                select(Game)
                .where(
                    Game.status == "FT",
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                    Game.commence_time < game.commence_time,
                )
                .order_by(Game.commence_time.desc())
                .limit(n)
            )
            recent = result.scalars().all()
            if recent:
                pts_scored = []
                pts_allowed = []
                h1_scored = []
                h1_allowed = []
                for g in recent:
                    if g.home_team_id == team_id:
                        pts_scored.append(g.home_score_fg or 0)
                        pts_allowed.append(g.away_score_fg or 0)
                        h1_scored.append(g.home_score_1h or 0)
                        h1_allowed.append(g.away_score_1h or 0)
                    else:
                        pts_scored.append(g.away_score_fg or 0)
                        pts_allowed.append(g.home_score_fg or 0)
                        h1_scored.append(g.away_score_1h or 0)
                        h1_allowed.append(g.home_score_1h or 0)
                features[f"{prefix}_{label}_pts_avg"] = float(np.mean(pts_scored))
                features[f"{prefix}_{label}_pts_allowed_avg"] = float(np.mean(pts_allowed))
                features[f"{prefix}_{label}_1h_pts_avg"] = float(np.mean(h1_scored))
                features[f"{prefix}_{label}_1h_allowed_avg"] = float(np.mean(h1_allowed))
            else:
                for k in ["pts_avg", "pts_allowed_avg", "1h_pts_avg", "1h_allowed_avg"]:
                    features[f"{prefix}_{label}_{k}"] = 0.0

    # ── Rest days ───────────────────────────────────────────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        result = await db.execute(
            select(Game.commence_time)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(1)
        )
        last_game_time = result.scalar_one_or_none()
        if last_game_time:
            rest_days = (game.commence_time - last_game_time).days
            features[f"{prefix}_rest_days"] = float(rest_days)
            features[f"{prefix}_b2b"] = 1.0 if rest_days <= 1 else 0.0
        else:
            features[f"{prefix}_rest_days"] = 3.0
            features[f"{prefix}_b2b"] = 0.0

        # Games in last 7 days
        week_ago = game.commence_time - timedelta(days=7)
        result = await db.execute(
            select(func.count(Game.id)).where(
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time.between(week_ago, game.commence_time),
                Game.status == "FT",
            )
        )
        features[f"{prefix}_games_7d"] = float(result.scalar() or 0)

    # ── Injury impact ───────────────────────────────────────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        result = await db.execute(
            select(Injury).where(Injury.team_id == team_id)
        )
        injuries = result.scalars().all()
        injury_impact = 0.0
        injured_count = 0
        for inj in injuries:
            weight = INJURY_WEIGHTS.get(inj.status.lower(), 0.0)
            if weight > 0:
                # Estimate player value from season averages
                result2 = await db.execute(
                    select(
                        func.avg(PlayerGameStats.points),
                        func.avg(PlayerGameStats.minutes),
                    ).where(PlayerGameStats.player_id == inj.player_id)
                )
                row = result2.one_or_none()
                avg_pts = float(row[0] or 0) if row else 0.0
                avg_min = float(row[1] or 0) if row else 0.0
                injury_impact += weight * (avg_min * avg_pts / max(avg_min, 1))
                injured_count += 1
        features[f"{prefix}_injury_impact"] = injury_impact
        features[f"{prefix}_injured_count"] = float(injured_count)

    # ── Market signals (latest odds) ────────────────────────────
    result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.game_id == game.id)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(200)
    )
    snapshots = result.scalars().all()
    if snapshots:
        spreads = [s.point for s in snapshots if s.market == "spreads" and s.point is not None]
        totals = [s.point for s in snapshots if s.market == "totals" and s.point is not None]
        h1_spreads = [s.point for s in snapshots if s.market == "spreads_1st_half" and s.point is not None]
        h1_totals = [s.point for s in snapshots if s.market == "totals_1st_half" and s.point is not None]

        features["mkt_spread_avg"] = float(np.mean(spreads)) if spreads else 0.0
        features["mkt_spread_std"] = float(np.std(spreads)) if len(spreads) > 1 else 0.0
        features["mkt_total_avg"] = float(np.mean(totals)) if totals else 0.0
        features["mkt_total_std"] = float(np.std(totals)) if len(totals) > 1 else 0.0
        features["mkt_1h_spread_avg"] = float(np.mean(h1_spreads)) if h1_spreads else 0.0
        features["mkt_1h_total_avg"] = float(np.mean(h1_totals)) if h1_totals else 0.0

        # Moneyline implied probability
        h2h = [s for s in snapshots if s.market == "h2h"]
        if h2h:
            home_prices = [s.price for s in h2h if "home" in (s.outcome_name or "").lower() or s.outcome_name == game.home_team.name]
            if home_prices:
                avg_price = np.mean(home_prices)
                if avg_price < 0:
                    features["mkt_home_ml_prob"] = abs(avg_price) / (abs(avg_price) + 100)
                else:
                    features["mkt_home_ml_prob"] = 100 / (avg_price + 100)
            else:
                features["mkt_home_ml_prob"] = 0.5
        else:
            features["mkt_home_ml_prob"] = 0.5
    else:
        for k in ["mkt_spread_avg", "mkt_spread_std", "mkt_total_avg", "mkt_total_std",
                   "mkt_1h_spread_avg", "mkt_1h_total_avg", "mkt_home_ml_prob"]:
            features[k] = 0.0

    return features


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature names for the model."""
    cols = []
    for prefix in ["home", "away"]:
        cols.extend([
            f"{prefix}_ppg", f"{prefix}_oppg", f"{prefix}_wins", f"{prefix}_losses",
            f"{prefix}_pace", f"{prefix}_off_rating", f"{prefix}_def_rating", f"{prefix}_win_pct",
        ])
        for label in ["l5", "l10"]:
            cols.extend([
                f"{prefix}_{label}_pts_avg", f"{prefix}_{label}_pts_allowed_avg",
                f"{prefix}_{label}_1h_pts_avg", f"{prefix}_{label}_1h_allowed_avg",
            ])
        cols.extend([
            f"{prefix}_rest_days", f"{prefix}_b2b", f"{prefix}_games_7d",
            f"{prefix}_injury_impact", f"{prefix}_injured_count",
        ])
    cols.extend([
        "mkt_spread_avg", "mkt_spread_std", "mkt_total_avg", "mkt_total_std",
        "mkt_1h_spread_avg", "mkt_1h_total_avg", "mkt_home_ml_prob",
    ])
    return cols
