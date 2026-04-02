"""Game-level feature builders: venue/streak, Elo/H2H/referee, derived, interactions."""

import logging
import math
from typing import Any, cast

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game, GameReferee, OddsSnapshot
from src.models.features._elo import build_elo_ratings
from src.models.features._helpers import (
    TEAM_TZ,
    NaN,
    _as_float,
    _as_str,
    _assigned_referee_names,
)

logger = logging.getLogger(__name__)


async def _venue_and_streak_features(
    db: AsyncSession, home_id: int, away_id: int, game: Any,
    prior: dict[str, float],
) -> dict[str, float]:
    features: dict[str, float] = {}

    # Expected game pace (interaction of prior features)
    home_pace = prior.get("home_pace", NaN)
    away_pace = prior.get("away_pace", NaN)
    if not (math.isfinite(home_pace) and math.isfinite(away_pace)):
        features["expected_pace"] = NaN
        features["pace_diff"] = NaN
    else:
        features["expected_pace"] = (home_pace + away_pace) / 2.0
        features["pace_diff"] = home_pace - away_pace

    # Venue splits
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        if prefix == "home":
            venue_filter = Game.home_team_id == team_id
        else:
            venue_filter = Game.away_team_id == team_id
        res_3 = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                venue_filter,
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(15)
        )
        venue_games = res_3.scalars().all()
        if venue_games:
            scored = []
            allowed = []
            for g in venue_games:
                if int(cast(Any, g.home_team_id)) == team_id:
                    scored.append(_as_float(cast(Any, g.home_score_fg)))
                    allowed.append(_as_float(cast(Any, g.away_score_fg)))
                else:
                    scored.append(_as_float(cast(Any, g.away_score_fg)))
                    allowed.append(_as_float(cast(Any, g.home_score_fg)))
            features[f"{prefix}_venue_ppg"] = float(np.mean(scored))
            features[f"{prefix}_venue_oppg"] = float(np.mean(allowed))
        else:
            features[f"{prefix}_venue_ppg"] = prior.get(f"{prefix}_ppg", NaN)
            features[f"{prefix}_venue_oppg"] = prior.get(f"{prefix}_oppg", NaN)

    # Win streak & L5/L10 record
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        res_4 = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(10)
        )
        streak_games = res_4.scalars().all()
        streak = 0
        if streak_games:
            first_won = None
            for g in streak_games:
                if int(cast(Any, g.home_team_id)) == team_id:
                    won = _as_float(cast(Any, g.home_score_fg)) > _as_float(
                        cast(Any, g.away_score_fg)
                    )
                else:
                    won = _as_float(cast(Any, g.away_score_fg)) > _as_float(
                        cast(Any, g.home_score_fg)
                    )
                if first_won is None:
                    first_won = won
                if won == first_won:
                    streak += 1 if won else -1
                else:
                    break
        features[f"{prefix}_win_streak"] = float(streak)

        l5_wins = 0
        l10_wins = 0
        for i, g in enumerate(streak_games):
            if int(cast(Any, g.home_team_id)) == team_id:
                won = _as_float(cast(Any, g.home_score_fg)) > _as_float(cast(Any, g.away_score_fg))
            else:
                won = _as_float(cast(Any, g.away_score_fg)) > _as_float(cast(Any, g.home_score_fg))
            if won:
                l10_wins += 1
                if i < 5:
                    l5_wins += 1
            elif i < 5:
                pass
        features[f"{prefix}_l5_record"] = float(l5_wins)
        features[f"{prefix}_l10_record"] = float(l10_wins)
    return features


async def _elo_h2h_referee_features(
    db: AsyncSession, game: Any, home_id: int, away_id: int,
) -> dict[str, float]:
    features: dict[str, float] = {}

    # Elo ratings
    elo = await build_elo_ratings(db)
    features["home_elo"] = elo.rating(home_id)
    features["away_elo"] = elo.rating(away_id)
    features["elo_diff"] = features["home_elo"] - features["away_elo"]

    # Head-to-head history
    res_5 = await db.execute(
        select(Game)
        .where(
            Game.status == "FT",
            ((Game.home_team_id == home_id) & (Game.away_team_id == away_id))
            | ((Game.home_team_id == away_id) & (Game.away_team_id == home_id)),
            Game.commence_time < game.commence_time,
        )
        .order_by(Game.commence_time.desc())
        .limit(10)
    )
    h2h_games = res_5.scalars().all()
    if h2h_games:
        h2h_wins = 0
        h2h_margins = []
        for g in h2h_games:
            if int(cast(Any, g.home_team_id)) == home_id:
                margin = _as_float(cast(Any, g.home_score_fg)) - _as_float(
                    cast(Any, g.away_score_fg)
                )
            else:
                margin = _as_float(cast(Any, g.away_score_fg)) - _as_float(
                    cast(Any, g.home_score_fg)
                )
            h2h_margins.append(margin)
            if margin > 0:
                h2h_wins += 1
        features["h2h_win_pct"] = h2h_wins / len(h2h_games)
        features["h2h_avg_margin"] = float(np.mean(h2h_margins))
    else:
        features["h2h_win_pct"] = 0.5
        features["h2h_avg_margin"] = NaN

    # Travel / timezone
    home_name = game.home_team.name if game.home_team is not None else ""
    away_name = game.away_team.name if game.away_team is not None else ""
    home_tz = TEAM_TZ.get(home_name, -5)
    away_tz = TEAM_TZ.get(away_name, -5)
    features["tz_diff"] = float(abs(home_tz - away_tz))

    # Referee History
    ref_names = _assigned_referee_names(game)
    if ref_names:
        ref_metrics = []
        for name in ref_names:
            ref_games_res = await db.execute(
                select(Game)
                .join(GameReferee)
                .where(
                    GameReferee.referee_name == name,
                    Game.status == "FT",
                    Game.commence_time < game.commence_time,
                    Game.home_score_fg.is_not(None),
                    Game.away_score_fg.is_not(None),
                )
            )
            ref_games = ref_games_res.scalars().all()
            if ref_games:
                pts = []
                hw_pct = []
                over_pct = []
                for rg in ref_games:
                    h_pts = _as_float(cast(Any, rg.home_score_fg))
                    a_pts = _as_float(cast(Any, rg.away_score_fg))
                    pts.append(h_pts + a_pts)
                    hw_pct.append(1.0 if h_pts > a_pts else 0.0)
                    total_res = await db.execute(
                        select(OddsSnapshot.point)
                        .where(
                            OddsSnapshot.game_id == rg.id,
                            OddsSnapshot.market == "totals",
                        )
                        .limit(1)
                    )
                    line = total_res.scalar()
                    if line:
                        over_pct.append(1.0 if (h_pts + a_pts) > _as_float(line) else 0.0)
                ref_metrics.append(
                    {
                        "pts": float(np.mean(pts)),
                        "hw": float(np.mean(hw_pct)),
                        "over": float(np.mean(over_pct)) if over_pct else NaN,
                    }
                )
        if ref_metrics:
            features["ref_avg_pts"] = float(np.mean([m["pts"] for m in ref_metrics]))
            features["ref_home_win_pct_bias"] = float(np.mean([m["hw"] for m in ref_metrics]))
            avg_over = [m["over"] for m in ref_metrics if not math.isnan(m["over"])]
            features["ref_over_pct"] = float(np.mean(avg_over)) if avg_over else NaN
        else:
            features["ref_avg_pts"] = NaN
            features["ref_home_win_pct_bias"] = NaN
            features["ref_over_pct"] = NaN
    else:
        features["ref_avg_pts"] = NaN
        features["ref_home_win_pct_bias"] = NaN
        features["ref_over_pct"] = NaN
    return features


def _derived_features(
    features: dict[str, float], game: Any,
) -> dict[str, float]:
    out: dict[str, float] = {}

    # Season progress
    _hw = features.get("home_wins", NaN)
    _hl = features.get("home_losses", NaN)
    _aw = features.get("away_wins", NaN)
    _al = features.get("away_losses", NaN)
    if any(math.isnan(v) for v in (_hw, _hl, _aw, _al)):
        out["season_progress"] = NaN
    else:
        out["season_progress"] = (_hw + _hl + _aw + _al) / (2.0 * 82.0)

    # Opponent-adjusted ratings
    _h_off = features.get("home_off_rating", NaN)
    _a_def = features.get("away_def_rating", NaN)
    _a_off = features.get("away_off_rating", NaN)
    _h_def = features.get("home_def_rating", NaN)
    out["home_adj_off"] = _h_off - _a_def
    out["home_adj_def"] = _a_off - _h_def
    out["rest_diff"] = features.get("home_rest_days", NaN) - features.get(
        "away_rest_days", NaN
    )

    # Matchup & Situational Styles
    _h_3pt = features.get("home_player_3pt_pct", NaN)
    _a_3pt = features.get("away_player_3pt_pct", NaN)
    out["matchup_3pt_diff"] = _h_3pt - _a_3pt

    month = game.commence_time.month if game.commence_time else 1
    urgency = 0.0
    if month in (3, 4):
        _h_win_pct = features.get("home_win_pct", 0.5)
        _a_win_pct = features.get("away_win_pct", 0.5)
        if _h_win_pct < 0.35:
            urgency -= 1.0
        elif _h_win_pct > 0.60:
            urgency += 0.5
        if _a_win_pct < 0.35:
            urgency += 1.0
        elif _a_win_pct > 0.60:
            urgency -= 0.5
    out["situational_urgency"] = urgency
    return out


def _interaction_features(
    features: dict[str, float], game: Any,
) -> dict[str, float]:
    out: dict[str, float] = {}

    _exp_pace = features.get("expected_pace", NaN)
    _m3pt = features.get("matchup_3pt_diff", NaN)
    out["pace_x_3pt_diff"] = _exp_pace * _m3pt

    _elo_d = features.get("elo_diff", NaN)
    _rest_d = features.get("rest_diff", NaN)
    out["elo_x_rest"] = _elo_d * _rest_d

    out["injury_diff"] = features.get("home_injury_impact", NaN) - features.get(
        "away_injury_impact", NaN
    )
    out["venue_scoring_edge"] = features.get("home_venue_ppg", NaN) - features.get(
        "away_venue_ppg", NaN
    )
    out["off_def_mismatch"] = features.get("home_adj_off", NaN) - features.get(
        "home_adj_def", NaN
    )
    out["streak_diff"] = features.get("home_win_streak", NaN) - features.get(
        "away_win_streak", NaN
    )

    # NaN prevalence check
    all_features = {**features, **out}
    total_feats = len(all_features)
    nan_count = sum(1 for v in all_features.values() if isinstance(v, float) and math.isnan(v))
    if total_feats > 0 and nan_count / total_feats > 0.3:
        nan_keys = [k for k, v in all_features.items() if isinstance(v, float) and math.isnan(v)]
        logger.warning(
            "High NaN prevalence in features for game %s: %d/%d (%.0f%%) are NaN. Missing keys: %s",
            game.id,
            nan_count,
            total_feats,
            nan_count / total_feats * 100,
            ", ".join(nan_keys[:20]) + ("..." if len(nan_keys) > 20 else ""),
        )
    return out
