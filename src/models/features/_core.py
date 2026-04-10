import logging
import math
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, cast

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.seasons import season_for_date
from src.db.models import (
    Game,
    GameReferee,
    Injury,
    OddsSnapshot,
    Player,
    PlayerGameStats,
    TeamSeasonStats,
)
from src.models.elo import EloSystem

logger = logging.getLogger(__name__)


NaN = float("nan")


def _as_float(value: Any, default: float = NaN) -> float:
    return float(value) if value is not None else default


def _as_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


def _assigned_referee_names(game: Any) -> list[str]:
    """Return assigned referee names when the relation is present."""
    referees = getattr(game, "referees", None) or []
    return [
        _as_str(getattr(referee, "referee_name", "")).strip()
        for referee in referees
        if _as_str(getattr(referee, "referee_name", "")).strip()
    ]


def _home_spreads(
    snapshots: Sequence[Any],
    home_team_name: str,
    market: str = "spreads",
    books: frozenset[str] | None = None,
) -> list[float]:
    """Extract the home team's spread values from odds snapshots.

    The Odds API returns two outcomes per bookmaker (one per team) with
    opposite-signed point values.  Averaging both cancels to ~0 and
    discards all directional information.

    This function keeps only the home team's outcome, preserving the
    standard betting convention::

        negative  -> home favorite  (e.g. -5.5 means home gives 5.5)
        positive  -> home underdog  (e.g. +3.0 means home gets 3)

    Parameters
    ----------
    books : optional frozenset — only include bookmakers in this set.
    """
    result: list[float] = []
    for s in snapshots:
        s_market = getattr(s, "market", None)
        s_outcome = getattr(s, "outcome_name", None)
        if s_market is None or s_outcome is None:
            continue
        if _as_str(s_market) != market or s.point is None:
            continue
        if _as_str(s_outcome) != home_team_name:
            continue
        if books is not None and _as_str(s.bookmaker).lower() not in books:
            continue
        result.append(_as_float(s.point))
    return result


# Injury status weights
INJURY_WEIGHTS = {"out": 1.0, "doubtful": 0.75, "questionable": 0.25, "probable": 0.05}

# ── NBA team timezone offsets (UTC) for travel features ─────
# Maps api-sports team IDs to UTC offset (EST=-5, CST=-6, MST=-7, PST=-8)
# Populated lazily by team *name* when IDs aren't known yet.
TEAM_TZ: dict[str, int] = {
    # Eastern (-5)
    "Boston Celtics": -5,
    "Brooklyn Nets": -5,
    "New York Knicks": -5,
    "Philadelphia 76ers": -5,
    "Toronto Raptors": -5,
    "Chicago Bulls": -6,
    "Cleveland Cavaliers": -5,
    "Detroit Pistons": -5,
    "Indiana Pacers": -5,
    "Milwaukee Bucks": -6,
    "Atlanta Hawks": -5,
    "Charlotte Hornets": -5,
    "Miami Heat": -5,
    "Orlando Magic": -5,
    "Washington Wizards": -5,
    # Central (-6)
    "Dallas Mavericks": -6,
    "Houston Rockets": -6,
    "Memphis Grizzlies": -6,
    "New Orleans Pelicans": -6,
    "San Antonio Spurs": -6,
    "Minnesota Timberwolves": -6,
    "Oklahoma City Thunder": -6,
    # Mountain (-7)
    "Denver Nuggets": -7,
    "Utah Jazz": -7,
    "Phoenix Suns": -7,
    # Pacific (-8)
    "Golden State Warriors": -8,
    "LA Clippers": -8,
    "Los Angeles Lakers": -8,
    "Portland Trail Blazers": -8,
    "Sacramento Kings": -8,
}

# ── Shared Elo instance (built once per trainer/predictor run) ──
_elo_system: EloSystem | None = None

# ── Sharp vs. Square book classification ────────────────────
# Sharp books: professional-facing, efficient lines, low vig
SHARP_BOOKS = frozenset(
    {
        "pinnacle",
        "lowvig",
        "betonlineag",
    }
)
# Square books: retail-facing, wider vig, public-driven lines
SQUARE_BOOKS = frozenset(
    {
        "fanduel",
        "draftkings",
        "betmgm",
        "pointsbetus",
        "caesars",
        "wynnbet",
        "unibet_us",
        "betrivers",
        "superbook",
        "twinspires",
        "betus",
    }
)


async def build_elo_ratings(db: AsyncSession) -> EloSystem:
    """Build Elo ratings from all completed games (called once per run)."""
    global _elo_system  # noqa: PLW0603
    if _elo_system is not None:
        return _elo_system

    res_1 = await db.execute(
        select(Game)
        .where(
            Game.status == "FT",
            Game.home_score_fg.is_not(None),
            Game.away_score_fg.is_not(None),
        )
        .order_by(Game.commence_time)
    )
    games = res_1.scalars().all()
    elo = EloSystem()
    for g in games:
        elo.update(
            int(cast(Any, g.home_team_id)),
            int(cast(Any, g.away_team_id)),
            _as_float(cast(Any, g.home_score_fg)),
            _as_float(cast(Any, g.away_score_fg)),
            season=_as_str(cast(Any, g.season)),
        )
    _elo_system = elo
    logger.info("Built Elo ratings from %d games", len(games))
    return elo


def reset_elo_cache() -> None:
    """Clear cached Elo so it's rebuilt on next call (e.g. after retrain)."""
    global _elo_system  # noqa: PLW0603
    _elo_system = None


# ── Feature-vector helper functions ─────────────────────────────


async def _team_season_stats(
    db: AsyncSession,
    home_id: int,
    away_id: int,
    season: str,
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
                "ppg",
                "oppg",
                "wins",
                "losses",
                "pace",
                "off_rating",
                "def_rating",
                "win_pct",
            ]:
                features[f"{prefix}_{key}"] = NaN
    return features


async def _recent_form(
    db: AsyncSession,
    home_id: int,
    away_id: int,
    game: Any,
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

                # Recency-weighted averages (exponential decay, λ=0.15)
                # Most recent game index=0 gets highest weight
                decay = 0.15
                weights = np.array([math.exp(-decay * i) for i in range(len(recent))])
                weights /= weights.sum()
                features[f"{prefix}_{label}_pts_wavg"] = float(np.dot(weights, pts_scored))
                features[f"{prefix}_{label}_pts_allowed_wavg"] = float(np.dot(weights, pts_allowed))
            else:
                for k in [
                    "pts_avg",
                    "pts_allowed_avg",
                    "1h_pts_avg",
                    "1h_allowed_avg",
                    "pts_wavg",
                    "pts_allowed_wavg",
                ]:
                    features[f"{prefix}_{label}_{k}"] = NaN
    return features


async def _schedule_features(
    db: AsyncSession,
    home_id: int,
    away_id: int,
    game: Any,
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
    db: AsyncSession,
    home_id: int,
    away_id: int,
    game: Any,
) -> dict[str, float]:
    features: dict[str, float] = {}
    game_time = cast(Any, game.commence_time)
    if game_time is None:
        for prefix in ("home", "away"):
            features[f"{prefix}_injury_impact"] = 0.0
            features[f"{prefix}_injured_count"] = 0.0
        return features

    window_start = game_time - timedelta(days=14)
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        inj_result = await db.execute(
            select(Injury).where(
                Injury.team_id == team_id,
                Injury.reported_at >= window_start,
                Injury.reported_at <= game_time,
                Injury.status.in_(["out", "doubtful", "questionable", "probable"]),
            )
        )
        injuries = inj_result.scalars().all()

        # Batch-fetch player averages for all injured players at once
        player_ids = [inj.player_id for inj in injuries]
        player_avg: dict[int, tuple[float, float]] = {}
        if player_ids:
            avg_result = await db.execute(
                select(
                    PlayerGameStats.player_id,
                    func.avg(PlayerGameStats.points),
                    func.avg(PlayerGameStats.minutes),
                )
                .where(PlayerGameStats.player_id.in_(player_ids))
                .group_by(PlayerGameStats.player_id)
            )
            for row in avg_result.all():
                player_avg[int(row[0])] = (float(row[1] or 0), float(row[2] or 0))

        injury_impact = 0.0
        injured_count = 0
        for inj in injuries:
            weight = INJURY_WEIGHTS.get(_as_str(cast(Any, inj.status)).lower(), 0.0)
            if weight > 0:
                avg_pts, avg_min = player_avg.get(int(inj.player_id), (0.0, 0.0))
                injury_impact += weight * avg_pts * (avg_min / 30.0)
                injured_count += 1
        features[f"{prefix}_injury_impact"] = injury_impact
        features[f"{prefix}_injured_count"] = float(injured_count)
    return features


async def _player_and_quarter_features(
    db: AsyncSession,
    home_id: int,
    away_id: int,
    game: Any,
) -> dict[str, float]:
    from collections import defaultdict

    features: dict[str, float] = {}
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        recent_game_ids_result = await db.execute(
            select(Game.id)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(5)
        )
        recent_game_ids = [r[0] for r in recent_game_ids_result.fetchall()]

        if recent_game_ids:
            pgs_result = await db.execute(
                select(PlayerGameStats)
                .join(Player, PlayerGameStats.player_id == Player.id)
                .where(
                    PlayerGameStats.game_id.in_(recent_game_ids),
                    Player.team_id == team_id,
                )
            )
            pgs_rows = pgs_result.scalars().all()
        else:
            pgs_rows = []

        if pgs_rows:
            all_pts = [_as_float(cast(Any, p.points)) for p in pgs_rows]
            all_ast = [_as_float(cast(Any, p.assists)) for p in pgs_rows]
            all_reb = [_as_float(cast(Any, p.rebounds)) for p in pgs_rows]
            all_fg = [_as_float(cast(Any, p.fg_pct)) for p in pgs_rows if p.fg_pct is not None]
            all_3pt = [
                _as_float(cast(Any, p.three_pct)) for p in pgs_rows if p.three_pct is not None
            ]
            all_min = [_as_float(cast(Any, p.minutes)) for p in pgs_rows]

            by_game: dict[int, list[Any]] = defaultdict(list)
            for p in pgs_rows:
                by_game[int(cast(Any, p.game_id))].append(p)

            starter_pts_list = []
            bench_pts_list = []
            for _gid, players in by_game.items():
                sorted_p = sorted(
                    players, key=lambda x: _as_float(cast(Any, x.minutes)), reverse=True
                )
                starters = sorted_p[:5]
                bench = sorted_p[5:]
                starter_pts_list.append(sum(_as_float(cast(Any, s.points)) for s in starters))
                bench_pts_list.append(sum(_as_float(cast(Any, b.points)) for b in bench))

            features[f"{prefix}_player_pts_avg"] = float(np.mean(all_pts))
            features[f"{prefix}_player_ast_avg"] = float(np.mean(all_ast))
            features[f"{prefix}_player_reb_avg"] = float(np.mean(all_reb))
            features[f"{prefix}_player_fg_pct"] = float(np.mean(all_fg)) if all_fg else NaN
            features[f"{prefix}_player_3pt_pct"] = float(np.mean(all_3pt)) if all_3pt else NaN
            features[f"{prefix}_starter_pts_avg"] = (
                float(np.mean(starter_pts_list)) if starter_pts_list else NaN
            )
            features[f"{prefix}_bench_pts_avg"] = (
                float(np.mean(bench_pts_list)) if bench_pts_list else NaN
            )
            features[f"{prefix}_bench_ratio"] = (
                features[f"{prefix}_bench_pts_avg"]
                / max(
                    features[f"{prefix}_starter_pts_avg"] + features[f"{prefix}_bench_pts_avg"],
                    1.0,
                )
                if (
                    math.isfinite(features[f"{prefix}_bench_pts_avg"])
                    and math.isfinite(features[f"{prefix}_starter_pts_avg"])
                )
                else NaN
            )
            features[f"{prefix}_min_std"] = float(np.std(all_min)) if len(all_min) > 1 else NaN
        else:
            for k in [
                "player_pts_avg",
                "player_ast_avg",
                "player_reb_avg",
                "player_fg_pct",
                "player_3pt_pct",
                "starter_pts_avg",
                "bench_pts_avg",
                "bench_ratio",
                "min_std",
            ]:
                features[f"{prefix}_{k}"] = NaN

    # Quarter scoring tendencies
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        res_2 = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
                Game.home_q1.is_not(None),
            )
            .order_by(Game.commence_time.desc())
            .limit(10)
        )
        qtr_games = res_2.scalars().all()
        if qtr_games:
            q1_scored = []
            q3_scored = []
            for g in qtr_games:
                if int(cast(Any, g.home_team_id)) == team_id:
                    q1_scored.append(_as_float(cast(Any, g.home_q1)))
                    q3_scored.append(_as_float(cast(Any, g.home_q3)))
                else:
                    q1_scored.append(_as_float(cast(Any, g.away_q1)))
                    q3_scored.append(_as_float(cast(Any, g.away_q3)))
            features[f"{prefix}_q1_avg"] = float(np.mean(q1_scored))
            features[f"{prefix}_q3_avg"] = float(np.mean(q3_scored))
        else:
            features[f"{prefix}_q1_avg"] = NaN
            features[f"{prefix}_q3_avg"] = NaN
    return features


async def _prop_consensus_features(
    db: AsyncSession,
    game: Any,
    odds_snapshots: list | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    prop_markets = [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_blocks",
        "player_steals",
        "player_turnovers",
        "player_points_rebounds_assists",
        "player_points_rebounds",
        "player_points_assists",
        "player_rebounds_assists",
        "player_double_double",
        "player_triple_double",
    ]
    if odds_snapshots is None:
        ranked_props = (
            select(
                OddsSnapshot.id.label("snapshot_id"),
                func.row_number()
                .over(
                    partition_by=(
                        OddsSnapshot.bookmaker,
                        OddsSnapshot.market,
                        OddsSnapshot.description,
                        OddsSnapshot.outcome_name,
                    ),
                    order_by=OddsSnapshot.captured_at.desc(),
                )
                .label("row_num"),
            )
            .where(
                OddsSnapshot.game_id == game.id,
                OddsSnapshot.market.in_(prop_markets),
            )
            .subquery()
        )
        prop_snap_result = await db.execute(
            select(OddsSnapshot)
            .join(ranked_props, OddsSnapshot.id == ranked_props.c.snapshot_id)
            .where(ranked_props.c.row_num == 1)
        )
        prop_snaps = prop_snap_result.scalars().all()
    else:
        prop_snaps = [s for s in odds_snapshots if _as_str(s.market) in prop_markets]

    if prop_snaps:
        _prop_best: dict[tuple[str, str, str, str], Any] = {}
        for s in prop_snaps:
            prop_key = (
                _as_str(s.bookmaker),
                _as_str(s.market),
                _as_str(getattr(s, "description", "")),
                _as_str(s.outcome_name),
            )
            existing = _prop_best.get(prop_key)
            if existing is None or cast(Any, s.captured_at) > cast(Any, existing.captured_at):
                _prop_best[prop_key] = s
        deduped_props = list(_prop_best.values())

        def _over_lines(market: str) -> list[float]:
            return [
                _as_float(s.point)
                for s in deduped_props
                if _as_str(s.market) == market
                and _as_str(s.outcome_name) == "Over"
                and s.point is not None
            ]

        pts_over_lines = _over_lines("player_points")
        features["prop_pts_lines_count"] = float(len(pts_over_lines))
        features["prop_pts_avg_line"] = float(np.mean(pts_over_lines)) if pts_over_lines else NaN

        ast_over_lines = _over_lines("player_assists")
        features["prop_ast_avg_line"] = float(np.mean(ast_over_lines)) if ast_over_lines else NaN

        reb_over_lines = _over_lines("player_rebounds")
        features["prop_reb_avg_line"] = float(np.mean(reb_over_lines)) if reb_over_lines else NaN

        threes_over = _over_lines("player_threes")
        features["prop_threes_avg_line"] = float(np.mean(threes_over)) if threes_over else NaN

        blk_over = _over_lines("player_blocks")
        features["prop_blk_avg_line"] = float(np.mean(blk_over)) if blk_over else NaN

        stl_over = _over_lines("player_steals")
        features["prop_stl_avg_line"] = float(np.mean(stl_over)) if stl_over else NaN

        tov_over = _over_lines("player_turnovers")
        features["prop_tov_avg_line"] = float(np.mean(tov_over)) if tov_over else NaN

        pra_over = _over_lines("player_points_rebounds_assists")
        features["prop_pra_avg_line"] = float(np.mean(pra_over)) if pra_over else NaN

        dd_yes = [
            s
            for s in deduped_props
            if _as_str(s.market) == "player_double_double" and _as_str(s.outcome_name) == "Yes"
        ]
        features["prop_dd_count"] = float(len(dd_yes))

        td_yes = [
            s
            for s in deduped_props
            if _as_str(s.market) == "player_triple_double" and _as_str(s.outcome_name) == "Yes"
        ]
        features["prop_td_count"] = float(len(td_yes))

        sharp_pts = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_points"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
            and _as_str(s.bookmaker).lower() in SHARP_BOOKS
        ]
        square_pts = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_points"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
            and _as_str(s.bookmaker).lower() in SQUARE_BOOKS
        ]
        sharp_avg = float(np.mean(sharp_pts)) if sharp_pts else features["prop_pts_avg_line"]
        square_avg = float(np.mean(square_pts)) if square_pts else features["prop_pts_avg_line"]
        features["prop_sharp_square_diff"] = sharp_avg - square_avg
    else:
        features["prop_pts_lines_count"] = 0.0
        features["prop_pts_avg_line"] = NaN
        features["prop_ast_avg_line"] = NaN
        features["prop_reb_avg_line"] = NaN
        features["prop_threes_avg_line"] = NaN
        features["prop_blk_avg_line"] = NaN
        features["prop_stl_avg_line"] = NaN
        features["prop_tov_avg_line"] = NaN
        features["prop_pra_avg_line"] = NaN
        features["prop_dd_count"] = 0.0
        features["prop_td_count"] = 0.0
        features["prop_sharp_square_diff"] = NaN
    return features


async def _venue_and_streak_features(
    db: AsyncSession,
    home_id: int,
    away_id: int,
    game: Any,
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
    db: AsyncSession,
    game: Any,
    home_id: int,
    away_id: int,
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

    # Referee History — single batch query replaces per-ref N+1 loops
    ref_names = _assigned_referee_names(game)
    if ref_names:
        # Fetch all games officiated by any assigned referee in one query
        ref_games_res = await db.execute(
            select(Game, GameReferee.referee_name)
            .join(GameReferee)
            .where(
                GameReferee.referee_name.in_(ref_names),
                Game.status == "FT",
                Game.commence_time < game.commence_time,
                Game.home_score_fg.is_not(None),
                Game.away_score_fg.is_not(None),
            )
        )
        ref_rows = ref_games_res.all()

        # Batch-fetch totals lines for all referee games in one query
        ref_game_ids = list({int(cast(Any, row[0].id)) for row in ref_rows})
        totals_by_game: dict[int, float] = {}
        if ref_game_ids:
            # Use a ranked subquery to get one totals line per game
            totals_res = await db.execute(
                select(OddsSnapshot.game_id, OddsSnapshot.point)
                .where(
                    OddsSnapshot.game_id.in_(ref_game_ids),
                    OddsSnapshot.market == "totals",
                    OddsSnapshot.point.is_not(None),
                )
                .distinct(OddsSnapshot.game_id)
            )
            for t_row in totals_res.all():
                totals_by_game[int(t_row[0])] = _as_float(t_row[1])

        # Group results by referee name
        by_ref: dict[str, list[Game]] = {}
        for rg, rname in ref_rows:
            by_ref.setdefault(rname, []).append(rg)

        ref_metrics = []
        for _ref_name in ref_names:
            ref_games = by_ref.get(_ref_name, [])
            if not ref_games:
                continue
            pts = []
            hw_pct = []
            over_pct = []
            for rg in ref_games:
                h_pts = _as_float(cast(Any, rg.home_score_fg))
                a_pts = _as_float(cast(Any, rg.away_score_fg))
                pts.append(h_pts + a_pts)
                hw_pct.append(1.0 if h_pts > a_pts else 0.0)
                line = totals_by_game.get(int(cast(Any, rg.id)))
                if line is not None:
                    over_pct.append(1.0 if (h_pts + a_pts) > line else 0.0)
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
    features: dict[str, float],
    game: Any,
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
    out["rest_diff"] = features.get("home_rest_days", NaN) - features.get("away_rest_days", NaN)

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


async def _market_features(
    db: AsyncSession,
    game: Any,
    odds_snapshots: list | None,
    home_name: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if odds_snapshots is None:
        res_6 = await db.execute(
            select(OddsSnapshot)
            .where(OddsSnapshot.game_id == game.id)
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(500)
        )
        snapshots = res_6.scalars().all()
    else:
        snapshots = odds_snapshots

    if snapshots:
        spreads = _home_spreads(snapshots, home_name)
        totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals" and s.point is not None
        ]
        h1_spreads = _home_spreads(snapshots, home_name, "spreads_h1")
        h1_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals_h1" and s.point is not None
        ]

        features["mkt_spread_avg"] = float(np.mean(spreads)) if spreads else NaN
        features["mkt_spread_std"] = float(np.std(spreads)) if len(spreads) > 1 else NaN
        features["mkt_total_avg"] = float(np.mean(totals)) if totals else NaN
        features["mkt_total_std"] = float(np.std(totals)) if len(totals) > 1 else NaN
        features["mkt_1h_spread_avg"] = float(np.mean(h1_spreads)) if h1_spreads else NaN
        features["mkt_1h_total_avg"] = float(np.mean(h1_totals)) if h1_totals else NaN

        # 1st-half moneyline implied probability
        h2h_h1 = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h_h1"
            and ("home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name)
        ]
        if h2h_h1:
            h1_prices = [_as_float(s.price) for s in h2h_h1]
            avg_h1_price = np.mean(h1_prices)
            if avg_h1_price < 0:
                features["mkt_1h_home_ml_prob"] = float(
                    abs(avg_h1_price) / (abs(avg_h1_price) + 100)
                )
            else:
                features["mkt_1h_home_ml_prob"] = float(100 / (avg_h1_price + 100))
        else:
            features["mkt_1h_home_ml_prob"] = 0.5

        # Moneyline implied probability (overall)
        h2h = [s for s in snapshots if _as_str(s.market) == "h2h"]
        if h2h:
            home_prices = [
                _as_float(s.price)
                for s in h2h
                if "home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name
            ]
            if home_prices:
                avg_price = np.mean(home_prices)
                if avg_price < 0:
                    features["mkt_home_ml_prob"] = float(abs(avg_price) / (abs(avg_price) + 100))
                else:
                    features["mkt_home_ml_prob"] = float(100 / (avg_price + 100))
            else:
                features["mkt_home_ml_prob"] = 0.5
        else:
            features["mkt_home_ml_prob"] = 0.5

        # Sharp vs. Square book analysis
        def _split_by_book_type(snaps: Sequence[Any], mkt: str, field: str = "point"):
            sharp_vals, square_vals = [], []
            for s in snaps:
                if _as_str(s.market) != mkt:
                    continue
                val = getattr(s, field)
                if val is None:
                    continue
                bk = _as_str(s.bookmaker).lower()
                if bk in SHARP_BOOKS:
                    sharp_vals.append(_as_float(val))
                elif bk in SQUARE_BOOKS:
                    square_vals.append(_as_float(val))
            return sharp_vals, square_vals

        def _price_to_implied(price: Any) -> float:
            price_f = _as_float(price)
            if price_f < 0:
                return abs(price_f) / (abs(price_f) + 100)
            return 100 / (price_f + 100)

        sharp_spr = _home_spreads(snapshots, home_name, books=SHARP_BOOKS)
        square_spr = _home_spreads(snapshots, home_name, books=SQUARE_BOOKS)
        features["sharp_spread"] = (
            float(np.mean(sharp_spr)) if sharp_spr else features["mkt_spread_avg"]
        )
        features["square_spread"] = (
            float(np.mean(square_spr)) if square_spr else features["mkt_spread_avg"]
        )
        features["sharp_square_spread_diff"] = features["sharp_spread"] - features["square_spread"]

        sharp_tot, square_tot = _split_by_book_type(snapshots, "totals")
        features["sharp_total"] = (
            float(np.mean(sharp_tot)) if sharp_tot else features["mkt_total_avg"]
        )
        features["square_total"] = (
            float(np.mean(square_tot)) if square_tot else features["mkt_total_avg"]
        )
        features["sharp_square_total_diff"] = features["sharp_total"] - features["square_total"]

        sharp_h2h = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h"
            and _as_str(s.bookmaker).lower() in SHARP_BOOKS
            and ("home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name)
        ]
        square_h2h = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h"
            and _as_str(s.bookmaker).lower() in SQUARE_BOOKS
            and ("home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name)
        ]
        if sharp_h2h:
            features["sharp_ml_prob"] = float(
                np.mean([_price_to_implied(s.price) for s in sharp_h2h])
            )
        else:
            features["sharp_ml_prob"] = features["mkt_home_ml_prob"]
        if square_h2h:
            features["square_ml_prob"] = float(
                np.mean([_price_to_implied(s.price) for s in square_h2h])
            )
        else:
            features["square_ml_prob"] = features["mkt_home_ml_prob"]
        features["sharp_square_ml_diff"] = float(
            features["sharp_ml_prob"] - features["square_ml_prob"]
        )

        # Line movement: opening = first capture day, current = most recent capture day.
        # Comparing per capture_date prevents stale multi-week comparisons when the
        # archive holds many days of data.
        spread_dates = sorted(
            {
                cast(Any, s.captured_at).date()
                if hasattr(cast(Any, s.captured_at), "date")
                else cast(Any, s.captured_at)
                for s in snapshots
                if _as_str(s.market) == "spreads"
                and s.point is not None
                and _as_str(s.outcome_name) == home_name
            }
        )
        total_dates = sorted(
            {
                cast(Any, s.captured_at).date()
                if hasattr(cast(Any, s.captured_at), "date")
                else cast(Any, s.captured_at)
                for s in snapshots
                if _as_str(s.market) == "totals" and s.point is not None
            }
        )
        oldest_spread_date = spread_dates[0] if spread_dates else None
        latest_spread_date = spread_dates[-1] if spread_dates else None
        oldest_total_date = total_dates[0] if total_dates else None
        latest_total_date = total_dates[-1] if total_dates else None

        def _date_of(ts: Any) -> Any:
            return ts.date() if hasattr(ts, "date") else ts

        opening_spreads = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and _date_of(cast(Any, s.captured_at)) == oldest_spread_date
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        current_spreads = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and _date_of(cast(Any, s.captured_at)) == latest_spread_date
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        opening_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals"
            and _date_of(cast(Any, s.captured_at)) == oldest_total_date
            and s.point is not None
        ]
        current_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals"
            and _date_of(cast(Any, s.captured_at)) == latest_total_date
            and s.point is not None
        ]

        open_spr = float(np.mean(opening_spreads)) if opening_spreads else NaN
        curr_spr = float(np.mean(current_spreads)) if current_spreads else NaN
        open_tot = float(np.mean(opening_totals)) if opening_totals else NaN
        curr_tot = float(np.mean(current_totals)) if current_totals else NaN

        features["spread_move"] = curr_spr - open_spr
        features["total_move"] = curr_tot - open_tot

        # Reverse line movement (RLM) indicator
        spread_moved_toward_sharp = (
            features["sharp_square_spread_diff"] > 0 and features["spread_move"] > 0
        ) or (features["sharp_square_spread_diff"] < 0 and features["spread_move"] < 0)
        features["rlm_flag"] = 1.0 if spread_moved_toward_sharp else 0.0
    else:
        for k in [
            "mkt_spread_avg",
            "mkt_spread_std",
            "mkt_total_avg",
            "mkt_total_std",
            "mkt_1h_spread_avg",
            "mkt_1h_total_avg",
            "mkt_1h_home_ml_prob",
            "mkt_home_ml_prob",
            "sharp_spread",
            "square_spread",
            "sharp_square_spread_diff",
            "sharp_total",
            "square_total",
            "sharp_square_total_diff",
            "sharp_ml_prob",
            "square_ml_prob",
            "sharp_square_ml_diff",
            "spread_move",
            "total_move",
            "rlm_flag",
        ]:
            features[k] = NaN
    return features


def _interaction_features(
    features: dict[str, float],
    game: Any,
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
    out["off_def_mismatch"] = features.get("home_adj_off", NaN) - features.get("home_adj_def", NaN)
    out["streak_diff"] = features.get("home_win_streak", NaN) - features.get("away_win_streak", NaN)

    # ── Additional interaction features for accuracy ────────────
    # Pace × total market line — captures tempo-adjusted total expectation
    _mkt_total = features.get("mkt_total_avg", NaN)
    out["pace_x_mkt_total"] = (
        _exp_pace * _mkt_total if (math.isfinite(_exp_pace) and math.isfinite(_mkt_total)) else NaN
    )

    # Recency momentum: weighted recent form differential
    _h_l5w = features.get("home_l5_pts_wavg", NaN)
    _a_l5w = features.get("away_l5_pts_wavg", NaN)
    out["recent_form_diff"] = _h_l5w - _a_l5w

    # Defensive matchup: home defense vs away offense
    _h_def = features.get("home_def_rating", NaN)
    _a_off = features.get("away_off_rating", NaN)
    _a_def = features.get("away_def_rating", NaN)
    _h_off = features.get("home_off_rating", NaN)
    out["def_matchup_diff"] = (_h_def - _a_off) - (_a_def - _h_off)

    # Market-model agreement signal: does market spread align with Elo?
    _mkt_spread = features.get("mkt_spread_avg", NaN)
    if math.isfinite(_elo_d) and math.isfinite(_mkt_spread):
        # Elo diff in spread units: ~25 Elo = 1 point
        elo_implied_spread = -_elo_d / 25.0
        out["market_elo_disagreement"] = _mkt_spread - elo_implied_spread
    else:
        out["market_elo_disagreement"] = NaN

    # Fatigue-scoring interaction: games in 7 days × scoring average
    _h_g7d = features.get("home_games_7d", NaN)
    _h_ppg = features.get("home_ppg", NaN)
    out["home_fatigue_scoring"] = (
        _h_g7d * _h_ppg if (math.isfinite(_h_g7d) and math.isfinite(_h_ppg)) else NaN
    )

    _a_g7d = features.get("away_games_7d", NaN)
    _a_ppg = features.get("away_ppg", NaN)
    out["away_fatigue_scoring"] = (
        _a_g7d * _a_ppg if (math.isfinite(_a_g7d) and math.isfinite(_a_ppg)) else NaN
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


async def build_feature_vector(
    game: Game,
    db: AsyncSession,
    odds_snapshots: list | None = None,
) -> dict[str, float] | None:
    """Build a feature dict for a single game.

    Args:
        game: The Game ORM object.
        db: Async database session (used for stats, recent form, injuries).
        odds_snapshots: Pre-fetched odds data.  When provided these are used
            instead of querying the ``OddsSnapshot`` table — this is the path
            used at prediction time so the model always runs on *fresh* odds.
            When ``None`` (training), historical cached odds from the DB are
            used instead.
    """
    home_id = int(cast(Any, game.home_team_id))
    away_id = int(cast(Any, game.away_team_id))
    season = _as_str(
        cast(Any, game.season),
        season_for_date(cast(Any, game.commence_time)),
    )
    home_name = game.home_team.name if game.home_team is not None else ""

    features: dict[str, float] = {}
    features.update(await _team_season_stats(db, home_id, away_id, season))
    features.update(await _recent_form(db, home_id, away_id, game))
    features.update(await _schedule_features(db, home_id, away_id, game))
    features.update(await _injury_features(db, home_id, away_id, game))
    features.update(await _player_and_quarter_features(db, home_id, away_id, game))
    features.update(await _prop_consensus_features(db, game, odds_snapshots))
    features.update(await _venue_and_streak_features(db, home_id, away_id, game, features))
    features.update(await _elo_h2h_referee_features(db, game, home_id, away_id))
    features.update(_derived_features(features, game))
    features.update(await _market_features(db, game, odds_snapshots, home_name))
    features.update(_interaction_features(features, game))

    return features


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature names for the model."""
    cols = []
    for prefix in ["home", "away"]:
        cols.extend(
            [
                f"{prefix}_ppg",
                f"{prefix}_oppg",
                f"{prefix}_wins",
                f"{prefix}_losses",
                f"{prefix}_pace",
                f"{prefix}_off_rating",
                f"{prefix}_def_rating",
                f"{prefix}_win_pct",
            ]
        )
        for label in ["l5", "l10"]:
            cols.extend(
                [
                    f"{prefix}_{label}_pts_avg",
                    f"{prefix}_{label}_pts_allowed_avg",
                    f"{prefix}_{label}_1h_pts_avg",
                    f"{prefix}_{label}_1h_allowed_avg",
                ]
            )
        cols.extend(
            [
                f"{prefix}_rest_days",
                f"{prefix}_b2b",
                f"{prefix}_games_7d",
                f"{prefix}_injury_impact",
                f"{prefix}_injured_count",
            ]
        )
        # Player-stat aggregated features
        cols.extend(
            [
                f"{prefix}_player_pts_avg",
                f"{prefix}_player_ast_avg",
                f"{prefix}_player_reb_avg",
                f"{prefix}_player_fg_pct",
                f"{prefix}_player_3pt_pct",
                f"{prefix}_starter_pts_avg",
                f"{prefix}_bench_pts_avg",
                f"{prefix}_bench_ratio",
                f"{prefix}_min_std",
            ]
        )
        # Quarter scoring tendencies
        cols.extend(
            [
                f"{prefix}_q1_avg",
                f"{prefix}_q3_avg",
            ]
        )
    # ── New per-team features ───────────────────────────────────
    for prefix in ["home", "away"]:
        cols.extend(
            [
                f"{prefix}_venue_ppg",
                f"{prefix}_venue_oppg",
                f"{prefix}_win_streak",
                f"{prefix}_l5_record",
                f"{prefix}_l10_record",
            ]
        )
    # ── Game-level features ─────────────────────────────────────
    cols.extend(
        [
            "expected_pace",
            "pace_diff",
            "season_progress",
            "home_elo",
            "away_elo",
            "elo_diff",
            "h2h_win_pct",
            "h2h_avg_margin",
            "tz_diff",
            "home_adj_off",
            "home_adj_def",
            "rest_diff",
        ]
    )
    # ── Market signals ──────────────────────────────────────────
    cols.extend(
        [
            "mkt_spread_avg",
            "mkt_spread_std",
            "mkt_total_avg",
            "mkt_total_std",
            "mkt_home_ml_prob",
            "mkt_1h_spread_avg",
            "mkt_1h_total_avg",
            "mkt_1h_home_ml_prob",
            # Sharp vs. Square analysis
            "sharp_spread",
            "square_spread",
            "sharp_square_spread_diff",
            "sharp_total",
            "square_total",
            "sharp_square_total_diff",
            "sharp_ml_prob",
            "square_ml_prob",
            "sharp_square_ml_diff",
            # Line movement & RLM
            "spread_move",
            "total_move",
            "rlm_flag",
        ]
    )
    # ── Player prop consensus signals ───────────────────────────
    cols.extend(
        [
            "prop_pts_lines_count",
            "prop_pts_avg_line",
            "prop_ast_avg_line",
            "prop_reb_avg_line",
            "prop_threes_avg_line",
            "prop_blk_avg_line",
            "prop_stl_avg_line",
            "prop_tov_avg_line",
            "prop_pra_avg_line",
            "prop_dd_count",
            "prop_td_count",
            "prop_sharp_square_diff",
        ]
    )
    # ── Matchup & Situational Styles ────────────────────────────
    cols.extend(
        [
            "matchup_3pt_diff",
            "situational_urgency",
        ]
    )
    # ── Cross-team interaction features ─────────────────────────
    cols.extend(
        [
            "pace_x_3pt_diff",
            "elo_x_rest",
            "injury_diff",
            "venue_scoring_edge",
            "off_def_mismatch",
            "streak_diff",
        ]
    )
    # ── NEW features (v6.6.0) ───────────────────────────────────
    # Appended AFTER the 126 features the v6.5.0 models were trained
    # on so compatibility-mode (first-N slice) stays aligned.
    # These will be ignored until the next retrain.
    for prefix in ["home", "away"]:
        for label in ["l5", "l10"]:
            cols.extend(
                [
                    f"{prefix}_{label}_pts_wavg",
                    f"{prefix}_{label}_pts_allowed_wavg",
                ]
            )
    cols.extend(
        [
            "pace_x_mkt_total",
            "recent_form_diff",
            "def_matchup_diff",
            "market_elo_disagreement",
            "home_fatigue_scoring",
            "away_fatigue_scoring",
        ]
    )
    # ── Referee history ─────────────────────────────────────────
    cols.extend(
        [
            "ref_avg_pts",
            "ref_home_win_pct_bias",
            "ref_over_pct",
        ]
    )
    return cols
