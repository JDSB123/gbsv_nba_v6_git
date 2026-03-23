import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, cast

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.seasons import season_for_date
from src.db.models import (
    Game,
    Injury,
    OddsSnapshot,
    Player,
    PlayerGameStats,
    TeamSeasonStats,
)
from src.models.elo import EloSystem

logger = logging.getLogger(__name__)


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _as_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


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
        if _as_str(s.market) != market or s.point is None:
            continue
        if _as_str(s.outcome_name) != home_team_name:
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

    result = await db.execute(
        select(Game)
        .where(
            Game.status == "FT",
            Game.home_score_fg.is_not(None),
            Game.away_score_fg.is_not(None),
        )
        .order_by(Game.commence_time)
    )
    games = result.scalars().all()
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
            win_pct = wins / max(games_played, 1) if games_played else 0.0
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
                features[f"{prefix}_{label}_pts_allowed_avg"] = float(
                    np.mean(pts_allowed)
                )
                features[f"{prefix}_{label}_1h_pts_avg"] = float(np.mean(h1_scored))
                features[f"{prefix}_{label}_1h_allowed_avg"] = float(
                    np.mean(h1_allowed)
                )
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
        if last_game_time is not None:
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
        result = await db.execute(select(Injury).where(Injury.team_id == team_id))
        injuries = result.scalars().all()
        injury_impact = 0.0
        injured_count = 0
        for inj in injuries:
            weight = INJURY_WEIGHTS.get(_as_str(cast(Any, inj.status)).lower(), 0.0)
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

    # ── Player-stat aggregated team features ────────────────────
    # Aggregate box-score data into team-level signals for each team's
    # recent games so the model can capture roster depth and form.
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        # Get player stats from last 5 games for this team
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
            all_tov = [_as_float(cast(Any, p.turnovers)) for p in pgs_rows]
            all_pm = [_as_float(cast(Any, p.plus_minus)) for p in pgs_rows]
            all_fg = [
                _as_float(cast(Any, p.fg_pct)) for p in pgs_rows if p.fg_pct is not None
            ]
            all_3pt = [
                _as_float(cast(Any, p.three_pct))
                for p in pgs_rows
                if p.three_pct is not None
            ]
            all_min = [_as_float(cast(Any, p.minutes)) for p in pgs_rows]

            # Per-game aggregation: starters (top-5 by minutes) vs bench
            from collections import defaultdict

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
                starter_pts_list.append(
                    sum(_as_float(cast(Any, s.points)) for s in starters)
                )
                bench_pts_list.append(
                    sum(_as_float(cast(Any, b.points)) for b in bench)
                )

            features[f"{prefix}_player_pts_avg"] = float(np.mean(all_pts))
            features[f"{prefix}_player_ast_avg"] = float(np.mean(all_ast))
            features[f"{prefix}_player_reb_avg"] = float(np.mean(all_reb))
            features[f"{prefix}_player_tov_avg"] = float(np.mean(all_tov))
            features[f"{prefix}_player_pm_avg"] = float(np.mean(all_pm))
            features[f"{prefix}_player_fg_pct"] = (
                float(np.mean(all_fg)) if all_fg else 0.0
            )
            features[f"{prefix}_player_3pt_pct"] = (
                float(np.mean(all_3pt)) if all_3pt else 0.0
            )
            features[f"{prefix}_starter_pts_avg"] = (
                float(np.mean(starter_pts_list)) if starter_pts_list else 0.0
            )
            features[f"{prefix}_bench_pts_avg"] = (
                float(np.mean(bench_pts_list)) if bench_pts_list else 0.0
            )
            features[f"{prefix}_bench_ratio"] = features[
                f"{prefix}_bench_pts_avg"
            ] / max(
                features[f"{prefix}_starter_pts_avg"]
                + features[f"{prefix}_bench_pts_avg"],
                1.0,
            )
            # Minutes concentration: std of minutes shows roster depth
            features[f"{prefix}_min_std"] = (
                float(np.std(all_min)) if len(all_min) > 1 else 0.0
            )
        else:
            for k in [
                "player_pts_avg",
                "player_ast_avg",
                "player_reb_avg",
                "player_tov_avg",
                "player_pm_avg",
                "player_fg_pct",
                "player_3pt_pct",
                "starter_pts_avg",
                "bench_pts_avg",
                "bench_ratio",
                "min_std",
            ]:
                features[f"{prefix}_{k}"] = 0.0

    # ── Quarter scoring tendencies ──────────────────────────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        result = await db.execute(
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
        qtr_games = result.scalars().all()
        if qtr_games:
            q1_scored = []
            q3_scored = []  # 2nd half opener
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
            features[f"{prefix}_q1_avg"] = 0.0
            features[f"{prefix}_q3_avg"] = 0.0

    # ── Player prop consensus signals ───────────────────────────
    # Aggregate player prop lines into team-level signals that capture
    # bookmaker expectations about individual performance.
    prop_markets = ["player_points", "player_rebounds", "player_assists"]
    if odds_snapshots is None:
        prop_snap_result = await db.execute(
            select(OddsSnapshot)
            .where(
                OddsSnapshot.game_id == game.id,
                OddsSnapshot.market.in_(prop_markets),
            )
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(500)
        )
        prop_snaps = prop_snap_result.scalars().all()
    else:
        prop_snaps = [s for s in odds_snapshots if _as_str(s.market) in prop_markets]

    if prop_snaps:
        # Deduplicate: keep latest per bookmaker+market+description+outcome
        _prop_best: dict[tuple[str, str, str, str], Any] = {}
        for s in prop_snaps:
            key = (
                _as_str(s.bookmaker),
                _as_str(s.market),
                _as_str(getattr(s, "description", "")),
                _as_str(s.outcome_name),
            )
            existing = _prop_best.get(key)
            if existing is None or cast(Any, s.captured_at) > cast(
                Any, existing.captured_at
            ):
                _prop_best[key] = s
        deduped_props = list(_prop_best.values())

        # Points props: sum of Over lines = implied team total
        pts_over_lines = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_points"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
        ]
        features["prop_pts_lines_count"] = float(len(pts_over_lines))
        features["prop_pts_avg_line"] = (
            float(np.mean(pts_over_lines)) if pts_over_lines else 0.0
        )

        # Assists props
        ast_over_lines = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_assists"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
        ]
        features["prop_ast_avg_line"] = (
            float(np.mean(ast_over_lines)) if ast_over_lines else 0.0
        )

        # Rebounds props
        reb_over_lines = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_rebounds"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
        ]
        features["prop_reb_avg_line"] = (
            float(np.mean(reb_over_lines)) if reb_over_lines else 0.0
        )

        # Sharp vs square prop divergence (points market)
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
        sharp_avg = (
            float(np.mean(sharp_pts)) if sharp_pts else features["prop_pts_avg_line"]
        )
        square_avg = (
            float(np.mean(square_pts)) if square_pts else features["prop_pts_avg_line"]
        )
        features["prop_sharp_square_diff"] = sharp_avg - square_avg
    else:
        features["prop_pts_lines_count"] = 0.0
        features["prop_pts_avg_line"] = 0.0
        features["prop_ast_avg_line"] = 0.0
        features["prop_reb_avg_line"] = 0.0
        features["prop_sharp_square_diff"] = 0.0

    # ── Expected game pace (interaction) ────────────────────────
    home_pace = features.get("home_pace", 0.0)
    away_pace = features.get("away_pace", 0.0)
    features["expected_pace"] = (
        (home_pace + away_pace) / 2.0
        if (home_pace != 0.0 and away_pace != 0.0)
        else 0.0
    )
    features["pace_diff"] = home_pace - away_pace

    # ── Home/away venue splits (PPG when home vs. away) ─────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        if prefix == "home":
            venue_filter = Game.home_team_id == team_id
        else:
            venue_filter = Game.away_team_id == team_id
        result = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                venue_filter,
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(15)
        )
        venue_games = result.scalars().all()
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
            features[f"{prefix}_venue_ppg"] = features.get(f"{prefix}_ppg", 0.0)
            features[f"{prefix}_venue_oppg"] = features.get(f"{prefix}_oppg", 0.0)

    # ── Win streak & L5/L10 record ──────────────────────────────
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        result = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(10)
        )
        streak_games = result.scalars().all()
        # Win streak (positive = wins, negative = losses)
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

        # L5 and L10 record (wins in last 5/10)
        l5_wins = 0
        l10_wins = 0
        for i, g in enumerate(streak_games):
            if int(cast(Any, g.home_team_id)) == team_id:
                won = _as_float(cast(Any, g.home_score_fg)) > _as_float(
                    cast(Any, g.away_score_fg)
                )
            else:
                won = _as_float(cast(Any, g.away_score_fg)) > _as_float(
                    cast(Any, g.home_score_fg)
                )
            if won:
                l10_wins += 1
                if i < 5:
                    l5_wins += 1
            elif i < 5:
                pass  # loss in L5, already 0
        features[f"{prefix}_l5_record"] = float(l5_wins)
        features[f"{prefix}_l10_record"] = float(l10_wins)

    # ── Season phase ────────────────────────────────────────────
    home_gp = features.get("home_wins", 0) + features.get("home_losses", 0)
    away_gp = features.get("away_wins", 0) + features.get("away_losses", 0)
    features["season_progress"] = (home_gp + away_gp) / (2.0 * 82.0)

    # ── Elo ratings ─────────────────────────────────────────────
    elo = await build_elo_ratings(db)
    features["home_elo"] = elo.rating(home_id)
    features["away_elo"] = elo.rating(away_id)
    features["elo_diff"] = features["home_elo"] - features["away_elo"]

    # ── Head-to-head history ────────────────────────────────────
    result = await db.execute(
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
    h2h_games = result.scalars().all()
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
        features["h2h_avg_margin"] = 0.0

    # ── Travel / timezone ───────────────────────────────────────
    home_name = game.home_team.name if game.home_team is not None else ""
    away_name = game.away_team.name if game.away_team is not None else ""
    home_tz = TEAM_TZ.get(home_name, -5)
    away_tz = TEAM_TZ.get(away_name, -5)
    features["tz_diff"] = float(abs(home_tz - away_tz))

    # ── Opponent-adjusted ratings ───────────────────────────────
    # Approximation: team off rating relative to opponent def rating
    features["home_adj_off"] = features.get("home_off_rating", 0.0) - features.get(
        "away_def_rating", 0.0
    )
    features["home_adj_def"] = features.get("away_off_rating", 0.0) - features.get(
        "home_def_rating", 0.0
    )
    features["rest_diff"] = features.get("home_rest_days", 0.0) - features.get(
        "away_rest_days", 0.0
    )

    # ── Market signals (latest odds) ────────────────────────────
    if odds_snapshots is None:
        # Training path: read historical cached odds from DB
        result = await db.execute(
            select(OddsSnapshot)
            .where(OddsSnapshot.game_id == game.id)
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(500)
        )
        snapshots = result.scalars().all()
    else:
        # Prediction path: use fresh odds passed in by the caller
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

        features["mkt_spread_avg"] = float(np.mean(spreads)) if spreads else 0.0
        features["mkt_spread_std"] = float(np.std(spreads)) if len(spreads) > 1 else 0.0
        features["mkt_total_avg"] = float(np.mean(totals)) if totals else 0.0
        features["mkt_total_std"] = float(np.std(totals)) if len(totals) > 1 else 0.0
        features["mkt_1h_spread_avg"] = (
            float(np.mean(h1_spreads)) if h1_spreads else 0.0
        )
        features["mkt_1h_total_avg"] = float(np.mean(h1_totals)) if h1_totals else 0.0

        # Moneyline implied probability (overall)
        h2h = [s for s in snapshots if _as_str(s.market) == "h2h"]
        if h2h:
            home_prices = [
                _as_float(s.price)
                for s in h2h
                if "home" in _as_str(s.outcome_name).lower()
                or _as_str(s.outcome_name) == home_name
            ]
            if home_prices:
                avg_price = np.mean(home_prices)
                if avg_price < 0:
                    features["mkt_home_ml_prob"] = float(
                        abs(avg_price) / (abs(avg_price) + 100)
                    )
                else:
                    features["mkt_home_ml_prob"] = float(100 / (avg_price + 100))
            else:
                features["mkt_home_ml_prob"] = 0.5
        else:
            features["mkt_home_ml_prob"] = 0.5

        # ── Sharp vs. Square book analysis ──────────────────────
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

        # Spread: sharp vs square (home-margin convention)
        sharp_spr = _home_spreads(snapshots, home_name, books=SHARP_BOOKS)
        square_spr = _home_spreads(snapshots, home_name, books=SQUARE_BOOKS)
        features["sharp_spread"] = (
            float(np.mean(sharp_spr)) if sharp_spr else features["mkt_spread_avg"]
        )
        features["square_spread"] = (
            float(np.mean(square_spr)) if square_spr else features["mkt_spread_avg"]
        )
        features["sharp_square_spread_diff"] = (
            features["sharp_spread"] - features["square_spread"]
        )

        # Total: sharp vs square
        sharp_tot, square_tot = _split_by_book_type(snapshots, "totals")
        features["sharp_total"] = (
            float(np.mean(sharp_tot)) if sharp_tot else features["mkt_total_avg"]
        )
        features["square_total"] = (
            float(np.mean(square_tot)) if square_tot else features["mkt_total_avg"]
        )
        features["sharp_square_total_diff"] = (
            features["sharp_total"] - features["square_total"]
        )

        # Moneyline implied prob: sharp vs square
        sharp_h2h = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h"
            and _as_str(s.bookmaker).lower() in SHARP_BOOKS
            and (
                "home" in _as_str(s.outcome_name).lower()
                or _as_str(s.outcome_name) == home_name
            )
        ]
        square_h2h = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h"
            and _as_str(s.bookmaker).lower() in SQUARE_BOOKS
            and (
                "home" in _as_str(s.outcome_name).lower()
                or _as_str(s.outcome_name) == home_name
            )
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

        # ── Line movement (opening → current) ──────────────────
        # Opening = earliest captured snapshot; Current = latest
        oldest_ts = min(cast(Any, s.captured_at) for s in snapshots)
        opening_spreads = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and cast(Any, s.captured_at) == oldest_ts
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        current_spreads = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and cast(Any, s.captured_at) == cast(Any, snapshots[0].captured_at)
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        opening_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals"
            and cast(Any, s.captured_at) == oldest_ts
            and s.point is not None
        ]
        current_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals"
            and cast(Any, s.captured_at) == cast(Any, snapshots[0].captured_at)
            and s.point is not None
        ]

        open_spr = float(np.mean(opening_spreads)) if opening_spreads else 0.0
        curr_spr = float(np.mean(current_spreads)) if current_spreads else 0.0
        open_tot = float(np.mean(opening_totals)) if opening_totals else 0.0
        curr_tot = float(np.mean(current_totals)) if current_totals else 0.0

        features["spread_move"] = curr_spr - open_spr
        features["total_move"] = curr_tot - open_tot

        # ── Reverse line movement (RLM) indicator ──────────────
        # All spread values use betting convention (negative = home fav).
        # sharp_square_spread_diff = sharp - square.
        #   Negative diff → sharps have home as bigger favorite.
        #   Positive diff → sharps have away as bigger favorite.
        # RLM = line moved in the same direction as the sharp-square
        # divergence (i.e. toward the sharp side, against public).
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
            features[k] = 0.0

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
                f"{prefix}_player_tov_avg",
                f"{prefix}_player_pm_avg",
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
            "mkt_1h_spread_avg",
            "mkt_1h_total_avg",
            "mkt_home_ml_prob",
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
            "prop_sharp_square_diff",
        ]
    )
    return cols
