import logging
from datetime import timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Game,
    Injury,
    OddsSnapshot,
    PlayerGameStats,
    TeamSeasonStats,
)
from src.models.elo import EloSystem

logger = logging.getLogger(__name__)

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
            g.home_team_id,
            g.away_team_id,
            float(g.home_score_fg),
            float(g.away_score_fg),
            season=g.season or "",
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
            win_pct = (
                stats.wins / max(stats.games_played, 1) if stats.games_played else 0.0
            )
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
        result = await db.execute(select(Injury).where(Injury.team_id == team_id))
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

    # ── Expected game pace (interaction) ────────────────────────
    home_pace = features.get("home_pace", 0.0)
    away_pace = features.get("away_pace", 0.0)
    features["expected_pace"] = (
        (home_pace + away_pace) / 2.0 if (home_pace and away_pace) else 0.0
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
                if g.home_team_id == team_id:
                    scored.append(g.home_score_fg or 0)
                    allowed.append(g.away_score_fg or 0)
                else:
                    scored.append(g.away_score_fg or 0)
                    allowed.append(g.home_score_fg or 0)
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
                if g.home_team_id == team_id:
                    won = (g.home_score_fg or 0) > (g.away_score_fg or 0)
                else:
                    won = (g.away_score_fg or 0) > (g.home_score_fg or 0)
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
            if g.home_team_id == team_id:
                won = (g.home_score_fg or 0) > (g.away_score_fg or 0)
            else:
                won = (g.away_score_fg or 0) > (g.home_score_fg or 0)
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
            if g.home_team_id == home_id:
                margin = (g.home_score_fg or 0) - (g.away_score_fg or 0)
            else:
                margin = (g.away_score_fg or 0) - (g.home_score_fg or 0)
            h2h_margins.append(margin)
            if margin > 0:
                h2h_wins += 1
        features["h2h_win_pct"] = h2h_wins / len(h2h_games)
        features["h2h_avg_margin"] = float(np.mean(h2h_margins))
    else:
        features["h2h_win_pct"] = 0.5
        features["h2h_avg_margin"] = 0.0

    # ── Travel / timezone ───────────────────────────────────────
    home_name = game.home_team.name if game.home_team else ""
    away_name = game.away_team.name if game.away_team else ""
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
        spreads = [
            s.point for s in snapshots if s.market == "spreads" and s.point is not None
        ]
        totals = [
            s.point for s in snapshots if s.market == "totals" and s.point is not None
        ]
        h1_spreads = [
            s.point
            for s in snapshots
            if s.market == "spreads_1st_half" and s.point is not None
        ]
        h1_totals = [
            s.point
            for s in snapshots
            if s.market == "totals_1st_half" and s.point is not None
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
        h2h = [s for s in snapshots if s.market == "h2h"]
        if h2h:
            home_prices = [
                s.price
                for s in h2h
                if "home" in (s.outcome_name or "").lower()
                or s.outcome_name == game.home_team.name
            ]
            if home_prices:
                avg_price = np.mean(home_prices)
                if avg_price < 0:
                    features["mkt_home_ml_prob"] = abs(avg_price) / (
                        abs(avg_price) + 100
                    )
                else:
                    features["mkt_home_ml_prob"] = 100 / (avg_price + 100)
            else:
                features["mkt_home_ml_prob"] = 0.5
        else:
            features["mkt_home_ml_prob"] = 0.5

        # ── Sharp vs. Square book analysis ──────────────────────
        home_team_name = game.home_team.name if game.home_team else ""

        def _split_by_book_type(snaps: list, mkt: str, field: str = "point"):
            sharp_vals, square_vals = [], []
            for s in snaps:
                if s.market != mkt:
                    continue
                val = getattr(s, field)
                if val is None:
                    continue
                bk = (s.bookmaker or "").lower()
                if bk in SHARP_BOOKS:
                    sharp_vals.append(val)
                elif bk in SQUARE_BOOKS:
                    square_vals.append(val)
            return sharp_vals, square_vals

        def _price_to_implied(price: float) -> float:
            if price < 0:
                return abs(price) / (abs(price) + 100)
            return 100 / (price + 100)

        # Spread: sharp vs square
        sharp_spr, square_spr = _split_by_book_type(snapshots, "spreads")
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
            if s.market == "h2h"
            and (s.bookmaker or "").lower() in SHARP_BOOKS
            and (
                "home" in (s.outcome_name or "").lower()
                or s.outcome_name == home_team_name
            )
        ]
        square_h2h = [
            s
            for s in snapshots
            if s.market == "h2h"
            and (s.bookmaker or "").lower() in SQUARE_BOOKS
            and (
                "home" in (s.outcome_name or "").lower()
                or s.outcome_name == home_team_name
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
        features["sharp_square_ml_diff"] = (
            features["sharp_ml_prob"] - features["square_ml_prob"]
        )

        # ── Line movement (opening → current) ──────────────────
        # Opening = earliest captured snapshot; Current = latest
        oldest_ts = min(s.captured_at for s in snapshots)
        opening_spreads = [
            s.point
            for s in snapshots
            if s.market == "spreads"
            and s.captured_at == oldest_ts
            and s.point is not None
        ]
        current_spreads = [
            s.point
            for s in snapshots
            if s.market == "spreads"
            and s.captured_at == snapshots[0].captured_at
            and s.point is not None
        ]
        opening_totals = [
            s.point
            for s in snapshots
            if s.market == "totals"
            and s.captured_at == oldest_ts
            and s.point is not None
        ]
        current_totals = [
            s.point
            for s in snapshots
            if s.market == "totals"
            and s.captured_at == snapshots[0].captured_at
            and s.point is not None
        ]

        open_spr = float(np.mean(opening_spreads)) if opening_spreads else 0.0
        curr_spr = float(np.mean(current_spreads)) if current_spreads else 0.0
        open_tot = float(np.mean(opening_totals)) if opening_totals else 0.0
        curr_tot = float(np.mean(current_totals)) if current_totals else 0.0

        features["spread_move"] = curr_spr - open_spr
        features["total_move"] = curr_tot - open_tot

        # ── Reverse line movement (RLM) indicator ──────────────
        # RLM = spread moved toward the sharp side despite public likely
        # being on the other side.  Proxy: if the sharp-square spread
        # divergence and the line movement direction disagree with what
        # public action would cause, flag RLM.
        # Positive sharp_square_spread_diff → sharps see home stronger.
        # If spread ALSO moved toward home (more negative = home favored
        # more), that aligns with sharp $, not public. Flag as RLM.
        spread_moved_toward_sharp = (
            features["sharp_square_spread_diff"] > 0 and features["spread_move"] < 0
        ) or (features["sharp_square_spread_diff"] < 0 and features["spread_move"] > 0)
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
    return cols
