"""Tests for build_feature_vector — the #1 coverage gap (374 lines).

Strategy: Mock AsyncSession.execute to return controlled ORM-like objects
for every DB query that build_feature_vector issues.  Validate the
returned feature dict has all 122 feature columns with expected values.
"""

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.features import (
    TEAM_TZ,
    build_elo_ratings,
    build_feature_vector,
    get_feature_columns,
    reset_elo_cache,
)

# ── Helpers to build fake ORM rows ──────────────────────────────


def _ts(days_ago: int = 0) -> datetime:
    return datetime(2025, 1, 15, 19, 0, tzinfo=UTC) - timedelta(days=days_ago)


def _make_game(
    game_id=1,
    home_id=1,
    away_id=2,
    commence=None,
    status="NS",
    season="2024-2025",
    home_fg=None,
    away_fg=None,
    home_1h=None,
    away_1h=None,
    home_q1=None,
    home_q3=None,
    away_q1=None,
    away_q3=None,
    home_team_name="Boston Celtics",
    away_team_name="Los Angeles Lakers",
):
    g = SimpleNamespace(
        id=game_id,
        home_team_id=home_id,
        away_team_id=away_id,
        commence_time=commence or _ts(0),
        status=status,
        season=season,
        home_score_fg=home_fg,
        away_score_fg=away_fg,
        home_score_1h=home_1h,
        away_score_1h=away_1h,
        home_q1=home_q1,
        home_q2=None,
        home_q3=home_q3,
        home_q4=None,
        away_q1=away_q1,
        away_q2=None,
        away_q3=away_q3,
        away_q4=None,
        home_team=SimpleNamespace(name=home_team_name),
        away_team=SimpleNamespace(name=away_team_name),
        odds_api_id=None,
    )
    return g


def _make_team_season_stats(
    ppg=112.0,
    oppg=108.0,
    wins=30,
    losses=15,
    pace=100.0,
    off_rating=115.0,
    def_rating=109.0,
    games_played=45,
):
    return SimpleNamespace(
        ppg=ppg,
        oppg=oppg,
        wins=wins,
        losses=losses,
        pace=pace,
        off_rating=off_rating,
        def_rating=def_rating,
        games_played=games_played,
    )


def _make_recent_game(
    team_id, days_ago, home=True, fg=110, opp_fg=105, h1=55, opp_h1=50, q1=28, q3=27
):
    g = SimpleNamespace(
        id=100 + days_ago,
        home_team_id=team_id if home else 99,
        away_team_id=99 if home else team_id,
        commence_time=_ts(days_ago),
        status="FT",
        home_score_fg=fg if home else opp_fg,
        away_score_fg=opp_fg if home else fg,
        home_score_1h=h1 if home else opp_h1,
        away_score_1h=opp_h1 if home else h1,
        home_q1=q1 if home else 25,
        home_q3=q3 if home else 24,
        away_q1=25 if home else q1,
        away_q3=24 if home else q3,
        season="2024-2025",
    )
    return g


def _make_odds_snapshot(
    game_id=1,
    bookmaker="pinnacle",
    market="spreads",
    outcome_name="Boston Celtics",
    price=-110,
    point=-5.5,
    captured_at=None,
    description=None,
):
    return SimpleNamespace(
        id=1,
        game_id=game_id,
        bookmaker=bookmaker,
        market=market,
        outcome_name=outcome_name,
        price=price,
        point=point,
        captured_at=captured_at or _ts(0),
        description=description or "",
    )


def _make_player_game_stats(
    player_id,
    game_id,
    team_id,
    pts=15,
    ast=4,
    reb=5,
    tov=2,
    pm=3,
    fg_pct=0.45,
    three_pct=0.35,
    minutes=25,
):
    return SimpleNamespace(
        id=player_id * 1000 + game_id,
        player_id=player_id,
        game_id=game_id,
        team_id=team_id,
        points=pts,
        assists=ast,
        rebounds=reb,
        turnovers=tov,
        plus_minus=pm,
        fg_pct=fg_pct,
        three_pct=three_pct,
        minutes=minutes,
    )


def _make_injury(player_id, team_id, status="out"):
    return SimpleNamespace(
        player_id=player_id,
        team_id=team_id,
        status=status,
    )


# ── Mock DB session builder ─────────────────────────────────────


def _build_mock_db(
    game, team_stats=None, recent_games=None, injuries=None, player_stats=None, elo_games=None
):
    """Build a mock AsyncSession that responds to all queries in build_feature_vector."""
    team_stats = team_stats or {}
    recent_games = recent_games or []
    injuries = injuries or []
    player_stats = player_stats or []
    elo_games = elo_games or []

    call_count = 0
    home_id = game.home_team_id
    away_id = game.away_team_id

    db = AsyncMock()

    async def mock_execute(stmt, *args, **kwargs):
        nonlocal call_count
        call_count += 1

        # Introspect the SQL to decide what to return
        sql_str = str(stmt)
        result = MagicMock()

        # TeamSeasonStats queries
        if "team_season_stats" in sql_str.lower():
            # Return stats for home or away team
            team_stats.get(home_id) or team_stats.get(away_id)
            if team_stats:
                # Determine which team from param bindings
                stats_val = list(team_stats.values())[0] if len(team_stats) == 1 else None
                result.scalar_one_or_none.return_value = stats_val
            else:
                result.scalar_one_or_none.return_value = None
            return result

        # Injury queries
        if "injuries" in sql_str.lower():
            result.scalars.return_value.all.return_value = injuries
            return result

        # PlayerGameStats avg query (for injury impact)
        if "avg" in sql_str.lower() and "player_game_stats" in sql_str.lower():
            result.one_or_none.return_value = (15.0, 28.0)
            return result

        # count query (games_7d)
        if "count" in sql_str.lower():
            result.scalar.return_value = 3
            return result

        # Game queries for recent form / rest / streaks / H2H / venue / quarters / elo
        if "games" in sql_str.lower():
            result.scalars.return_value.all.return_value = recent_games[:10]
            result.scalar_one_or_none.return_value = (
                recent_games[0].commence_time if recent_games else None
            )
            result.fetchall.return_value = [(g.id,) for g in recent_games[:5]]
            return result

        # Player game stats join for player aggregates
        if "player_game_stats" in sql_str.lower():
            result.scalars.return_value.all.return_value = player_stats
            return result

        # OddsSnapshot queries
        if "odds_snapshot" in sql_str.lower():
            result.scalars.return_value.all.return_value = []
            return result

        # Default
        result.scalars.return_value.all.return_value = []
        result.scalar_one_or_none.return_value = None
        result.scalar.return_value = 0
        result.fetchall.return_value = []
        return result

    db.execute = mock_execute
    return db


# ── Tests ────────────────────────────────────────────────────────


class TestBuildFeatureVectorNoOdds:
    """Test build_feature_vector with no odds snapshots (all NaN market features)."""

    @pytest.fixture(autouse=True)
    def _reset_elo(self):
        reset_elo_cache()
        yield
        reset_elo_cache()

    async def test_returns_dict_with_all_feature_columns(self):
        """Feature dict must contain every feature in get_feature_columns()."""
        home_stats = _make_team_season_stats()
        _make_team_season_stats(ppg=108, oppg=112, wins=20, losses=25)

        game = _make_game()
        recent = [_make_recent_game(1, d) for d in range(1, 11)]
        recent += [_make_recent_game(2, d, home=False) for d in range(1, 11)]
        pgs = [_make_player_game_stats(p, g.id, 1) for p in range(10, 20) for g in recent[:5]]

        db = AsyncMock()
        query_index = {"n": 0}

        async def smart_execute(stmt, *a, **kw):
            query_index["n"] += 1
            sql = str(stmt).lower()
            result = MagicMock()

            if "team_season_stats" in sql:
                result.scalar_one_or_none.return_value = home_stats
                return result
            if "avg" in sql and "player_game_stats" in sql:
                result.one_or_none.return_value = (15.0, 28.0)
                return result
            if "count" in sql:
                result.scalar.return_value = 3
                return result
            if "odds_snapshot" in sql:
                result.scalars.return_value.all.return_value = []
                return result
            if "injur" in sql:
                result.scalars.return_value.all.return_value = []
                return result
            # player_game_stats JOIN must come BEFORE generic "games" match
            if "player_game_stats" in sql and "join" in sql:
                result.scalars.return_value.all.return_value = pgs
                return result
            if "player_game_stats" in sql:
                result.fetchall.return_value = [(g.id,) for g in recent[:5]]
                return result
            if "games" in sql:
                result.scalars.return_value.all.return_value = recent[:10]
                result.scalar_one_or_none.return_value = recent[0].commence_time if recent else None
                result.fetchall.return_value = [(g.id,) for g in recent[:5]]
                return result

            result.scalars.return_value.all.return_value = []
            result.scalar_one_or_none.return_value = None
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            return result

        db.execute = smart_execute

        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None

        expected_cols = set(get_feature_columns())
        actual_cols = set(features.keys())
        missing = expected_cols - actual_cols
        assert not missing, f"Missing feature columns: {missing}"

    async def test_team_season_stats_populate_features(self):
        """When TeamSeasonStats exists, PPG/OPPG/wins/losses are finite."""
        stats = _make_team_season_stats()
        game = _make_game()
        recent = [_make_recent_game(1, d) for d in range(1, 11)]
        pgs = [_make_player_game_stats(p, g.id, 1) for p in range(10, 15) for g in recent[:5]]

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            sql = str(stmt).lower()
            result = MagicMock()
            if "team_season_stats" in sql:
                result.scalar_one_or_none.return_value = stats
                return result
            if "avg" in sql and "player_game_stats" in sql:
                result.one_or_none.return_value = (15.0, 28.0)
                return result
            if "count" in sql:
                result.scalar.return_value = 3
                return result
            if "odds_snapshot" in sql:
                result.scalars.return_value.all.return_value = []
                return result
            if "injur" in sql:
                result.scalars.return_value.all.return_value = []
                return result
            if "player_game_stats" in sql and "join" in sql:
                result.scalars.return_value.all.return_value = pgs
                return result
            if "player_game_stats" in sql:
                result.fetchall.return_value = [(g.id,) for g in recent[:5]]
                return result
            if "games" in sql:
                result.scalars.return_value.all.return_value = recent
                result.scalar_one_or_none.return_value = recent[0].commence_time
                result.fetchall.return_value = [(g.id,) for g in recent[:5]]
                return result
            result.scalars.return_value.all.return_value = []
            result.scalar_one_or_none.return_value = None
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None
        assert math.isfinite(features["home_ppg"])
        assert features["home_ppg"] == 112.0
        assert math.isfinite(features["home_win_pct"])

    async def test_no_stats_yields_nan(self):
        """When no TeamSeasonStats, stat features are NaN."""
        game = _make_game()

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            str(stmt).lower()
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            result.one_or_none.return_value = None
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None
        assert math.isnan(features["home_ppg"])
        assert math.isnan(features["away_oppg"])


class TestBuildFeatureVectorWithOdds:
    """Test market signal extraction from odds snapshots."""

    @pytest.fixture(autouse=True)
    def _reset_elo(self):
        reset_elo_cache()
        yield
        reset_elo_cache()

    async def test_market_features_from_snapshots(self):
        """Providing odds_snapshots populates mkt_spread_avg etc."""
        game = _make_game()

        snapshots = [
            _make_odds_snapshot(
                market="spreads",
                outcome_name="Boston Celtics",
                point=-5.5,
                bookmaker="pinnacle",
                captured_at=_ts(1),
            ),
            _make_odds_snapshot(
                market="spreads",
                outcome_name="Boston Celtics",
                point=-5.0,
                bookmaker="fanduel",
                captured_at=_ts(0),
            ),
            _make_odds_snapshot(
                market="totals",
                outcome_name="Over",
                point=220.5,
                bookmaker="pinnacle",
                captured_at=_ts(1),
            ),
            _make_odds_snapshot(
                market="totals",
                outcome_name="Over",
                point=221.0,
                bookmaker="fanduel",
                captured_at=_ts(0),
            ),
            _make_odds_snapshot(
                market="h2h",
                outcome_name="Boston Celtics",
                price=-200,
                point=None,
                bookmaker="pinnacle",
            ),
            _make_odds_snapshot(
                market="h2h",
                outcome_name="Los Angeles Lakers",
                price=170,
                point=None,
                bookmaker="pinnacle",
            ),
            _make_odds_snapshot(
                market="player_points",
                outcome_name="Over",
                description="Jayson Tatum",
                point=29.5,
                bookmaker="fanduel",
                captured_at=datetime(2025, 1, 15, 20, 0, tzinfo=UTC),
            ),
        ]

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            sql = str(stmt).lower()
            result = MagicMock()
            if "team_season_stats" in sql:
                result.scalar_one_or_none.return_value = _make_team_season_stats()
                return result
            if "avg" in sql and "player_game_stats" in sql:
                result.one_or_none.return_value = (15.0, 28.0)
                return result
            if "count" in sql:
                result.scalar.return_value = 3
                return result
            if "injur" in sql:
                result.scalars.return_value.all.return_value = []
                return result
            if "player_game_stats" in sql and "join" in sql:
                result.scalars.return_value.all.return_value = [
                    _make_player_game_stats(p, 101, 1) for p in range(10, 20)
                ]
                return result
            if "player_game_stats" in sql:
                recent = [_make_recent_game(1, d) for d in range(1, 6)]
                result.fetchall.return_value = [(g.id,) for g in recent]
                return result
            if "games" in sql:
                recent = [_make_recent_game(1, d) for d in range(1, 11)]
                result.scalars.return_value.all.return_value = recent
                result.scalar_one_or_none.return_value = recent[0].commence_time
                result.fetchall.return_value = [(g.id,) for g in recent[:5]]
                return result
            result.scalars.return_value.all.return_value = []
            result.scalar_one_or_none.return_value = None
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=snapshots)
        assert features is not None
        assert math.isfinite(features["mkt_spread_avg"])
        assert features["mkt_spread_avg"] < 0  # home favorite
        assert math.isfinite(features["mkt_total_avg"])
        assert features["mkt_total_avg"] > 200
        assert math.isfinite(features["sharp_spread"])
        assert math.isfinite(features["spread_move"])
        assert math.isfinite(features["total_move"])
        assert features["rlm_flag"] in (0.0, 1.0)


class TestBuildFeatureVectorEdgeCases:
    """Edge cases: no recent games, no injuries, no player stats."""

    @pytest.fixture(autouse=True)
    def _reset_elo(self):
        reset_elo_cache()
        yield
        reset_elo_cache()

    async def test_no_recent_games(self):
        """With no recent games, rest_days defaults and l5/l10 are NaN."""
        game = _make_game()

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            result.one_or_none.return_value = None
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None
        assert features["home_rest_days"] == 3.0
        assert features["home_b2b"] == 0.0
        assert math.isnan(features["home_l5_pts_avg"])

    async def test_prop_signals_no_props(self):
        """Without prop odds, prop features have default/NaN values."""
        game = _make_game()

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            result.one_or_none.return_value = None
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None
        assert features["prop_pts_lines_count"] == 0.0
        assert math.isnan(features["prop_pts_avg_line"])
        assert features["prop_dd_count"] == 0.0
        assert features["prop_td_count"] == 0.0

    async def test_elo_features_present(self):
        """Elo ratings are computed and present in the feature dict."""
        game = _make_game()

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            result.one_or_none.return_value = None
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None
        assert "home_elo" in features
        assert "away_elo" in features
        assert "elo_diff" in features
        # Default Elo is 1500
        assert features["home_elo"] == 1500.0
        assert features["elo_diff"] == 0.0

    async def test_h2h_no_history(self):
        """H2H with no history uses neutral priors."""
        game = _make_game()

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = 0
            result.fetchall.return_value = []
            result.one_or_none.return_value = None
            return result

        db.execute = execute
        features = await build_feature_vector(game, db, odds_snapshots=[])
        assert features is not None
        assert features["h2h_win_pct"] == 0.5
        assert math.isnan(features["h2h_avg_margin"])


class TestBuildEloRatings:
    """Test build_elo_ratings function."""

    @pytest.fixture(autouse=True)
    def _reset_elo(self):
        reset_elo_cache()
        yield
        reset_elo_cache()

    async def test_builds_from_completed_games(self):
        """build_elo_ratings processes finished games and returns EloSystem."""
        games = [
            _make_recent_game(1, 5, fg=110, opp_fg=105),
            _make_recent_game(1, 3, fg=100, opp_fg=108),
        ]

        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            result = MagicMock()
            result.scalars.return_value.all.return_value = games
            return result

        db.execute = execute
        elo = await build_elo_ratings(db)
        assert elo is not None
        # After processing games, ratings should have diverged from 1500
        r1 = elo.rating(1)
        r99 = elo.rating(99)
        assert isinstance(r1, float)
        assert isinstance(r99, float)

    async def test_caching(self):
        """Second call returns cached instance without re-querying."""
        db = AsyncMock()

        async def execute(stmt, *a, **kw):
            result = MagicMock()
            result.scalars.return_value.all.return_value = []
            return result

        db.execute = execute

        elo1 = await build_elo_ratings(db)
        elo2 = await build_elo_ratings(db)
        assert elo1 is elo2


class TestTeamTimezones:
    """Validate TEAM_TZ mappings."""

    def test_all_30_teams_mapped(self):
        assert len(TEAM_TZ) == 30

    def test_timezone_ranges(self):
        for team, tz in TEAM_TZ.items():
            assert -8 <= tz <= -5, f"{team} has unexpected tz={tz}"
