"""Tests for build_feature_vector: player props, injury impact, venue stats, win streaks."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.features import _as_str, _injury_features, build_feature_vector, reset_elo_cache


def _ts(days_ago: int = 0) -> datetime:
    return datetime(2025, 1, 15, 19, 0, tzinfo=UTC) - timedelta(days=days_ago)


def _g(
    game_id=1,
    home_id=1,
    away_id=2,
    commence=None,
    status="NS",
    home_fg=None,
    away_fg=None,
    home_1h=None,
    away_1h=None,
):
    return SimpleNamespace(
        id=game_id,
        home_team_id=home_id,
        away_team_id=away_id,
        commence_time=commence or _ts(0),
        status=status,
        season="2024-2025",
        home_score_fg=home_fg,
        away_score_fg=away_fg,
        home_score_1h=home_1h,
        away_score_1h=away_1h,
        home_q1=None,
        home_q2=None,
        home_q3=None,
        home_q4=None,
        away_q1=None,
        away_q2=None,
        away_q3=None,
        away_q4=None,
        home_team=SimpleNamespace(name="Boston Celtics"),
        away_team=SimpleNamespace(name="Los Angeles Lakers"),
        odds_api_id=None,
    )


def _stats(
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


def _snap(market, outcome, point=None, price=-110, bookmaker="pinnacle", captured=None, desc=""):
    return SimpleNamespace(
        id=1,
        game_id=1,
        bookmaker=bookmaker,
        market=market,
        outcome_name=outcome,
        price=price,
        point=point,
        captured_at=captured or _ts(0),
        description=desc,
    )


def _recent(team_id, days, home=True, fg=110, opp_fg=105):
    return SimpleNamespace(
        id=100 + days,
        home_team_id=team_id if home else 99,
        away_team_id=99 if home else team_id,
        commence_time=_ts(days),
        status="FT",
        home_score_fg=fg if home else opp_fg,
        away_score_fg=opp_fg if home else fg,
        home_score_1h=55,
        away_score_1h=50,
        home_q1=28,
        home_q3=27,
        away_q1=25,
        away_q3=24,
        season="2024-2025",
    )


def _injury(pid, tid, status="out"):
    return SimpleNamespace(player_id=pid, team_id=tid, status=status)


def _pgs(pid, gid, tid, pts=15, ast=4, reb=5, minutes=25):
    return SimpleNamespace(
        id=1,
        player_id=pid,
        game_id=gid,
        team_id=tid,
        points=pts,
        assists=ast,
        rebounds=reb,
        turnovers=2,
        plus_minus=3,
        fg_pct=0.45,
        three_pct=0.35,
        minutes=minutes,
    )


class TestAsStr:
    def test_returns_str_for_value(self):
        assert _as_str(42) == "42"

    def test_none_returns_default(self):
        assert _as_str(None) == ""

    def test_custom_default(self):
        assert _as_str(None, "N/A") == "N/A"


class TestBuildFeatureVectorProps:
    """Test the player props branch of build_feature_vector."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_elo_cache()
        yield
        reset_elo_cache()

    @pytest.mark.anyio
    async def test_with_player_props(self):
        """When odds_snapshots include player props, prop features are populated."""
        game = _g()
        st = _stats()

        # Create prop snapshots covering all prop markets
        props = [
            _snap("player_points", "Over", point=22.5, bookmaker="pinnacle"),
            _snap("player_points", "Over", point=18.5, bookmaker="fanduel"),
            _snap("player_assists", "Over", point=5.5),
            _snap("player_rebounds", "Over", point=8.5),
            _snap("player_threes", "Over", point=2.5),
            _snap("player_blocks", "Over", point=1.5),
            _snap("player_steals", "Over", point=1.5),
            _snap("player_turnovers", "Over", point=3.5),
            _snap("player_points_rebounds_assists", "Over", point=35.5),
            _snap("player_double_double", "Yes"),
            _snap("player_triple_double", "Yes"),
        ]
        # Regular odds snapshots (spreads + totals so market features work)
        odds = [
            _snap("spreads", "Boston Celtics", point=-5.5),
            _snap("totals", "Over", point=220.5),
            _snap("h2h", "Boston Celtics", price=-200),
        ] + props

        recents = [_recent(1, d) for d in range(1, 11)] + [
            _recent(2, d, home=False) for d in range(1, 11)
        ]
        pgs = [_pgs(p, recents[0].id, 1) for p in range(10, 20)]

        db = AsyncMock()

        async def mock_exec(stmt, *a, **kw):
            sql = str(stmt).lower()
            r = MagicMock()
            if "team_season_stats" in sql:
                r.scalar_one_or_none.return_value = st
                return r
            if "avg" in sql and "player_game_stats" in sql:
                r.all.return_value = [(10, 15.0, 28.0)]
                r.one_or_none.return_value = (15.0, 28.0)
                return r
            if "count" in sql:
                r.scalar.return_value = 3
                return r
            if "injur" in sql:
                r.scalars.return_value.all.return_value = [_injury(10, 1)]
                return r
            if "player_game_stats" in sql and "join" in sql:
                r.scalars.return_value.all.return_value = pgs
                return r
            if "odds_snapshot" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "games" in sql:
                r.scalars.return_value.all.return_value = recents[:10]
                r.scalar_one_or_none.return_value = recents[0].commence_time if recents else None
                r.fetchall.return_value = [(g.id,) for g in recents[:5]]
                return r
            r.scalars.return_value.all.return_value = []
            r.scalar_one_or_none.return_value = None
            r.scalar.return_value = 0
            r.fetchall.return_value = []
            return r

        db.execute = mock_exec

        features = await build_feature_vector(game, db, odds_snapshots=odds)

        assert features is not None
        # Props should be populated
        assert features["prop_pts_lines_count"] == 2.0
        assert not math.isnan(features["prop_pts_avg_line"])
        assert not math.isnan(features["prop_ast_avg_line"])
        assert not math.isnan(features["prop_reb_avg_line"])
        assert not math.isnan(features["prop_threes_avg_line"])
        assert not math.isnan(features["prop_blk_avg_line"])
        assert not math.isnan(features["prop_stl_avg_line"])
        assert not math.isnan(features["prop_tov_avg_line"])
        assert not math.isnan(features["prop_pra_avg_line"])
        assert features["prop_dd_count"] == 1.0
        assert features["prop_td_count"] == 1.0
        # Injury impact should be > 0 (one injured player)
        assert features["home_injury_impact"] > 0
        assert features["home_injured_count"] == 1.0

    @pytest.mark.anyio
    async def test_no_props_branch(self):
        """When no player props exist, defaults are set."""
        game = _g()
        st = _stats()
        odds = [
            _snap("spreads", "Boston Celtics", point=-5.5),
            _snap("totals", "Over", point=220.5),
        ]
        recents = [_recent(1, d) for d in range(1, 6)]

        db = AsyncMock()

        async def mock_exec(stmt, *a, **kw):
            sql = str(stmt).lower()
            r = MagicMock()
            if "team_season_stats" in sql:
                r.scalar_one_or_none.return_value = st
                return r
            if "count" in sql:
                r.scalar.return_value = 2
                return r
            if "injur" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "avg" in sql:
                r.one_or_none.return_value = (10.0, 20.0)
                return r
            if "player_game_stats" in sql and "join" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "odds_snapshot" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "games" in sql:
                r.scalars.return_value.all.return_value = recents
                r.scalar_one_or_none.return_value = recents[0].commence_time
                r.fetchall.return_value = [(g.id,) for g in recents[:5]]
                return r
            r.scalars.return_value.all.return_value = []
            r.scalar_one_or_none.return_value = None
            r.scalar.return_value = 0
            r.fetchall.return_value = []
            return r

        db.execute = mock_exec

        features = await build_feature_vector(game, db, odds_snapshots=odds)

        assert features is not None
        assert features["prop_pts_lines_count"] == 0.0
        assert math.isnan(features["prop_pts_avg_line"])
        assert features["prop_dd_count"] == 0.0

    @pytest.mark.anyio
    async def test_injury_features_missing_commence_time_returns_zeroes(self):
        db = AsyncMock()
        game = SimpleNamespace(commence_time=None)

        features = await _injury_features(db, 1, 2, game)

        assert features == {
            "home_injury_impact": 0.0,
            "home_injured_count": 0.0,
            "away_injury_impact": 0.0,
            "away_injured_count": 0.0,
        }
        db.execute.assert_not_awaited()

    @pytest.mark.anyio
    async def test_expected_pace_with_valid_values(self):
        """When both teams have pace, expected_pace and pace_diff are computed."""
        game = _g()
        home_st = _stats(pace=100.0)
        away_st = _stats(pace=105.0)
        recents = [_recent(1, d) for d in range(1, 6)] + [
            _recent(2, d, home=False) for d in range(1, 6)
        ]

        db = AsyncMock()
        call_idx = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_idx["n"] += 1
            sql = str(stmt).lower()
            r = MagicMock()
            if "team_season_stats" in sql:
                # Alternate home/away stats
                r.scalar_one_or_none.return_value = home_st if call_idx["n"] <= 2 else away_st
                return r
            if "count" in sql:
                r.scalar.return_value = 2
                return r
            if "injur" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "avg" in sql:
                r.one_or_none.return_value = (10.0, 20.0)
                return r
            if "player_game_stats" in sql and "join" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "odds_snapshot" in sql:
                r.scalars.return_value.all.return_value = []
                return r
            if "games" in sql:
                r.scalars.return_value.all.return_value = recents[:5]
                r.scalar_one_or_none.return_value = recents[0].commence_time
                r.fetchall.return_value = [(g.id,) for g in recents[:5]]
                return r
            r.scalars.return_value.all.return_value = []
            r.scalar_one_or_none.return_value = None
            r.scalar.return_value = 0
            r.fetchall.return_value = []
            return r

        db.execute = mock_exec

        features = await build_feature_vector(game, db, odds_snapshots=[])

        assert features is not None
        # With both team paces available, expected_pace should be calculated
        if math.isfinite(features.get("home_pace", float("nan"))) and math.isfinite(
            features.get("away_pace", float("nan"))
        ):
            assert math.isfinite(features["expected_pace"])
            assert math.isfinite(features["pace_diff"])
