"""Tests for feature engineering helper functions.

The full build_feature_vector is async and DB-heavy so we test the
pure-logic helpers and the feature column registry here.
"""

import math
from types import SimpleNamespace

from src.models.features import (
    INJURY_WEIGHTS,
    SHARP_BOOKS,
    SQUARE_BOOKS,
    TEAM_TZ,
    _as_float,
    _as_str,
    _home_spreads,
    get_feature_columns,
    reset_elo_cache,
)

# ── _as_float ──────────────────────────────────────────────────


def test_as_float_with_value():
    assert _as_float(3.5) == 3.5
    assert _as_float(0) == 0.0
    assert _as_float("7.2") == 7.2


def test_as_float_none_returns_nan():
    result = _as_float(None)
    assert math.isnan(result)


def test_as_float_none_with_custom_default():
    assert _as_float(None, default=0.0) == 0.0
    assert _as_float(None, default=-1.0) == -1.0


# ── _as_str ────────────────────────────────────────────────────


def test_as_str_with_value():
    assert _as_str("hello") == "hello"
    assert _as_str(42) == "42"


def test_as_str_none_returns_default():
    assert _as_str(None) == ""
    assert _as_str(None, "fallback") == "fallback"


# ── _home_spreads ──────────────────────────────────────────────


def _snap(market="spreads", outcome_name="Boston Celtics", point=-5.5, bookmaker="fanduel"):
    return SimpleNamespace(
        market=market, outcome_name=outcome_name, point=point, bookmaker=bookmaker
    )


def test_home_spreads_extracts_home_team():
    snaps = [
        _snap(outcome_name="Boston Celtics", point=-5.5),
        _snap(outcome_name="Los Angeles Lakers", point=5.5),
    ]
    result = _home_spreads(snaps, "Boston Celtics")
    assert result == [-5.5]


def test_home_spreads_multiple_books():
    snaps = [
        _snap(outcome_name="Celtics", point=-5.5, bookmaker="fanduel"),
        _snap(outcome_name="Celtics", point=-6.0, bookmaker="draftkings"),
        _snap(outcome_name="Lakers", point=5.5, bookmaker="fanduel"),
    ]
    result = _home_spreads(snaps, "Celtics")
    assert result == [-5.5, -6.0]


def test_home_spreads_filters_by_market():
    snaps = [
        _snap(market="spreads", outcome_name="Team", point=-3.0),
        _snap(market="totals", outcome_name="Team", point=220.5),
    ]
    result = _home_spreads(snaps, "Team", market="spreads")
    assert result == [-3.0]


def test_home_spreads_filters_by_books():
    snaps = [
        _snap(outcome_name="Team", point=-3.0, bookmaker="pinnacle"),
        _snap(outcome_name="Team", point=-4.0, bookmaker="fanduel"),
        _snap(outcome_name="Team", point=-3.5, bookmaker="draftkings"),
    ]
    result = _home_spreads(snaps, "Team", books=SHARP_BOOKS)
    assert result == [-3.0]


def test_home_spreads_skips_none_point():
    snaps = [
        _snap(outcome_name="Team", point=None),
        _snap(outcome_name="Team", point=-4.0),
    ]
    result = _home_spreads(snaps, "Team")
    assert result == [-4.0]


def test_home_spreads_empty_when_no_match():
    snaps = [_snap(outcome_name="Other Team", point=-3.0)]
    result = _home_spreads(snaps, "My Team")
    assert result == []


# ── Constants ──────────────────────────────────────────────────


def test_injury_weights_keys():
    assert set(INJURY_WEIGHTS.keys()) == {"out", "doubtful", "questionable", "probable"}
    assert INJURY_WEIGHTS["out"] == 1.0
    assert INJURY_WEIGHTS["probable"] == 0.05


def test_team_tz_has_30_teams():
    assert len(TEAM_TZ) == 30


def test_sharp_and_square_books_disjoint():
    assert SHARP_BOOKS.isdisjoint(SQUARE_BOOKS)
    assert len(SHARP_BOOKS) >= 2
    assert len(SQUARE_BOOKS) >= 5


# ── Feature column registry ───────────────────────────────────


def test_feature_columns_include_market_signals():
    cols = get_feature_columns()
    assert "mkt_spread_avg" in cols
    assert "mkt_total_avg" in cols
    assert "sharp_spread" in cols
    assert "rlm_flag" in cols


def test_feature_columns_include_prop_signals():
    cols = get_feature_columns()
    assert "prop_pts_avg_line" in cols
    assert "prop_pra_avg_line" in cols
    assert "prop_dd_count" in cols
    assert "prop_sharp_square_diff" in cols


def test_feature_columns_include_elo():
    cols = get_feature_columns()
    assert "home_elo" in cols
    assert "away_elo" in cols
    assert "elo_diff" in cols


def test_feature_columns_include_rest_and_b2b():
    cols = get_feature_columns()
    assert "home_rest_days" in cols
    assert "away_b2b" in cols
    assert "rest_diff" in cols


def test_feature_columns_exclude_live_unavailable_box_score_fields():
    cols = get_feature_columns()
    assert "home_player_tov_avg" not in cols
    assert "home_player_pm_avg" not in cols
    assert "away_player_tov_avg" not in cols
    assert "away_player_pm_avg" not in cols


def test_feature_columns_include_1h_market_and_prop_fields():
    """v6.5.0: formerly-orphaned features are now registered in the feature vector."""
    cols = get_feature_columns()
    assert "mkt_1h_spread_avg" in cols
    assert "mkt_1h_total_avg" in cols
    assert "mkt_1h_home_ml_prob" in cols
    assert "prop_blk_avg_line" in cols
    assert "prop_stl_avg_line" in cols
    assert "prop_tov_avg_line" in cols


# ── Reset Elo cache ────────────────────────────────────────────


def test_reset_elo_cache():
    """reset_elo_cache should clear the global _elo_system without error."""
    reset_elo_cache()  # just ensure it doesn't raise
