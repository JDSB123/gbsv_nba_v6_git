"""Tests for src.notifications._helpers — pure helper functions."""

import os
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.notifications._helpers import (
    Pick,
    _app_build_stamp,
    _edge_color,
    _fire_count,
    _fire_emojis,
    _fmt_time_cst,
    _get_model_modified_at,
    _team_record,
)

# ── _fire_emojis ────────────────────────────────────────────────


class TestFireEmojis:
    def test_5_fires_for_edge_9(self):
        assert _fire_emojis(9.0).count("\U0001f525") == 5

    def test_4_fires_for_edge_7(self):
        assert _fire_emojis(7.0).count("\U0001f525") == 4

    def test_3_fires_for_edge_5(self):
        assert _fire_emojis(5.0).count("\U0001f525") == 3

    def test_2_fires_for_edge_3_5(self):
        assert _fire_emojis(3.5).count("\U0001f525") == 2

    def test_1_fire_for_low_edge(self):
        assert _fire_emojis(1.0).count("\U0001f525") == 1


# ── _fire_count ─────────────────────────────────────────────────


class TestFireCount:
    def test_5_at_extreme(self):
        assert _fire_count(10.0) == 5

    def test_4_at_7(self):
        assert _fire_count(7.0) == 4

    def test_3_at_5(self):
        assert _fire_count(5.0) == 3

    def test_2_at_3_5(self):
        assert _fire_count(3.5) == 2

    def test_1_at_low(self):
        assert _fire_count(2.0) == 1


# ── _edge_color ─────────────────────────────────────────────────


class TestEdgeColor:
    def test_attention_at_high(self):
        assert _edge_color(8.0) == "Attention"

    def test_good_at_medium(self):
        assert _edge_color(5.0) == "Good"

    def test_accent_at_low(self):
        assert _edge_color(2.0) == "Accent"


# ── _fmt_time_cst ───────────────────────────────────────────────


class TestFmtTimeCst:
    def test_none_returns_tbd(self):
        assert _fmt_time_cst(None) == "TBD"

    def test_aware_datetime(self):
        dt = datetime(2025, 3, 1, 18, 30, tzinfo=UTC)
        result = _fmt_time_cst(dt)
        assert "CT" in result

    def test_naive_datetime_treated_utc(self):
        dt = datetime(2025, 6, 15, 22, 0)
        result = _fmt_time_cst(dt)
        assert "CT" in result


# ── _team_record ────────────────────────────────────────────────


class TestTeamRecord:
    def test_from_season_stats_list(self):
        stats = SimpleNamespace(wins=42, losses=18)
        team = SimpleNamespace(season_stats=[stats])
        assert _team_record(team) == "42-18"

    def test_from_direct_attrs(self):
        team = SimpleNamespace(wins=30, losses=30, season_stats=None)
        assert _team_record(team) == "30-30"

    def test_no_data(self):
        team = SimpleNamespace(season_stats=None, wins=None, losses=None)
        assert _team_record(team) == ""

    def test_none_team(self):
        assert _team_record(None) == ""

    def test_from_season_stats_dict(self):
        stats = SimpleNamespace(wins=55, losses=27)
        team = SimpleNamespace(season_stats=stats)
        assert _team_record(team) == "55-27"


# ── _app_build_stamp ────────────────────────────────────────────


def test_app_build_stamp_from_env(monkeypatch):
    monkeypatch.setenv("APP_BUILD_TIMESTAMP", "2025-01-01 00:00 UTC")
    assert _app_build_stamp() == "2025-01-01 00:00 UTC"


def test_app_build_stamp_default(monkeypatch):
    monkeypatch.delenv("APP_BUILD_TIMESTAMP", raising=False)
    stamp = _app_build_stamp()
    assert "UTC" in stamp


# ── _get_model_modified_at ──────────────────────────────────────


def test_model_modified_at_no_files():
    # Should return 'unknown' if no model files exist
    with patch("src.notifications._helpers._ARTIFACTS_DIR") as mock_dir:
        mock_dir.glob.return_value = []
        result = _get_model_modified_at()
        assert isinstance(result, str)


# ── Pick dataclass ──────────────────────────────────────────────


def test_pick_fields():
    p = Pick(
        label="Lakers -3.5",
        edge=5.2,
        time_cst="7:00 PM CT",
        matchup="Celtics @ Lakers",
        segment="FG",
        market_type="SPREAD",
        market_line="-3.5",
        model_scores="Lakers 112, Celtics 108",
        home_record="42-18",
        away_record="28-32",
        confidence=3,
        odds="-110",
        rationale="Model edge 5.2",
    )
    assert p.label == "Lakers -3.5"
    assert p.edge == 5.2
    assert p.confidence == 3
    assert p.odds == "-110"
