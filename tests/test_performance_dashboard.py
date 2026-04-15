"""Tests for the performance dashboard HTML builder."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.api.routes.performance import (
    GradedPick,
    _build_dashboard_html,
    _pct_class,
    _roi_class,
)


def _pred_game(
    home_fg=110,
    away_fg=105,
    spread=-3.5,
    total=220.0,
    h1_spread=None,
    h1_total=None,
    home_1h=55,
    away_1h=52,
    clv_spread=1.0,
    clv_total=-0.5,
):
    pred = MagicMock()
    pred.predicted_home_fg = home_fg
    pred.predicted_away_fg = away_fg
    pred.predicted_home_1h = home_1h
    pred.predicted_away_1h = away_1h
    pred.opening_spread = spread
    pred.opening_total = total
    pred.opening_h1_spread = h1_spread
    pred.opening_h1_total = h1_total
    pred.fg_home_ml_prob = 0.65
    pred.h1_home_ml_prob = 0.60
    pred.clv_spread = clv_spread
    pred.clv_total = clv_total
    pred.predicted_at = MagicMock()
    # Store 1H lines in odds_sourced for _grade_game
    odds_sourced_dict = {}
    if h1_spread is not None:
        odds_sourced_dict["opening_h1_spread"] = h1_spread
    if h1_total is not None:
        odds_sourced_dict["opening_h1_total"] = h1_total
    pred.odds_sourced = odds_sourced_dict if odds_sourced_dict else None

    game = MagicMock()
    game.id = 1
    game.home_score_fg = 112
    game.away_score_fg = 108
    game.home_score_1h = 56
    game.away_score_1h = 50
    game.home_team = MagicMock(name="Lakers")
    game.away_team = MagicMock(name="Celtics")
    game.status = "FT"
    return pred, game


class TestBuildDashboardHtml:
    def test_empty_graded_shows_accumulating(self):
        html = _build_dashboard_html([], [])
        assert "Accumulating Data" in html

    def test_with_graded_picks_has_overview(self):
        graded = [
            GradedPick("FG", "SPREAD", 3.5, "W", "Celtics @ Lakers", "Lakers -3.5"),
            GradedPick("FG", "TOTAL", 2.0, "L", "Celtics @ Lakers", "Over 220"),
            GradedPick("1H", "SPREAD", 1.5, "P", "Celtics @ Lakers", "Lakers -1.5"),
        ]
        rows = [_pred_game()]
        html = _build_dashboard_html(rows, graded)
        assert "GBSV Performance Dashboard" in html
        assert "Overview" in html
        assert "Performance by Market" in html
        assert "Prediction Accuracy" in html
        assert "Performance by Edge Threshold" in html
        assert "Closing Line Value" in html
        assert "Recent Results" in html

    def test_with_clv_data(self):
        graded = [GradedPick("FG", "SPREAD", 3.0, "W")]
        rows = [_pred_game(clv_spread=2.5, clv_total=-1.0)]
        html = _build_dashboard_html(rows, graded)
        assert "CLV" in html

    def test_recent_results_classes(self):
        graded = [
            GradedPick("FG", "SPREAD", 3.0, "W", "A @ B", "B -3"),
            GradedPick("FG", "TOTAL", 2.0, "L", "A @ B", "Over 220"),
            GradedPick("1H", "SPREAD", 1.0, "P", "A @ B", "B -1"),
        ]
        html = _build_dashboard_html([_pred_game()], graded)
        assert "win-row" in html
        assert "loss-row" in html

    def test_threshold_rows(self):
        graded = [GradedPick("FG", "SPREAD", 5.0, "W")] * 5
        html = _build_dashboard_html([_pred_game()], graded)
        assert "Edge" in html


class TestPctClass:
    def test_positive(self):
        assert _pct_class(55.0) == "positive"

    def test_negative(self):
        assert _pct_class(45.0) == "negative"

    def test_neutral(self):
        assert _pct_class(50.0) == ""

    def test_none(self):
        assert _pct_class(None) == ""


class TestRoiClass:
    def test_positive(self):
        assert _roi_class(5.0) == "positive"

    def test_negative(self):
        assert _roi_class(-5.0) == "negative"

    def test_zero(self):
        assert _roi_class(0.0) == ""

    def test_none(self):
        assert _roi_class(None) == ""
