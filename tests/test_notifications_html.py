"""Tests for src.notifications._html — HTML slate builder helpers."""

from types import SimpleNamespace

from src.notifications._html import (
    _build_html_odds_section,
    _confidence_badge,
    _edge_css_color,
    _esc,
    _pick_side_border,
    _segment_pill,
)


class TestEsc:
    def test_escapes_html(self):
        assert _esc("<script>") == "&lt;script&gt;"

    def test_ampersand(self):
        assert _esc("a & b") == "a &amp; b"

    def test_plain_text(self):
        assert _esc("Lakers") == "Lakers"

    def test_numeric_input(self):
        assert _esc(42) == "42"


class TestEdgeCssColor:
    def test_high_edge(self):
        assert _edge_css_color(7.0) == "#16a34a"

    def test_medium_edge(self):
        assert _edge_css_color(5.0) == "#ca8a04"

    def test_low_edge(self):
        assert _edge_css_color(3.0) == "#2563eb"

    def test_minimal_edge(self):
        assert _edge_css_color(1.0) == "#6b7280"


class TestConfidenceBadge:
    def test_high_fires(self):
        html = _confidence_badge(4)
        assert "#dcfce7" in html  # green bg
        assert "\U0001f525\U0001f525\U0001f525\U0001f525" in html

    def test_medium_fires(self):
        html = _confidence_badge(3)
        assert "#fef9c3" in html  # yellow bg

    def test_low_fires(self):
        html = _confidence_badge(1)
        assert "#fef2f2" in html  # red bg


class TestSegmentPill:
    def test_fg_segment(self):
        html = _segment_pill("FG")
        assert "#34a853" in html  # green colour
        assert "FG" in html

    def test_1h_segment(self):
        html = _segment_pill("1H")
        assert "#4285F4" in html  # blue colour
        assert "1H" in html


class TestPickSideBorder:
    def test_total_over(self):
        pick = SimpleNamespace(label="OVER 220.5", market_type="TOTAL", matchup="Celtics @ Lakers")
        assert _pick_side_border(pick) == "#16a34a"

    def test_total_under(self):
        pick = SimpleNamespace(label="UNDER 220.5", market_type="TOTAL", matchup="Celtics @ Lakers")
        assert _pick_side_border(pick) == "#2563eb"

    def test_home_team_spread(self):
        pick = SimpleNamespace(
            label="Lakers -5.0", market_type="SPREAD", matchup="Celtics @ Lakers"
        )
        assert _pick_side_border(pick) == "#16a34a"

    def test_away_team_spread(self):
        pick = SimpleNamespace(
            label="Celtics +5.0", market_type="SPREAD", matchup="Celtics @ Lakers"
        )
        assert _pick_side_border(pick) == "#2563eb"


class TestBuildHtmlOddsSection:
    def test_empty_odds(self):
        assert _build_html_odds_section({}, {}) == ""

    def test_single_game_single_book(self):
        odds = {
            1: {
                "books": {
                    "dk": {
                        "spread": -5.0,
                        "spread_price": -110,
                        "total": 220.5,
                        "total_price": -110,
                    }
                }
            }
        }
        labels = {1: "Celtics @ Lakers"}
        html = _build_html_odds_section(odds, labels)
        assert "Celtics @ Lakers" in html
        assert "dk" in html
        assert "-5.0" in html

    def test_1h_lines_shown(self):
        odds = {
            1: {
                "books": {
                    "fd": {
                        "spread": -3.0,
                        "total": 218.0,
                        "spread_h1": -1.5,
                        "spread_h1_price": -115,
                        "total_h1": 112.5,
                        "total_h1_price": -108,
                    }
                }
            }
        }
        labels = {1: "Nets @ Hawks"}
        html = _build_html_odds_section(odds, labels)
        assert "-1.5" in html
        assert "112.5" in html

    def test_missing_fields_show_dash(self):
        odds = {1: {"books": {"bm": {}}}}
        labels = {1: "A @ B"}
        html = _build_html_odds_section(odds, labels)
        assert "\u2014" in html  # em dash for missing

    def test_captured_at_timestamp(self):
        odds = {1: {"books": {"dk": {"spread": -2.0}}, "captured_at": "2025-03-15T20:30:00+00:00"}}
        labels = {1: "A @ B"}
        html = _build_html_odds_section(odds, labels)
        assert "As of" in html
