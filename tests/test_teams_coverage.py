"""Tests for teams.py: _get_model_modified_at, build_html_slate, build_slate_csv,
_odds_source_block with 1H, _edge_css_color, and performance dashboard HTML."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from src.notifications.teams import (
    _edge_css_color,
    _get_model_modified_at,
    _odds_source_block,
    build_html_slate,
    build_slate_csv,
    build_teams_card,
)


def _make_pred_game(
    home_fg=110.0, away_fg=105.0, home_1h=55.0, away_1h=52.0,
    opening_spread=-3.5, opening_total=220.0,
    opening_h1_spread=None, opening_h1_total=None,
    home_name="Lakers", away_name="Celtics",
    home_record="30-10", away_record="25-15",
    odds_sourced=None,
):
    pred = SimpleNamespace(
        predicted_home_fg=home_fg,
        predicted_away_fg=away_fg,
        predicted_home_1h=home_1h,
        predicted_away_1h=away_1h,
        fg_spread=home_fg - away_fg,
        fg_total=home_fg + away_fg,
        h1_spread=home_1h - away_1h,
        h1_total=home_1h + away_1h,
        fg_home_ml_prob=0.65,
        h1_home_ml_prob=0.60,
        opening_spread=opening_spread,
        opening_total=opening_total,
        opening_h1_spread=opening_h1_spread,
        opening_h1_total=opening_h1_total,
        game_id=1,
        odds_sourced=odds_sourced,
    )
    home_team = SimpleNamespace(name=home_name)
    away_team = SimpleNamespace(name=away_name)
    game = SimpleNamespace(
        id=1,
        home_team=home_team,
        away_team=away_team,
        commence_time=datetime(2024, 12, 1, 19, 0, tzinfo=UTC),
    )
    # Attach records
    home_team.team_season_stats = [SimpleNamespace(wins=30, losses=10)]
    away_team.team_season_stats = [SimpleNamespace(wins=25, losses=15)]
    return pred, game


# ── _get_model_modified_at ──────────────────────────────────────


class TestGetModelModifiedAt:
    def test_returns_timestamp_when_files_exist(self, tmp_path):
        model_file = tmp_path / "model_home_fg.json"
        model_file.write_text("{}")

        with patch("src.notifications.teams._ARTIFACTS_DIR", tmp_path):
            result = _get_model_modified_at()
        assert "UTC" in result
        assert result != "unknown"

    def test_returns_unknown_when_no_files(self, tmp_path):
        with patch("src.notifications.teams._ARTIFACTS_DIR", tmp_path):
            result = _get_model_modified_at()
        assert result == "unknown"

    def test_returns_unknown_on_exception(self):
        with patch("src.notifications.teams._ARTIFACTS_DIR") as mock_dir:
            mock_dir.glob.side_effect = PermissionError("denied")
            result = _get_model_modified_at()
        assert result == "unknown"


# ── _edge_css_color ─────────────────────────────────────────────


class TestEdgeCssColor:
    def test_hot(self):
        assert _edge_css_color(7.0) == "#16a34a"

    def test_warm(self):
        assert _edge_css_color(5.0) == "#ca8a04"

    def test_mild(self):
        assert _edge_css_color(3.0) == "#2563eb"

    def test_flat(self):
        assert _edge_css_color(1.0) == "#6b7280"


# ── _odds_source_block with 1H data ────────────────────────────


class TestOddsSourceBlock1H:
    def test_with_1h_markets(self):
        odds = {
            "books": {
                "fanduel": {
                    "spread": -3.5,
                    "total": 220.5,
                    "spread_h1": -1.5,
                    "total_h1": 110.5,
                    "spread_h1_price": -110,
                    "total_h1_price": -105,
                    "home_ml": -150,
                },
            },
            "captured_at": "2024-12-01T18:00:00Z",
        }
        blocks = _odds_source_block(odds)
        assert len(blocks) == 1
        text = blocks[0]["text"]
        assert "1H Sprd" in text
        assert "1H O/U" in text
        assert "Sprd -3.5" in text
        assert "ML -150" in text

    def test_with_timestamp_parsing(self):
        odds = {
            "books": {"dk": {"spread": -2.0}},
            "captured_at": "2024-12-01T23:30:00Z",
        }
        blocks = _odds_source_block(odds)
        assert len(blocks) == 1
        assert "CT" in blocks[0]["text"]

    def test_with_invalid_timestamp_omits_ct_suffix(self):
        odds = {
            "books": {"dk": {"total_h1": 109.5, "total_h1_price": -105}},
            "captured_at": "not-a-timestamp",
        }
        blocks = _odds_source_block(odds)
        assert len(blocks) == 1
        assert "1H O/U" in blocks[0]["text"]
        assert "CT" not in blocks[0]["text"]


# ── build_html_slate ────────────────────────────────────────────


class TestBuildHtmlSlate:
    def test_empty_picks_message(self):
        pred, game = _make_pred_game(
            home_fg=115, away_fg=115,  # no edge
            home_1h=57.5, away_1h=57.5,
            opening_spread=0, opening_total=230,
            opening_h1_spread=0, opening_h1_total=115,
        )
        pred.fg_home_ml_prob = 0.50  # no ML edge either
        html = build_html_slate([(pred, game)], min_edge=99)
        assert "No qualified picks today" in html

    def test_with_picks_has_table(self):
        pred, game = _make_pred_game()
        html = build_html_slate([(pred, game)])
        assert "<table" in html
        assert "NBA Daily Slate" in html
        assert "Lakers" in html
        assert "Celtics" in html

    def test_with_odds_pulled_at(self):
        pred, game = _make_pred_game()
        pulled = datetime(2024, 12, 1, 17, 30, tzinfo=UTC)
        html = build_html_slate([(pred, game)], odds_pulled_at=pulled)
        assert "Odds pulled" in html

    def test_filter_dropdowns_present(self):
        pred, game = _make_pred_game()
        html = build_html_slate([(pred, game)])
        assert "fMatchup" in html
        assert "fSeg" in html
        assert "applyFilters" in html

    def test_with_odds_sourced(self):
        odds_sourced = {
            "books": {
                "dk": {
                    "spread": -3.0, "spread_price": -110,
                    "total": 218.0, "total_price": -105,
                    "home_ml": -150, "away_ml": 130,
                    "spread_h1": -1.5, "spread_h1_price": -115,
                    "total_h1": 109.0, "total_h1_price": -108,
                },
            },
            "captured_at": "2024-12-01T18:00:00Z",
        }
        pred, game = _make_pred_game(odds_sourced=odds_sourced)
        html = build_html_slate([(pred, game)])
        assert "NBA Daily Slate" in html
        # Odds table should be rendered
        assert "-3.0" in html
        assert "218.0" in html

    def test_with_partial_odds_rows_uses_placeholders(self):
        odds_sourced = {
            "books": {
                "fanduel": {"spread": -3.0},
                "draftkings": {"total_h1": 109.5, "away_ml": 125},
            },
            "captured_at": "bad-ts",
        }
        pred, game = _make_pred_game(odds_sourced=odds_sourced)
        html = build_html_slate([(pred, game)])
        assert "Odds Sources" in html
        assert "—" in html
        assert "As of" not in html


# ── build_slate_csv ─────────────────────────────────────────────


class TestBuildSlateCsv:
    def test_csv_has_headers(self):
        pred, game = _make_pred_game()
        result = build_slate_csv([(pred, game)])
        reader = csv.reader(io.StringIO(result))
        headers = next(reader)
        assert "Time (CT)" in headers
        assert "Matchup" in headers
        assert "Odds Source" in headers
        assert "Odds Pulled" in headers

    def test_csv_with_odds_sourced(self):
        odds_sourced = {
            "books": {
                "fanduel": {"spread": -3.5, "total": 220.0},
                "draftkings": {"spread": -3.0, "total": 219.5},
            },
            "captured_at": "2024-12-01T18:00:00Z",
        }
        pred, game = _make_pred_game(odds_sourced=odds_sourced)
        result = build_slate_csv([(pred, game)])
        assert "fanduel" in result.lower() or "draftkings" in result.lower()

    def test_csv_rows_populated(self):
        pred, game = _make_pred_game()
        result = build_slate_csv([(pred, game)])
        reader = csv.reader(io.StringIO(result))
        next(reader)
        rows = list(reader)
        assert len(rows) > 0  # At least one pick row

    def test_csv_with_odds_map(self):
        pred, game = _make_pred_game()
        odds_map = {"FG_SPREAD": "-110", "FG_TOTAL": "-110"}
        result = build_slate_csv([(pred, game, odds_map)])
        assert len(result) > 0


# ── build_teams_card with odds_by_game ──────────────────────────


class TestBuildTeamsCardOdds:
    def test_card_with_download_urls(self):
        pred, game = _make_pred_game()
        result = build_teams_card(
            [(pred, game)],
            max_games=10,
            download_url="https://example.com/slate.html",
            csv_download_url="https://example.com/slate.csv",
        )
        card = result["attachments"][0]["content"]
        assert any(a.get("title", "").startswith("📊") for a in card.get("actions", []))
        assert any(a.get("title", "").startswith("📥") for a in card.get("actions", []))

    def test_card_with_odds_by_game(self):
        odds_sourced = {
            "books": {
                "fanduel": {
                    "spread": -3.5, "spread_price": -110,
                    "total": 220.0, "total_price": -108,
                    "home_ml": -160, "away_ml": 140,
                    "spread_h1": -1.5, "spread_h1_price": -105,
                    "total_h1": 109.5, "total_h1_price": -112,
                },
            },
            "captured_at": "2024-12-01T18:00:00Z",
        }
        pred, game = _make_pred_game(odds_sourced=odds_sourced)
        result = build_teams_card([(pred, game)], max_games=10)
        body = result["attachments"][0]["content"]["body"]
        # Should contain odds source blocks
        texts = [b.get("text", "") for b in body if isinstance(b, dict)]
        assert any("Odds" in t for t in texts)
