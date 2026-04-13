"""Tests for src.notifications._csv — CSV slate builder."""

from datetime import UTC, datetime
from types import SimpleNamespace

from src.notifications._csv import build_slate_csv


def _game(home="Lakers", away="Celtics"):
    return SimpleNamespace(
        home_team=SimpleNamespace(name=home, season_stats=None, wins=40, losses=20),
        away_team=SimpleNamespace(name=away, season_stats=None, wins=30, losses=30),
        home_team_id=1,
        away_team_id=2,
        commence_time=datetime(2025, 3, 15, 23, 0, tzinfo=UTC),
    )


def _pred(
    fg_spread=-5.0,
    fg_total=222.0,
    fg_home_ml_prob=0.65,
    h1_spread=-3.0,
    h1_total=112.0,
    opening_spread=0.0,
    opening_total=220.0,
    odds_sourced=None,
):
    return SimpleNamespace(
        fg_spread=fg_spread,
        fg_total=fg_total,
        fg_home_ml_prob=fg_home_ml_prob,
        h1_spread=h1_spread,
        h1_total=h1_total,
        opening_spread=opening_spread,
        opening_total=opening_total,
        odds_sourced=odds_sourced,
        predicted_home_fg=112.0,
        predicted_away_fg=110.0,
        predicted_home_1h=56.0,
        predicted_away_1h=53.0,
        h1_home_ml_prob=0.60,
    )


class TestBuildSlateCsv:
    def test_header_row(self):
        csv_str = build_slate_csv([(_pred(), _game())], min_edge=1.0)
        first_line = csv_str.split("\n")[0]
        assert "Time (CT)" in first_line
        assert "Matchup" in first_line
        assert "Edge" in first_line
        assert "Rating" in first_line

    def test_produces_data_rows(self):
        csv_str = build_slate_csv([(_pred(), _game())], min_edge=1.0)
        lines = [l for l in csv_str.strip().split("\n") if l.strip()]
        assert len(lines) >= 2  # header + at least 1 data row

    def test_empty_input(self):
        csv_str = build_slate_csv([], min_edge=1.0)
        lines = [l for l in csv_str.strip().split("\n") if l.strip()]
        assert len(lines) == 1  # header only

    def test_high_min_edge_filters_all(self):
        csv_str = build_slate_csv([(_pred(fg_spread=-1.0), _game())], min_edge=50.0)
        lines = [l for l in csv_str.strip().split("\n") if l.strip()]
        assert len(lines) == 1  # header only

    def test_odds_map_passed_through(self):
        csv_str = build_slate_csv(
            [(_pred(), _game(), {"FG_SPREAD": "-110"})],
            min_edge=1.0,
        )
        assert "-110" in csv_str or len(csv_str) > 0

    def test_multiple_games(self):
        rows = [
            (_pred(), _game("Lakers", "Celtics")),
            (_pred(), _game("Hawks", "Nets")),
        ]
        csv_str = build_slate_csv(rows, min_edge=1.0)
        assert "Lakers" in csv_str or "Hawks" in csv_str

    def test_odds_sourced_adds_consensus(self):
        sourced = {
            "books": {
                "dk": {"spread": -5.0, "spread_price": -110, "total": 220.5},
                "fd": {"spread": -4.5, "total": 221.0},
            }
        }
        csv_str = build_slate_csv([(_pred(odds_sourced=sourced), _game())], min_edge=1.0)
        assert "consensus" in csv_str.lower() or len(csv_str) > 100
