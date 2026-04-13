"""Tests for src.notifications._picks — pick extraction logic."""

from datetime import UTC, datetime
from types import SimpleNamespace

from src.notifications._picks import extract_picks


def _game(home="Lakers", away="Celtics", commence=None):
    """Build a minimal game-like object."""
    ct = commence or datetime(2025, 3, 15, 23, 0, tzinfo=UTC)
    return SimpleNamespace(
        home_team=SimpleNamespace(name=home, season_stats=None, wins=40, losses=20),
        away_team=SimpleNamespace(name=away, season_stats=None, wins=30, losses=30),
        home_team_id=1,
        away_team_id=2,
        commence_time=ct,
    )


def _pred(
    fg_spread=-5.0,
    fg_total=222.0,
    fg_home_ml_prob=0.65,
    h1_spread=-3.0,
    h1_total=112.0,
    opening_spread=None,
    opening_total=None,
    odds_sourced=None,
    predicted_home_fg=112.0,
    predicted_away_fg=110.0,
    predicted_home_1h=56.0,
    predicted_away_1h=53.0,
    h1_home_ml_prob=0.60,
):
    """Build a minimal prediction-like object."""
    return SimpleNamespace(
        fg_spread=fg_spread,
        fg_total=fg_total,
        fg_home_ml_prob=fg_home_ml_prob,
        h1_spread=h1_spread,
        h1_total=h1_total,
        opening_spread=opening_spread,
        opening_total=opening_total,
        odds_sourced=odds_sourced,
        predicted_home_fg=predicted_home_fg,
        predicted_away_fg=predicted_away_fg,
        predicted_home_1h=predicted_home_1h,
        predicted_away_1h=predicted_away_1h,
        h1_home_ml_prob=h1_home_ml_prob,
    )


class TestExtractPicks:
    def test_returns_list(self):
        picks = extract_picks(_pred(), _game(), min_edge=1.0)
        assert isinstance(picks, list)

    def test_picks_have_required_attrs(self):
        picks = extract_picks(_pred(), _game(), min_edge=1.0)
        for p in picks:
            assert hasattr(p, "label")
            assert hasattr(p, "edge")
            assert hasattr(p, "segment")
            assert hasattr(p, "market_type")

    def test_fg_spread_pick_generated(self):
        """Model projects -5 vs no market line → pick generated for edge 5."""
        picks = extract_picks(_pred(fg_spread=-5.0), _game(), min_edge=3.0)
        spread_picks = [p for p in picks if p.segment == "FG" and p.market_type == "SPREAD"]
        assert len(spread_picks) >= 1

    def test_fg_total_with_market_line(self):
        """Model total 230 vs market 220 → 10pt over pick."""
        pred = _pred(fg_total=230.0, opening_total=220.0)
        picks = extract_picks(pred, _game(), min_edge=3.0)
        total_picks = [p for p in picks if p.segment == "FG" and p.market_type == "TOTAL"]
        assert len(total_picks) >= 1
        assert "OVER" in total_picks[0].label

    def test_fg_total_under_pick(self):
        """Model total 210 vs market 220 → 10pt under pick."""
        pred = _pred(fg_total=210.0, opening_total=220.0)
        picks = extract_picks(pred, _game(), min_edge=3.0)
        total_picks = [p for p in picks if p.segment == "FG" and p.market_type == "TOTAL"]
        assert len(total_picks) >= 1
        assert "UNDER" in total_picks[0].label

    def test_1h_spread_pick(self):
        pred = _pred(h1_spread=-5.0, opening_spread=None)
        game = _game()
        picks = extract_picks(pred, game, min_edge=3.0)
        h1_spread_picks = [p for p in picks if p.segment == "1H" and p.market_type == "SPREAD"]
        assert len(h1_spread_picks) >= 1

    def test_ml_pick_generated(self):
        """Strong model edge on ML → ML pick generated."""
        pred = _pred(fg_spread=8.0, fg_home_ml_prob=0.80)
        picks = extract_picks(pred, _game(), min_edge=1.0, odds_map={"FG_ML_HOME": "-110"})
        ml_picks = [p for p in picks if p.market_type == "ML"]
        assert len(ml_picks) >= 1

    def test_low_edge_filtered(self):
        """With high min_edge, weak picks should be filtered."""
        pred = _pred(fg_spread=-1.0, fg_total=230.0, h1_spread=-0.5, h1_total=115.0)
        picks = extract_picks(pred, _game(), min_edge=20.0)
        assert len(picks) == 0

    def test_matchup_format(self):
        picks = extract_picks(_pred(), _game(), min_edge=1.0)
        if picks:
            assert " @ " in picks[0].matchup

    def test_none_team_fallback(self):
        """When team objects are None, should use team IDs."""
        game = SimpleNamespace(
            home_team=None,
            away_team=None,
            home_team_id=100,
            away_team_id=200,
            commence_time=datetime(2025, 3, 15, 23, 0, tzinfo=UTC),
        )
        picks = extract_picks(_pred(), game, min_edge=1.0)
        assert isinstance(picks, list)

    def test_odds_map_passed_through(self):
        """When odds_map is provided, it should be used for odds strings."""
        odds_map = {"FG_SPREAD": "-110", "FG_TOTAL": "-105"}
        pred = _pred(fg_spread=-5.0, opening_spread=0.0, opening_total=220.0, fg_total=230.0)
        picks = extract_picks(pred, _game(), min_edge=3.0, odds_map=odds_map)
        spread_picks = [p for p in picks if p.segment == "FG" and p.market_type == "SPREAD"]
        if spread_picks:
            assert spread_picks[0].odds == "-110"

    def test_with_books_in_odds_sourced(self):
        """odds_sourced with books should generate consensus odds."""
        sourced = {
            "books": {
                "dk": {"spread_price": -110, "total_price": -110},
                "fd": {"spread_price": -108, "total_price": -112},
            }
        }
        pred = _pred(
            fg_spread=-5.0,
            opening_spread=0.0,
            opening_total=220.0,
            fg_total=230.0,
            odds_sourced=sourced,
        )
        picks = extract_picks(pred, _game(), min_edge=3.0)
        assert len(picks) >= 1

    def test_1h_total_pick(self):
        """1H total edge should generate a pick."""
        pred = _pred(h1_total=120.0)
        game = _game()
        # Use odds_sourced to provide 1H total line
        pred.odds_sourced = {"books": {"dk": {"total_h1": 110.0}}}
        picks = extract_picks(pred, game, min_edge=3.0)
        h1_total = [p for p in picks if p.segment == "1H" and p.market_type == "TOTAL"]
        assert len(h1_total) >= 1

    def test_1h_ml_pick(self):
        """Strong 1H ML edge should generate a pick."""
        pred = _pred(h1_spread=6.0, h1_home_ml_prob=0.80)
        picks = extract_picks(pred, _game(), min_edge=1.0, odds_map={"1H_ML_HOME": "-110"})
        h1_ml = [p for p in picks if p.segment == "1H" and p.market_type == "ML"]
        assert len(h1_ml) >= 1
