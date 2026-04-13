"""Tests for src.models.odds_utils — odds conversion and snapshot helpers."""

from types import SimpleNamespace

from src.models.odds_utils import (
    american_to_prob,
    build_odds_detail,
    consensus_line,
    consensus_price,
    latest_snapshots,
    prob_to_american,
)

# ── prob_to_american ────────────────────────────────────────────


class TestProbToAmerican:
    def test_favourite(self):
        result = prob_to_american(0.7)
        assert result.startswith("-")

    def test_underdog(self):
        result = prob_to_american(0.3)
        assert result.startswith("+")

    def test_coin_flip(self):
        result = prob_to_american(0.5)
        assert result == "-100" or result == "+100"

    def test_extreme_low_returns_empty(self):
        assert prob_to_american(0.005) == ""

    def test_extreme_high_returns_empty(self):
        assert prob_to_american(0.995) == ""


# ── american_to_prob ────────────────────────────────────────────


class TestAmericanToProb:
    def test_negative_odds(self):
        p = american_to_prob("-150")
        assert p is not None
        assert 0.55 < p < 0.65

    def test_positive_odds(self):
        p = american_to_prob("+200")
        assert p is not None
        assert 0.3 < p < 0.4

    def test_empty_string(self):
        assert american_to_prob("") is None

    def test_none(self):
        assert american_to_prob(None) is None

    def test_invalid(self):
        assert american_to_prob("abc") is None

    def test_zero_odds(self):
        p = american_to_prob("0")
        assert p == 0.5

    def test_plus_with_sign(self):
        p = american_to_prob("+100")
        assert p is not None
        assert abs(p - 0.5) < 0.01


# ── consensus_line ──────────────────────────────────────────────


class TestConsensusLine:
    def test_basic(self):
        books = {"dk": {"spread": -3.5}, "fd": {"spread": -4.5}}
        result = consensus_line(books, "spread")
        assert result == -4.0

    def test_missing_key(self):
        books = {"dk": {"total": 220.5}, "fd": {}}
        result = consensus_line(books, "total")
        assert result == 220.5

    def test_empty_books(self):
        assert consensus_line({}, "spread") is None

    def test_all_none(self):
        books = {"dk": {"spread": None}, "fd": {"spread": None}}
        assert consensus_line(books, "spread") is None


# ── consensus_price ─────────────────────────────────────────────


class TestConsensusPrice:
    def test_basic(self):
        books = {"dk": {"spread_price": -110}, "fd": {"spread_price": -108}}
        result = consensus_price(books, "spread_price")
        assert result == "-109"

    def test_empty(self):
        assert consensus_price({}, "spread_price") == ""


# ── latest_snapshots ────────────────────────────────────────────


def _snap(bk: str, mkt: str, outcome: str, ts_str: str):
    from datetime import datetime

    ts = datetime.fromisoformat(ts_str)
    return SimpleNamespace(
        bookmaker=bk,
        market=mkt,
        outcome_name=outcome,
        captured_at=ts,
    )


class TestLatestSnapshots:
    def test_deduplicates(self):
        snaps = [
            _snap("dk", "spreads", "Lakers", "2025-03-01T10:00:00"),
            _snap("dk", "spreads", "Lakers", "2025-03-01T12:00:00"),
        ]
        result, newest = latest_snapshots(snaps)
        assert len(result) == 1
        assert newest.hour == 12

    def test_empty(self):
        result, newest = latest_snapshots([])
        assert result == []
        assert newest is None

    def test_different_bookmakers(self):
        snaps = [
            _snap("dk", "spreads", "Lakers", "2025-03-01T10:00:00"),
            _snap("fd", "spreads", "Lakers", "2025-03-01T10:00:00"),
        ]
        result, _ = latest_snapshots(snaps)
        assert len(result) == 2


# ── build_odds_detail ───────────────────────────────────────────


def _full_snap(bk, mkt, outcome, price, point, ts_str):
    from datetime import datetime

    return SimpleNamespace(
        bookmaker=bk,
        market=mkt,
        outcome_name=outcome,
        price=price,
        point=point,
        captured_at=datetime.fromisoformat(ts_str),
    )


class TestBuildOddsDetail:
    def test_spread(self):
        snaps = [_full_snap("dk", "spreads", "Lakers", -110, -3.5, "2025-03-01T10:00:00")]
        from datetime import datetime

        result = build_odds_detail(snaps, "Lakers", "Celtics", datetime(2025, 3, 1))
        assert "books" in result
        assert "dk" in result["books"]
        assert result["books"]["dk"]["spread"] == -3.5

    def test_totals(self):
        snaps = [_full_snap("dk", "totals", "Over", -110, 220.5, "2025-03-01T10:00:00")]
        from datetime import datetime

        result = build_odds_detail(snaps, "Lakers", "Celtics", datetime(2025, 3, 1))
        assert result["books"]["dk"]["total"] == 220.5

    def test_h2h(self):
        snaps = [
            _full_snap("dk", "h2h", "Lakers", -150, None, "2025-03-01T10:00:00"),
            _full_snap("dk", "h2h", "Celtics", 130, None, "2025-03-01T10:00:00"),
        ]
        from datetime import datetime

        result = build_odds_detail(snaps, "Lakers", "Celtics", datetime(2025, 3, 1))
        assert result["books"]["dk"]["home_ml"] == -150
        assert result["books"]["dk"]["away_ml"] == 130

    def test_1h_markets(self):
        snaps = [
            _full_snap("dk", "spreads_h1", "Lakers", -110, -1.5, "2025-03-01T10:00:00"),
            _full_snap("dk", "totals_h1", "Over", -110, 110.5, "2025-03-01T10:00:00"),
            _full_snap("dk", "h2h_h1", "Lakers", -130, None, "2025-03-01T10:00:00"),
            _full_snap("dk", "h2h_h1", "Celtics", 110, None, "2025-03-01T10:00:00"),
        ]
        from datetime import datetime

        result = build_odds_detail(snaps, "Lakers", "Celtics", datetime(2025, 3, 1))
        bk = result["books"]["dk"]
        assert bk["spread_h1"] == -1.5
        assert bk["total_h1"] == 110.5
        assert bk["home_ml_h1"] == -130
        assert bk["away_ml_h1"] == 110

    def test_empty_snapshots(self):
        from datetime import datetime

        result = build_odds_detail([], "Lakers", "Celtics", datetime(2025, 3, 1))
        assert result["books"] == {}
