"""Tests for src.data.reconciliation — game reconciliation helpers."""

from types import SimpleNamespace

from src.data.reconciliation import _copy_prediction_payload


class TestCopyPredictionPayload:
    def test_copies_from_preferred(self):
        target = SimpleNamespace(
            predicted_home_fg=None,
            predicted_away_fg=None,
            predicted_home_1h=None,
            predicted_away_1h=None,
            fg_spread=None,
            fg_total=None,
            fg_home_ml_prob=None,
            h1_spread=None,
            h1_total=None,
            h1_home_ml_prob=None,
            opening_spread=None,
            opening_total=None,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            odds_sourced=None,
            predicted_at=None,
        )
        preferred = SimpleNamespace(
            predicted_home_fg=112.0,
            predicted_away_fg=108.0,
            predicted_home_1h=56.0,
            predicted_away_1h=54.0,
            fg_spread=-4.0,
            fg_total=220.0,
            fg_home_ml_prob=0.65,
            h1_spread=-2.0,
            h1_total=110.0,
            h1_home_ml_prob=0.60,
            opening_spread=-3.0,
            opening_total=218.0,
            closing_spread=-4.5,
            closing_total=219.0,
            clv_spread=-1.5,
            clv_total=1.0,
            odds_sourced={"books": {}},
            predicted_at="2025-03-15T20:00:00",
        )
        _copy_prediction_payload(target, preferred)
        assert target.predicted_home_fg == 112.0
        assert target.fg_spread == -4.0
        assert target.odds_sourced == {"books": {}}

    def test_falls_back_when_preferred_is_none(self):
        target = SimpleNamespace(
            predicted_home_fg=None,
            predicted_away_fg=None,
            predicted_home_1h=None,
            predicted_away_1h=None,
            fg_spread=None,
            fg_total=None,
            fg_home_ml_prob=None,
            h1_spread=None,
            h1_total=None,
            h1_home_ml_prob=None,
            opening_spread=None,
            opening_total=None,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            odds_sourced=None,
            predicted_at=None,
        )
        preferred = SimpleNamespace(
            predicted_home_fg=None,
            predicted_away_fg=None,
            predicted_home_1h=None,
            predicted_away_1h=None,
            fg_spread=None,
            fg_total=None,
            fg_home_ml_prob=None,
            h1_spread=None,
            h1_total=None,
            h1_home_ml_prob=None,
            opening_spread=None,
            opening_total=None,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            odds_sourced=None,
            predicted_at=None,
        )
        fallback = SimpleNamespace(
            predicted_home_fg=100.0,
            predicted_away_fg=95.0,
            predicted_home_1h=50.0,
            predicted_away_1h=48.0,
            fg_spread=-5.0,
            fg_total=195.0,
            fg_home_ml_prob=0.55,
            h1_spread=-2.5,
            h1_total=98.0,
            h1_home_ml_prob=0.53,
            opening_spread=-4.0,
            opening_total=196.0,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            odds_sourced=None,
            predicted_at="fallback-ts",
        )
        _copy_prediction_payload(target, preferred, fallback)
        assert target.predicted_home_fg == 100.0
        assert target.fg_spread == -5.0
        assert target.predicted_at == "fallback-ts"

    def test_no_fallback_leaves_none(self):
        target = SimpleNamespace(
            predicted_home_fg=99.0,
            predicted_away_fg=None,
            predicted_home_1h=None,
            predicted_away_1h=None,
            fg_spread=None,
            fg_total=None,
            fg_home_ml_prob=None,
            h1_spread=None,
            h1_total=None,
            h1_home_ml_prob=None,
            opening_spread=None,
            opening_total=None,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            odds_sourced=None,
            predicted_at=None,
        )
        preferred = SimpleNamespace(
            predicted_home_fg=None,
            predicted_away_fg=None,
            predicted_home_1h=None,
            predicted_away_1h=None,
            fg_spread=None,
            fg_total=None,
            fg_home_ml_prob=None,
            h1_spread=None,
            h1_total=None,
            h1_home_ml_prob=None,
            opening_spread=None,
            opening_total=None,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            odds_sourced=None,
            predicted_at=None,
        )
        _copy_prediction_payload(target, preferred, fallback=None)
        # Target retains original value only if not overwritten
        assert target.predicted_home_fg == 99.0
