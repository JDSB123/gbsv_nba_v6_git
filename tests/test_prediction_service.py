"""Tests for src.services.predictions — PredictionService.format_prediction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.services.predictions import PredictionService, _as_float, _parse_iso_utc


class TestAsFloat:
    def test_normal(self):
        assert _as_float(3.14) == 3.14

    def test_none_returns_default(self):
        assert _as_float(None) == 0.0

    def test_none_custom_default(self):
        assert _as_float(None, 99.0) == 99.0

    def test_string_numeric(self):
        assert _as_float("5.5") == 5.5


class TestParseIsoUtc:
    def test_valid_iso(self):
        result = _parse_iso_utc("2025-03-15T20:00:00+00:00")
        assert result is not None
        assert result.year == 2025

    def test_zulu_format(self):
        result = _parse_iso_utc("2025-03-15T20:00:00Z")
        assert result is not None

    def test_none_input(self):
        assert _parse_iso_utc(None) is None

    def test_empty_string(self):
        assert _parse_iso_utc("") is None

    def test_non_string(self):
        assert _parse_iso_utc(12345) is None

    def test_invalid_string(self):
        assert _parse_iso_utc("not-a-date") is None


class TestPredictionServiceFormatPrediction:
    def _make_pred(self, **kwargs):
        defaults = dict(
            fg_spread=-5.0,
            fg_total=222.0,
            fg_home_ml_prob=0.65,
            h1_spread=-3.0,
            h1_total=112.0,
            h1_home_ml_prob=0.60,
            opening_spread=None,
            opening_total=None,
            odds_sourced=None,
            predicted_home_fg=112.0,
            predicted_away_fg=110.0,
            predicted_home_1h=56.0,
            predicted_away_1h=54.0,
            clv_spread=None,
            clv_total=None,
            closing_spread=None,
            closing_total=None,
            model_version="6.0.0",
            created_at=None,
            predicted_at=None,
            game_id=1,
            id=1,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _make_game(self, **kwargs):
        defaults = dict(
            id=1,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
            home_team_id=1,
            away_team_id=2,
            odds_api_id=None,
            commence_time=None,
            status="NS",
            home_score_fg=None,
            away_score_fg=None,
            home_score_1h=None,
            away_score_1h=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _make_settings(self, **kwargs):
        defaults = dict(
            min_edge_spread=3.0,
            min_edge_total=3.0,
            min_edge_ml=2.0,
            market_blend_alpha=1.0,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _make_service(self, settings_overrides=None):
        repo = AsyncMock()
        predictor = MagicMock()
        settings = self._make_settings(**(settings_overrides or {}))
        return PredictionService(repo, predictor, settings)

    def test_basic_format(self):
        svc = self._make_service()
        result = svc.format_prediction(self._make_pred(), self._make_game())
        assert isinstance(result, dict)
        assert "fg_spread" in result or "markets" in result or "matchup" in result

    def test_home_away_names_used(self):
        svc = self._make_service()
        pred = self._make_pred()
        game = self._make_game()
        result = svc.format_prediction(pred, game)
        result_str = str(result)
        assert "Lakers" in result_str or "Celtics" in result_str

    def test_none_teams_fallback(self):
        svc = self._make_service()
        pred = self._make_pred()
        game = self._make_game(home_team=None, away_team=None)
        result = svc.format_prediction(pred, game)
        result_str = str(result)
        assert "Team 1" in result_str or "Team 2" in result_str

    def test_with_opening_lines(self):
        svc = self._make_service()
        pred = self._make_pred(opening_spread=-3.0, opening_total=220.0)
        game = self._make_game()
        result = svc.format_prediction(pred, game)
        assert isinstance(result, dict)

    def test_with_odds_sourced(self):
        sourced = {
            "books": {
                "dk": {
                    "spread": -5.0,
                    "spread_price": -110,
                    "total": 220.5,
                    "total_price": -110,
                    "home_ml": -200,
                    "away_ml": 170,
                }
            },
            "captured_at": "2025-03-15T20:00:00Z",
        }
        svc = self._make_service()
        pred = self._make_pred(odds_sourced=sourced)
        game = self._make_game()
        result = svc.format_prediction(pred, game)
        assert isinstance(result, dict)
