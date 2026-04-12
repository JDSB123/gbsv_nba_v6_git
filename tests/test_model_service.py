"""Tests for src.services.model — ModelService.get_performance."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.services.model import ModelService


def _pred(
    game_id=1,
    model_version="6.0.0",
    predicted_home_fg=112.0,
    predicted_away_fg=108.0,
    predicted_home_1h=56.0,
    predicted_away_1h=54.0,
    opening_spread=-3.0,
    opening_total=220.0,
    closing_spread=-4.0,
    closing_total=221.0,
    clv_spread=-1.0,
    clv_total=1.0,
    fg_spread=-4.0,
    fg_total=220.0,
):
    return SimpleNamespace(
        game_id=game_id,
        model_version=model_version,
        predicted_home_fg=predicted_home_fg,
        predicted_away_fg=predicted_away_fg,
        predicted_home_1h=predicted_home_1h,
        predicted_away_1h=predicted_away_1h,
        opening_spread=opening_spread,
        opening_total=opening_total,
        closing_spread=closing_spread,
        closing_total=closing_total,
        clv_spread=clv_spread,
        clv_total=clv_total,
        fg_spread=fg_spread,
        fg_total=fg_total,
        created_at=None,
    )


def _game(
    home_score_fg=110,
    away_score_fg=105,
    home_score_1h=55,
    away_score_1h=52,
):
    return SimpleNamespace(
        home_score_fg=home_score_fg,
        away_score_fg=away_score_fg,
        home_score_1h=home_score_1h,
        away_score_1h=away_score_1h,
    )


@pytest.mark.asyncio
class TestModelServicePerformance:
    async def test_empty_results(self):
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = []
        svc = ModelService(repo)
        result = await svc.get_performance(limit=200)
        assert result["window"] == 200
        assert result["models"] == []

    async def test_single_model_basic_mae(self):
        pred = _pred(predicted_home_fg=112.0, predicted_away_fg=108.0)
        game = _game(home_score_fg=110, away_score_fg=105)
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = [(pred, game)]
        svc = ModelService(repo)
        result = await svc.get_performance()
        assert len(result["models"]) == 1
        model = result["models"][0]
        assert model["model_version"] == "6.0.0"
        assert model["sample_size"] == 1
        assert model["mae_home_fg"] == 2.0  # |112-110|
        assert model["mae_away_fg"] == 3.0  # |108-105|

    async def test_1h_scores_calculated(self):
        pred = _pred(predicted_home_1h=56.0, predicted_away_1h=54.0)
        game = _game(home_score_1h=55, away_score_1h=52)
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = [(pred, game)]
        svc = ModelService(repo)
        result = await svc.get_performance()
        model = result["models"][0]
        assert model["mae_home_1h"] == 1.0
        assert model["mae_away_1h"] == 2.0

    async def test_clv_averaged(self):
        p1 = _pred(game_id=1, clv_spread=-1.0, clv_total=2.0)
        g1 = _game()
        p2 = _pred(game_id=2, clv_spread=-3.0, clv_total=4.0)
        g2 = _game()
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = [(p1, g1), (p2, g2)]
        svc = ModelService(repo)
        result = await svc.get_performance()
        model = result["models"][0]
        assert model["avg_clv_spread"] == -2.0
        assert model["avg_clv_total"] == 3.0

    async def test_no_1h_scores_returns_none(self):
        pred = _pred()
        game = _game(home_score_1h=None, away_score_1h=None)
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = [(pred, game)]
        svc = ModelService(repo)
        result = await svc.get_performance()
        model = result["models"][0]
        assert model["mae_home_1h"] is None
        assert model["mae_away_1h"] is None

    async def test_multiple_model_versions_separated(self):
        p1 = _pred(game_id=1, model_version="6.0.0")
        p2 = _pred(game_id=2, model_version="5.0.0")
        g = _game()
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = [(p1, g), (p2, g)]
        svc = ModelService(repo)
        result = await svc.get_performance()
        versions = {m["model_version"] for m in result["models"]}
        assert "6.0.0" in versions
        assert "5.0.0" in versions

    async def test_sorted_by_sample_size(self):
        pairs = [(_pred(game_id=i, model_version="6.0.0"), _game()) for i in range(5)]
        pairs += [(_pred(game_id=10, model_version="5.0.0"), _game())]
        repo = AsyncMock()
        repo.get_finished_game_predictions.return_value = pairs
        svc = ModelService(repo)
        result = await svc.get_performance()
        assert result["models"][0]["sample_size"] >= result["models"][-1]["sample_size"]
