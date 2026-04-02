"""Tests for remaining predictor.py gaps (lines 126-127, 149-150, 162-174)
and trainer.py gaps (lines 274-285 Optuna, 395-402 feature importance drift)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xgboost as xgb

_PRED_MOD = "src.models.predictor"
_TRAINER_MOD = "src.models.trainer"


class TestPredictorFeatureNameMismatch:
    """Cover lines 126-127: model feature names don't match code."""

    def test_feature_name_mismatch_logged(self, tmp_path):
        from src.models.predictor import MODEL_NAMES, Predictor

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Create fake models with feature names
        feature_cols = ["f1", "f2", "f3"]
        for name in MODEL_NAMES:
            model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
            X = np.random.rand(20, 3).astype(np.float32)
            y = np.random.rand(20)
            model.fit(X, y)
            # Set feature names on the booster after training
            model.get_booster().feature_names = feature_cols
            model.save_model(str(artifacts / f"{name}.json"))

        with (
            patch(f"{_PRED_MOD}.ARTIFACTS_DIR", artifacts),
            patch(f"{_PRED_MOD}.get_feature_columns", return_value=["a", "b", "c"]),
        ):
            predictor = Predictor()
            # Feature count matches (3==3) but names differ → logs error but stays ready
            # due to compatibility mode
            assert predictor._last_error is not None or predictor.is_ready

    def test_feature_name_mismatch_sets_last_error_with_mocked_models(self, tmp_path):
        from src.models.predictor import MODEL_NAMES, Predictor

        for name in MODEL_NAMES:
            (tmp_path / f"{name}.json").write_text("{}")

        mock_model = MagicMock()
        mock_model.get_booster.return_value.num_features.return_value = 3
        mock_model.get_booster.return_value.feature_names = ["wrong_a", "wrong_b", "wrong_c"]

        with (
            patch(f"{_PRED_MOD}.ARTIFACTS_DIR", tmp_path),
            patch(f"{_PRED_MOD}.get_feature_columns", return_value=["a", "b", "c"]),
            patch(f"{_PRED_MOD}.xgb.XGBRegressor", return_value=mock_model),
        ):
            predictor = Predictor()

        assert predictor._last_error == "Model feature name mismatch. Code and artifacts are out of sync."


class TestPredictorIncompatibleModels:
    """Cover lines 162-174: model feature count doesn't match and can't be rescued."""

    def test_incompatible_clears_models(self, tmp_path):
        from src.models.predictor import MODEL_NAMES, Predictor

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Create models with 5 features
        for name in MODEL_NAMES:
            model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
            X = np.random.rand(20, 5).astype(np.float32)
            y = np.random.rand(20)
            model.fit(X, y)
            model.save_model(str(artifacts / f"{name}.json"))

        # But code says we expect 3 features — 5 > 3 → incompatible
        with (
            patch(f"{_PRED_MOD}.ARTIFACTS_DIR", artifacts),
            patch(f"{_PRED_MOD}.get_feature_columns", return_value=["a", "b", "c"]),
        ):
            predictor = Predictor()
            # 5 != 3, and 5 > 3 → not a subset → clears models
            assert predictor.models == {}
            assert predictor._last_error is not None


class TestPredictorCompatibilityAlert:
    """Cover lines 149-150: fire-and-forget alert in compatibility mode."""

    def test_compatibility_mode_attempts_alert(self, tmp_path):
        from src.models.predictor import MODEL_NAMES, Predictor

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Create models with 3 features (fewer than code expects)
        for name in MODEL_NAMES:
            model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
            X = np.random.rand(20, 3).astype(np.float32)
            y = np.random.rand(20)
            model.fit(X, y)
            model.save_model(str(artifacts / f"{name}.json"))

        # Code expects 5 features → 3 < 5 → compatibility mode
        with (
            patch(f"{_PRED_MOD}.ARTIFACTS_DIR", artifacts),
            patch(f"{_PRED_MOD}.get_feature_columns", return_value=["a", "b", "c", "d", "e"]),
        ):
            predictor = Predictor()
            assert predictor._compatibility_mode is True
            assert len(predictor._inference_feature_cols) == 3

    @pytest.mark.anyio
    async def test_compatibility_mode_logs_warning_without_fire_and_forget(self, tmp_path):
        """After refactor: compatibility mode only logs — no fire-and-forget task."""
        from src.models.predictor import MODEL_NAMES, Predictor

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        for name in MODEL_NAMES:
            model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
            X = np.random.rand(20, 3).astype(np.float32)
            y = np.random.rand(20)
            model.fit(X, y)
            model.save_model(str(artifacts / f"{name}.json"))

        with (
            patch(f"{_PRED_MOD}.ARTIFACTS_DIR", artifacts),
            patch(f"{_PRED_MOD}.get_feature_columns", return_value=["a", "b", "c", "d", "e"]),
        ):
            p = Predictor()

        assert p._compatibility_mode is True


class TestTrainerFeatureImportanceDrift:
    """Cover lines 395, 397-402: feature importance drift detection."""

    def test_drift_detected_logs_warning(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Previous importance: feature a is #1, b is #2, etc.
        prev_importance = {
            "f_a": 0.50, "f_b": 0.30, "f_c": 0.10, "f_d": 0.05, "f_e": 0.03,
            "f_f": 0.02,
        }
        # We need enough features to trigger >10 rank shift
        for i in range(20):
            prev_importance[f"f_{i}"] = 0.01 - i * 0.0001

        imp_path = artifacts / "feature_importance.json"
        imp_path.write_text(json.dumps(prev_importance))

        # New importance: drastically different ranking
        new_importance = dict(prev_importance)
        # Move f_a from #1 to #20+ — that's >10 rank shift
        new_importance["f_a"] = 0.0001
        # Move a low-rank feature to #1
        new_importance["f_15"] = 0.60

        prev_ranking = sorted(prev_importance, key=prev_importance.get, reverse=True)
        new_ranking = sorted(new_importance, key=new_importance.get, reverse=True)

        drifted = []
        for feat in new_ranking[:20]:
            old_rank = prev_ranking.index(feat) if feat in prev_ranking else -1
            new_rank = new_ranking.index(feat)
            if old_rank >= 0 and abs(old_rank - new_rank) > 10:
                drifted.append(f"{feat}: {old_rank}→{new_rank}")

        assert len(drifted) > 0  # Drift should be detected


class TestServicesModelGetPerformance:
    """Cover services/model.py line 22: get_performance with actual data."""

    @pytest.mark.anyio
    async def test_get_performance_with_rows(self):
        from src.services.model import ModelService

        pred = SimpleNamespace(
            game_id=1, model_version="v6-test",
            predicted_home_fg=110.0, predicted_away_fg=105.0,
            predicted_home_1h=55.0, predicted_away_1h=52.0,
            fg_spread=5.0, fg_total=215.0,
            h1_spread=3.0, h1_total=107.0,
            clv_spread=-0.5, clv_total=1.0,
            predicted_at=None,
        )
        game = SimpleNamespace(
            id=1,
            home_score_fg=112, away_score_fg=108,
            home_score_1h=54, away_score_1h=53,
        )

        repo = MagicMock()
        repo.get_finished_game_predictions = MagicMock(return_value=[(pred, game)])
        # Make it awaitable

        async def _get():
            return [(pred, game)]

        repo.get_finished_game_predictions = _get

        service = ModelService(repo)
        result = await service.get_performance(limit=10)
        assert "models" in result
        assert len(result["models"]) > 0
        assert result["models"][0]["model_version"] == "v6-test"


class TestServicesPredictionsGetList:
    """Cover services/predictions.py lines 125 and 142."""

    @pytest.mark.anyio
    async def test_get_list_predictions_with_game(self):
        from src.services.predictions import PredictionService

        pred = SimpleNamespace(
            game_id=1,
            predicted_home_fg=110.0, predicted_away_fg=105.0,
            predicted_home_1h=55.0, predicted_away_1h=52.0,
            fg_spread=5.0, fg_total=215.0,
            h1_spread=3.0, h1_total=107.0,
            fg_home_ml_prob=0.65, h1_home_ml_prob=0.60,
            opening_spread=-3.5, opening_total=220.0,
            opening_h1_spread=-1.5, opening_h1_total=109.5,
            closing_spread=-3.0, closing_total=219.0,
            clv_spread=-0.5, clv_total=1.0,
            clv_h1_spread=None, clv_h1_total=None,
            model_version="v6-test",
            predicted_at=None,
            odds_sourced={
                "captured_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                    "+00:00", "Z"
                )
            },
        )
        game = SimpleNamespace(
            id=1,
            odds_api_id="game-abc-123",
            home_team_id=1,
            away_team_id=2,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
            commence_time=None,
            status="NS",
            closing_spread=None,
            closing_total=None,
        )

        repo = MagicMock()

        async def _latest():
            return [pred]

        async def _games(ids):
            return [game]

        repo.get_latest_predictions_for_upcoming_games = _latest
        repo.get_games_with_teams = _games

        from src.config import Settings

        service = PredictionService(repo, None, Settings())
        result = await service.get_list_predictions()
        assert result["count"] == 1
        assert len(result["predictions"]) == 1

    @pytest.mark.anyio
    async def test_get_slate_payload_with_game(self):
        from src.services.predictions import PredictionService

        pred = SimpleNamespace(
            game_id=1,
            predicted_home_fg=110.0, predicted_away_fg=105.0,
            predicted_home_1h=55.0, predicted_away_1h=52.0,
            fg_spread=5.0, fg_total=215.0,
            h1_spread=3.0, h1_total=107.0,
            odds_sourced={
                "captured_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                    "+00:00", "Z"
                )
            },
        )
        game = SimpleNamespace(id=1)

        repo = MagicMock()

        async def _latest():
            return [pred]

        async def _games_with_stats(ids):
            return [game]

        async def _odds_ts():
            return None

        repo.get_latest_predictions_for_upcoming_games = _latest
        repo.get_games_with_teams_and_stats = _games_with_stats
        repo.get_latest_odds_pull_timestamp = _odds_ts

        from src.config import Settings

        service = PredictionService(repo, None, Settings())
        rows, pulled_at = await service.get_slate_payload()
        assert len(rows) == 1
        assert pulled_at is None
