"""Tests for ModelTrainer — _build_dataset, _sync_model_registry, train, _optuna_objective."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.models.trainer import (
    DEFAULT_XGB_PARAMS,
    MODEL_NAMES,
    TARGETS,
    ModelTrainer,
    _calibrate_probabilities,
    _evaluate_promotion,
    _optuna_objective,
)


# ── _optuna_objective ────────────────────────────────────────────


class TestOptunaObjective:
    def test_returns_finite_mae(self):
        """Objective returns a finite float (MAE) from a trial."""
        import optuna

        study = optuna.create_study(direction="minimize")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = X[:, 0] * 2 + rng.standard_normal(100) * 0.5

        study.optimize(
            lambda t: _optuna_objective(t, X, y, n_splits=2),
            n_trials=2,
        )
        assert study.best_value > 0
        assert np.isfinite(study.best_value)


# ── ModelTrainer._sync_model_registry ────────────────────────────


class TestSyncModelRegistry:
    async def test_creates_new_registry_entry(self):
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1", "f2"]

        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None  # no existing entry
        db.execute = AsyncMock(return_value=result)
        db.add = MagicMock()
        db.commit = AsyncMock()

        await trainer._sync_model_registry(
            db,
            metrics={"mae": 8.0},
            best_params_all={"model_home_fg": {}},
            should_promote=True,
            promotion_reason="all gate checks passed",
        )

        db.add.assert_called_once()
        db.commit.assert_awaited_once()

    async def test_updates_existing_entry(self):
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]

        existing = SimpleNamespace(
            model_version="v6",
            is_active=False,
            promoted_at=None,
            retired_at=None,
            metrics_json=None,
            params_json=None,
            promotion_reason=None,
        )

        active_entry = SimpleNamespace(
            is_active=True,
            model_version="v5",
            retired_at=None,
        )

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            if call_n["n"] == 1:
                result.scalar_one_or_none.return_value = existing
            else:
                result.scalars.return_value.all.return_value = [active_entry]
            return result

        db = AsyncMock()
        db.execute = mock_exec
        db.add = MagicMock()
        db.commit = AsyncMock()

        await trainer._sync_model_registry(
            db,
            metrics={"mae": 7.0},
            best_params_all={},
            should_promote=True,
            promotion_reason="passed",
        )

        db.commit.assert_awaited()
        assert existing.is_active is True


# ── ModelTrainer._build_dataset ──────────────────────────────────


class TestBuildDataset:
    @patch("src.models.trainer.reset_elo_cache")
    @patch("src.models.trainer.build_feature_vector")
    async def test_builds_dataframe(self, mock_bfv, mock_reset):
        game = SimpleNamespace(
            id=1,
            home_team_id=1,
            away_team_id=2,
            commence_time=datetime(2025, 1, 10, 19, 0),
            status="FT",
            home_score_fg=110,
            away_score_fg=105,
            home_score_1h=55,
            away_score_1h=50,
            home_team=SimpleNamespace(name="A"),
            away_team=SimpleNamespace(name="B"),
        )

        mock_bfv.return_value = {"f1": 1.0, "f2": 2.0}

        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [game]
        db.execute = AsyncMock(return_value=result)

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1", "f2"]

        df = await trainer._build_dataset(db)
        assert not df.empty
        assert "home_score_fg" in df.columns
        assert "f1" in df.columns
        assert len(df) == 1

    @patch("src.models.trainer.reset_elo_cache")
    @patch("src.models.trainer.build_feature_vector", return_value=None)
    async def test_skips_games_with_no_features(self, mock_bfv, mock_reset):
        game = SimpleNamespace(
            id=1, home_team_id=1, away_team_id=2,
            commence_time=datetime(2025, 1, 10), status="FT",
            home_score_fg=110, away_score_fg=105,
            home_score_1h=55, away_score_1h=50,
            home_team=SimpleNamespace(name="A"),
            away_team=SimpleNamespace(name="B"),
        )

        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [game]
        db.execute = AsyncMock(return_value=result)

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]

        df = await trainer._build_dataset(db)
        assert df.empty


# ── ModelTrainer.train ───────────────────────────────────────────


class TestTrain:
    @patch("src.models.trainer.reset_elo_cache")
    @patch("src.models.trainer.build_feature_vector")
    async def test_train_insufficient_data(self, mock_bfv, mock_reset):
        """train() returns {} when data is too small."""
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        trainer = ModelTrainer(run_optuna=False)
        metrics = await trainer.train(db)
        assert metrics == {}

    @patch("src.models.trainer.ARTIFACTS_DIR")
    @patch("src.models.trainer.reset_elo_cache")
    @patch("src.models.trainer.build_feature_vector")
    async def test_train_full_pipeline(self, mock_bfv, mock_reset, mock_artifacts, tmp_path):
        """Full training pipeline with synthetic data (no Optuna)."""
        mock_artifacts.__truediv__ = lambda self, name: tmp_path / name
        mock_artifacts.mkdir = MagicMock()

        # Use a deterministic feature dict
        feature_cols = [f"f{i}" for i in range(5)]
        rng = np.random.default_rng(42)

        games = []
        for i in range(100):
            g = SimpleNamespace(
                id=i,
                home_team_id=1, away_team_id=2,
                commence_time=datetime(2024, 1, 1) + timedelta(days=i),
                status="FT",
                home_score_fg=100 + rng.integers(-15, 15),
                away_score_fg=100 + rng.integers(-15, 15),
                home_score_1h=50 + rng.integers(-10, 10),
                away_score_1h=50 + rng.integers(-10, 10),
                home_team=SimpleNamespace(name="A"),
                away_team=SimpleNamespace(name="B"),
            )
            games.append(g)

        mock_bfv.side_effect = lambda g, db, **kw: {f"f{j}": float(rng.standard_normal()) for j in range(5)}

        db = AsyncMock()
        game_result = MagicMock()
        game_result.scalars.return_value.all.return_value = games
        reg_result = MagicMock()
        reg_result.scalar_one_or_none.return_value = None
        reg_result.scalars.return_value.all.return_value = []

        call_n = {"n": 0}
        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return game_result
            return reg_result

        db.execute = mock_exec
        db.add = MagicMock()
        db.commit = AsyncMock()

        trainer = ModelTrainer(run_optuna=False)
        trainer.feature_cols = feature_cols

        metrics = await trainer.train(db)
        assert metrics  # non-empty
        assert "model_home_fg_mae" in metrics
        assert "model_away_fg_mae" in metrics
        assert metrics["model_home_fg_mae"] > 0
        # Verify model files were "saved" (xgb saves to the tmp_path)
        assert len(trainer.models) == 4


# ── Constants ────────────────────────────────────────────────────


class TestTrainerConstants:
    def test_targets_and_model_names_aligned(self):
        assert len(TARGETS) == len(MODEL_NAMES) == 4

    def test_default_xgb_params(self):
        assert DEFAULT_XGB_PARAMS["objective"] == "reg:squarederror"
        assert DEFAULT_XGB_PARAMS["max_depth"] == 6
