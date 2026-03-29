"""Tests for Trainer: outlier detection, feature importance drift, dataset build."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

_MOD = "src.models.trainer"


class TestBuildDataset:
    @pytest.mark.anyio
    async def test_build_dataset_returns_dataframe(self):
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1", "f2"]
        trainer.run_optuna = False
        trainer.models = {}

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
            home_score_fg=110,
            away_score_fg=105,
            home_score_1h=55,
            away_score_1h=52,
            commence_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="FT",
        )

        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [game]
        db.execute.return_value = mock_result

        with (
            patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat,
            patch(f"{_MOD}.reset_elo_cache"),
        ):
            mock_feat.return_value = {"f1": 1.0, "f2": 2.0}
            df = await trainer._build_dataset(db)

        assert not df.empty
        assert "f1" in df.columns
        assert "home_score_fg" in df.columns

    @pytest.mark.anyio
    async def test_build_dataset_none_1h_scores_become_nan(self):
        """Games with None 1H scores get NaN (not 0.0)."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]
        trainer.run_optuna = False
        trainer.models = {}

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="A"),
            away_team=SimpleNamespace(name="B"),
            home_score_fg=100,
            away_score_fg=95,
            home_score_1h=None,
            away_score_1h=None,
            commence_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="FT",
        )

        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [game]
        db.execute.return_value = mock_result

        with (
            patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat,
            patch(f"{_MOD}.reset_elo_cache"),
        ):
            mock_feat.return_value = {"f1": 1.0}
            df = await trainer._build_dataset(db)

        assert np.isnan(df.iloc[0]["home_score_1h"])
        assert np.isnan(df.iloc[0]["away_score_1h"])


class TestTrainOutlierAndDrift:
    @pytest.mark.anyio
    async def test_train_too_few_rows_returns_empty(self):
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]
        trainer.run_optuna = False
        trainer.models = {}

        db = AsyncMock()
        with patch.object(trainer, "_build_dataset", new_callable=AsyncMock) as mock_ds:
            import pandas as pd

            mock_ds.return_value = pd.DataFrame()
            result = await trainer.train(db)
        assert result == {}

    @pytest.mark.anyio
    async def test_train_detects_outliers(self, tmp_path):
        """Outlier detection runs on target variables without crashing."""
        from src.models.trainer import ARTIFACTS_DIR, ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]
        trainer.run_optuna = False
        trainer.models = {}

        import pandas as pd

        np.random.seed(42)
        n = 100
        data = {
            "f1": np.random.randn(n),
            "home_score_fg": np.random.normal(105, 10, n),
            "away_score_fg": np.random.normal(102, 10, n),
            "home_score_1h": np.random.normal(52, 5, n),
            "away_score_1h": np.random.normal(50, 5, n),
            "commence_time": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
        # Add extreme outlier
        data["home_score_fg"][0] = 250.0
        df = pd.DataFrame(data)

        db = AsyncMock()

        with (
            patch.object(trainer, "_build_dataset", new_callable=AsyncMock, return_value=df),
            patch(f"{_MOD}.ARTIFACTS_DIR", tmp_path),
            patch.object(trainer, "_sync_model_registry", new_callable=AsyncMock),
        ):
            result = await trainer.train(db)

        assert len(result) > 0
        assert "model_home_fg_mae" in result

    @pytest.mark.anyio
    async def test_train_feature_importance_drift(self, tmp_path):
        """Feature importance drift detection when previous importance file exists."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1", "f2"]
        trainer.run_optuna = False
        trainer.models = {}

        import pandas as pd

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "home_score_fg": np.random.normal(105, 10, n),
            "away_score_fg": np.random.normal(102, 10, n),
            "home_score_1h": np.random.normal(52, 5, n),
            "away_score_1h": np.random.normal(50, 5, n),
            "commence_time": pd.date_range("2024-01-01", periods=n, freq="D"),
        })

        # Create a previous importance file with very different rankings
        prev_importance = {"f2": 0.9, "f1": 0.1}
        imp_path = tmp_path / "feature_importance.json"
        imp_path.write_text(json.dumps(prev_importance))

        db = AsyncMock()

        with (
            patch.object(trainer, "_build_dataset", new_callable=AsyncMock, return_value=df),
            patch(f"{_MOD}.ARTIFACTS_DIR", tmp_path),
            patch.object(trainer, "_sync_model_registry", new_callable=AsyncMock),
        ):
            result = await trainer.train(db)

        assert len(result) > 0
        # New importance file should be written
        assert imp_path.exists()

    @pytest.mark.anyio
    async def test_train_runs_optuna_branch(self, tmp_path):
        from src.models.trainer import ModelTrainer

        import pandas as pd

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1", "f2"]
        trainer.run_optuna = True
        trainer.models = {}

        n = 220
        df = pd.DataFrame(
            {
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
                "home_score_fg": np.random.normal(105, 10, n),
                "away_score_fg": np.random.normal(102, 10, n),
                "home_score_1h": np.random.normal(52, 5, n),
                "away_score_1h": np.random.normal(50, 5, n),
                "commence_time": pd.date_range("2024-01-01", periods=n, freq="D"),
            }
        )

        class _FakeModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.feature_importances_ = np.array([0.6, 0.4])

            def fit(self, X, y, eval_set=None, verbose=False):
                return self

            def predict(self, X):
                return np.full(len(X), 100.0)

            def save_model(self, path):
                Path(path).write_text("{}")

        study = MagicMock()
        study.best_params = {"max_depth": 2, "n_estimators": 5}
        study.best_value = 4.2

        db = AsyncMock()

        with (
            patch.object(trainer, "_build_dataset", new_callable=AsyncMock, return_value=df),
            patch(f"{_MOD}.ARTIFACTS_DIR", tmp_path),
            patch.object(trainer, "_sync_model_registry", new_callable=AsyncMock),
            patch(f"{_MOD}.optuna.create_study", return_value=study),
            patch(f"{_MOD}.xgb.XGBRegressor", side_effect=lambda **kwargs: _FakeModel(**kwargs)),
            patch(f"{_MOD}._calibrate_probabilities", return_value=(0.1, -0.2)),
            patch(f"{_MOD}._evaluate_promotion", return_value=(False, "ok")),
        ):
            result = await trainer.train(db)

        assert study.optimize.call_count == 4
        assert "model_home_fg_mae" in result
        best_params = json.loads((tmp_path / "best_params.json").read_text())
        assert best_params["model_home_fg"]["objective"] == "reg:squarederror"

    @pytest.mark.anyio
    async def test_train_logs_feature_importance_drift(self, tmp_path):
        from src.models.trainer import ModelTrainer

        import pandas as pd

        feature_cols = [f"f{i}" for i in range(25)]
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = feature_cols
        trainer.run_optuna = False
        trainer.models = {}

        n = 120
        df = pd.DataFrame(
            {
                **{name: np.random.randn(n) for name in feature_cols},
                "home_score_fg": np.random.normal(105, 10, n),
                "away_score_fg": np.random.normal(102, 10, n),
                "home_score_1h": np.random.normal(52, 5, n),
                "away_score_1h": np.random.normal(50, 5, n),
                "commence_time": pd.date_range("2024-01-01", periods=n, freq="D"),
            }
        )

        prev_importance = {name: float(100 - idx) for idx, name in enumerate(feature_cols)}
        (tmp_path / "feature_importance.json").write_text(json.dumps(prev_importance))

        reversed_importance = np.array(list(range(1, 26)), dtype=float)
        reversed_importance = reversed_importance / reversed_importance.sum()

        class _FakeModel:
            def __init__(self, **kwargs):
                self.feature_importances_ = reversed_importance

            def fit(self, X, y, eval_set=None, verbose=False):
                return self

            def predict(self, X):
                return np.full(len(X), 100.0)

            def save_model(self, path):
                Path(path).write_text("{}")

        db = AsyncMock()

        with (
            patch.object(trainer, "_build_dataset", new_callable=AsyncMock, return_value=df),
            patch(f"{_MOD}.ARTIFACTS_DIR", tmp_path),
            patch.object(trainer, "_sync_model_registry", new_callable=AsyncMock),
            patch(f"{_MOD}.xgb.XGBRegressor", side_effect=lambda **kwargs: _FakeModel(**kwargs)),
            patch(f"{_MOD}._calibrate_probabilities", return_value=(0.1, -0.2)),
            patch(f"{_MOD}._evaluate_promotion", return_value=(False, "ok")),
            patch(f"{_MOD}.logger.warning") as mock_warning,
        ):
            await trainer.train(db)

        assert mock_warning.called
        assert "Feature importance drift detected (>10 rank shift): %s" in mock_warning.call_args_list[-1].args[0]


class TestSyncModelRegistry:
    @pytest.mark.anyio
    async def test_sync_creates_new_entry(self):
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]
        trainer.run_optuna = False
        trainer.models = {}

        db = AsyncMock()
        db.add = MagicMock()

        # No existing registry entry
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute.return_value = mock_result

        await trainer._sync_model_registry(
            db,
            metrics={"model_home_fg_mae": 5.0},
            best_params_all={"model_home_fg": {"max_depth": 6}},
            should_promote=False,
            promotion_reason="metrics ok",
        )
        db.add.assert_called_once()
        db.commit.assert_awaited()

    @pytest.mark.anyio
    async def test_sync_promotes_existing(self):
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.feature_cols = ["f1"]
        trainer.run_optuna = False
        trainer.models = {}

        db = AsyncMock()
        existing = MagicMock()
        existing.is_active = False

        # Return existing for version lookup, then return list for deactivation
        call_count = 0
        async def fake_execute(stmt, *a, **kw):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 1:
                r.scalar_one_or_none.return_value = existing
            else:
                # Existing active entries to deactivate
                r.scalars.return_value.all.return_value = []
            return r

        db.execute = fake_execute

        await trainer._sync_model_registry(
            db,
            metrics={"model_home_fg_mae": 3.0},
            best_params_all={},
            should_promote=True,
            promotion_reason="quality threshold met",
        )
        assert existing.is_active is True
        db.commit.assert_awaited()
