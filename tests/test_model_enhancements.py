"""Tests for ensemble stacking, SHAP explainability, OOD detection,
and quantile regression — the v6.5.0 prediction pipeline enhancements.
"""

import json

import numpy as np
import pytest

# ── OOD Detector ─────────────────────────────────────────────────


class TestOODDetector:
    def test_fit_and_score_inlier(self):
        from src.models.ood import OODDetector

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 10))

        ood = OODDetector()
        ood.fit(X_train)

        assert ood.is_ready

        # Score a sample near the mean — should be inlier
        sample = np.zeros((1, 10))
        dist, is_ood = ood.score(sample)
        assert dist >= 0
        assert is_ood is False

    def test_fit_and_score_outlier(self):
        from src.models.ood import OODDetector

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 10))

        ood = OODDetector()
        ood.fit(X_train, threshold_percentile=95.0)

        # Score a sample far from the centroid — should be OOD
        outlier = np.full((1, 10), 20.0)
        dist, is_ood = ood.score(outlier)
        assert dist > 0
        assert is_ood is True

    def test_not_ready_before_fit(self):
        from src.models.ood import OODDetector

        ood = OODDetector()
        assert ood.is_ready is False
        dist, is_ood = ood.score(np.zeros((1, 5)))
        assert dist == 0.0
        assert is_ood is False

    def test_insufficient_data_skips_fit(self):
        from src.models.ood import OODDetector

        ood = OODDetector()
        ood.fit(np.zeros((3, 5)))
        assert ood.is_ready is False

    def test_save_and_load(self, tmp_path, monkeypatch):
        from src.models.ood import OODDetector

        monkeypatch.setattr("src.models.ood.ARTIFACTS_DIR", tmp_path)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))

        ood = OODDetector()
        ood.fit(X)
        assert ood.is_ready

        # Load into a new instance
        ood2 = OODDetector()
        loaded = ood2.load()
        assert loaded
        assert ood2.is_ready
        assert ood2._threshold == pytest.approx(ood._threshold, abs=0.01)

    def test_load_missing_file(self, tmp_path, monkeypatch):
        from src.models.ood import OODDetector

        monkeypatch.setattr("src.models.ood.ARTIFACTS_DIR", tmp_path)
        ood = OODDetector()
        assert ood.load() is False


# ── Explainability ───────────────────────────────────────────────


class TestExplainer:
    def _make_xgb_model(self, n_features: int):
        import xgboost as xgb

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, n_features)).astype(np.float32)
        y = rng.standard_normal(50).astype(np.float32)
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        return model, X

    def test_initialize_and_explain(self):
        from src.models.explainability import Explainer

        model, X = self._make_xgb_model(5)
        feature_cols = [f"f{i}" for i in range(5)]

        explainer = Explainer()
        explainer.initialize({"model_home_fg": model}, feature_cols)
        assert explainer.is_ready

        result = explainer.explain_prediction(X[:1], "model_home_fg")
        assert result is not None
        assert "base_value" in result
        assert "shap_values" in result
        assert "top_drivers" in result
        assert len(result["top_drivers"]) <= 10
        assert len(result["shap_values"]) == 5

    def test_explain_unknown_model(self):
        from src.models.explainability import Explainer

        explainer = Explainer()
        result = explainer.explain_prediction(np.zeros((1, 3)), "nonexistent")
        assert result is None

    def test_not_ready_before_init(self):
        from src.models.explainability import Explainer

        explainer = Explainer()
        assert explainer.is_ready is False

    def test_global_importance(self):
        from src.models.explainability import Explainer

        model, X = self._make_xgb_model(5)
        feature_cols = [f"f{i}" for i in range(5)]

        explainer = Explainer()
        explainer.initialize({"model_home_fg": model}, feature_cols)

        imp = explainer.compute_global_importance(X, "model_home_fg")
        assert len(imp) == 5
        assert all(v >= 0 for v in imp.values())

    def test_pruning_candidates(self):
        from src.models.explainability import Explainer

        model, X = self._make_xgb_model(5)
        feature_cols = [f"f{i}" for i in range(5)]

        explainer = Explainer()
        explainer.initialize({"model_home_fg": model}, feature_cols)
        explainer.compute_global_importance(X, "model_home_fg")

        candidates = explainer.get_pruning_candidates(threshold_pct=0.5)
        assert isinstance(candidates, list)

    def test_save_global_importance(self, tmp_path, monkeypatch):
        from src.models.explainability import Explainer

        monkeypatch.setattr("src.models.explainability.ARTIFACTS_DIR", tmp_path)

        model, X = self._make_xgb_model(5)
        feature_cols = [f"f{i}" for i in range(5)]

        explainer = Explainer()
        explainer.initialize({"model_home_fg": model}, feature_cols)
        explainer.compute_global_importance(X, "model_home_fg")
        explainer.save_global_importance()

        path = tmp_path / "shap_importance.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "model_home_fg" in data


# ── Ensemble Stack ───────────────────────────────────────────────


class TestEnsembleStack:
    def _make_models_and_data(self, n_features: int = 10, n_samples: int = 300):
        import xgboost as xgb

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y = rng.standard_normal(n_samples).astype(np.float32) * 10 + 100

        model = xgb.XGBRegressor(
            n_estimators=20, max_depth=3, random_state=42, early_stopping_rounds=5
        )
        split = int(n_samples * 0.85)
        model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])], verbose=False)

        xgb_models = {}
        y_dict = {}
        for name in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]:
            xgb_models[name] = model
            y_dict[name] = y

        masks = {
            "fg": np.ones(n_samples, dtype=bool),
            "1h": np.ones(n_samples, dtype=bool),
        }
        return X, y_dict, xgb_models, masks

    def test_train_produces_metrics(self):
        from src.models.ensemble import EnsembleStack

        X, y_dict, xgb_models, masks = self._make_models_and_data()
        ensemble = EnsembleStack()
        metrics = ensemble.train(X, y_dict, xgb_models, masks)

        assert ensemble.is_ready
        assert any("ensemble_mae" in k for k in metrics)

    def test_predict_blends_xgb_and_lgb(self):
        from src.models.ensemble import EnsembleStack

        X, y_dict, xgb_models, masks = self._make_models_and_data()
        ensemble = EnsembleStack()
        ensemble.train(X, y_dict, xgb_models, masks)

        sample = X[:1]
        xgb_pred = float(xgb_models["model_home_fg"].predict(sample)[0])
        blended = ensemble.predict(sample, "model_home_fg", xgb_pred)

        assert blended is not None
        assert isinstance(blended, float)

    def test_predict_returns_none_for_unloaded(self):
        from src.models.ensemble import EnsembleStack

        ensemble = EnsembleStack()
        result = ensemble.predict(np.zeros((1, 5)), "model_home_fg", 100.0)
        assert result is None

    def test_not_ready_before_training(self):
        from src.models.ensemble import EnsembleStack

        ensemble = EnsembleStack()
        assert ensemble.is_ready is False

    def test_save_and_load(self, tmp_path, monkeypatch):
        from src.models.ensemble import EnsembleStack

        monkeypatch.setattr("src.models.ensemble.ARTIFACTS_DIR", tmp_path)

        X, y_dict, xgb_models, masks = self._make_models_and_data()
        ensemble = EnsembleStack()
        ensemble.train(X, y_dict, xgb_models, masks)

        ensemble2 = EnsembleStack()
        loaded = ensemble2.load()
        assert loaded
        assert ensemble2.is_ready

    def test_insufficient_data_skips(self):
        import xgboost as xgb

        from src.models.ensemble import EnsembleStack

        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 5)).astype(np.float32)
        y = rng.standard_normal(30).astype(np.float32)

        model = xgb.XGBRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        y_dict = {
            name: y for name in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
        }
        xgb_models = {name: model for name in y_dict}
        masks = {"fg": np.ones(30, dtype=bool), "1h": np.ones(30, dtype=bool)}

        ensemble = EnsembleStack()
        ensemble.train(X, y_dict, xgb_models, masks)
        # With only 30 rows, ensemble should skip
        assert ensemble.is_ready is False


# ── Feature column registry ─────────────────────────────────────


class TestFeatureColumnsV65:
    def test_new_interaction_features_present(self):
        from src.models.features import get_feature_columns

        cols = get_feature_columns()
        new_features = [
            "pace_x_3pt_diff",
            "elo_x_rest",
            "injury_diff",
            "venue_scoring_edge",
            "off_def_mismatch",
            "streak_diff",
        ]
        for feat in new_features:
            assert feat in cols, f"Missing interaction feature: {feat}"

    def test_formerly_orphaned_features_present(self):
        from src.models.features import get_feature_columns

        cols = get_feature_columns()
        orphaned = [
            "ref_avg_pts",
            "ref_home_win_pct_bias",
            "ref_over_pct",
            "mkt_1h_spread_avg",
            "mkt_1h_total_avg",
            "mkt_1h_home_ml_prob",
            "prop_blk_avg_line",
            "prop_stl_avg_line",
            "prop_tov_avg_line",
        ]
        for feat in orphaned:
            assert feat in cols, f"Missing orphaned feature: {feat}"

    def test_feature_count(self):
        from src.models.features import get_feature_columns

        cols = get_feature_columns()
        assert len(cols) == 143

    def test_no_duplicates(self):
        from src.models.features import get_feature_columns

        cols = get_feature_columns()
        assert len(cols) == len(set(cols))


# ── Version bump ─────────────────────────────────────────────────


class TestVersioning:
    def test_version_bumped(self):
        from src.models.versioning import MODEL_VERSION

        assert MODEL_VERSION == "v6.6.0"


# ── Predictor runtime status includes new components ─────────────


class TestPredictorRuntimeStatusV65:
    def test_status_includes_new_keys(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a", "b"]
        p._inference_feature_cols = ["a", "b"]
        p.models = {}
        p._model_feature_counts = {}
        p._incompatible_models = {}
        p._last_error = None
        p._compatibility_mode = False
        p._calibration = {}
        p.model_version = "test"

        status = p.get_runtime_status()
        assert "ensemble_ready" in status
        assert "quantile_models" in status
        assert "ood_ready" in status
        assert "shap_ready" in status
        assert status["ensemble_ready"] is False
        assert status["ood_ready"] is False
        assert status["shap_ready"] is False

    def test_predict_scores_works_without_ensemble(self):
        from unittest.mock import MagicMock

        from src.models.predictor import MODEL_NAMES, Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a", "b"]
        p._inference_feature_cols = ["a", "b"]
        p._calibration = {}
        p._last_error = None

        for name in MODEL_NAMES:
            m = MagicMock()
            m.predict.return_value = np.array([100.0])
            p.models = p.models if hasattr(p, "models") and p.models else {}
            p.models[name] = m

        p._incompatible_models = {}

        X = np.array([[1.0, 2.0]])
        home_fg, away_fg, home_1h, away_1h = p._predict_scores(X, "test")
        assert home_fg == 100.0
        assert away_fg == 100.0

    def test_predict_quantiles_empty_without_models(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        X = np.array([[1.0, 2.0]])
        result = p._predict_quantiles(X)
        assert result == {}
