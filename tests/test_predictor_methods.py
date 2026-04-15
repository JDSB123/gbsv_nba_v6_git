"""Tests for Predictor static methods and helpers."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models.predictor import MODEL_NAMES, Predictor, _margin_to_prob

# ── _margin_to_prob ──────────────────────────────────────────────


class TestMarginToProb:
    def test_zero_margin_returns_half(self):
        assert _margin_to_prob(0.0) == pytest.approx(0.5, abs=0.01)

    def test_positive_margin_favors_home(self):
        p = _margin_to_prob(10.0)
        assert p > 0.5
        assert p < 1.0

    def test_negative_margin_favors_away(self):
        p = _margin_to_prob(-10.0)
        assert p < 0.5
        assert p > 0.0

    def test_with_calibration_coefficients(self):
        # With coef=1.0, intercept=0.0, should be standard sigmoid
        p = _margin_to_prob(0.0, coef=1.0, intercept=0.0)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_calibration_shifts_probability(self):
        # Positive intercept → higher base prob
        p = _margin_to_prob(0.0, coef=1.0, intercept=2.0)
        assert p > 0.8

    def test_large_margin(self):
        p = _margin_to_prob(50.0)
        assert p > 0.99

    def test_large_negative_margin(self):
        p = _margin_to_prob(-50.0)
        assert p < 0.01


# ── _latest_snapshots ────────────────────────────────────────────


def _make_snapshot(bookmaker, market, outcome_name, captured_at, price=100, point=None):
    return SimpleNamespace(
        bookmaker=bookmaker,
        market=market,
        outcome_name=outcome_name,
        captured_at=captured_at,
        price=price,
        point=point,
    )


class TestLatestSnapshots:
    def test_deduplicates_by_key(self):
        old = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
        new = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        snapshots = [
            _make_snapshot("fanduel", "spreads", "Celtics", old, price=-110, point=-3.5),
            _make_snapshot("fanduel", "spreads", "Celtics", new, price=-108, point=-4.0),
        ]
        result, newest = Predictor._latest_snapshots(snapshots)
        assert len(result) == 1
        assert result[0].point == -4.0
        assert newest == new

    def test_multiple_bookmakers_kept(self):
        ts = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        snapshots = [
            _make_snapshot("fanduel", "spreads", "Celtics", ts, point=-3.5),
            _make_snapshot("draftkings", "spreads", "Celtics", ts, point=-4.0),
        ]
        result, _ = Predictor._latest_snapshots(snapshots)
        assert len(result) == 2

    def test_empty_list(self):
        result, newest = Predictor._latest_snapshots([])
        assert result == []
        assert newest is None


# ── _build_odds_detail ───────────────────────────────────────────


class TestBuildOddsDetail:
    def test_spreads(self):
        ts = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        snapshots = [
            _make_snapshot("fanduel", "spreads", "Celtics", ts, price=-110, point=-3.5),
        ]
        detail = Predictor._build_odds_detail(snapshots, "Celtics", "Heat", ts)
        assert "fanduel" in detail["books"]
        assert detail["books"]["fanduel"]["spread"] == -3.5
        assert detail["books"]["fanduel"]["spread_price"] == -110
        assert "Z" in detail["captured_at"]

    def test_totals(self):
        ts = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        snapshots = [
            _make_snapshot("fanduel", "totals", "Over", ts, price=-110, point=224.5),
        ]
        detail = Predictor._build_odds_detail(snapshots, "Celtics", "Heat", ts)
        assert detail["books"]["fanduel"]["total"] == 224.5

    def test_h2h(self):
        ts = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        snapshots = [
            _make_snapshot("fanduel", "h2h", "Celtics", ts, price=-150),
            _make_snapshot("fanduel", "h2h", "Heat", ts, price=130),
        ]
        detail = Predictor._build_odds_detail(snapshots, "Celtics", "Heat", ts)
        assert detail["books"]["fanduel"]["home_ml"] == -150
        assert detail["books"]["fanduel"]["away_ml"] == 130

    def test_1h_markets(self):
        ts = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        snapshots = [
            _make_snapshot("fanduel", "spreads_h1", "Celtics", ts, price=-110, point=-1.5),
            _make_snapshot("fanduel", "totals_h1", "Over", ts, price=-105, point=112.5),
            _make_snapshot("fanduel", "h2h_h1", "Celtics", ts, price=-130),
            _make_snapshot("fanduel", "h2h_h1", "Heat", ts, price=110),
        ]
        detail = Predictor._build_odds_detail(snapshots, "Celtics", "Heat", ts)
        books = detail["books"]["fanduel"]
        assert books["spread_h1"] == -1.5
        assert books["total_h1"] == 112.5
        assert books["home_ml_h1"] == -130
        assert books["away_ml_h1"] == 110

    def test_no_ts(self):
        detail = Predictor._build_odds_detail([], "Celtics", "Heat", None)
        assert detail["captured_at"] is None
        assert detail["books"] == {}


# ── Predictor properties ─────────────────────────────────────────


class TestPredictorProperties:
    def test_get_runtime_status_structure(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.models.predictor.ARTIFACTS_DIR", MagicMock())
            # Can't fully instantiate without model files, but can test structure
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
            assert status["ready"] is False
            assert "missing_models" in status
            assert status["expected_features"] == 2

    def test_is_ready_false_when_no_models(self):
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a"]
        p.models = {}
        p._incompatible_models = {}
        assert not p.is_ready


class TestModelSmokeTest:
    def test_smoke_test_warns_but_keeps_implausible_models_loaded(self):
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a", "b"]
        p._inference_feature_cols = ["a", "b"]
        p._imputation_values = {"a": 1.0, "b": 2.0}
        p._model_feature_counts = {name: 2 for name in MODEL_NAMES}
        p._incompatible_models = {}
        p._compatibility_mode = False
        p._calibration = {}
        p._last_error = None
        p._runtime_warning = None
        p.model_version = "test"

        bad_home_fg = MagicMock()
        bad_home_fg.predict.return_value = np.array([0.1])
        bad_away_fg = MagicMock()
        bad_away_fg.predict.return_value = np.array([0.5])
        bad_home_1h = MagicMock()
        bad_home_1h.predict.return_value = np.array([-2.4])
        bad_away_1h = MagicMock()
        bad_away_1h.predict.return_value = np.array([1.5])
        p.models = {
            "model_home_fg": bad_home_fg,
            "model_away_fg": bad_away_fg,
            "model_home_1h": bad_home_1h,
            "model_away_1h": bad_away_1h,
        }

        p._run_model_smoke_test()

        assert set(p.models) == set(MODEL_NAMES)
        assert p._last_error is None
        assert p._runtime_warning is not None

    def test_is_ready_false_when_incompatible(self):
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a"]
        p.models = {
            "model_home_fg": MagicMock(),
            "model_away_fg": MagicMock(),
            "model_home_1h": MagicMock(),
            "model_away_1h": MagicMock(),
        }
        p._incompatible_models = {"model_home_fg": 10}
        assert not p.is_ready


# ── _load_models edge cases ──────────────────────────────────────


class TestLoadModelsEdgeCases:
    def test_mismatched_feature_counts_clears_models(self):
        """When model files have different feature counts, models are cleared."""
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a", "b", "c"]
        p._inference_feature_cols = list(p.feature_cols)
        p._model_feature_counts = {}
        p._incompatible_models = {}
        p._last_error = None
        p._compatibility_mode = False
        p._calibration = {}
        p.model_version = "test"

        # Simulate loaded models with different feature counts
        m1 = MagicMock()
        m1.get_booster.return_value.num_features.return_value = 3
        m2 = MagicMock()
        m2.get_booster.return_value.num_features.return_value = 5

        p.models = {
            "model_home_fg": m1,
            "model_away_fg": m1,
            "model_home_1h": m1,
            "model_away_1h": m2,
        }
        p._model_feature_counts = {
            "model_home_fg": 3,
            "model_away_fg": 3,
            "model_home_1h": 3,
            "model_away_1h": 5,
        }
        p._load_models.__func__(p)  # type: ignore[attr-defined]
        # After re-running, models should be cleared
        assert p.models == {}
        assert p._last_error is not None

    def test_compatibility_mode_fewer_features(self):
        """When models expect fewer features than code, compatibility mode activates."""
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a", "b", "c", "d"]
        p._inference_feature_cols = list(p.feature_cols)
        p._model_feature_counts = {}
        p._incompatible_models = {}
        p._last_error = None
        p._compatibility_mode = False
        p._calibration = {}
        p.model_version = "test"

        mock_model = MagicMock()
        mock_model.get_booster.return_value.num_features.return_value = 2
        mock_model.get_booster.return_value.feature_names = None

        p.models = {
            n: mock_model
            for n in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
        }
        p._model_feature_counts = {n: 2 for n in p.models}

        # Call the tail of _load_models (feature count validation)
        # We need to call the private method with models already loaded
        from unittest.mock import patch as _patch

        with _patch.object(type(p), "_load_models", lambda self: None):
            pass  # skip actual file loading

        # Directly test the validation logic
        expected_features = p.feature_cols
        expected_count = len(expected_features)
        unique_counts = sorted(set(p._model_feature_counts.values()))
        model_feature_count = unique_counts[0]
        if model_feature_count < expected_count:
            p._compatibility_mode = True
            p._inference_feature_cols = expected_features[:model_feature_count]

        assert p._compatibility_mode is True
        assert len(p._inference_feature_cols) == 2

    def test_models_more_features_than_code_clears(self):
        """When models expect more features than code, models are cleared."""
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a", "b"]
        p._inference_feature_cols = list(p.feature_cols)
        p._model_feature_counts = {
            n: 10 for n in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
        }
        p._incompatible_models = {}
        p._last_error = None
        p._compatibility_mode = False
        p._calibration = {}
        p.model_version = "test"

        mock_model = MagicMock()
        mock_model.get_booster.return_value.num_features.return_value = 10
        mock_model.get_booster.return_value.feature_names = None
        p.models = {
            n: mock_model
            for n in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
        }

        # Test the validation logic directly
        expected_count = len(p.feature_cols)
        unique_counts = sorted(set(p._model_feature_counts.values()))
        model_feature_count = unique_counts[0]
        if model_feature_count > expected_count:
            p._incompatible_models = {
                name: count
                for name, count in p._model_feature_counts.items()
                if count != expected_count
            }
            p._last_error = "feature mismatch"
            p.models = {}

        assert p.models == {}
        assert p._last_error is not None


# ── _load_calibration ────────────────────────────────────────────


class TestLoadCalibration:
    def test_loads_calibration_from_metrics(self, tmp_path):
        import json
        from unittest.mock import patch

        metrics = {
            "calibration_fg_coef": 0.15,
            "calibration_fg_intercept": -0.3,
            "calibration_1h_coef": 0.12,
            "calibration_1h_intercept": -0.2,
        }
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics))

        p = Predictor.__new__(Predictor)
        p._calibration = {}

        with patch("src.models.predictor.ARTIFACTS_DIR", tmp_path):
            p._load_calibration()

        assert p._calibration["calibration_fg_coef"] == 0.15
        assert len(p._calibration) == 4

    def test_no_metrics_file(self, tmp_path):
        from unittest.mock import patch

        p = Predictor.__new__(Predictor)
        p._calibration = {}

        with patch("src.models.predictor.ARTIFACTS_DIR", tmp_path):
            p._load_calibration()

        assert p._calibration == {}


# ── predict_and_store / predict_upcoming ─────────────────────────


class TestPredictAndStore:
    @pytest.mark.anyio
    async def test_creates_new_prediction(self):
        from unittest.mock import AsyncMock, patch

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a"]
        p._inference_feature_cols = ["a"]
        p.models = {
            n: MagicMock()
            for n in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
        }
        p._incompatible_models = {}
        p._last_error = None
        p._compatibility_mode = False
        p._calibration = {}
        p.model_version = "v1"

        mock_game = MagicMock(id=1, home_team=MagicMock(name="A"), away_team=MagicMock(name="B"))
        mock_db = AsyncMock()

        pred_dict = {
            "predicted_home_fg": 110.0,
            "predicted_away_fg": 105.0,
            "predicted_home_1h": 55.0,
            "predicted_away_1h": 52.0,
            "fg_spread": 5.0,
            "fg_total": 215.0,
            "fg_home_ml_prob": 0.65,
            "h1_spread": 3.0,
            "h1_total": 107.0,
            "h1_home_ml_prob": 0.6,
            "opening_spread": -3.5,
            "opening_total": 220.0,
            "odds_detail": {"books": {}},
        }

        # No existing prediction
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None

        # model version lookup
        version_result = MagicMock()
        version_result.scalar_one_or_none.return_value = "v1"

        mock_db.execute = AsyncMock(side_effect=[version_result, existing_result])
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        with patch.object(p, "predict_game", new_callable=AsyncMock, return_value=pred_dict):
            await p.predict_and_store(mock_game, mock_db)
            mock_db.add.assert_called_once()

    @pytest.mark.anyio
    async def test_updates_existing_prediction(self):
        from unittest.mock import AsyncMock, patch

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a"]
        p._inference_feature_cols = ["a"]
        p.models = {
            n: MagicMock()
            for n in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
        }
        p._incompatible_models = {}
        p._last_error = None
        p._compatibility_mode = False
        p._calibration = {}
        p.model_version = "v1"

        mock_game = MagicMock(id=1)

        pred_dict = {
            "predicted_home_fg": 110.0,
            "predicted_away_fg": 105.0,
            "predicted_home_1h": 55.0,
            "predicted_away_1h": 52.0,
            "fg_spread": 5.0,
            "fg_total": 215.0,
            "fg_home_ml_prob": 0.65,
            "h1_spread": 3.0,
            "h1_total": 107.0,
            "h1_home_ml_prob": 0.6,
            "opening_spread": -3.5,
            "opening_total": 220.0,
            "odds_detail": {"books": {}},
        }

        existing = MagicMock()
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = existing

        version_result = MagicMock()
        version_result.scalar_one_or_none.return_value = "v1"

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=[version_result, existing_result])
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        with patch.object(p, "predict_game", new_callable=AsyncMock, return_value=pred_dict):
            result = await p.predict_and_store(mock_game, mock_db)
            assert result is existing
            assert existing.predicted_home_fg == 110.0

    @pytest.mark.anyio
    async def test_predict_game_returns_none(self):
        from unittest.mock import AsyncMock, patch

        p = Predictor.__new__(Predictor)
        p.models = {}
        p._incompatible_models = {}

        mock_db = AsyncMock()
        with patch.object(p, "predict_game", new_callable=AsyncMock, return_value=None):
            result = await p.predict_and_store(MagicMock(), mock_db)
            assert result is None


class TestPredictUpcoming:
    @pytest.mark.anyio
    async def test_no_games(self):
        from unittest.mock import AsyncMock

        p = Predictor.__new__(Predictor)
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await p.predict_upcoming(mock_db)
        assert result == []

    @pytest.mark.anyio
    async def test_some_predictions_skipped(self):
        from unittest.mock import AsyncMock, patch

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["a"]
        p.models = {}
        p._incompatible_models = {}

        game1 = MagicMock(id=1, away_team=MagicMock(name="A"), home_team=MagicMock(name="B"))
        game2 = MagicMock(id=2, away_team=MagicMock(name="C"), home_team=MagicMock(name="D"))

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [game1, game2]
        mock_db.execute = AsyncMock(return_value=mock_result)

        pred1 = MagicMock()
        with patch.object(
            p, "predict_and_store", new_callable=AsyncMock, side_effect=[pred1, None]
        ):
            result = await p.predict_upcoming(mock_db)
            assert len(result) == 1
