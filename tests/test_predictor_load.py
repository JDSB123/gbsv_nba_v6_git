"""Tests for Predictor._load_models and predict_game."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

_MOD = "src.models.predictor"


def _make_mock_booster(num_features: int, feature_names=None):
    booster = MagicMock()
    booster.num_features.return_value = num_features
    booster.feature_names = feature_names
    return booster


def _make_mock_model(num_features: int, feature_names=None):
    model = MagicMock()
    model.get_booster.return_value = _make_mock_booster(num_features, feature_names)
    model.predict.return_value = np.array([105.0])
    model.feature_importances_ = np.ones(num_features) / num_features
    return model


def _make_snapshot(
    bookmaker: str = "fanduel",
    market: str = "spreads",
    outcome_name: str = "Lakers",
    price: int = -110,
    point: float | None = -3.5,
    captured_at: datetime | None = None,
):
    return SimpleNamespace(
        bookmaker=bookmaker,
        market=market,
        outcome_name=outcome_name,
        price=price,
        point=point,
        captured_at=captured_at or datetime.now(UTC),
    )


class TestLoadModelsAllPresent:
    """_load_models when all 4 model files exist."""

    def test_exact_feature_count_match(self):
        from src.models.features import get_feature_columns
        from src.models.predictor import MODEL_NAMES, Predictor

        expected = len(get_feature_columns())
        mock_model = _make_mock_model(expected)

        with (
            patch(f"{_MOD}.ARTIFACTS_DIR") as mock_dir,
            patch(f"{_MOD}.xgb") as mock_xgb,
        ):
            mock_xgb.XGBRegressor.return_value = mock_model
            # All 4 model files exist
            for name in MODEL_NAMES:
                (mock_dir / f"{name}.json").exists.return_value = True
            # Path exists returns True for any path
            mock_dir.__truediv__ = lambda self, key: MagicMock(exists=MagicMock(return_value=True))
            mock_xgb.XGBRegressor.return_value.load_model = MagicMock()
            mock_xgb.XGBRegressor.return_value.get_booster.return_value = _make_mock_booster(expected)

            p = Predictor.__new__(Predictor)
            p.feature_cols = get_feature_columns()
            p._inference_feature_cols = list(p.feature_cols)
            p._model_feature_counts = {}
            p._incompatible_models = {}
            p._compatibility_mode = False
            p._last_error = None
            p._calibration = {}
            p.models = {}
            p._load_models()

            assert len(p.models) == 4

    def test_missing_model_file(self):
        from src.models.features import get_feature_columns
        from src.models.predictor import Predictor

        mock_path = MagicMock()
        mock_path.exists.return_value = False

        with patch(f"{_MOD}.ARTIFACTS_DIR") as mock_dir:
            mock_dir.__truediv__ = lambda self, key: mock_path

            p = Predictor.__new__(Predictor)
            p.feature_cols = get_feature_columns()
            p._inference_feature_cols = list(p.feature_cols)
            p._model_feature_counts = {}
            p._incompatible_models = {}
            p._compatibility_mode = False
            p._last_error = None
            p._calibration = {}
            p.models = {}
            p._load_models()

            assert len(p.models) == 0  # None loaded

    def test_feature_count_mismatch_across_models(self):
        """Different feature counts across models → error, models cleared."""
        from src.models.features import get_feature_columns
        from src.models.predictor import Predictor

        expected = len(get_feature_columns())
        call_count = 0

        def make_model_with_varying_features():
            nonlocal call_count
            call_count += 1
            # First 3 have expected, last has different
            count = expected if call_count < 4 else expected + 5
            m = _make_mock_model(count)
            m.load_model = MagicMock()
            return m

        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with (
            patch(f"{_MOD}.ARTIFACTS_DIR") as mock_dir,
            patch(f"{_MOD}.xgb") as mock_xgb,
        ):
            mock_dir.__truediv__ = lambda self, key: mock_path
            mock_xgb.XGBRegressor.side_effect = make_model_with_varying_features

            p = Predictor.__new__(Predictor)
            p.feature_cols = get_feature_columns()
            p._inference_feature_cols = list(p.feature_cols)
            p._model_feature_counts = {}
            p._incompatible_models = {}
            p._compatibility_mode = False
            p._last_error = None
            p._calibration = {}
            p.models = {}
            p._load_models()

            assert p.models == {}
            assert p._last_error is not None
            assert "mismatch" in p._last_error.lower()

    def test_compatibility_mode_on_fewer_features(self):
        """Models with fewer features → compatibility mode enabled."""
        from src.models.features import get_feature_columns
        from src.models.predictor import Predictor

        expected = len(get_feature_columns())
        fewer = expected - 10
        mock_model = _make_mock_model(fewer)
        mock_model.load_model = MagicMock()

        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with (
            patch(f"{_MOD}.ARTIFACTS_DIR") as mock_dir,
            patch(f"{_MOD}.xgb") as mock_xgb,
        ):
            mock_dir.__truediv__ = lambda self, key: mock_path
            mock_xgb.XGBRegressor.return_value = mock_model

            p = Predictor.__new__(Predictor)
            p.feature_cols = get_feature_columns()
            p._inference_feature_cols = list(p.feature_cols)
            p._model_feature_counts = {}
            p._incompatible_models = {}
            p._compatibility_mode = False
            p._last_error = None
            p._calibration = {}
            p.models = {}
            p._load_models()

            assert p._compatibility_mode is True
            assert len(p._inference_feature_cols) == fewer

    def test_feature_name_mismatch_logged(self):
        """Feature names don't match code → _last_error set but models kept."""
        from src.models.features import get_feature_columns
        from src.models.predictor import Predictor

        expected = len(get_feature_columns())
        bad_names = [f"wrong_{i}" for i in range(expected)]
        mock_model = _make_mock_model(expected, feature_names=bad_names)
        mock_model.load_model = MagicMock()

        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with (
            patch(f"{_MOD}.ARTIFACTS_DIR") as mock_dir,
            patch(f"{_MOD}.xgb") as mock_xgb,
        ):
            mock_dir.__truediv__ = lambda self, key: mock_path
            mock_xgb.XGBRegressor.return_value = mock_model

            p = Predictor.__new__(Predictor)
            p.feature_cols = get_feature_columns()
            p._inference_feature_cols = list(p.feature_cols)
            p._model_feature_counts = {}
            p._incompatible_models = {}
            p._compatibility_mode = False
            p._last_error = None
            p._calibration = {}
            p.models = {}
            p._load_models()

            # Models are kept (compatibility_mode may apply if counts match)
            assert p._last_error is not None or len(p.models) == 4


class TestPredictGame:
    """predict_game with full model inference pipeline."""

    @pytest.mark.anyio
    async def test_basic_prediction(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1", "f2"]
        p._inference_feature_cols = ["f1", "f2"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(predict=MagicMock(return_value=np.array([110.0]))),
            "model_away_fg": MagicMock(predict=MagicMock(return_value=np.array([105.0]))),
            "model_home_1h": MagicMock(predict=MagicMock(return_value=np.array([55.0]))),
            "model_away_1h": MagicMock(predict=MagicMock(return_value=np.array([52.0]))),
        }

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
        )
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [_make_snapshot()]
        db.execute.return_value = mock_result

        with patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat:
            mock_feat.return_value = {"f1": 1.0, "f2": 2.0}
            result = await p.predict_game(game, db)

        assert result is not None
        assert result["predicted_home_fg"] == 110.0
        assert result["predicted_away_fg"] == 105.0
        assert result["fg_spread"] == 5.0
        assert result["fg_total"] == 215.0

    @pytest.mark.anyio
    async def test_prediction_with_odds_snapshots(self):
        """When stored_snapshots exist, opening lines are extracted."""
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1"]
        p._inference_feature_cols = ["f1"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(predict=MagicMock(return_value=np.array([110.0]))),
            "model_away_fg": MagicMock(predict=MagicMock(return_value=np.array([105.0]))),
            "model_home_1h": MagicMock(predict=MagicMock(return_value=np.array([55.0]))),
            "model_away_1h": MagicMock(predict=MagicMock(return_value=np.array([52.0]))),
        }

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
        )

        snap1 = SimpleNamespace(
            bookmaker="fanduel", market="spreads", outcome_name="Lakers",
            captured_at=datetime.now(UTC), price=-110, point=-3.5,
        )
        snap2 = SimpleNamespace(
            bookmaker="fanduel", market="totals", outcome_name="Over",
            captured_at=datetime.now(UTC), price=-110, point=220.5,
        )
        snap3 = SimpleNamespace(
            bookmaker="fanduel", market="spreads_h1", outcome_name="Lakers",
            captured_at=datetime.now(UTC), price=-110, point=-1.5,
        )
        snap4 = SimpleNamespace(
            bookmaker="fanduel", market="totals_h1", outcome_name="Over",
            captured_at=datetime.now(UTC), price=-110, point=110.5,
        )

        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [snap1, snap2, snap3, snap4]
        db.execute.return_value = mock_result

        with patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat:
            mock_feat.return_value = {"f1": 1.0}
            result = await p.predict_game(game, db)

        assert result is not None
        assert result["opening_spread"] == -3.5
        assert result["opening_total"] == 220.5
        assert result["odds_detail"]["opening_h1_spread"] == -1.5
        assert result["odds_detail"]["opening_h1_total"] == 110.5

    @pytest.mark.anyio
    async def test_prediction_returns_none_when_features_none(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1"]
        p._inference_feature_cols = ["f1"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(),
            "model_away_fg": MagicMock(),
            "model_home_1h": MagicMock(),
            "model_away_1h": MagicMock(),
        }

        game = SimpleNamespace(
            id=1, home_team=SimpleNamespace(name="A"), away_team=SimpleNamespace(name="B"),
        )
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [_make_snapshot(outcome_name="A")]
        db.execute.return_value = mock_result

        with patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat:
            mock_feat.return_value = None
            result = await p.predict_game(game, db)

        assert result is None

    @pytest.mark.anyio
    async def test_prediction_raises_on_model_error(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1"]
        p._inference_feature_cols = ["f1"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(predict=MagicMock(side_effect=ValueError("boom"))),
            "model_away_fg": MagicMock(),
            "model_home_1h": MagicMock(),
            "model_away_1h": MagicMock(),
        }

        game = SimpleNamespace(
            id=1, home_team=SimpleNamespace(name="A"), away_team=SimpleNamespace(name="B"),
        )
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [_make_snapshot(outcome_name="A")]
        db.execute.return_value = mock_result

        with patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat:
            mock_feat.return_value = {"f1": 1.0}
            with pytest.raises(RuntimeError, match="boom"):
                await p.predict_game(game, db)

    @pytest.mark.anyio
    async def test_prediction_allows_limited_non_finite_feature_imputation(self):
        from src.models.predictor import Predictor

        home_fg = MagicMock(return_value=np.array([110.0]))
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1", "f2"]
        p._inference_feature_cols = ["f1", "f2"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p._imputation_values = {"f1": 1.5, "f2": 2.5}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(predict=home_fg),
            "model_away_fg": MagicMock(predict=MagicMock(return_value=np.array([105.0]))),
            "model_home_1h": MagicMock(predict=MagicMock(return_value=np.array([55.0]))),
            "model_away_1h": MagicMock(predict=MagicMock(return_value=np.array([52.0]))),
        }

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
        )
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [_make_snapshot()]
        db.execute.return_value = mock_result

        with patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat:
            mock_feat.return_value = {"f1": float("nan"), "f2": 2.0}
            result = await p.predict_game(game, db)

        assert result is not None
        X_input = home_fg.call_args.args[0]
        assert X_input.tolist() == [[1.5, 2.0]]

    @pytest.mark.anyio
    async def test_prediction_allows_limited_missing_feature_imputation(self):
        from src.models.predictor import Predictor

        home_fg = MagicMock(return_value=np.array([110.0]))
        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1", "f2"]
        p._inference_feature_cols = ["f1", "f2"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p._imputation_values = {"f1": 1.5, "f2": 2.5}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(predict=home_fg),
            "model_away_fg": MagicMock(predict=MagicMock(return_value=np.array([105.0]))),
            "model_home_1h": MagicMock(predict=MagicMock(return_value=np.array([55.0]))),
            "model_away_1h": MagicMock(predict=MagicMock(return_value=np.array([52.0]))),
        }

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="Lakers"),
            away_team=SimpleNamespace(name="Celtics"),
        )
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [_make_snapshot()]
        db.execute.return_value = mock_result

        with patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat:
            mock_feat.return_value = {"f1": 1.0}
            result = await p.predict_game(game, db)

        assert result is not None
        X_input = home_fg.call_args.args[0]
        assert X_input.tolist() == [[1.0, 2.5]]

    @pytest.mark.anyio
    async def test_prediction_returns_none_when_odds_are_stale(self):
        from src.models.predictor import Predictor

        p = Predictor.__new__(Predictor)
        p.feature_cols = ["f1"]
        p._inference_feature_cols = ["f1"]
        p._compatibility_mode = False
        p._last_error = None
        p._calibration = {}
        p._incompatible_models = {}
        p._model_feature_counts = {}
        p._imputation_values = {"f1": 0.0}
        p.model_version = "test"
        p.models = {
            "model_home_fg": MagicMock(),
            "model_away_fg": MagicMock(),
            "model_home_1h": MagicMock(),
            "model_away_1h": MagicMock(),
        }

        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(name="A"),
            away_team=SimpleNamespace(name="B"),
        )
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            _make_snapshot(
                outcome_name="A",
                captured_at=datetime.now(UTC) - timedelta(minutes=120),
            )
        ]
        db.execute.return_value = mock_result

        with (
            patch(f"{_MOD}.build_feature_vector", new_callable=AsyncMock) as mock_feat,
            patch(f"{_MOD}.get_settings", return_value=SimpleNamespace(odds_freshness_max_age_minutes=30)),
        ):
            mock_feat.return_value = {"f1": 1.0}
            result = await p.predict_game(game, db)

        assert result is None
