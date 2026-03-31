"""Tests for Predictor inference path — predict_game, predict_and_store,
predict_upcoming, _latest_snapshots, _build_odds_detail, _resolve_model_version,
get_metrics, get_feature_importance.
"""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from src.models.predictor import MODEL_NAMES, Predictor

# ── Helpers ──────────────────────────────────────────────────────


def _ts(days_ago=0):
    return datetime.now(UTC) - timedelta(days=days_ago)


def _make_snapshot(
    bk="pinnacle", mkt="spreads", outcome="Boston Celtics", price=-110, point=-5.5, ts=None
):
    return SimpleNamespace(
        bookmaker=bk,
        market=mkt,
        outcome_name=outcome,
        price=price,
        point=point,
        captured_at=ts or _ts(0),
    )


# ── _latest_snapshots ───────────────────────────────────────────


class TestLatestSnapshots:
    def test_deduplicates_by_key(self):
        old = _make_snapshot(ts=_ts(2))
        new = _make_snapshot(ts=_ts(0))
        result, newest = Predictor._latest_snapshots([old, new])
        assert len(result) == 1
        assert result[0].captured_at == new.captured_at
        assert newest == new.captured_at

    def test_keeps_different_keys(self):
        s1 = _make_snapshot(bk="pinnacle", mkt="spreads", outcome="A")
        s2 = _make_snapshot(bk="fanduel", mkt="spreads", outcome="A")
        s3 = _make_snapshot(bk="pinnacle", mkt="totals", outcome="Over")
        result, newest = Predictor._latest_snapshots([s1, s2, s3])
        assert len(result) == 3

    def test_empty_input(self):
        result, newest = Predictor._latest_snapshots([])
        assert result == []
        assert newest is None


# ── _build_odds_detail ───────────────────────────────────────────


class TestBuildOddsDetail:
    def test_spreads_extraction(self):
        snaps = [
            _make_snapshot(bk="pinnacle", mkt="spreads", outcome="BOS", point=-5.5, price=-110)
        ]
        detail = Predictor._build_odds_detail(snaps, "BOS", "LAL", _ts(0))
        assert "pinnacle" in detail["books"]
        assert detail["books"]["pinnacle"]["spread"] == -5.5

    def test_totals_extraction(self):
        snaps = [
            _make_snapshot(bk="fanduel", mkt="totals", outcome="Over", point=220.5, price=-110)
        ]
        detail = Predictor._build_odds_detail(snaps, "BOS", "LAL", _ts(0))
        assert detail["books"]["fanduel"]["total"] == 220.5

    def test_h2h_extraction(self):
        snaps = [
            _make_snapshot(bk="dk", mkt="h2h", outcome="BOS", point=None, price=-200),
            _make_snapshot(bk="dk", mkt="h2h", outcome="LAL", point=None, price=170),
        ]
        detail = Predictor._build_odds_detail(snaps, "BOS", "LAL", _ts(0))
        assert detail["books"]["dk"]["home_ml"] == -200
        assert detail["books"]["dk"]["away_ml"] == 170

    def test_1h_markets(self):
        snaps = [
            _make_snapshot(bk="pin", mkt="spreads_h1", outcome="BOS", point=-2.5, price=-110),
            _make_snapshot(bk="pin", mkt="totals_h1", outcome="Over", point=108.5, price=-110),
            _make_snapshot(bk="pin", mkt="h2h_h1", outcome="BOS", point=None, price=-150),
            _make_snapshot(bk="pin", mkt="h2h_h1", outcome="LAL", point=None, price=130),
        ]
        detail = Predictor._build_odds_detail(snaps, "BOS", "LAL", _ts(0))
        books = detail["books"]["pin"]
        assert books["spread_h1"] == -2.5
        assert books["total_h1"] == 108.5
        assert books["home_ml_h1"] == -150
        assert books["away_ml_h1"] == 130

    def test_no_odds_ts(self):
        detail = Predictor._build_odds_detail([], "BOS", "LAL", None)
        assert detail["captured_at"] is None
        assert detail["books"] == {}


# ── _resolve_model_version ───────────────────────────────────────


class TestResolveModelVersion:
    async def test_uses_active_registry(self):
        predictor = Predictor.__new__(Predictor)
        predictor.model_version = "v0"

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = "v6.0.5"
        db.execute.return_value = result_mock

        version = await predictor._resolve_model_version(db)
        assert version == "v6.0.5"
        assert predictor.model_version == "v6.0.5"

    async def test_fallback_to_constant(self):
        from src.models.versioning import MODEL_VERSION

        predictor = Predictor.__new__(Predictor)
        predictor.model_version = "old"

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        db.execute.return_value = result_mock

        version = await predictor._resolve_model_version(db)
        assert version == MODEL_VERSION


# ── get_metrics / get_feature_importance ─────────────────────────


class TestGetMetricsAndImportance:
    def test_get_metrics_file_exists(self, tmp_path, monkeypatch):
        metrics = {"model_home_fg_mae": 8.5}
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))
        monkeypatch.setattr("src.models.predictor.ARTIFACTS_DIR", tmp_path)

        pred = Predictor.__new__(Predictor)
        result = pred.get_metrics()
        assert result["model_home_fg_mae"] == 8.5

    def test_get_metrics_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.models.predictor.ARTIFACTS_DIR", tmp_path)
        pred = Predictor.__new__(Predictor)
        assert pred.get_metrics() == {}

    def test_get_feature_importance_exists(self, tmp_path, monkeypatch):
        imp = {"home_ppg": 0.05, "away_ppg": 0.04}
        (tmp_path / "feature_importance.json").write_text(json.dumps(imp))
        monkeypatch.setattr("src.models.predictor.ARTIFACTS_DIR", tmp_path)

        pred = Predictor.__new__(Predictor)
        result = pred.get_feature_importance()
        assert "home_ppg" in result

    def test_get_feature_importance_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.models.predictor.ARTIFACTS_DIR", tmp_path)
        pred = Predictor.__new__(Predictor)
        assert pred.get_feature_importance() == {}


# ── predict_game ─────────────────────────────────────────────────


class TestPredictGame:
    async def test_returns_none_when_not_ready(self):
        pred = Predictor.__new__(Predictor)
        pred.models = {}
        pred._incompatible_models = {}
        pred.feature_cols = []
        pred._inference_feature_cols = []
        pred._calibration = {}

        db = AsyncMock()
        game = SimpleNamespace(
            id=1, home_team=SimpleNamespace(name="A"), away_team=SimpleNamespace(name="B")
        )
        result = await pred.predict_game(game, db)
        assert result is None

    async def test_returns_prediction_dict(self, monkeypatch):
        """When models are loaded, predict_game returns full dict."""
        pred = Predictor.__new__(Predictor)
        pred.feature_cols = ["f1", "f2"]
        pred._inference_feature_cols = ["f1", "f2"]
        pred._calibration = {}
        pred._last_error = None
        pred._imputation_values = {"f1": 0.0, "f2": 0.0}

        pred.models = {
            "model_home_fg": MagicMock(),
            "model_away_fg": MagicMock(),
            "model_home_1h": MagicMock(),
            "model_away_1h": MagicMock(),
        }
        pred.models["model_home_fg"].predict.return_value = np.array([110.0])
        pred.models["model_away_fg"].predict.return_value = np.array([105.0])
        pred.models["model_home_1h"].predict.return_value = np.array([55.0])
        pred.models["model_away_1h"].predict.return_value = np.array([52.0])

        pred._incompatible_models = {}
        pred._model_feature_counts = {n: 2 for n in MODEL_NAMES}

        game = SimpleNamespace(
            id=1,
            home_team_id=1,
            away_team_id=2,
            home_team=SimpleNamespace(name="BOS"),
            away_team=SimpleNamespace(name="LAL"),
            commence_time=_ts(0),
            season="2024-2025",
        )

        # Mock DB
        db = AsyncMock()
        snap_result = MagicMock()
        snap_result.scalars.return_value.all.return_value = [_make_snapshot(outcome="BOS")]
        db.execute.return_value = snap_result

        # Mock build_feature_vector
        monkeypatch.setattr(
            "src.models.predictor.build_feature_vector",
            AsyncMock(return_value={"f1": 1.0, "f2": 2.0}),
        )

        result = await pred.predict_game(game, db)
        assert result is not None
        assert "predicted_home_fg" in result
        assert "fg_spread" in result
        assert "fg_home_ml_prob" in result
        assert "h1_spread" in result

    async def test_allows_limited_imputation(self, monkeypatch):
        pred = Predictor.__new__(Predictor)
        pred.feature_cols = ["f1", "f2"]
        pred._inference_feature_cols = ["f1", "f2"]
        pred._calibration = {}
        pred._last_error = None
        pred._imputation_values = {"f1": 1.5, "f2": 2.0}

        home_fg_model = MagicMock()
        home_fg_model.predict.return_value = np.array([111.0])
        away_fg_model = MagicMock()
        away_fg_model.predict.return_value = np.array([106.0])
        home_1h_model = MagicMock()
        home_1h_model.predict.return_value = np.array([56.0])
        away_1h_model = MagicMock()
        away_1h_model.predict.return_value = np.array([52.0])
        pred.models = {
            "model_home_fg": home_fg_model,
            "model_away_fg": away_fg_model,
            "model_home_1h": home_1h_model,
            "model_away_1h": away_1h_model,
        }

        pred._incompatible_models = {}
        pred._model_feature_counts = {n: 2 for n in MODEL_NAMES}

        game = SimpleNamespace(
            id=1,
            home_team_id=1,
            away_team_id=2,
            home_team=SimpleNamespace(name="BOS"),
            away_team=SimpleNamespace(name="LAL"),
            commence_time=_ts(0),
            season="2024-2025",
        )

        db = AsyncMock()
        snap_result = MagicMock()
        snap_result.scalars.return_value.all.return_value = [_make_snapshot(outcome="BOS")]
        db.execute.return_value = snap_result

        monkeypatch.setattr(
            "src.models.predictor.build_feature_vector",
            AsyncMock(return_value={"f1": float("nan"), "f2": 2.5}),
        )
        monkeypatch.setattr(
            "src.models.predictor.get_settings",
            lambda: SimpleNamespace(odds_freshness_max_age_minutes=30),
        )

        result = await pred.predict_game(game, db)
        assert result is not None
        X_input = home_fg_model.predict.call_args.args[0]
        assert X_input.tolist() == [[1.5, 2.5]]

    async def test_skips_when_too_many_features_require_imputation(self, monkeypatch):
        pred = Predictor.__new__(Predictor)
        pred.feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        pred._inference_feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        pred._calibration = {}
        pred._last_error = None
        pred._imputation_values = {f"f{i}": float(i) for i in range(1, 6)}

        home_fg_model = MagicMock()
        home_fg_model.predict.return_value = np.array([111.0])
        pred.models = {
            "model_home_fg": home_fg_model,
            "model_away_fg": MagicMock(),
            "model_home_1h": MagicMock(),
            "model_away_1h": MagicMock(),
        }

        pred._incompatible_models = {}
        pred._model_feature_counts = {n: 5 for n in MODEL_NAMES}

        game = SimpleNamespace(
            id=1,
            home_team_id=1,
            away_team_id=2,
            home_team=SimpleNamespace(name="BOS"),
            away_team=SimpleNamespace(name="LAL"),
            commence_time=_ts(0),
            season="2024-2025",
        )

        db = AsyncMock()
        snap_result = MagicMock()
        snap_result.scalars.return_value.all.return_value = [_make_snapshot(outcome="BOS")]
        db.execute.return_value = snap_result

        monkeypatch.setattr(
            "src.models.predictor.build_feature_vector",
            AsyncMock(
                return_value={
                    "f1": float("nan"),
                    "f2": float("nan"),
                    "f3": 3.0,
                    "f4": 4.0,
                    "f5": 5.0,
                }
            ),
        )
        monkeypatch.setattr(
            "src.models.predictor.get_settings",
            lambda: SimpleNamespace(odds_freshness_max_age_minutes=30),
        )

        result = await pred.predict_game(game, db)
        assert result is None
        home_fg_model.predict.assert_not_called()

    async def test_skips_generated_payload_that_fails_integrity(self, monkeypatch):
        pred = Predictor.__new__(Predictor)
        pred.feature_cols = ["f1", "f2"]
        pred._inference_feature_cols = ["f1", "f2"]
        pred._calibration = {}
        pred._last_error = None
        pred._imputation_values = {"f1": 0.0, "f2": 0.0}

        home_fg_model = MagicMock()
        home_fg_model.predict.return_value = np.array([-5.0])
        away_fg_model = MagicMock()
        away_fg_model.predict.return_value = np.array([4.0])
        home_1h_model = MagicMock()
        home_1h_model.predict.return_value = np.array([-5.0])
        away_1h_model = MagicMock()
        away_1h_model.predict.return_value = np.array([4.0])
        pred.models = {
            "model_home_fg": home_fg_model,
            "model_away_fg": away_fg_model,
            "model_home_1h": home_1h_model,
            "model_away_1h": away_1h_model,
        }

        pred._incompatible_models = {}
        pred._model_feature_counts = {n: 2 for n in MODEL_NAMES}

        game = SimpleNamespace(
            id=1,
            home_team_id=1,
            away_team_id=2,
            home_team=SimpleNamespace(name="BOS"),
            away_team=SimpleNamespace(name="LAL"),
            commence_time=_ts(0),
            season="2024-2025",
        )

        db = AsyncMock()
        snap_result = MagicMock()
        snap_result.scalars.return_value.all.return_value = [_make_snapshot(outcome="BOS")]
        db.execute.return_value = snap_result

        monkeypatch.setattr(
            "src.models.predictor.build_feature_vector",
            AsyncMock(return_value={"f1": 1.0, "f2": 2.5}),
        )
        monkeypatch.setattr(
            "src.models.predictor.get_settings",
            lambda: SimpleNamespace(odds_freshness_max_age_minutes=30),
        )

        result = await pred.predict_game(game, db)
        assert result is None


# ── predict_and_store ────────────────────────────────────────────


class TestPredictAndStore:
    async def test_creates_new_prediction(self, monkeypatch):
        pred = Predictor.__new__(Predictor)
        pred.model_version = "v6"
        pred.models = {n: MagicMock() for n in MODEL_NAMES}
        pred._incompatible_models = {}
        pred.feature_cols = ["f1"]
        pred._inference_feature_cols = ["f1"]
        pred._calibration = {}
        pred._last_error = None
        pred._model_feature_counts = {n: 1 for n in MODEL_NAMES}

        pred.models["model_home_fg"].predict.return_value = np.array([110.0])
        pred.models["model_away_fg"].predict.return_value = np.array([105.0])
        pred.models["model_home_1h"].predict.return_value = np.array([55.0])
        pred.models["model_away_1h"].predict.return_value = np.array([52.0])

        game = SimpleNamespace(
            id=1,
            home_team_id=1,
            away_team_id=2,
            home_team=SimpleNamespace(name="A"),
            away_team=SimpleNamespace(name="B"),
            commence_time=_ts(0),
            season="2024-2025",
        )

        monkeypatch.setattr(
            "src.models.predictor.build_feature_vector",
            AsyncMock(return_value={"f1": 1.0}),
        )

        db = AsyncMock()

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            if call_n["n"] <= 1:
                # snapshot query
                result.scalars.return_value.all.return_value = [_make_snapshot(outcome="A")]
            elif call_n["n"] == 2:
                # resolve model version
                result.scalar_one_or_none.return_value = "v6"
            else:
                # existing prediction check
                result.scalar_one_or_none.return_value = None
            return result

        db.execute = mock_exec

        MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        await pred.predict_and_store(game, db)
        db.commit.assert_awaited()


# ── predict_upcoming ─────────────────────────────────────────────


class TestPredictUpcoming:
    async def test_no_upcoming_games(self):
        pred = Predictor.__new__(Predictor)

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        db.execute.return_value = result_mock

        predictions = await pred.predict_upcoming(db)
        assert predictions == []
