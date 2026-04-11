from __future__ import annotations

import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.models.ensemble import EnsembleStack
    from src.models.explainability import Explainer
    from src.models.ood import OODDetector

import numpy as np
import xgboost as xgb
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.db.models import Game, ModelRegistry, OddsSnapshot, Prediction
from src.models.features import build_feature_vector, get_feature_columns
from src.models.odds_utils import build_odds_detail, latest_snapshots
from src.models.versioning import MODEL_VERSION
from src.services.prediction_integrity import prediction_has_valid_score_payload

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_NAMES = ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
_MAX_IMPUTED_FEATURE_RATIO = 0.2
CORE_ODDS_MARKETS = ("h2h", "spreads", "totals", "h2h_h1", "spreads_h1", "totals_h1")
PROP_ODDS_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_double_double",
    "player_triple_double",
)


def _margin_to_prob(
    margin: float,
    coef: float | None = None,
    intercept: float | None = None,
    scale: float = 7.5,
) -> float:
    """Convert predicted point margin to win probability.

    Uses Platt-scaled logistic coefficients when available (trained by
    ``ModelTrainer``), otherwise falls back to a standard sigmoid.

    Args:
        margin: Predicted margin (home - away).
        coef: Logistic coefficient (from training).
        intercept: Logistic intercept (from training).
        scale: Fallback scale factor for fallback sigmoid.
    """
    if coef is not None and intercept is not None:
        # P(Win) = 1 / (1 + exp(-(coef*margin + intercept)))
        z = coef * margin + intercept
        return 1.0 / (1.0 + math.exp(-z))
    return 1.0 / (1.0 + math.exp(-margin / scale))


class Predictor:
    # Class-level defaults for optional components so __new__ instances
    # (used in tests) don't raise AttributeError.
    _ensemble: EnsembleStack | None = None
    _ood: OODDetector | None = None
    _explainer: Explainer | None = None
    _quantile_models: dict = {}  # noqa: RUF012

    def __init__(self) -> None:
        self.feature_cols = get_feature_columns()
        self._inference_feature_cols = list(self.feature_cols)
        self.models: dict[str, xgb.XGBRegressor] = {}
        self._imputation_values: dict[str, float] = {}
        self._model_feature_counts: dict[str, int] = {}
        self._incompatible_models: dict[str, int] = {}
        self._last_error: str | None = None
        self._runtime_warning: str | None = None
        self._compatibility_mode = False
        self._calibration: dict[str, float] = {}
        self.model_version = MODEL_VERSION
        self._load_calibration()
        self._load_imputation()
        self._download_blob_artifacts()
        self._load_models()
        self._load_quantile_models()
        self._load_ensemble()
        self._load_ood_detector()
        self._load_explainer()
        self._run_model_smoke_test()

    def _download_blob_artifacts(self) -> None:
        """Try to download model artifacts from blob storage before loading locally."""
        has_blob_config = bool(
            (os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip())
            or (os.getenv("AZURE_STORAGE_ACCOUNT_URL", "").strip())
        )
        if not has_blob_config:
            logger.warning(
                "Blob storage model sync disabled: set AZURE_STORAGE_ACCOUNT_URL or "
                "AZURE_STORAGE_CONNECTION_STRING to pull remote artifacts."
            )
            return
        try:
            from src.models.blob_storage import sync_artifacts_down

            count = sync_artifacts_down()
            if count:
                logger.info("Downloaded %d artifacts from blob storage", count)
            else:
                logger.warning(
                    "Blob storage sync ran but downloaded 0 artifacts; "
                    "falling back to local model artifacts.",
                )
        except Exception:
            logger.debug("Blob storage download skipped or failed", exc_info=True)

    async def _resolve_model_version(self, db: AsyncSession) -> str:
        result = await db.execute(
            select(ModelRegistry.model_version)
            .where(ModelRegistry.is_active.is_(True))
            .order_by(ModelRegistry.promoted_at.desc())
            .limit(1)
        )
        active_version = result.scalar_one_or_none()
        if active_version:
            self.model_version = active_version
        else:
            self.model_version = MODEL_VERSION
        return self.model_version

    def _load_models(self) -> None:
        expected_features = self.feature_cols
        expected_count = len(expected_features)

        for name in MODEL_NAMES:
            path = ARTIFACTS_DIR / f"{name}.json"
            if path.exists():
                model = xgb.XGBRegressor()
                model.load_model(str(path))
                actual_features = int(model.get_booster().num_features())
                self._model_feature_counts[name] = actual_features
                self.models[name] = model
                logger.info("Loaded model %s", name)
            else:
                logger.warning("Model file not found: %s", path)

        if len(self.models) != len(MODEL_NAMES):
            return

        unique_counts = sorted(set(self._model_feature_counts.values()))
        if len(unique_counts) > 1:
            self._incompatible_models = {
                name: count
                for name, count in self._model_feature_counts.items()
                if count != expected_count
            }
            mismatch = ", ".join(
                f"{name}={count}" for name, count in self._model_feature_counts.items()
            )
            self._last_error = (
                "Model artifact feature shape mismatch across models: "
                f"expected {expected_count}, got [{mismatch}]"
            )
            logger.error(self._last_error)
            self.models = {}
            return

        model_feature_count = unique_counts[0]

        # Check if the feature names in code match what the model expects (if we had feature_names)
        # XGBoost models saved as JSON might have feature_names if set during training.
        try:
            model_features = self.models[MODEL_NAMES[0]].get_booster().feature_names
            if model_features and list(model_features) != expected_features:
                self._last_error = (
                    "Model feature name mismatch. Code and artifacts are out of sync."
                )
                logger.error(self._last_error)
                # We don't nullify models yet to allow 'compatibility_mode' if counts match
        except Exception:
            pass

        if model_feature_count == expected_count:
            return

        if 0 < model_feature_count < expected_count:
            self._compatibility_mode = True
            # Prefer the authoritative feature list saved by the trainer.
            trained_cols_path = ARTIFACTS_DIR / "trained_feature_cols.json"
            if trained_cols_path.exists():
                import json as _json

                saved_cols = _json.loads(trained_cols_path.read_text())
                if len(saved_cols) == model_feature_count:
                    self._inference_feature_cols = saved_cols
                    logger.warning(
                        "Compatibility mode enabled: models trained on %d/%d features "
                        "(loaded from trained_feature_cols.json).",
                        model_feature_count,
                        expected_count,
                    )
                else:
                    self._inference_feature_cols = expected_features[:model_feature_count]
                    logger.warning(
                        "trained_feature_cols.json has %d cols but model expects %d; "
                        "falling back to first-%d. Retrain recommended.",
                        len(saved_cols),
                        model_feature_count,
                        model_feature_count,
                    )
            else:
                self._inference_feature_cols = expected_features[:model_feature_count]
                logger.warning(
                    "Compatibility mode enabled: models expect %d features, current vector has %d. "
                    "Using first %d features for inference. Retrain recommended.",
                    model_feature_count,
                    expected_count,
                    model_feature_count,
                )
            # Fire-and-forget alert — import here to avoid circular dependency
            try:
                import asyncio

                from src.notifications.teams import send_alert

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = loop.create_task(
                        send_alert(
                            "Model Compatibility Mode Active",
                            f"Models expect {model_feature_count} features but code provides "
                            f"{expected_features}. Running in degraded mode — retrain recommended.",
                            "warning",
                        ),
                        name="model_compat_alert",
                    )
                    task.add_done_callback(
                        lambda t: t.result() if not t.cancelled() and not t.exception() else None
                    )
            except Exception:
                pass  # alerting is best-effort
            return

        self._incompatible_models = {
            name: count
            for name, count in self._model_feature_counts.items()
            if count != expected_features
        }
        mismatch = ", ".join(
            f"{name}={count}" for name, count in self._model_feature_counts.items()
        )
        self._last_error = (
            f"Model artifact feature shape mismatch: expected {expected_features}, got [{mismatch}]"
        )
        logger.error(self._last_error)
        self.models = {}

    def _load_calibration(self) -> None:
        """Load Platt-scaling coefficients from metrics.json."""
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            for key in (
                "calibration_fg_coef",
                "calibration_fg_intercept",
                "calibration_1h_coef",
                "calibration_1h_intercept",
            ):
                if key in m:
                    self._calibration[key] = m[key]
        if self._calibration:
            logger.info("Loaded calibration coefficients")

    def _load_imputation(self) -> None:
        """Load saved feature fill values so inference never accepts NaNs."""
        imputation_path = ARTIFACTS_DIR / "imputation.json"
        if not imputation_path.exists():
            logger.warning("Imputation file not found: %s", imputation_path)
            return

        payload = json.loads(imputation_path.read_text())
        self._imputation_values = {}
        for col in self.feature_cols:
            value = payload.get(col)
            try:
                self._imputation_values[col] = float(value)
            except (TypeError, ValueError):
                self._imputation_values[col] = 0.0

        logger.info("Loaded imputation values for %d features", len(self._imputation_values))

    def _load_quantile_models(self) -> None:
        """Load quantile regression models for confidence intervals."""
        self._quantile_models: dict[str, dict[str, xgb.XGBRegressor]] = {}
        for name in MODEL_NAMES:
            self._quantile_models[name] = {}
            for label in ("q10", "q90"):
                path = ARTIFACTS_DIR / f"{name}_{label}.json"
                if path.exists():
                    model = xgb.XGBRegressor()
                    model.load_model(str(path))
                    self._quantile_models[name][label] = model
        loaded = sum(len(q) for q in self._quantile_models.values())
        if loaded:
            logger.info("Loaded %d quantile models", loaded)

    def _load_ensemble(self) -> None:
        """Load ensemble stacking artifacts."""
        self._ensemble = None
        try:
            from src.models.ensemble import EnsembleStack

            ensemble = EnsembleStack()
            if ensemble.load():
                self._ensemble = ensemble
        except Exception:
            logger.debug("Ensemble not available", exc_info=True)

    def _load_ood_detector(self) -> None:
        """Load OOD detector artifacts."""
        self._ood = None
        try:
            from src.models.ood import OODDetector

            ood = OODDetector()
            if ood.load():
                self._ood = ood
        except Exception:
            logger.debug("OOD detector not available", exc_info=True)

    def _load_explainer(self) -> None:
        """Initialize SHAP explainer for per-prediction attribution."""
        self._explainer = None
        try:
            from src.models.explainability import Explainer

            explainer = Explainer()
            if self.models:
                explainer.initialize(self.models, self._inference_feature_cols)
                self._explainer = explainer
        except Exception:
            logger.debug("SHAP explainer not available", exc_info=True)

    def _has_imputation_values(self) -> bool:
        imputation_values = getattr(self, "_imputation_values", {})
        return all(col in imputation_values for col in self._inference_feature_cols)

    def _allowed_imputed_feature_count(self) -> int:
        total_features = len(self._inference_feature_cols)
        if total_features <= 1:
            return 0
        return max(1, math.floor(total_features * _MAX_IMPUTED_FEATURE_RATIO))

    def _can_tolerate_imputation(self, imputed_count: int) -> bool:
        total_features = len(self._inference_feature_cols)
        if imputed_count == 0:
            return True
        if total_features <= 0 or imputed_count >= total_features:
            return False
        return imputed_count <= self._allowed_imputed_feature_count()

    def _predict_scores(
        self,
        X: np.ndarray,
        context: str,
    ) -> tuple[float, float, float, float]:
        try:
            home_fg = float(self.models["model_home_fg"].predict(X)[0])
            away_fg = float(self.models["model_away_fg"].predict(X)[0])
            home_1h = float(self.models["model_home_1h"].predict(X)[0])
            away_1h = float(self.models["model_away_1h"].predict(X)[0])
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Prediction failed for %s", context)
            raise RuntimeError(str(exc)) from exc

        # Blend with ensemble when available
        if self._ensemble and self._ensemble.is_ready:
            for name, xgb_val in [
                ("model_home_fg", home_fg),
                ("model_away_fg", away_fg),
                ("model_home_1h", home_1h),
                ("model_away_1h", away_1h),
            ]:
                blended = self._ensemble.predict(X, name, xgb_val)
                if blended is not None:
                    if name == "model_home_fg":
                        home_fg = blended
                    elif name == "model_away_fg":
                        away_fg = blended
                    elif name == "model_home_1h":
                        home_1h = blended
                    elif name == "model_away_1h":
                        away_1h = blended

        # Half-time scores should never exceed the full-game projection.
        home_1h = min(home_1h, home_fg)
        away_1h = min(away_1h, away_fg)
        return home_fg, away_fg, home_1h, away_1h

    def _predict_quantiles(
        self,
        X: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Predict 10th and 90th percentile intervals."""
        intervals: dict[str, dict[str, float]] = {}
        for name in MODEL_NAMES:
            q_models = self._quantile_models.get(name, {})
            if "q10" in q_models and "q90" in q_models:
                low = float(q_models["q10"].predict(X)[0])
                high = float(q_models["q90"].predict(X)[0])
                intervals[name] = {"q10": round(low, 1), "q90": round(high, 1)}
        return intervals

    def _run_model_smoke_test(self) -> None:
        if not self.is_ready or not self._has_imputation_values():
            return

        X = np.array(
            [[self._imputation_values[col] for col in self._inference_feature_cols]],
            dtype=float,
        )
        try:
            home_fg, away_fg, home_1h, away_1h = self._predict_scores(X, "model smoke test")
        except RuntimeError:
            self._last_error = f"Model smoke test failed: {self._last_error}"
            logger.error(self._last_error)
            self.models = {}
            return

        smoke_payload = SimpleNamespace(
            predicted_home_fg=round(home_fg, 1),
            predicted_away_fg=round(away_fg, 1),
            predicted_home_1h=round(home_1h, 1),
            predicted_away_1h=round(away_1h, 1),
            fg_spread=round(home_fg - away_fg, 1),
            fg_total=round(home_fg + away_fg, 1),
            h1_spread=round(home_1h - away_1h, 1),
            h1_total=round(home_1h + away_1h, 1),
        )
        if prediction_has_valid_score_payload(smoke_payload):
            return

        self._runtime_warning = (
            "Model smoke test produced an implausible baseline payload; "
            "continuing with per-request integrity checks enabled"
        )
        logger.warning(
            "%s (home_fg=%.1f away_fg=%.1f home_1h=%.1f away_1h=%.1f)",
            self._runtime_warning,
            smoke_payload.predicted_home_fg,
            smoke_payload.predicted_away_fg,
            smoke_payload.predicted_home_1h,
            smoke_payload.predicted_away_1h,
        )

    @property
    def is_ready(self) -> bool:
        return len(self.models) == len(MODEL_NAMES) and not self._incompatible_models

    def get_runtime_status(self) -> dict[str, Any]:
        expected_features = len(self.feature_cols)
        missing = [name for name in MODEL_NAMES if name not in self.models]
        imputation_values = getattr(self, "_imputation_values", {})
        missing_imputation = [
            col for col in self._inference_feature_cols if col not in imputation_values
        ]
        reason = self._last_error
        if reason is None and missing:
            reason = "Models not loaded"
        return {
            "ready": self.is_ready,
            "expected_features": expected_features,
            "inference_features": len(self._inference_feature_cols),
            "compatibility_mode": self._compatibility_mode,
            "loaded_models": sorted(self.models.keys()),
            "missing_models": missing,
            "imputation_features": len(imputation_values),
            "missing_imputation_features": missing_imputation[:10],
            "incompatible_models": self._incompatible_models,
            "model_feature_counts": self._model_feature_counts,
            "ensemble_ready": self._ensemble.is_ready if self._ensemble else False,
            "quantile_models": sum(len(q) for q in self._quantile_models.values()),
            "ood_ready": self._ood.is_ready if self._ood else False,
            "shap_ready": self._explainer.is_ready if self._explainer else False,
            "reason": reason,
            "warning": getattr(self, "_runtime_warning", None),
        }

    def get_metrics(self) -> dict:
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return {}

    def get_feature_importance(self) -> dict:
        imp_path = ARTIFACTS_DIR / "feature_importance.json"
        if imp_path.exists():
            return json.loads(imp_path.read_text())
        return {}

    def _sanitize_features(self, features: dict[str, Any]) -> tuple[dict[str, float], int]:
        """Normalize features to floats and count any invalid entries."""
        imputation_values = getattr(self, "_imputation_values", {})
        sanitized: dict[str, float] = {}
        imputed_count = 0
        for col in self._inference_feature_cols:
            fill_value = imputation_values.get(col, 0.0)
            if col not in features:
                sanitized[col] = fill_value
                imputed_count += 1
                continue
            raw_value = features[col]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = fill_value
                imputed_count += 1
            else:
                if not math.isfinite(value):
                    value = fill_value
                    imputed_count += 1
            sanitized[col] = value
        return sanitized, imputed_count

    @staticmethod
    def _latest_snapshots(
        snapshots: list[OddsSnapshot],
    ) -> tuple[list[OddsSnapshot], datetime | None]:
        """Keep only the most-recent capture per bookmaker+market+outcome."""
        return latest_snapshots(snapshots)

    @staticmethod
    def _build_odds_detail(
        snapshots: list[OddsSnapshot],
        home_name: str,
        away_name: str,
        odds_ts: datetime | None,
    ) -> dict:
        """Build per-book odds breakdown for transparency."""
        return build_odds_detail(snapshots, home_name, away_name, odds_ts)

    async def _load_prediction_snapshots(
        self,
        db: AsyncSession,
        game_id: int,
    ) -> tuple[list[OddsSnapshot], list[OddsSnapshot]]:
        core_result = await db.execute(
            select(OddsSnapshot)
            .where(
                OddsSnapshot.game_id == game_id,
                OddsSnapshot.market.in_(CORE_ODDS_MARKETS),
            )
            .order_by(OddsSnapshot.captured_at.desc())
        )
        core_snapshots = list(core_result.scalars().all())

        ranked_props = (
            select(
                OddsSnapshot.id.label("snapshot_id"),
                func.row_number()
                .over(
                    partition_by=(
                        OddsSnapshot.bookmaker,
                        OddsSnapshot.market,
                        OddsSnapshot.description,
                        OddsSnapshot.outcome_name,
                    ),
                    order_by=OddsSnapshot.captured_at.desc(),
                )
                .label("row_num"),
            )
            .where(
                OddsSnapshot.game_id == game_id,
                OddsSnapshot.market.in_(PROP_ODDS_MARKETS),
            )
            .subquery()
        )
        prop_result = await db.execute(
            select(OddsSnapshot)
            .join(ranked_props, OddsSnapshot.id == ranked_props.c.snapshot_id)
            .where(ranked_props.c.row_num == 1)
        )
        prop_snapshots = list(prop_result.scalars().all())

        return core_snapshots, prop_snapshots

    async def predict_game(
        self,
        game: Game,
        db: AsyncSession,
    ) -> dict | None:
        """Generate predictions for a single game using stored DB odds.

        Uses only the latest capture per bookmaker+market+outcome so
        predictions always reflect the most-recent odds.
        """
        if not self.is_ready:
            logger.warning("Models not loaded, cannot predict")
            return None

        # ── Use stored odds from DB (refreshed by polling jobs) ──
        core_snapshots, prop_snapshots = await self._load_prediction_snapshots(db, int(game.id))
        if not core_snapshots:
            logger.error(
                "Prediction skipped for game %s: no stored odds snapshots available",
                game.id,
            )
            return None
        stored_snapshots, odds_ts = self._latest_snapshots(core_snapshots)
        if odds_ts is None:
            logger.error(
                "Prediction skipped for game %s: stored odds snapshots have no capture timestamp",
                game.id,
            )
            return None
        if odds_ts.tzinfo is None:
            odds_ts_utc = odds_ts.replace(tzinfo=UTC)
        else:
            odds_ts_utc = odds_ts.astimezone(UTC)
        max_age_minutes = get_settings().odds_freshness_max_age_minutes
        odds_age_minutes = (datetime.now(UTC) - odds_ts_utc).total_seconds() / 60.0
        if odds_age_minutes > max_age_minutes:
            logger.error(
                "Prediction skipped for game %s: stale odds snapshot %.1f minutes old "
                "(limit=%d minutes)",
                game.id,
                odds_age_minutes,
                max_age_minutes,
            )
            return None

        home_name = game.home_team.name if game.home_team else ""
        away_name = game.away_team.name if game.away_team else ""
        odds_detail = self._build_odds_detail(stored_snapshots, home_name, away_name, odds_ts)

        features = await build_feature_vector(
            game,
            db,
            odds_snapshots=core_snapshots + prop_snapshots,
        )
        if features is None:
            return None

        sanitized_features, imputed_count = self._sanitize_features(features)
        if imputed_count:
            allowed_imputed = self._allowed_imputed_feature_count()
            if not self._can_tolerate_imputation(imputed_count):
                logger.error(
                    "Prediction skipped for game %s: %d/%d features required imputation (limit=%d)",
                    game.id,
                    imputed_count,
                    len(self._inference_feature_cols),
                    allowed_imputed,
                )
                return None

            logger.warning(
                "Prediction for game %s used saved imputation for %d/%d features",
                game.id,
                imputed_count,
                len(self._inference_feature_cols),
            )

        X = np.array([[sanitized_features[c] for c in self._inference_feature_cols]])
        home_fg, away_fg, home_1h, away_1h = self._predict_scores(X, f"game {game.id}")

        fg_margin = home_fg - away_fg
        h1_margin = home_1h - away_1h

        fg_prob = _margin_to_prob(
            fg_margin,
            coef=self._calibration.get("calibration_fg_coef"),
            intercept=self._calibration.get("calibration_fg_intercept"),
        )
        h1_prob = _margin_to_prob(
            h1_margin,
            coef=self._calibration.get("calibration_1h_coef"),
            intercept=self._calibration.get("calibration_1h_intercept"),
            scale=5.0,
        )

        # Extract opening market lines from stored odds for CLV tracking.
        # Spreads use betting convention (negative = home favorite).
        opening_spread = None
        opening_total = None
        if stored_snapshots:
            mkt_spreads = [
                float(cast(Any, s.point))
                for s in stored_snapshots
                if cast(Any, s.market) == "spreads"
                and s.point is not None
                and cast(Any, s.outcome_name) == home_name
            ]
            mkt_totals = [
                float(cast(Any, s.point))
                for s in stored_snapshots
                if cast(Any, s.market) == "totals" and s.point is not None
            ]
            if mkt_spreads:
                opening_spread = round(float(np.mean(mkt_spreads)), 1)
            if mkt_totals:
                opening_total = round(float(np.mean(mkt_totals)), 1)

        # ── 1H opening lines ──
        opening_h1_spread = None
        opening_h1_total = None
        if stored_snapshots:
            mkt_h1_spreads = [
                float(cast(Any, s.point))
                for s in stored_snapshots
                if cast(Any, s.market) == "spreads_h1"
                and s.point is not None
                and cast(Any, s.outcome_name) == home_name
            ]
            mkt_h1_totals = [
                float(cast(Any, s.point))
                for s in stored_snapshots
                if cast(Any, s.market) == "totals_h1" and s.point is not None
            ]
            if mkt_h1_spreads:
                opening_h1_spread = round(float(np.mean(mkt_h1_spreads)), 1)
            if mkt_h1_totals:
                opening_h1_total = round(float(np.mean(mkt_h1_totals)), 1)

        # Embed 1H opening lines in odds_detail for downstream use
        odds_detail["opening_h1_spread"] = opening_h1_spread
        odds_detail["opening_h1_total"] = opening_h1_total

        prediction_payload = {
            "predicted_home_fg": round(home_fg, 1),
            "predicted_away_fg": round(away_fg, 1),
            "predicted_home_1h": round(home_1h, 1),
            "predicted_away_1h": round(away_1h, 1),
            "fg_spread": round(fg_margin, 1),
            "fg_total": round(home_fg + away_fg, 1),
            "fg_home_ml_prob": round(fg_prob, 3),
            "h1_spread": round(h1_margin, 1),
            "h1_total": round(home_1h + away_1h, 1),
            "h1_home_ml_prob": round(h1_prob, 3),
            "opening_spread": opening_spread,
            "opening_total": opening_total,
            "odds_detail": odds_detail,
        }

        # ── Quantile confidence intervals ────────────────────────
        quantiles = self._predict_quantiles(X)
        if quantiles:
            ci = {}
            for name in ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]:
                if name in quantiles:
                    ci[name] = quantiles[name]
            if ci:
                # Derive total/spread intervals from component intervals
                if "model_home_fg" in ci and "model_away_fg" in ci:
                    ci["fg_total"] = {
                        "q10": round(ci["model_home_fg"]["q10"] + ci["model_away_fg"]["q10"], 1),
                        "q90": round(ci["model_home_fg"]["q90"] + ci["model_away_fg"]["q90"], 1),
                    }
                prediction_payload["confidence_intervals"] = ci

        # ── OOD detection ────────────────────────────────────────
        if self._ood and self._ood.is_ready:
            ood_dist, is_ood = self._ood.score(X)
            prediction_payload["ood_score"] = round(ood_dist, 2)
            prediction_payload["ood_flag"] = is_ood
            if is_ood:
                logger.warning(
                    "Game %s flagged as out-of-distribution (score=%.2f)",
                    game.id,
                    ood_dist,
                )

        # ── SHAP explanation (top drivers) ───────────────────────
        if self._explainer and self._explainer.is_ready:
            explanation = self._explainer.explain_prediction(X, "model_home_fg")
            if explanation:
                prediction_payload["explanation"] = {
                    "base_value": explanation["base_value"],
                    "top_drivers": explanation["top_drivers"],
                }
        if not prediction_has_valid_score_payload(SimpleNamespace(**prediction_payload)):
            logger.error(
                "Prediction skipped for game %s: generated payload failed integrity validation",
                game.id,
            )
            return None
        return prediction_payload

    async def predict_and_store(
        self,
        game: Game,
        db: AsyncSession,
    ) -> Prediction | None:
        """Predict and persist to database."""
        pred_dict = await self.predict_game(game, db)
        if pred_dict is None:
            return None

        model_version = await self._resolve_model_version(db)

        existing_result = await db.execute(
            select(Prediction).where(
                Prediction.game_id == game.id,
                Prediction.model_version == model_version,
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing is not None:
            existing.predicted_home_fg = pred_dict["predicted_home_fg"]
            existing.predicted_away_fg = pred_dict["predicted_away_fg"]
            existing.predicted_home_1h = pred_dict["predicted_home_1h"]
            existing.predicted_away_1h = pred_dict["predicted_away_1h"]
            existing.fg_spread = pred_dict["fg_spread"]
            existing.fg_total = pred_dict["fg_total"]
            existing.fg_home_ml_prob = pred_dict["fg_home_ml_prob"]
            existing.h1_spread = pred_dict["h1_spread"]
            existing.h1_total = pred_dict["h1_total"]
            existing.h1_home_ml_prob = pred_dict["h1_home_ml_prob"]
            existing.opening_spread = pred_dict.get("opening_spread")  # type: ignore[assignment]
            existing.opening_total = pred_dict.get("opening_total")  # type: ignore[assignment]
            existing.odds_sourced = pred_dict.get("odds_detail")  # type: ignore[assignment]
            existing.predicted_at = datetime.now(UTC).replace(tzinfo=None)  # type: ignore[assignment]
            await db.commit()
            await db.refresh(existing)
            logger.info("Prediction updated for game %d", game.id)
            return existing

        prediction = Prediction(
            game_id=game.id,
            model_version=model_version,
            predicted_home_fg=pred_dict["predicted_home_fg"],
            predicted_away_fg=pred_dict["predicted_away_fg"],
            predicted_home_1h=pred_dict["predicted_home_1h"],
            predicted_away_1h=pred_dict["predicted_away_1h"],
            fg_spread=pred_dict["fg_spread"],
            fg_total=pred_dict["fg_total"],
            fg_home_ml_prob=pred_dict["fg_home_ml_prob"],
            h1_spread=pred_dict["h1_spread"],
            h1_total=pred_dict["h1_total"],
            h1_home_ml_prob=pred_dict["h1_home_ml_prob"],
            opening_spread=pred_dict.get("opening_spread"),
            opening_total=pred_dict.get("opening_total"),
            odds_sourced=pred_dict.get("odds_detail"),
            predicted_at=datetime.now(UTC).replace(tzinfo=None),
        )
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)
        logger.info("Prediction stored for game %d", game.id)
        return prediction

    async def predict_upcoming(self, db: AsyncSession) -> list[Prediction]:
        """Predict all upcoming (NS) games and store.

        Uses stored odds from the database (refreshed by background
        polling jobs) instead of hitting the Odds API at prediction time.
        """
        result = await db.execute(
            select(Game)
            .options(
                selectinload(Game.home_team),
                selectinload(Game.away_team),
                selectinload(Game.referees),
            )
            .where(
                Game.status == "NS",
                Game.odds_api_id.is_not(None),
            )
            .order_by(Game.commence_time)
        )
        games = result.scalars().all()
        if not games:
            return []

        predictions = []
        for game in games:
            pred = await self.predict_and_store(game, db)
            if pred:
                predictions.append(pred)
            else:
                logger.warning(
                    "Skipped prediction for game %d (%s vs %s)",
                    game.id,
                    getattr(game.away_team, "name", "?"),
                    getattr(game.home_team, "name", "?"),
                )
        if len(predictions) < len(games):
            logger.warning(
                "predict_upcoming: %d/%d games predicted (%d skipped)",
                len(predictions),
                len(games),
                len(games) - len(predictions),
            )
        return predictions
