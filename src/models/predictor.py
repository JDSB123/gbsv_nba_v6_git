import json
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import xgboost as xgb
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.db.models import Game, ModelRegistry, OddsSnapshot, Prediction
from src.models.features import build_feature_vector, get_feature_columns
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
    def __init__(self) -> None:
        self.feature_cols = get_feature_columns()
        self._inference_feature_cols = list(self.feature_cols)
        self.models: dict[str, xgb.XGBRegressor] = {}
        self._imputation_values: dict[str, float] = {}
        self._model_feature_counts: dict[str, int] = {}
        self._incompatible_models: dict[str, int] = {}
        self._last_error: str | None = None
        self._compatibility_mode = False
        self._calibration: dict[str, float] = {}
        self.model_version = MODEL_VERSION
        self._load_models()
        self._load_calibration()
        self._load_imputation()
        self._run_model_smoke_test()

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
                    loop.create_task(
                        send_alert(
                            "Model Compatibility Mode Active",
                            f"Models expect {model_feature_count} features but code provides "
                            f"{expected_features}. Running in degraded mode — retrain recommended.",
                            "warning",
                        )
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

        # Half-time scores should never exceed the full-game projection.
        home_1h = min(home_1h, home_fg)
        away_1h = min(away_1h, away_fg)
        return home_fg, away_fg, home_1h, away_1h

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

        self._last_error = "Model smoke test failed: implausible prediction payload"
        logger.error(self._last_error)
        self.models = {}

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
            "reason": reason,
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
        """Keep only the most-recent capture per bookmaker+market+outcome.

        Returns the deduplicated list and the newest ``captured_at``.
        """
        best: dict[tuple[str, str, str], OddsSnapshot] = {}
        newest: datetime | None = None
        for s in snapshots:
            key = (
                cast(Any, s.bookmaker),
                cast(Any, s.market),
                cast(Any, s.outcome_name),
            )
            existing = best.get(key)
            s_ts = cast(Any, s.captured_at)
            if existing is None or s_ts > cast(Any, existing.captured_at):
                best[key] = s
            if newest is None or s_ts > newest:
                newest = s_ts
        return list(best.values()), newest

    @staticmethod
    def _build_odds_detail(
        snapshots: list[OddsSnapshot],
        home_name: str,
        away_name: str,
        odds_ts: datetime | None,
    ) -> dict:
        """Build per-book odds breakdown for transparency."""
        books: dict[str, dict[str, Any]] = defaultdict(dict)
        for s in snapshots:
            bk = cast(Any, s.bookmaker)
            mkt = cast(Any, s.market)
            outcome = cast(Any, s.outcome_name)
            price = cast(Any, s.price)
            point = cast(Any, s.point)
            if mkt == "spreads" and outcome == home_name and point is not None:
                books[bk]["spread"] = float(point)
                books[bk]["spread_price"] = int(price) if price else None
            elif mkt == "totals" and point is not None:
                if outcome in ("Over", home_name):
                    books[bk]["total"] = float(point)
                    books[bk]["total_price"] = int(price) if price else None
            elif mkt == "h2h":
                if outcome == home_name:
                    books[bk]["home_ml"] = int(price) if price else None
                elif outcome == away_name:
                    books[bk]["away_ml"] = int(price) if price else None
            # ── 1H markets ──
            elif mkt == "spreads_h1" and outcome == home_name and point is not None:
                books[bk]["spread_h1"] = float(point)
                books[bk]["spread_h1_price"] = int(price) if price else None
            elif mkt == "totals_h1" and point is not None:
                if outcome in ("Over", home_name):
                    books[bk]["total_h1"] = float(point)
                    books[bk]["total_h1_price"] = int(price) if price else None
            elif mkt == "h2h_h1":
                if outcome == home_name:
                    books[bk]["home_ml_h1"] = int(price) if price else None
                elif outcome == away_name:
                    books[bk]["away_ml_h1"] = int(price) if price else None
        return {
            "captured_at": odds_ts.isoformat() + "Z" if odds_ts else None,
            "books": dict(books),
        }

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
        core_snapshots, prop_snapshots = await self._load_prediction_snapshots(db, game.id)
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
                    "Prediction skipped for game %s: %d/%d features required imputation "
                    "(limit=%d)",
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
