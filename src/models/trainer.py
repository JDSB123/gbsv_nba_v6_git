import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.db.models import Game, ModelRegistry
from src.models.features import (
    build_feature_vector,
    get_feature_columns,
    reset_elo_cache,
)
from src.models.versioning import MODEL_VERSION

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Target columns: what each model predicts
TARGETS = ["home_score_fg", "away_score_fg", "home_score_1h", "away_score_1h"]
MODEL_NAMES = ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]

# Baseline params (used when Optuna is skipped or as starting point)
DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "early_stopping_rounds": 30,
}

# Optuna search bounds
OPTUNA_N_TRIALS = 30


def _evaluate_promotion(metrics: dict[str, float], row_count: int) -> tuple[bool, str]:
    settings = get_settings()
    if row_count < settings.model_gate_min_rows:
        return False, f"insufficient rows: {row_count} < {settings.model_gate_min_rows}"

    checks = {
        "model_home_fg_mae": settings.model_gate_max_mae_fg,
        "model_away_fg_mae": settings.model_gate_max_mae_fg,
        "model_home_1h_mae": settings.model_gate_max_mae_1h,
        "model_away_1h_mae": settings.model_gate_max_mae_1h,
        "model_home_fg_rmse": settings.model_gate_max_rmse_fg,
        "model_away_fg_rmse": settings.model_gate_max_rmse_fg,
        "model_home_1h_rmse": settings.model_gate_max_rmse_1h,
        "model_away_1h_rmse": settings.model_gate_max_rmse_1h,
    }
    failures: list[str] = []
    for key, threshold in checks.items():
        value = metrics.get(key)
        if value is None:
            failures.append(f"{key} missing")
            continue
        if value > threshold:
            failures.append(f"{key}={value:.3f} > {threshold:.3f}")

    if failures:
        return False, "; ".join(failures)
    return True, "all gate checks passed"


def _optuna_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Optuna objective — minimise CV MAE."""
    params = {
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "early_stopping_rounds": 30,
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = model.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], preds))
    return float(np.mean(maes))


def _calibrate_probabilities(
    margins: np.ndarray,
    actuals: np.ndarray,
) -> tuple[float, float]:
    """Fit a logistic regression on predicted margin → actual W/L.

    Returns (coef, intercept) so the predictor can use them.
    """
    lr = LogisticRegression()
    lr.fit(margins.reshape(-1, 1), actuals)
    coef = float(np.ravel(lr.coef_)[0])
    intercept = float(np.ravel(lr.intercept_)[0])
    return coef, intercept


class ModelTrainer:
    def __init__(self, run_optuna: bool = True) -> None:
        self.feature_cols = get_feature_columns()
        self.models: dict[str, xgb.XGBRegressor] = {}
        self.run_optuna = run_optuna

    async def _sync_model_registry(
        self,
        db: AsyncSession,
        metrics: dict[str, Any],
        best_params_all: dict[str, dict],
        should_promote: bool,
        promotion_reason: str,
    ) -> None:
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.model_version == MODEL_VERSION)
        )
        current = result.scalar_one_or_none()

        now = datetime.utcnow()
        metrics_json = json.dumps(metrics)
        params_json = json.dumps(best_params_all)

        if current is None:
            current = ModelRegistry(
                model_version=MODEL_VERSION,
                created_at=now,
            )
            db.add(current)

        current.metrics_json = metrics_json  # type: ignore[assignment]
        current.params_json = params_json  # type: ignore[assignment]
        current.promotion_reason = promotion_reason  # type: ignore[assignment]

        if should_promote:
            active = await db.execute(
                select(ModelRegistry).where(
                    ModelRegistry.is_active.is_(True),
                    ModelRegistry.model_version != MODEL_VERSION,
                )
            )
            for row in active.scalars().all():
                row.is_active = False  # type: ignore[assignment]
                row.retired_at = now  # type: ignore[assignment]

            current.is_active = True  # type: ignore[assignment]
            current.promoted_at = now  # type: ignore[assignment]
            current.retired_at = None  # type: ignore[assignment]
        elif current.is_active is None:
            current.is_active = False

        await db.commit()

    async def _build_dataset(self, db: AsyncSession) -> pd.DataFrame:
        """Build training dataset from completed games."""
        # Reset Elo cache so it's rebuilt fresh for this training run
        reset_elo_cache()

        result = await db.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(
                Game.status == "FT",
                Game.home_score_fg.is_not(None),
                Game.away_score_fg.is_not(None),
            )
            .order_by(Game.commence_time)
        )
        games = result.scalars().all()
        logger.info("Building dataset from %d completed games", len(games))

        rows: list[dict[str, Any]] = []
        for game in games:
            features = await build_feature_vector(game, db)
            if features is None:
                continue
            row: dict[str, Any] = dict(features)
            row["home_score_fg"] = float(cast(Any, game.home_score_fg))
            row["away_score_fg"] = float(cast(Any, game.away_score_fg))
            home_score_1h = cast(Any, game.home_score_1h)
            away_score_1h = cast(Any, game.away_score_1h)
            row["home_score_1h"] = float(home_score_1h) if home_score_1h is not None else 0.0
            row["away_score_1h"] = float(away_score_1h) if away_score_1h is not None else 0.0
            row["commence_time"] = cast(Any, game.commence_time)
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info("Dataset shape: %s", df.shape)
        return df

    async def train(self, db: AsyncSession) -> dict[str, float]:
        """Train all 4 models with optional Optuna tuning. Returns metrics."""
        df = await self._build_dataset(db)
        if df.empty or len(df) < 50:
            logger.warning("Not enough data to train (%d rows)", len(df))
            return {}

        df = df.sort_values("commence_time").reset_index(drop=True)
        X: np.ndarray = np.asarray(
            df[self.feature_cols].fillna(-999.0).to_numpy(dtype=float)
        )  # sentinel, not 0
        metrics: dict[str, float] = {}
        best_params_all: dict[str, dict] = {}

        for target, model_name in zip(TARGETS, MODEL_NAMES, strict=True):
            y: np.ndarray = np.asarray(df[target].to_numpy(dtype=float))

            # ── Optuna hyperparameter search ────────────────────
            if self.run_optuna and len(df) >= 200:
                logger.info("Running Optuna for %s (%d trials)...", model_name, OPTUNA_N_TRIALS)
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial, _X=X, _y=y: _optuna_objective(trial, _X, _y),  # type: ignore
                    n_trials=OPTUNA_N_TRIALS,
                )
                best = study.best_params
                best["objective"] = "reg:squarederror"
                best["random_state"] = 42
                best["early_stopping_rounds"] = 30
                best_params_all[model_name] = best
                logger.info("%s best params: %s (MAE=%.2f)", model_name, best, study.best_value)
            else:
                best = dict(DEFAULT_XGB_PARAMS)
                best_params_all[model_name] = best

            # ── Cross-validation with early stopping ────────────
            tscv = TimeSeriesSplit(n_splits=5)
            maes, rmses = [], []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = xgb.XGBRegressor(**best)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                preds = model.predict(X_val)
                maes.append(mean_absolute_error(y_val, preds))
                rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

            # ── Final fit on all data (with 80/20 holdout for early stopping)
            split_idx = int(len(X) * 0.85)
            X_fit, X_hold = X[:split_idx], X[split_idx:]
            y_fit, y_hold = y[:split_idx], y[split_idx:]
            final_params = dict(best)
            model = xgb.XGBRegressor(**final_params)
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_hold, y_hold)],
                verbose=False,
            )
            self.models[model_name] = model

            # Save model
            model_path = ARTIFACTS_DIR / f"{model_name}.json"
            model.save_model(str(model_path))

            avg_mae = float(np.mean(maes))
            avg_rmse = float(np.mean(rmses))
            metrics[f"{model_name}_mae"] = avg_mae
            metrics[f"{model_name}_rmse"] = avg_rmse
            logger.info("%s — MAE: %.2f, RMSE: %.2f", model_name, avg_mae, avg_rmse)

        # ── Calibrated probability coefficients (Platt scaling) ──
        if "model_home_fg" in self.models and "model_away_fg" in self.models:
            home_preds = self.models["model_home_fg"].predict(X)
            away_preds = self.models["model_away_fg"].predict(X)
            fg_margins = home_preds - away_preds
            fg_actuals = (
                np.asarray(df["home_score_fg"].to_numpy(dtype=float))
                > np.asarray(df["away_score_fg"].to_numpy(dtype=float))
            ).astype(float)
            fg_coef, fg_intercept = _calibrate_probabilities(fg_margins, fg_actuals)
            metrics["calibration_fg_coef"] = fg_coef
            metrics["calibration_fg_intercept"] = fg_intercept

        if "model_home_1h" in self.models and "model_away_1h" in self.models:
            h1_home = self.models["model_home_1h"].predict(X)
            h1_away = self.models["model_away_1h"].predict(X)
            h1_margins = h1_home - h1_away
            h1_actuals = (
                np.asarray(df["home_score_1h"].to_numpy(dtype=float))
                > np.asarray(df["away_score_1h"].to_numpy(dtype=float))
            ).astype(float)
            h1_coef, h1_intercept = _calibrate_probabilities(h1_margins, h1_actuals)
            metrics["calibration_1h_coef"] = h1_coef
            metrics["calibration_1h_intercept"] = h1_intercept

        # Save metrics
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # Save best hyperparams
        params_path = ARTIFACTS_DIR / "best_params.json"
        params_path.write_text(json.dumps(best_params_all, indent=2))

        should_promote, promotion_reason = _evaluate_promotion(metrics, len(df))
        metrics["row_count"] = float(len(df))
        metrics["promoted"] = 1.0 if should_promote else 0.0

        # Save feature importance
        if self.models:
            first_model = next(iter(self.models.values()))
            importance = dict(
                zip(
                    self.feature_cols,
                    first_model.feature_importances_.tolist(),
                    strict=True,
                )
            )
            imp_path = ARTIFACTS_DIR / "feature_importance.json"
            imp_path.write_text(json.dumps(importance, indent=2))

        await self._sync_model_registry(
            db,
            metrics=metrics,
            best_params_all=best_params_all,
            should_promote=should_promote,
            promotion_reason=promotion_reason,
        )

        metrics_path.write_text(json.dumps(metrics, indent=2))

        logger.info(
            "Training complete. Models saved to %s. promoted=%s reason=%s",
            ARTIFACTS_DIR,
            should_promote,
            promotion_reason,
        )
        return metrics
