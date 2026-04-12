"""Ensemble stacking layer for NBA prediction models.

Combines XGBoost base-model predictions with a LightGBM + Ridge
meta-learner to reduce variance and capture complementary signal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# LightGBM base-learner defaults (intentionally different from XGB
# to promote model diversity in the ensemble).
DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_samples": 10,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

MODEL_NAMES = ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]


class EnsembleStack:
    """Two-layer stacking ensemble.

    Layer 0: XGBoost (existing) + LightGBM base learners
    Layer 1: Ridge regression meta-learner blending L0 outputs
    """

    def __init__(self) -> None:
        self.lgb_models: dict[str, lgb.LGBMRegressor] = {}
        self.meta_models: dict[str, Ridge] = {}
        self._feature_cols: list[str] | None = None
        self._ready = False

    # ── Training ────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y_dict: dict[str, np.ndarray],
        xgb_models: dict[str, Any],
        valid_masks: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Train LightGBM base learners and Ridge meta-learners.

        Args:
            X: Full feature matrix (all rows, already imputed).
            y_dict: Mapping of model_name → target array (unmasked).
            xgb_models: Trained XGB models keyed by model_name.
            valid_masks: Boolean masks for valid target rows per pair.

        Returns:
            Metrics dict with ensemble MAE for each target.
        """
        # Store feature column names so predict() can wrap X in a DataFrame
        # and suppress "X does not have valid feature names" warnings.
        self._feature_cols = [f"f{i}" for i in range(X.shape[1])]

        metrics: dict[str, float] = {}

        for model_name in MODEL_NAMES:
            pair_key = "1h" if model_name.endswith("_1h") else "fg"
            mask = valid_masks[pair_key]
            X_valid = X[mask]
            y = y_dict[model_name]

            if len(y) < 100:
                logger.warning(
                    "Skipping ensemble for %s: insufficient rows (%d)",
                    model_name,
                    len(y),
                )
                continue

            # ── Train LightGBM base learner via TimeSeriesSplit ──
            lgb_model = lgb.LGBMRegressor(**DEFAULT_LGB_PARAMS)
            split_idx = int(len(X_valid) * 0.85)
            X_fit, X_hold = X_valid[:split_idx], X_valid[split_idx:]
            y_fit, y_hold = y[:split_idx], y[split_idx:]

            X_fit_df = pd.DataFrame(X_fit, columns=self._feature_cols)
            X_hold_df = pd.DataFrame(X_hold, columns=self._feature_cols)
            lgb_model.fit(
                X_fit_df,
                y_fit,
                eval_set=[(X_hold_df, y_hold)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            self.lgb_models[model_name] = lgb_model

            # ── Build OOF predictions for meta-learner ───────────
            tscv = TimeSeriesSplit(n_splits=5)
            oof_xgb = np.full(len(X_valid), np.nan)
            oof_lgb = np.full(len(X_valid), np.nan)

            for train_idx, val_idx in tscv.split(X_valid):
                # XGB OOF
                xgb_model = xgb_models[model_name]
                oof_xgb[val_idx] = xgb_model.predict(X_valid[val_idx])

                # LGB OOF (refit on fold)
                lgb_fold = lgb.LGBMRegressor(**DEFAULT_LGB_PARAMS)
                X_train_df = pd.DataFrame(X_valid[train_idx], columns=self._feature_cols)
                X_val_df = pd.DataFrame(X_valid[val_idx], columns=self._feature_cols)
                lgb_fold.fit(
                    X_train_df,
                    y[train_idx],
                    eval_set=[(X_val_df, y[val_idx])],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                oof_lgb[val_idx] = lgb_fold.predict(X_val_df)

            # Only use rows where both OOF predictions exist
            valid_oof = np.isfinite(oof_xgb) & np.isfinite(oof_lgb)
            if valid_oof.sum() < 50:
                logger.warning(
                    "Skipping meta-learner for %s: insufficient OOF rows (%d)",
                    model_name,
                    int(valid_oof.sum()),
                )
                continue

            meta_X = np.column_stack([oof_xgb[valid_oof], oof_lgb[valid_oof]])
            meta_y = y[valid_oof]

            meta = Ridge(alpha=1.0)
            meta.fit(meta_X, meta_y)
            self.meta_models[model_name] = meta

            # Ensemble MAE on held-out OOF
            meta_preds = meta.predict(meta_X)
            ensemble_mae = float(np.mean(np.abs(meta_preds - meta_y)))
            metrics[f"{model_name}_ensemble_mae"] = ensemble_mae
            logger.info(
                "%s ensemble MAE: %.2f (meta weights: XGB=%.3f, LGB=%.3f)",
                model_name,
                ensemble_mae,
                meta.coef_[0],
                meta.coef_[1],
            )

        self._ready = bool(self.meta_models)
        self._save()
        return metrics

    # ── Inference ───────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        model_name: str,
        xgb_pred: float,
    ) -> float | None:
        """Blend XGB + LGB predictions through the meta-learner.

        Returns None if ensemble is not available for this target.
        """
        if model_name not in self.lgb_models or model_name not in self.meta_models:
            return None

        try:
            X_df = pd.DataFrame(X, columns=self._feature_cols) if self._feature_cols else X
            lgb_pred = float(self.lgb_models[model_name].predict(X_df)[0])
            meta_X = np.array([[xgb_pred, lgb_pred]])
            return float(self.meta_models[model_name].predict(meta_X)[0])
        except Exception:
            logger.debug("Ensemble predict failed for %s", model_name, exc_info=True)
            return None

    # ── Persistence ─────────────────────────────────────────────

    def _save(self) -> None:
        for name, model in self.lgb_models.items():
            path = ARTIFACTS_DIR / f"{name}_lgb.txt"
            model.booster_.save_model(str(path))

        meta_data = {}
        for name, meta in self.meta_models.items():
            meta_data[name] = {
                "coef": meta.coef_.tolist(),
                "intercept": float(meta.intercept_),
            }
        if self._feature_cols:
            meta_data["_feature_cols"] = self._feature_cols
        meta_path = ARTIFACTS_DIR / "ensemble_meta.json"
        meta_path.write_text(json.dumps(meta_data, indent=2))
        logger.info("Ensemble artifacts saved to %s", ARTIFACTS_DIR)

    def load(self) -> bool:
        meta_path = ARTIFACTS_DIR / "ensemble_meta.json"
        if not meta_path.exists():
            return False

        try:
            meta_data = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        loaded = 0
        self._feature_cols = meta_data.pop("_feature_cols", None)
        for name in MODEL_NAMES:
            lgb_path = ARTIFACTS_DIR / f"{name}_lgb.txt"
            if not lgb_path.exists() or name not in meta_data:
                continue

            try:
                booster = lgb.Booster(model_file=str(lgb_path))
                lgb_model = lgb.LGBMRegressor(**DEFAULT_LGB_PARAMS)
                lgb_model._Booster = booster
                lgb_model.fitted_ = True
                lgb_model._n_features = booster.num_feature()
                lgb_model._n_features_in = booster.num_feature()
                self.lgb_models[name] = lgb_model
            except Exception:
                logger.warning("Failed to load LGB model %s", name, exc_info=True)
                continue

            meta = Ridge(alpha=1.0)
            meta.coef_ = np.array(meta_data[name]["coef"])
            meta.intercept_ = float(meta_data[name]["intercept"])
            meta.n_features_in_ = 2
            self.meta_models[name] = meta
            loaded += 1

        self._ready = loaded > 0
        if self._ready:
            logger.info("Loaded ensemble artifacts for %d models", loaded)
        return self._ready

    @property
    def is_ready(self) -> bool:
        return self._ready
