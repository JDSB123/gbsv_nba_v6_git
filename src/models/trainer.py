import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game
from src.models.features import build_feature_vector, get_feature_columns

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "v6.0.0"

# Target columns: what each model predicts
TARGETS = ["home_score_fg", "away_score_fg", "home_score_1h", "away_score_1h"]
MODEL_NAMES = ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


class ModelTrainer:
    def __init__(self) -> None:
        self.feature_cols = get_feature_columns()
        self.models: dict[str, xgb.XGBRegressor] = {}

    async def _build_dataset(self, db: AsyncSession) -> pd.DataFrame:
        """Build training dataset from completed games."""
        result = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                Game.home_score_fg.is_not(None),
                Game.away_score_fg.is_not(None),
            )
            .order_by(Game.commence_time)
        )
        games = result.scalars().all()
        logger.info("Building dataset from %d completed games", len(games))

        rows = []
        for game in games:
            features = await build_feature_vector(game, db)
            if features is None:
                continue
            features["home_score_fg"] = float(game.home_score_fg)
            features["away_score_fg"] = float(game.away_score_fg)
            features["home_score_1h"] = float(game.home_score_1h or 0)
            features["away_score_1h"] = float(game.away_score_1h or 0)
            features["commence_time"] = game.commence_time
            rows.append(features)

        df = pd.DataFrame(rows)
        logger.info("Dataset shape: %s", df.shape)
        return df

    async def train(self, db: AsyncSession) -> dict[str, float]:
        """Train all 4 models. Returns metrics dict."""
        df = await self._build_dataset(db)
        if df.empty or len(df) < 50:
            logger.warning("Not enough data to train (%d rows)", len(df))
            return {}

        df = df.sort_values("commence_time").reset_index(drop=True)
        X = df[self.feature_cols].fillna(0).values
        metrics: dict[str, float] = {}

        for target, model_name in zip(TARGETS, MODEL_NAMES, strict=True):
            y = df[target].values
            model = xgb.XGBRegressor(**XGB_PARAMS)

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            maes, rmses = [], []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                maes.append(mean_absolute_error(y_val, preds))
                rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

            # Final fit on all data
            model.fit(X, y, verbose=False)
            self.models[model_name] = model

            # Save model
            model_path = ARTIFACTS_DIR / f"{model_name}.json"
            model.save_model(str(model_path))

            avg_mae = float(np.mean(maes))
            avg_rmse = float(np.mean(rmses))
            metrics[f"{model_name}_mae"] = avg_mae
            metrics[f"{model_name}_rmse"] = avg_rmse
            logger.info("%s — MAE: %.2f, RMSE: %.2f", model_name, avg_mae, avg_rmse)

        # Save metrics
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # Save feature importance
        if self.models:
            first_model = next(iter(self.models.values()))
            importance = dict(
                zip(self.feature_cols, first_model.feature_importances_.tolist(), strict=True)
            )
            imp_path = ARTIFACTS_DIR / "feature_importance.json"
            imp_path.write_text(json.dumps(importance, indent=2))

        logger.info("Training complete. Models saved to %s", ARTIFACTS_DIR)
        return metrics
