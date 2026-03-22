import json
import logging
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xgboost as xgb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game, Prediction
from src.models.features import build_feature_vector, get_feature_columns

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_NAMES = ["model_home_fg", "model_away_fg", "model_home_1h", "model_away_1h"]
MODEL_VERSION = "v6.0.0"


def _margin_to_prob(margin: float, scale: float = 7.5) -> float:
    """Convert predicted point margin to win probability via logistic function."""
    return 1.0 / (1.0 + math.exp(-margin / scale))


def _odds_response_to_snapshots(odds_data: list[dict], game: Game) -> list:
    """Convert raw Odds API JSON into OddsSnapshot-like objects.

    Returns lightweight SimpleNamespace objects that quack like
    OddsSnapshot (same attribute names).  This keeps the feature
    builder's downstream code identical for both fresh and cached data.
    """
    now = datetime.utcnow()
    snapshots: list = []
    for event in odds_data:
        if event.get("id") != game.odds_api_id:
            continue
        for bookmaker in event.get("bookmakers", []):
            bk_name = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                for outcome in market.get("outcomes", []):
                    snapshots.append(
                        SimpleNamespace(
                            game_id=game.id,
                            bookmaker=bk_name,
                            market=market_key,
                            outcome_name=outcome["name"],
                            price=outcome["price"],
                            point=outcome.get("point"),
                            captured_at=now,
                        )
                    )
    return snapshots


class Predictor:
    def __init__(self) -> None:
        self.feature_cols = get_feature_columns()
        self.models: dict[str, xgb.XGBRegressor] = {}
        self._load_models()

    def _load_models(self) -> None:
        for name in MODEL_NAMES:
            path = ARTIFACTS_DIR / f"{name}.json"
            if path.exists():
                model = xgb.XGBRegressor()
                model.load_model(str(path))
                self.models[name] = model
                logger.info("Loaded model %s", name)
            else:
                logger.warning("Model file not found: %s", path)

    @property
    def is_ready(self) -> bool:
        return len(self.models) == len(MODEL_NAMES)

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

    async def predict_game(self, game: Game, db: AsyncSession) -> dict | None:
        """Generate predictions for a single game using *fresh* live odds.

        Always fetches the latest odds from The Odds API so predictions
        are never stale.  Falls back gracefully if the game has no
        ``odds_api_id`` or the fetch fails (uses empty odds in that case).
        """
        if not self.is_ready:
            logger.warning("Models not loaded, cannot predict")
            return None

        # ── Fetch fresh odds (single source of truth) ──────────
        fresh_snapshots: list = []
        if game.odds_api_id:
            try:
                from src.data.odds_client import OddsClient

                client = OddsClient()
                # Full-game odds
                fg_data = await client.fetch_odds()
                fresh_snapshots.extend(_odds_response_to_snapshots(fg_data, game))
                # 1st-half odds for this event
                h1_data = await client.fetch_event_odds(game.odds_api_id)
                if h1_data and h1_data.get("bookmakers"):
                    fresh_snapshots.extend(_odds_response_to_snapshots([h1_data], game))
            except Exception:
                logger.exception(
                    "Failed to fetch fresh odds for game %s; using empty odds",
                    game.odds_api_id,
                )

        features = await build_feature_vector(game, db, odds_snapshots=fresh_snapshots)
        if features is None:
            return None

        X = np.array([[features.get(c, 0.0) for c in self.feature_cols]])

        home_fg = float(self.models["model_home_fg"].predict(X)[0])
        away_fg = float(self.models["model_away_fg"].predict(X)[0])
        home_1h = float(self.models["model_home_1h"].predict(X)[0])
        away_1h = float(self.models["model_away_1h"].predict(X)[0])

        fg_margin = home_fg - away_fg
        h1_margin = home_1h - away_1h

        return {
            "predicted_home_fg": round(home_fg, 1),
            "predicted_away_fg": round(away_fg, 1),
            "predicted_home_1h": round(home_1h, 1),
            "predicted_away_1h": round(away_1h, 1),
            "fg_spread": round(fg_margin, 1),
            "fg_total": round(home_fg + away_fg, 1),
            "fg_home_ml_prob": round(_margin_to_prob(fg_margin), 3),
            "h1_spread": round(h1_margin, 1),
            "h1_total": round(home_1h + away_1h, 1),
            "h1_home_ml_prob": round(_margin_to_prob(h1_margin, scale=5.0), 3),
        }

    async def predict_and_store(
        self, game: Game, db: AsyncSession
    ) -> Prediction | None:
        """Predict and persist to database."""
        pred_dict = await self.predict_game(game, db)
        if pred_dict is None:
            return None

        prediction = Prediction(
            game_id=game.id,
            model_version=MODEL_VERSION,
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
            predicted_at=datetime.utcnow(),
        )
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)
        logger.info("Prediction stored for game %d", game.id)
        return prediction

    async def predict_upcoming(self, db: AsyncSession) -> list[Prediction]:
        """Predict all upcoming (NS) games and store."""
        result = await db.execute(
            select(Game).where(Game.status == "NS").order_by(Game.commence_time)
        )
        games = result.scalars().all()
        predictions = []
        for game in games:
            pred = await self.predict_and_store(game, db)
            if pred:
                predictions.append(pred)
        return predictions
