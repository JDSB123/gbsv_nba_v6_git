from typing import Any

import numpy as np

from src.db.models import Game, Prediction
from src.db.repositories.models import ModelRepository
from src.services.prediction_integrity import (
    prediction_has_valid_score_payload,
    prediction_score_rank,
)


class ModelService:
    def __init__(self, repo: ModelRepository):
        self.repo = repo

    async def get_performance(self, limit: int = 200) -> dict[str, Any]:
        rows = await self.repo.get_finished_game_predictions()

        latest_per_game_version: dict[tuple[int, str], tuple[Prediction, Game]] = {}
        for pred, game in rows:
            key = (int(pred.game_id), str(pred.model_version))
            existing = latest_per_game_version.get(key)
            if existing is None or prediction_score_rank(pred) > prediction_score_rank(existing[0]):
                latest_per_game_version[key] = (pred, game)
            if len(latest_per_game_version) >= limit:
                break

        by_model: dict[str, Any] = {}
        for pred, game in latest_per_game_version.values():
            if not prediction_has_valid_score_payload(pred):
                continue
            if game.home_score_fg is None or game.away_score_fg is None:
                continue

            model_version = str(pred.model_version)
            slot = by_model.setdefault(
                model_version,
                {
                    "home_fg_abs": [],
                    "away_fg_abs": [],
                    "home_1h_abs": [],
                    "away_1h_abs": [],
                    "clv_spread": [],
                    "clv_total": [],
                    "count": 0,
                },
            )
            slot["count"] = int(slot["count"]) + 1

            slot["home_fg_abs"].append(
                abs(float(pred.predicted_home_fg) - float(game.home_score_fg))
            )
            slot["away_fg_abs"].append(
                abs(float(pred.predicted_away_fg) - float(game.away_score_fg))
            )

            if game.home_score_1h is not None and game.away_score_1h is not None:
                slot["home_1h_abs"].append(
                    abs(float(pred.predicted_home_1h) - float(game.home_score_1h))
                )
                slot["away_1h_abs"].append(
                    abs(float(pred.predicted_away_1h) - float(game.away_score_1h))
                )

            if pred.clv_spread is not None:
                slot["clv_spread"].append(float(pred.clv_spread))
            if pred.clv_total is not None:
                slot["clv_total"].append(float(pred.clv_total))

        def _avg(values: list[float]) -> float | None:
            if not values:
                return None
            return round(float(np.mean(values)), 4)

        performance: list[dict[str, Any]] = []
        for version, vals in by_model.items():
            home_fg = vals["home_fg_abs"]
            away_fg = vals["away_fg_abs"]
            home_1h = vals["home_1h_abs"]
            away_1h = vals["away_1h_abs"]
            performance.append(
                {
                    "model_version": version,
                    "sample_size": vals["count"],
                    "mae_home_fg": _avg(home_fg),
                    "mae_away_fg": _avg(away_fg),
                    "mae_home_1h": _avg(home_1h),
                    "mae_away_1h": _avg(away_1h),
                    "mae_fg_combined": _avg(home_fg + away_fg),
                    "mae_1h_combined": _avg(home_1h + away_1h),
                    "avg_clv_spread": _avg(vals["clv_spread"]),
                    "avg_clv_total": _avg(vals["clv_total"]),
                }
            )

        performance.sort(key=lambda row: row.get("sample_size", 0), reverse=True)
        return {"window": limit, "models": performance}
