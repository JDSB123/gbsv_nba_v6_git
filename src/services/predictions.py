from datetime import UTC, datetime
from typing import Any, cast

import numpy as np

from src.config import Settings
from src.db.models import Game, Prediction
from src.db.repositories.predictions import PredictionRepository
from src.models.predictor import Predictor
from src.services.prediction_integrity import prediction_has_valid_payload


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _parse_iso_utc(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


class PredictionService:
    def __init__(self, repo: PredictionRepository, predictor: Predictor, settings: Settings):
        self.repo = repo
        self.predictor = predictor
        self.settings = settings

    def format_prediction(self, pred: Prediction, game: Game) -> dict:
        home_name = (
            game.home_team.name if game.home_team is not None else f"Team {game.home_team_id}"
        )
        away_name = (
            game.away_team.name if game.away_team is not None else f"Team {game.away_team_id}"
        )
        fg_home_ml_prob = _as_float(pred.fg_home_ml_prob, 0.5)
        h1_home_ml_prob = _as_float(pred.h1_home_ml_prob, 0.5)
        return {
            "game_id": game.id,
            "odds_api_id": game.odds_api_id,
            "commence_time": game.commence_time.isoformat()
            if game.commence_time is not None
            else None,
            "away_team": away_name,
            "home_team": home_name,
            "predicted_scores": {
                "full_game": {"away": pred.predicted_away_fg, "home": pred.predicted_home_fg},
                "first_half": {"away": pred.predicted_away_1h, "home": pred.predicted_home_1h},
            },
            "markets": {
                "fg_spread": {"prediction": pred.fg_spread},
                "fg_total": {"prediction": pred.fg_total},
                "fg_moneyline": {
                    "home_prob": fg_home_ml_prob,
                    "away_prob": round(1 - fg_home_ml_prob, 3),
                },
                "h1_spread": {"prediction": pred.h1_spread},
                "h1_total": {"prediction": pred.h1_total},
                "h1_moneyline": {
                    "home_prob": h1_home_ml_prob,
                    "away_prob": round(1 - h1_home_ml_prob, 3),
                },
            },
            "model_version": pred.model_version,
            "predicted_at": pred.predicted_at.isoformat()
            if pred.predicted_at is not None
            else None,
            "odds_sourced": pred.odds_sourced,
            "clv": {
                "opening_spread": pred.opening_spread,
                "opening_total": pred.opening_total,
                "closing_spread": pred.closing_spread,
                "closing_total": pred.closing_total,
                "clv_spread": pred.clv_spread,
                "clv_total": pred.clv_total,
            },
        }

    def evaluate_odds_freshness(self, predictions: list[dict]) -> dict[str, Any]:
        max_age = self.settings.odds_freshness_max_age_minutes
        now_utc = datetime.now(UTC)
        ages_minutes = []
        missing_odds_sourced = 0
        missing_captured_at = 0

        for row in predictions:
            odds_sourced = row.get("odds_sourced")
            if not isinstance(odds_sourced, dict):
                missing_odds_sourced += 1
                continue
            captured_at = _parse_iso_utc(odds_sourced.get("captured_at"))
            if captured_at is None:
                missing_captured_at += 1
                continue
            ages_minutes.append((now_utc - captured_at).total_seconds() / 60.0)

        stale_count = sum(1 for age in ages_minutes if age > max_age)
        usable = len(ages_minutes)
        status = "warning" if missing_odds_sourced or missing_captured_at or stale_count else "ok"

        return {
            "status": status,
            "max_allowed_age_minutes": max_age,
            "evaluated_predictions": len(predictions),
            "usable_captured_at_count": usable,
            "missing_odds_sourced": missing_odds_sourced,
            "missing_captured_at": missing_captured_at,
            "stale_count": stale_count,
            "freshest_age_minutes": round(min(ages_minutes), 2) if ages_minutes else None,
            "stale_threshold_minutes": max_age,
            "stale_ratio": round(stale_count / usable, 3) if usable else None,
        }

    def _prediction_has_fresh_odds(self, pred_or_row: Prediction | dict[str, Any]) -> bool:
        if isinstance(pred_or_row, dict):
            odds_sourced = pred_or_row.get("odds_sourced")
        else:
            odds_sourced = getattr(pred_or_row, "odds_sourced", None)

        if not isinstance(odds_sourced, dict):
            return False

        captured_at = _parse_iso_utc(odds_sourced.get("captured_at"))
        if captured_at is None:
            return False

        age_minutes = (datetime.now(UTC) - captured_at).total_seconds() / 60.0
        return age_minutes <= self.settings.odds_freshness_max_age_minutes

    async def get_list_predictions(self):
        latest = await self.repo.get_latest_predictions_for_upcoming_games()
        game_ids = [int(cast(Any, p.game_id)) for p in latest if p.game_id is not None]
        games = await self.repo.get_games_with_teams(game_ids)
        game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

        output = []
        for pred in latest:
            if pred.game_id is None:
                continue
            if not prediction_has_valid_payload(pred):
                continue
            game = game_by_id.get(int(cast(Any, pred.game_id)))
            if game:
                output.append(self.format_prediction(pred, game))

        freshness = self.evaluate_odds_freshness(output)
        fresh_output = [row for row in output if self._prediction_has_fresh_odds(row)]
        freshness["filtered_out_non_fresh"] = len(output) - len(fresh_output)
        return {"predictions": fresh_output, "count": len(fresh_output), "freshness": freshness}

    async def get_slate_payload(self) -> tuple[list, Any]:
        latest = await self.repo.get_latest_predictions_for_upcoming_games()
        game_ids = [int(cast(Any, p.game_id)) for p in latest if p.game_id is not None]
        games = await self.repo.get_games_with_teams_and_stats(game_ids)
        game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

        rows = []
        for pred in latest:
            if pred.game_id is None:
                continue
            if not prediction_has_valid_payload(pred):
                continue
            if not self._prediction_has_fresh_odds(pred):
                continue
            game = game_by_id.get(int(cast(Any, pred.game_id)))
            if game:
                rows.append((pred, game))

        odds_pulled_at = await self.repo.get_latest_odds_pull_timestamp()
        return rows, odds_pulled_at

    async def get_prediction_detail(self, game_id: int):
        game = await self.repo.get_game_with_teams(game_id)
        if not game:
            return None

        pred = await self.repo.get_latest_prediction_for_game(game_id)
        if not pred or not prediction_has_valid_payload(pred) or not self._prediction_has_fresh_odds(pred):
            return {"game": game, "pred": None}

        result = self.format_prediction(pred, game)
        odds = await self.repo.get_recent_odds_snapshots(game_id, limit=50)

        if odds:
            spreads = [
                float(cast(Any, o.point))
                for o in odds
                if cast(Any, o.market) == "spreads" and o.point is not None
            ]
            totals = [
                float(cast(Any, o.point))
                for o in odds
                if cast(Any, o.market) == "totals" and o.point is not None
            ]

            if spreads:
                mkt_spread = float(np.mean(spreads))
                result["markets"]["fg_spread"]["market_line"] = mkt_spread
                # Fixed opposing sign conventions: pred.fg_spread (margin, positive=Home) and mkt_spread (betting, negative=Home favored)
                result["markets"]["fg_spread"]["edge"] = round(
                    _as_float(pred.fg_spread) + mkt_spread, 1
                )
            if totals:
                mkt_total = float(np.mean(totals))
                result["markets"]["fg_total"]["market_line"] = mkt_total
        return {"game": game, "pred": pred, "result": result}
