from datetime import UTC, datetime
from typing import Any, cast

from src.config import Settings
from src.db.models import Game, Prediction
from src.db.repositories.predictions import PredictionRepository
from src.models.predictor import Predictor
from src.services.prediction_integrity import prediction_has_valid_payload


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _consensus(books: dict, key: str) -> float | None:
    """Average a numeric field across all books."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _consensus_price(books: dict, key: str) -> str:
    """Average a price field across all books, return American odds string."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    return f"{int(round(sum(vals) / len(vals))):+d}" if vals else ""


def _prob_to_american(prob: float) -> str:
    if prob <= 0.01 or prob >= 0.99:
        return ""
    if prob >= 0.5:
        return f"{int(round(-(prob / (1 - prob)) * 100)):+d}"
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def _american_to_prob(odds_str: str) -> float | None:
    if not odds_str:
        return None
    try:
        odds = float(odds_str.replace("+", ""))
        if odds == 0:
            return 0.5
        return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)
    except ValueError:
        return None


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

        fg_spread = _as_float(pred.fg_spread)
        fg_total = _as_float(pred.fg_total)
        h1_spread = _as_float(pred.h1_spread)
        h1_total = _as_float(pred.h1_total)

        # ── Consensus lines from stored book data ───────────────
        odds_sourced = pred.odds_sourced if isinstance(pred.odds_sourced, dict) else {}
        books = odds_sourced.get("books", {})
        captured_at = odds_sourced.get("captured_at")

        mkt_spread = (
            _as_float(pred.opening_spread)
            if pred.opening_spread is not None
            else _consensus(books, "spread")
        )
        mkt_total = (
            _as_float(pred.opening_total)
            if pred.opening_total is not None
            else _consensus(books, "total")
        )
        h1_mkt_spread = (
            _as_float(odds_sourced.get("opening_h1_spread"))
            if odds_sourced.get("opening_h1_spread") is not None
            else _consensus(books, "spread_h1")
        )
        h1_mkt_total = (
            _as_float(odds_sourced.get("opening_h1_total"))
            if odds_sourced.get("opening_h1_total") is not None
            else _consensus(books, "total_h1")
        )

        min_edge = self.settings.min_edge

        # ── Edge calculations ───────────────────────────────────
        # Spread: edge = market_line + model_spread (+ = HOME better than market)
        fg_spread_edge = round(mkt_spread + fg_spread, 1) if mkt_spread else None
        h1_spread_edge = round(h1_mkt_spread + h1_spread, 1) if h1_mkt_spread else None
        fg_total_edge = round(fg_total - mkt_total, 1) if mkt_total else None
        h1_total_edge = round(h1_total - h1_mkt_total, 1) if h1_mkt_total else None

        # ML: compare model prob vs consensus implied prob
        home_ml_odds = _consensus_price(books, "home_ml")
        away_ml_odds = _consensus_price(books, "away_ml")
        ml_pick_home = fg_spread > 0
        ml_prob = fg_home_ml_prob if ml_pick_home else (1 - fg_home_ml_prob)
        ml_odds_str = home_ml_odds if ml_pick_home else away_ml_odds
        mkt_prob = _american_to_prob(ml_odds_str)
        ml_prob_edge = (ml_prob - mkt_prob) if mkt_prob is not None else None

        def _spread_pick(edge_val: float | None, mkt_line: float | None, seg: str) -> dict:
            if edge_val is None or mkt_line is None:
                return {}
            pick_home = edge_val > 0
            side = home_name if pick_home else away_name
            e = round(abs(edge_val), 1)
            line = mkt_line if pick_home else -mkt_line
            return {
                "pick": f"{side} {line:+.1f}",
                "edge": e,
                "side": side,
                "consensus_line": round(mkt_line, 1),
                "actionable": e >= min_edge,
            }

        def _total_pick(edge_val: float | None, mkt_line: float | None, seg: str) -> dict:
            if edge_val is None or mkt_line is None:
                return {}
            direction = "OVER" if edge_val > 0 else "UNDER"
            e = round(abs(edge_val), 1)
            return {
                "pick": f"{direction} {mkt_line:.1f}",
                "edge": e,
                "direction": direction,
                "consensus_line": round(mkt_line, 1),
                "actionable": e >= min_edge,
            }

        fg_sp = _spread_pick(fg_spread_edge, mkt_spread, "FG")
        h1_sp = _spread_pick(h1_spread_edge, h1_mkt_spread, "1H")
        fg_tp = _total_pick(fg_total_edge, mkt_total, "FG")
        h1_tp = _total_pick(h1_total_edge, h1_mkt_total, "1H")

        # ML pick
        ml_pick: dict[str, Any] = {}
        if ml_prob_edge is not None and ml_prob_edge > 0.02:
            ml_side = home_name if ml_pick_home else away_name
            ml_pts = round(ml_prob_edge * 33.3, 1)
            ml_pick = {
                "pick": f"{ml_side} ML",
                "edge": ml_pts,
                "win_prob": round(ml_prob, 3),
                "implied_prob": round(mkt_prob, 3) if mkt_prob else None,
                "odds": ml_odds_str,
                "actionable": ml_pts >= min_edge,
            }

        # Collect actionable edges for status
        edges = [
            d.get("edge", 0)
            for d in (fg_sp, h1_sp, fg_tp, h1_tp, ml_pick)
            if d.get("actionable")
        ]
        best_edge = max(edges) if edges else 0.0
        status = "actionable" if edges else "monitoring"

        # ── Consensus odds summary (replaces raw book dump) ─────
        consensus = {}
        if books:
            for k in ("spread", "total", "spread_h1", "total_h1"):
                v = _consensus(books, k)
                if v is not None:
                    consensus[k] = v
            for k in ("spread_price", "total_price", "home_ml", "away_ml",
                       "spread_h1_price", "total_h1_price", "home_ml_h1", "away_ml_h1"):
                v = _consensus_price(books, k)
                if v:
                    consensus[k] = v

        return {
            "game_id": game.id,
            "odds_api_id": game.odds_api_id,
            "commence_time": game.commence_time.isoformat()
            if game.commence_time is not None
            else None,
            "home_team": home_name,
            "away_team": away_name,
            "status": status,
            "best_edge": best_edge,
            "predicted_scores": {
                "full_game": {"home": pred.predicted_home_fg, "away": pred.predicted_away_fg},
                "first_half": {"home": pred.predicted_home_1h, "away": pred.predicted_away_1h},
            },
            "markets": {
                "fg_spread": {"prediction": pred.fg_spread, **fg_sp},
                "fg_total": {"prediction": pred.fg_total, **fg_tp},
                "fg_moneyline": {
                    "home_prob": fg_home_ml_prob,
                    "away_prob": round(1 - fg_home_ml_prob, 3),
                    **ml_pick,
                },
                "h1_spread": {"prediction": pred.h1_spread, **h1_sp},
                "h1_total": {"prediction": pred.h1_total, **h1_tp},
                "h1_moneyline": {
                    "home_prob": h1_home_ml_prob,
                    "away_prob": round(1 - h1_home_ml_prob, 3),
                },
            },
            "model_version": pred.model_version,
            "predicted_at": pred.predicted_at.isoformat()
            if pred.predicted_at is not None
            else None,
            "odds": {
                "captured_at": captured_at,
                "book_count": len(books),
                "consensus": consensus,
            },
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
            odds_block = row.get("odds") or row.get("odds_sourced")
            if not isinstance(odds_block, dict):
                missing_odds_sourced += 1
                continue
            captured_at = _parse_iso_utc(odds_block.get("captured_at"))
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
            odds_sourced = pred_or_row.get("odds") or pred_or_row.get("odds_sourced")
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
        return {"game": game, "pred": pred, "result": result}
