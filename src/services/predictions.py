from datetime import UTC, datetime
from typing import Any, cast

from src.config import Settings
from src.db.models import Game, Prediction
from src.db.repositories.predictions import PredictionRepository
from src.models.odds_utils import (
    american_to_prob,
    consensus_line,
    consensus_price,
    prob_to_american,
)
from src.models.predictor import Predictor
from src.services.prediction_integrity import prediction_has_valid_payload


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


# Backward-compat aliases used internally
_consensus = consensus_line
_consensus_price = consensus_price
_prob_to_american = prob_to_american
_american_to_prob = american_to_prob


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
        odds_sourced: dict[str, Any] = (
            pred.odds_sourced if isinstance(pred.odds_sourced, dict) else {}
        )
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
        fg_spread_edge = round(mkt_spread + fg_spread, 1) if mkt_spread is not None else None
        h1_spread_edge = round(h1_mkt_spread + h1_spread, 1) if h1_mkt_spread is not None else None
        fg_total_edge = round(fg_total - mkt_total, 1) if mkt_total is not None else None
        h1_total_edge = round(h1_total - h1_mkt_total, 1) if h1_mkt_total is not None else None

        # ML: compare model prob vs consensus implied prob
        home_ml_odds = _consensus_price(books, "home_ml")
        away_ml_odds = _consensus_price(books, "away_ml")

        # FG ML
        fg_ml_pick_home = fg_spread > 0
        fg_ml_prob = fg_home_ml_prob if fg_ml_pick_home else (1 - fg_home_ml_prob)
        fg_ml_odds_str = home_ml_odds if fg_ml_pick_home else away_ml_odds
        fg_mkt_prob = _american_to_prob(fg_ml_odds_str)
        fg_ml_prob_edge = (fg_ml_prob - fg_mkt_prob) if fg_mkt_prob is not None else None

        # H1 ML
        h1_home_ml_odds = _consensus_price(books, "home_ml_h1")
        h1_away_ml_odds = _consensus_price(books, "away_ml_h1")
        h1_ml_pick_home = h1_spread > 0
        h1_ml_prob = h1_home_ml_prob if h1_ml_pick_home else (1 - h1_home_ml_prob)
        h1_ml_odds_str = h1_home_ml_odds if h1_ml_pick_home else h1_away_ml_odds
        h1_mkt_prob = _american_to_prob(h1_ml_odds_str)
        h1_ml_prob_edge = (h1_ml_prob - h1_mkt_prob) if h1_mkt_prob is not None else None

        # ── Spread pick builder ─────────────────────────────────
        def _spread_market(
            model_val: float,
            edge_val: float | None,
            mkt_line: float | None,
            seg: str,
        ) -> dict:
            base: dict[str, Any] = {
                "prediction": model_val,
                "consensus_line": round(mkt_line, 1) if mkt_line is not None else None,
                "pick": None,
                "edge": 0.0,
                "side": None,
                "actionable": False,
                "rationale": None,
            }
            if edge_val is None or mkt_line is None:
                return base
            pick_home = edge_val > 0
            side = home_name if pick_home else away_name
            e = round(abs(edge_val), 1)
            line = mkt_line if pick_home else -mkt_line
            base.update(
                {
                    "pick": f"{side} {line:+.1f}",
                    "edge": e,
                    "side": side,
                    "actionable": e >= min_edge,
                    "rationale": (
                        f"Model: {home_name} by {model_val:+.1f} vs line {mkt_line:+.1f} "
                        f"→ {e:.1f}pt edge on {side}"
                    ),
                }
            )
            return base

        # ── Total pick builder ──────────────────────────────────
        def _total_market(
            model_val: float,
            edge_val: float | None,
            mkt_line: float | None,
            seg: str,
        ) -> dict:
            base: dict[str, Any] = {
                "prediction": model_val,
                "consensus_line": round(mkt_line, 1) if mkt_line is not None else None,
                "pick": None,
                "edge": 0.0,
                "direction": None,
                "actionable": False,
                "rationale": None,
            }
            if edge_val is None or mkt_line is None:
                return base
            direction = "OVER" if edge_val > 0 else "UNDER"
            e = round(abs(edge_val), 1)
            base.update(
                {
                    "pick": f"{direction} {mkt_line:.1f}",
                    "edge": e,
                    "direction": direction,
                    "actionable": e >= min_edge,
                    "rationale": (
                        f"Model total {model_val:.1f} vs line {mkt_line:.1f} "
                        f"→ {e:.1f}pt {direction.lower()}"
                    ),
                }
            )
            return base

        # ── ML pick builder ─────────────────────────────────────
        def _ml_market(
            home_prob: float,
            pick_home: bool,
            model_prob: float,
            market_prob: float | None,
            prob_edge: float | None,
            odds_str: str,
            model_spread: float,
        ) -> dict:
            side = home_name if pick_home else away_name
            base: dict[str, Any] = {
                "home_prob": round(home_prob, 3),
                "away_prob": round(1 - home_prob, 3),
                "pick": None,
                "edge": 0.0,
                "win_prob": round(model_prob, 3),
                "implied_prob": round(market_prob, 3) if market_prob is not None else None,
                "odds": odds_str or None,
                "actionable": False,
                "rationale": None,
            }
            if prob_edge is not None and prob_edge > 0.02:
                ml_pts = round(prob_edge * 33.3, 1)
                m_str = f"{market_prob:.0%}" if market_prob is not None else "N/A"
                base.update(
                    {
                        "pick": f"{side} ML",
                        "edge": ml_pts,
                        "actionable": ml_pts >= min_edge,
                        "rationale": (
                            f"Model projects {side} by {abs(model_spread):.1f}pts. "
                            f"Edge: {model_prob:.0%} win prob vs {m_str} implied ({odds_str or 'n/a'})"
                        ),
                    }
                )
            return base

        fg_sp = _spread_market(fg_spread, fg_spread_edge, mkt_spread, "FG")
        h1_sp = _spread_market(h1_spread, h1_spread_edge, h1_mkt_spread, "1H")
        fg_tp = _total_market(fg_total, fg_total_edge, mkt_total, "FG")
        h1_tp = _total_market(h1_total, h1_total_edge, h1_mkt_total, "1H")
        fg_ml = _ml_market(
            fg_home_ml_prob,
            fg_ml_pick_home,
            fg_ml_prob,
            fg_mkt_prob,
            fg_ml_prob_edge,
            fg_ml_odds_str,
            fg_spread,
        )
        h1_ml = _ml_market(
            h1_home_ml_prob,
            h1_ml_pick_home,
            h1_ml_prob,
            h1_mkt_prob,
            h1_ml_prob_edge,
            h1_ml_odds_str,
            h1_spread,
        )

        # Collect actionable edges for status
        all_markets = [fg_sp, h1_sp, fg_tp, h1_tp, fg_ml, h1_ml]
        edges = [m["edge"] for m in all_markets if m.get("actionable")]
        best_edge = max(edges) if edges else 0.0
        status = "actionable" if edges else "monitoring"

        # ── Consensus odds summary (replaces raw book dump) ─────
        consensus: dict[str, Any] = {}
        if books:
            for k in ("spread", "total", "spread_h1", "total_h1"):
                v = _consensus(books, k)
                if v is not None:
                    consensus[k] = v
            for k in (
                "spread_price",
                "total_price",
                "home_ml",
                "away_ml",
                "spread_h1_price",
                "total_h1_price",
                "home_ml_h1",
                "away_ml_h1",
            ):
                p = _consensus_price(books, k)
                if p:
                    consensus[k] = p

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
                "fg_spread": fg_sp,
                "fg_total": fg_tp,
                "fg_moneyline": fg_ml,
                "h1_spread": h1_sp,
                "h1_total": h1_tp,
                "h1_moneyline": h1_ml,
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
        if (
            not pred
            or not prediction_has_valid_payload(pred)
            or not self._prediction_has_fresh_odds(pred)
        ):
            return {"game": game, "pred": None}

        result = self.format_prediction(pred, game)
        return {"game": game, "pred": pred, "result": result}
