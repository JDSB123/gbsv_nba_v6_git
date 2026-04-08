"""Odds-processing utilities extracted from Predictor for reuse and testability."""

from collections import defaultdict
from datetime import datetime
from typing import Any, cast

from src.db.models import OddsSnapshot


# ── Probability / American-odds conversions ────────────────────────


def prob_to_american(prob: float) -> str:
    """Convert win probability (0-1) to American odds string."""
    if prob <= 0.01 or prob >= 0.99:
        return ""
    if prob >= 0.5:
        return f"{int(round(-(prob / (1 - prob)) * 100)):+d}"
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def american_to_prob(odds_str: str) -> float | None:
    """Convert American odds string to implied probability (0-1)."""
    if not odds_str:
        return None
    try:
        odds = float(odds_str.replace("+", ""))
        if odds == 0:
            return 0.5
        return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)
    except ValueError:
        return None


def consensus_line(books: dict, key: str) -> float | None:
    """Average a numeric field across all books."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def consensus_price(books: dict, key: str) -> str:
    """Average a price field across all books, return American odds string."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    return f"{int(round(sum(vals) / len(vals))):+d}" if vals else ""


# ── Snapshot deduplication ─────────────────────────────────────────


def latest_snapshots(
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


def build_odds_detail(
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
