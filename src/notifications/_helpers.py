"""Shared helpers, constants, and the Pick dataclass for notifications."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.config import get_nba_avg_total
from src.config import get_settings as _get_settings
from src.models.versioning import MODEL_VERSION

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

# Minimum edge (in points) for a pick to qualify — from central config
MIN_EDGE = _get_settings().min_edge
EDGE_THRESHOLDS = _get_settings().edge_thresholds

_get_nba_avg_total = get_nba_avg_total  # backward-compat alias

_CST = ZoneInfo("US/Central")


def _get_model_modified_at() -> str:
    """Timestamp of the newest model artifact file."""
    try:
        latest = max(
            (f.stat().st_mtime for f in _ARTIFACTS_DIR.glob("model_*.json") if f.exists()),
            default=0,
        )
        if latest > 0:
            return datetime.fromtimestamp(latest, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return "unknown"


def _app_build_stamp() -> str:
    """Container / app build timestamp (set via env or fallback)."""
    return os.environ.get(
        "APP_BUILD_TIMESTAMP",
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def _fire_emojis(edge: float) -> str:
    """Return fire emojis proportional to edge strength."""
    if edge >= 9:
        return " 🔥🔥🔥🔥🔥"
    if edge >= 7:
        return " 🔥🔥🔥🔥"
    if edge >= 5:
        return " 🔥🔥🔥"
    if edge >= 3.5:
        return " 🔥🔥"
    return " 🔥"


def _fire_count(edge: float) -> int:
    """Return 1-5 fire level based on edge strength."""
    if edge >= 9:
        return 5
    if edge >= 7:
        return 4
    if edge >= 5:
        return 3
    if edge >= 3.5:
        return 2
    return 1


def _edge_color(edge: float) -> str:
    """Adaptive Card colour for an edge value."""
    if edge >= 7:
        return "Attention"
    if edge >= 4:
        return "Good"
    return "Accent"


def _fmt_time_cst(dt: datetime | None) -> str:
    """Format a datetime as 'YYYY-MM-DD 6:30 PM CT' for clarity and sortability."""
    if not dt:
        return "TBD"
    aware = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    ct = aware.astimezone(_CST)
    raw_time = ct.strftime("%I:%M %p").lstrip("0")
    return f"{ct.strftime('%Y-%m-%d')} {raw_time} CT"


def _team_record(team: Any) -> str:
    """Extract W-L record string from a team object."""
    stats = getattr(team, "season_stats", None)
    if stats:
        s = stats[0] if isinstance(stats, list) and stats else stats
        w = getattr(s, "wins", None)
        loss = getattr(s, "losses", None)
        if w is not None and loss is not None:
            return f"{w}-{loss}"
    w = getattr(team, "wins", None)
    loss = getattr(team, "losses", None)
    if w is not None and loss is not None:
        return f"{w}-{loss}"
    return ""


@dataclass(frozen=True)
class Pick:
    """A single actionable pick derived from a prediction."""

    label: str        # e.g. "UNDER 222.5" or "Celtics -3.5"
    edge: float       # positive value, higher = stronger
    time_cst: str     # "6:30 PM CT"
    matchup: str      # "Heat @ Celtics"
    segment: str      # "FG" or "1H"
    market_type: str  # "SPREAD", "TOTAL", "ML"
    market_line: str  # "-3.5", "222.5", "-160"
    model_scores: str  # "Celtics 112, Heat 108"
    home_record: str  # "42-18" or ""
    away_record: str  # "28-32" or ""
    confidence: int   # 1-5 fire count
    odds: str = ""    # American odds, e.g. "-110", "+150"
    rationale: str = ""  # Brief explanation of the edge
