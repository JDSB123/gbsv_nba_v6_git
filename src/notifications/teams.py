"""Teams notifications — thin re-export shim.

All implementation lives in the private submodules:
  _helpers, _picks, _cards, _csv, _html, _text, _delivery

This module re-exports every public (and commonly-tested private) name
so existing ``from src.notifications.teams import X`` statements
continue to work unchanged.
"""
from __future__ import annotations

# ── helpers ────────────────────────────────────────────────────────
from src.notifications._helpers import (  # noqa: F401
    _ARTIFACTS_DIR,
    MIN_EDGE,
    Pick,
    _app_build_stamp,
    _CST,
    _edge_color,
    _fire_count,
    _fire_emojis,
    _fmt_time_cst,
    _get_model_modified_at,
    _team_record,
)

# ── odds utils (aliased for backward compat) ──────────────────────
from src.models.odds_utils import (  # noqa: F401
    american_to_prob as _american_to_prob,
    consensus_line as _consensus_line,
    consensus_price as _consensus_price,
    prob_to_american as _prob_to_american,
)

# ── picks ──────────────────────────────────────────────────────────
from src.notifications._picks import extract_picks  # noqa: F401

# ── cards ──────────────────────────────────────────────────────────
from src.notifications._cards import (  # noqa: F401
    _odds_source_block,
    _pick_row,
    build_teams_card,
)

# ── csv ────────────────────────────────────────────────────────────
from src.notifications._csv import build_slate_csv  # noqa: F401

# ── html ───────────────────────────────────────────────────────────
from src.notifications._html import (  # noqa: F401
    _confidence_badge,
    _edge_css_color,
    _esc,
    _pick_side_border,
    _segment_pill,
    build_html_slate,
)

# ── text ───────────────────────────────────────────────────────────
from src.notifications._text import (  # noqa: F401
    _format_game_line,
    build_teams_text,
)

# ── delivery ───────────────────────────────────────────────────────
from src.notifications._delivery import (  # noqa: F401
    _chunk_card_payload,
    _payload_size_bytes,
    send_alert,
    send_card_to_teams,
    send_card_via_graph,
    send_html_via_graph,
    send_text_to_teams,
    upload_csv_to_channel,
)

__all__ = [
    # helpers
    "MIN_EDGE", "Pick", "_app_build_stamp", "_CST", "_edge_color",
    "_fire_count", "_fire_emojis", "_fmt_time_cst", "_get_model_modified_at",
    "_team_record",
    # odds utils
    "_american_to_prob", "_consensus_line", "_consensus_price",
    "_prob_to_american",
    # picks
    "extract_picks",
    # cards
    "_odds_source_block", "_pick_row", "build_teams_card",
    # csv
    "build_slate_csv",
    # html
    "_confidence_badge", "_edge_css_color", "_esc", "_pick_side_border",
    "_segment_pill", "build_html_slate",
    # text
    "_format_game_line", "build_teams_text",
    # delivery
    "_chunk_card_payload", "_payload_size_bytes", "send_alert",
    "send_card_to_teams", "send_card_via_graph", "send_html_via_graph",
    "send_text_to_teams", "upload_csv_to_channel",
]
