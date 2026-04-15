"""Adaptive Card builder for Teams notifications."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.models.versioning import MODEL_VERSION
from src.notifications._helpers import (
    _CST,
    MIN_EDGE,
    Pick,
    _edge_color,
    _fire_emojis,
)
from src.notifications._picks import extract_picks


def _pick_row(pick: Pick) -> dict:
    """Build a ColumnSet for a single pick row."""
    fires = _fire_emojis(pick.edge)
    odds_tag = f" ({pick.odds})" if pick.odds else ""
    if pick.away_record and pick.home_record:
        matchup_line = (
            f"{pick.matchup.split(' @ ')[0]} ({pick.away_record}) @ "
            f"{pick.matchup.split(' @ ')[1]} ({pick.home_record})"
        )
    else:
        matchup_line = pick.matchup
    detail_line = (
        f"{pick.segment} {pick.market_type} · Line: {pick.market_line} · Model: {pick.model_scores}"
    )

    items: list[dict] = [
        {
            "type": "TextBlock",
            "text": f"**{pick.label}**{odds_tag}{fires}",
            "wrap": True,
            "weight": "Bolder",
        },
        {
            "type": "TextBlock",
            "text": f"{pick.time_cst} · {matchup_line}",
            "size": "Small",
            "isSubtle": True,
            "spacing": "None",
            "wrap": True,
        },
        {
            "type": "TextBlock",
            "text": detail_line,
            "size": "Small",
            "isSubtle": True,
            "spacing": "None",
            "wrap": True,
        },
    ]
    if pick.rationale:
        items.append(
            {
                "type": "TextBlock",
                "text": f"_{pick.rationale}_",
                "size": "Small",
                "isSubtle": True,
                "spacing": "None",
                "wrap": True,
            }
        )

    return {
        "type": "ColumnSet",
        "separator": True,
        "spacing": "Medium",
        "columns": [
            {"type": "Column", "width": "stretch", "items": items},
            {
                "type": "Column",
                "width": "auto",
                "verticalContentAlignment": "Center",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"**{pick.edge:.1f}**",
                        "size": "Large",
                        "weight": "Bolder",
                        "color": _edge_color(pick.edge),
                        "horizontalAlignment": "Right",
                    }
                ],
            },
        ],
    }


def _odds_source_block(odds_sourced: dict | None) -> list[dict]:
    """Build Adaptive Card elements showing per-book odds breakdown."""
    if not odds_sourced:
        return []
    books = odds_sourced.get("books", {})
    if not books:
        return []
    parts: list[str] = []
    for bk, lines in sorted(books.items()):
        pieces = []
        if "spread" in lines:
            price_str = f" ({lines['spread_price']:+d})" if lines.get("spread_price") else ""
            pieces.append(f"Sprd {lines['spread']:+.1f}{price_str}")
        if "total" in lines:
            price_str = f" ({lines['total_price']:+d})" if lines.get("total_price") else ""
            pieces.append(f"O/U {lines['total']:.1f}{price_str}")
        if "home_ml" in lines:
            pieces.append(f"ML {lines['home_ml']:+d}")
        if "spread_h1" in lines:
            price_str = f" ({lines['spread_h1_price']:+d})" if lines.get("spread_h1_price") else ""
            pieces.append(f"1H Sprd {lines['spread_h1']:+.1f}{price_str}")
        if "total_h1" in lines:
            price_str = f" ({lines['total_h1_price']:+d})" if lines.get("total_h1_price") else ""
            pieces.append(f"1H O/U {lines['total_h1']:.1f}{price_str}")
        if pieces:
            parts.append(f"**{bk}**: {' · '.join(pieces)}")
    if not parts:
        return []
    ts_raw = odds_sourced.get("captured_at", "")
    ts_display = ""
    if ts_raw:
        try:
            dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(_CST)
            ts_display = f" (as of {dt.strftime('%I:%M %p').lstrip('0')} CT)"
        except Exception:
            pass
    return [
        {
            "type": "TextBlock",
            "text": f"📊 {' | '.join(parts)}{ts_display}",
            "size": "Small",
            "isSubtle": True,
            "wrap": True,
            "spacing": "None",
        }
    ]


def build_teams_card(
    predictions_with_games: list[tuple[Any, Any]],
    max_games: int,
    odds_pulled_at: datetime | None = None,
    min_edge: float = MIN_EDGE,
    download_url: str | None = None,
    csv_download_url: str | None = None,
) -> dict[str, Any]:
    """Build a pick-focused Adaptive Card sorted by edge."""
    now_cst = datetime.now(_CST)
    now_str = now_cst.strftime("%Y-%m-%d %I:%M %p CT")
    odds_ts = (
        odds_pulled_at.astimezone(_CST).strftime("%Y-%m-%d %I:%M %p CT")
        if odds_pulled_at
        else now_str
    )

    all_picks: list[Pick] = []
    game_ids: set[int] = set()
    odds_by_game: dict[int, dict] = {}
    game_labels: dict[int, str] = {}
    for pred, game in predictions_with_games:
        gid = getattr(game, "id", id(game))
        game_ids.add(gid)
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge))
        sourced = getattr(pred, "odds_sourced", None)
        if sourced:
            odds_by_game[gid] = sourced
        home = game.home_team.name if game.home_team else "Home"
        away = game.away_team.name if game.away_team else "Away"
        game_labels[gid] = f"{away} @ {home}"

    all_picks.sort(key=lambda p: -p.edge)

    n_games = len(game_ids)
    total_picks = len(all_picks)
    max_picks = max_games * 2
    display_picks = all_picks[:max_picks]
    remaining = total_picks - len(display_picks)

    body: list[dict] = [
        {
            "type": "TextBlock",
            "text": "🏀 **NBA Daily Slate**",
            "size": "Large",
            "weight": "Bolder",
            "wrap": True,
        },
        {
            "type": "TextBlock",
            "text": f"Model {MODEL_VERSION} · Odds pulled {odds_ts}",
            "wrap": True,
            "size": "Small",
            "isSubtle": True,
            "spacing": "None",
        },
        {
            "type": "TextBlock",
            "text": f"{n_games} games · {total_picks} qualified picks",
            "wrap": True,
            "size": "Medium",
            "weight": "Bolder",
            "spacing": "Small",
        },
        {
            "type": "ColumnSet",
            "spacing": "Small",
            "columns": [
                {
                    "type": "Column",
                    "width": "stretch",
                    "items": [
                        {"type": "TextBlock", "text": "**PICK**", "size": "Small", "isSubtle": True}
                    ],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": "**EDGE**",
                            "size": "Small",
                            "isSubtle": True,
                            "horizontalAlignment": "Right",
                        }
                    ],
                },
            ],
        },
    ]

    for pick in display_picks:
        body.append(_pick_row(pick))

    if remaining > 0:
        body.append(
            {
                "type": "TextBlock",
                "text": f"• {remaining} more picks — download full slate below",
                "size": "Small",
                "isSubtle": True,
                "spacing": "Medium",
                "wrap": True,
            }
        )

    if odds_by_game:
        body.append(
            {
                "type": "TextBlock",
                "text": "📊 **Odds Sources**",
                "size": "Small",
                "weight": "Bolder",
                "spacing": "Large",
                "separator": True,
            }
        )
        for gid, detail in odds_by_game.items():
            label = game_labels.get(gid, "")
            blocks = _odds_source_block(detail)
            if blocks and label:
                body.append(
                    {
                        "type": "TextBlock",
                        "text": f"**{label}**",
                        "size": "Small",
                        "spacing": "Small",
                        "wrap": True,
                    }
                )
                body.extend(blocks)

    card: dict[str, Any] = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }

    actions: list[dict[str, str]] = []
    if download_url:
        actions.append(
            {"type": "Action.OpenUrl", "title": "📊 View Full Slate", "url": download_url}
        )
    if csv_download_url:
        actions.append(
            {"type": "Action.OpenUrl", "title": "📥 Download CSV", "url": csv_download_url}
        )
    if actions:
        card["actions"] = actions

    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card,
            }
        ],
    }
