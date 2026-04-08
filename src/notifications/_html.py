"""HTML slate builder for NBA pick tables posted to Teams."""

from __future__ import annotations

import html as html_mod
from datetime import UTC, datetime
from typing import Any

from src.models.versioning import MODEL_VERSION
from src.notifications._helpers import MIN_EDGE, Pick, _CST
from src.notifications._picks import extract_picks


# ── Small CSS/HTML helpers ─────────────────────────────────────────


def _esc(val: object) -> str:
    return html_mod.escape(str(val))


def _edge_css_color(edge: float) -> str:
    if edge >= 7:
        return "#16a34a"
    if edge >= 5:
        return "#ca8a04"
    if edge >= 3:
        return "#2563eb"
    return "#6b7280"


def _confidence_badge(fires: int) -> str:
    """Inline-styled confidence badge from fire count (1-5)."""
    if fires >= 4:
        bg, fg = "#dcfce7", "#15803d"
    elif fires >= 3:
        bg, fg = "#fef9c3", "#854d0e"
    else:
        bg, fg = "#fef2f2", "#dc2626"
    label = "\U0001f525" * fires
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 6px;'
        f'border-radius:4px;font-weight:700;font-size:13px">{label}</span>'
    )


def _segment_pill(seg: str) -> str:
    if seg == "FG":
        bg, fg, bd = "rgba(52,168,83,.15)", "#34a853", "rgba(52,168,83,.3)"
    else:
        bg, fg, bd = "rgba(66,133,244,.15)", "#4285F4", "rgba(66,133,244,.3)"
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {bd};'
        f"padding:2px 6px;border-radius:4px;font-weight:600;font-size:11px;"
        f'text-transform:uppercase;letter-spacing:.5px">{_esc(seg)}</span>'
    )


def _pick_side_border(pick: Pick) -> str:
    """Return left-border colour: green for home/over, blue for away/under."""
    label_lower = pick.label.lower()
    if pick.market_type == "TOTAL":
        return "#16a34a" if "over" in label_lower else "#2563eb"
    matchup_parts = pick.matchup.split(" @ ")
    home_name = matchup_parts[1] if len(matchup_parts) == 2 else ""
    if home_name and pick.label.startswith(home_name):
        return "#16a34a"
    return "#2563eb"


def _build_html_odds_section(odds_by_game: dict[int, dict], game_labels: dict[int, str]) -> str:
    if not odds_by_game:
        return ""

    odds_rows: list[str] = []
    oth = (
        "background:#1a2332;color:#d4af37;font-weight:600;font-size:11px;"
        "letter-spacing:1px;text-transform:uppercase;padding:6px 8px;"
        "text-align:left;white-space:nowrap"
    )
    for gid, detail in odds_by_game.items():
        label = game_labels.get(gid, "")
        books = detail.get("books", {})
        if not books:
            continue
        first = True
        for bk, lines in sorted(books.items()):
            pieces: list[str] = []
            if "spread" in lines:
                price = f" ({lines['spread_price']:+d})" if lines.get("spread_price") else ""
                pieces.append(f"{lines['spread']:+.1f}{price}")
            else:
                pieces.append("—")
            if "total" in lines:
                price = f" ({lines['total_price']:+d})" if lines.get("total_price") else ""
                pieces.append(f"{lines['total']:.1f}{price}")
            else:
                pieces.append("—")
            if "home_ml" in lines:
                pieces.append(f"{lines['home_ml']:+d}")
            else:
                pieces.append("—")
            if "away_ml" in lines:
                pieces.append(f"{lines['away_ml']:+d}")
            else:
                pieces.append("—")
            if "spread_h1" in lines:
                price = (
                    f" ({lines['spread_h1_price']:+d})" if lines.get("spread_h1_price") else ""
                )
                pieces.append(f"{lines['spread_h1']:+.1f}{price}")
            else:
                pieces.append("—")
            if "total_h1" in lines:
                price = (
                    f" ({lines['total_h1_price']:+d})" if lines.get("total_h1_price") else ""
                )
                pieces.append(f"{lines['total_h1']:.1f}{price}")
            else:
                pieces.append("—")

            otd = 'style="padding:4px 8px;border-bottom:1px solid #e9ecef;font-size:12px"'
            matchup_cell = f"<td {otd}><b>{_esc(label)}</b></td>" if first else f"<td {otd}></td>"
            odds_rows.append(
                f"<tr>"
                f"{matchup_cell}"
                f"<td {otd}><b>{_esc(bk)}</b></td>"
                f"<td {otd}>{_esc(pieces[0])}</td>"
                f"<td {otd}>{_esc(pieces[1])}</td>"
                f"<td {otd}>{_esc(pieces[2])}</td>"
                f"<td {otd}>{_esc(pieces[3])}</td>"
                f"<td {otd}>{_esc(pieces[4])}</td>"
                f"<td {otd}>{_esc(pieces[5])}</td>"
                f"</tr>"
            )
            first = False

    if not odds_rows:
        return ""

    any_detail = next(iter(odds_by_game.values()), {})
    ts_raw = any_detail.get("captured_at", "")
    ts_display = ""
    if ts_raw:
        try:
            dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(_CST)
            ts_display = f" &middot; As of {dt.strftime('%I:%M %p').lstrip('0')} CT"
        except Exception:
            pass

    return (
        '<div style="margin-top:16px">'
        f'<div style="font-size:14px;font-weight:700;color:#1a2332;margin-bottom:4px">'
        f"\U0001f4ca Odds Sources{ts_display}</div>"
        '<table style="width:100%;border-collapse:collapse;border:1px solid #dee2e6">'
        f"<thead><tr>"
        f'<th style="{oth}">Game</th>'
        f'<th style="{oth}">Book</th>'
        f'<th style="{oth}">Spread</th>'
        f'<th style="{oth}">Total</th>'
        f'<th style="{oth}">Home ML</th>'
        f'<th style="{oth}">Away ML</th>'
        f'<th style="{oth}">1H Spread</th>'
        f'<th style="{oth}">1H Total</th>'
        f"</tr></thead>"
        f"<tbody>{''.join(odds_rows)}</tbody>"
        "</table></div>"
    )


# ── Main HTML slate builder ────────────────────────────────────────


def build_html_slate(
    predictions_with_games: list[tuple[Any, ...]],
    odds_pulled_at: datetime | None = None,
    min_edge: float = MIN_EDGE,
    empty_message: str | None = None,
) -> str:
    """Build a styled HTML table of the full slate for posting directly to Teams.

    Teams HTML messages support inline CSS — no JS, no external fonts.
    Formatting draws from NCAAM v2.0 / NBA v5.0 visual patterns.

    Each element may be ``(pred, game)`` or ``(pred, game, odds_map)``.
    """
    now_cst = datetime.now(_CST)
    now_str = now_cst.strftime("%Y-%m-%d %I:%M %p CT")
    odds_ts = (
        odds_pulled_at.astimezone(_CST).strftime("%Y-%m-%d %I:%M %p CT")
        if odds_pulled_at
        else now_str
    )
    today_display = now_cst.strftime("%A, %B %d, %Y")

    all_picks: list[Pick] = []
    game_ids: set[int] = set()
    odds_by_game: dict[int, dict] = {}
    game_labels: dict[int, str] = {}
    for row in predictions_with_games:
        pred, game = row[0], row[1]
        om: dict[str, str] = row[2] if len(row) > 2 else {}  # type: ignore[index]
        gid = getattr(game, "id", id(game))
        game_ids.add(gid)
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge, odds_map=om))
        sourced = getattr(pred, "odds_sourced", None)
        if sourced:
            odds_by_game[gid] = sourced
        home = game.home_team.name if game.home_team else "Home"
        away = game.away_team.name if game.away_team else "Away"
        game_labels[gid] = f"{away} @ {home}"
    all_picks.sort(key=lambda p: -p.edge)

    n_games = len(game_ids)
    odds_section = _build_html_odds_section(odds_by_game, game_labels)

    # ── header ─────────────────────────────────────
    header = (
        '<div style="margin-bottom:12px">'
        '<div style="font-size:20px;font-weight:700;color:#1a2332">'
        "\U0001f3c0 NBA Daily Slate</div>"
        f'<div style="font-size:13px;color:#6b7280">{_esc(today_display)}</div>'
        f'<div style="font-size:12px;color:#6b7280">'
        f"Model {_esc(MODEL_VERSION)} &middot; Odds pulled {_esc(odds_ts)} "
        f"&middot; {n_games} games &middot; {len(all_picks)} picks</div>"
        "</div>"
    )

    if not all_picks:
        message = empty_message or "No qualified picks today."
        return header + f'<p style="color:#6b7280">{_esc(message)}</p>' + odds_section

    matchups = sorted({p.matchup for p in all_picks})
    segments = sorted({p.segment for p in all_picks})
    markets = sorted({p.market_type for p in all_picks})

    # ── filter bar ─────────────────────────────────
    sel_style = (
        "padding:6px 10px;border:1px solid #dee2e6;border-radius:6px;"
        "font-size:13px;background:#fff;color:#1a2332;cursor:pointer"
    )
    matchup_opts = "".join(f'<option value="{_esc(m)}">{_esc(m)}</option>' for m in matchups)
    seg_opts = "".join(f'<option value="{_esc(s)}">{_esc(s)}</option>' for s in segments)
    mkt_opts = "".join(f'<option value="{_esc(m)}">{_esc(m)}</option>' for m in markets)
    filter_bar = (
        '<div id="filters" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:center">'
        f'<select id="fMatchup" style="{sel_style}" onchange="applyFilters()">'
        f'<option value="">All Matchups</option>{matchup_opts}</select>'
        f'<select id="fSeg" style="{sel_style}" onchange="applyFilters()">'
        f'<option value="">All Segments</option>{seg_opts}</select>'
        f'<select id="fMkt" style="{sel_style}" onchange="applyFilters()">'
        f'<option value="">All Markets</option>{mkt_opts}</select>'
        f'<input id="fEdge" type="number" step="0.5" min="0" placeholder="Min Edge" '
        f'style="{sel_style};width:100px" oninput="applyFilters()">'
        '<button onclick="resetFilters()" style="padding:6px 14px;border:none;'
        "border-radius:6px;background:#1a2332;color:#d4af37;font-weight:600;"
        'font-size:12px;cursor:pointer;letter-spacing:.5px">RESET</button>'
        "</div>"
    )

    # ── table ──────────────────────────────────────
    th_style = (
        "background:#1a2332;color:#d4af37;font-weight:600;font-size:11px;"
        "letter-spacing:1px;text-transform:uppercase;padding:8px 6px;"
        "text-align:left;white-space:nowrap;cursor:pointer;user-select:none"
    )

    rows_html: list[str] = []
    for i, p in enumerate(all_picks):
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fa"
        border_color = _pick_side_border(p)
        edge_color = _edge_css_color(p.edge)

        parts = p.matchup.split(" @ ")
        away_display = f'<span style="color:#2563eb">{_esc(parts[0])}</span>'
        if p.away_record:
            away_display += (
                f' <span style="font-size:11px;color:#9ca3af">({_esc(p.away_record)})</span>'
            )
        home_display = (
            f'<span style="color:#16a34a;font-weight:700">'
            f'{_esc(parts[1]) if len(parts) > 1 else ""}</span>'
        )
        if p.home_record:
            home_display += (
                f' <span style="font-size:11px;color:#9ca3af">({_esc(p.home_record)})</span>'
            )
        matchup_cell = f"{away_display} @ {home_display}"

        td = (
            'style="padding:6px;border-bottom:1px solid #e9ecef;'
            'font-size:13px;vertical-align:middle"'
        )

        odds_html = ""
        if p.odds:
            odds_fg = "#16a34a" if p.odds.startswith("+") else "#dc2626"
            odds_html = (
                f' <span style="font-size:12px;font-weight:700;color:{odds_fg};'
                f'background:rgba(0,0,0,.04);padding:1px 4px;border-radius:3px">'
                f"({_esc(p.odds)})</span>"
            )

        rationale_html = ""
        if p.rationale:
            rationale_html = (
                f'<div style="font-size:11px;color:#6b7280;font-style:italic;'
                f'margin-top:2px">{_esc(p.rationale)}</div>'
            )

        rows_html.append(
            f'<tr style="background:{bg}" data-matchup="{_esc(p.matchup)}" '
            f'data-seg="{_esc(p.segment)}" data-mkt="{_esc(p.market_type)}" '
            f'data-edge="{p.edge:.1f}" data-time="{_esc(p.time_cst)}" '
            f'data-rating="{p.confidence}">'
            f"<td {td}>{_esc(p.time_cst)}</td>"
            f"<td {td}>{matchup_cell}</td>"
            f"<td {td}>{_segment_pill(p.segment)}</td>"
            f'<td {td}><span style="font-weight:600;color:#374151;font-size:12px">'
            f"{_esc(p.market_type)}</span></td>"
            f'<td style="padding:6px;border-bottom:1px solid #e9ecef;font-size:13px;'
            f'vertical-align:middle;border-left:3px solid {border_color};font-weight:700">'
            f"{_esc(p.label)}{odds_html}"
            f' <span style="font-size:11px;color:#9ca3af;font-weight:400">'
            f"{_esc(p.market_line)}</span>"
            f"{rationale_html}</td>"
            f'<td {td}><span style="font-weight:600">{_esc(p.model_scores)}</span></td>'
            f'<td style="padding:6px;border-bottom:1px solid #e9ecef;text-align:center;'
            f'font-weight:700;font-size:14px;color:{edge_color}">{p.edge:.1f}</td>'
            f"<td {td}>{_confidence_badge(p.confidence)}</td>"
            "</tr>"
        )

    sort_arrow = (
        '<span class="sort-arrow" style="margin-left:4px;font-size:10px">\u25b2\u25bc</span>'
    )
    table = (
        '<table id="slate" style="width:100%;border-collapse:collapse;border:1px solid #dee2e6">'
        "<thead><tr>"
        f'<th style="{th_style}" onclick="sortTable(0,\'text\')" data-col="0">Time{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(1,\'text\')" data-col="1">Matchup{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(2,\'text\')" data-col="2">Seg{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(3,\'text\')" data-col="3">Market{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(4,\'text\')" data-col="4">Pick{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(5,\'text\')" data-col="5">Model{sort_arrow}</th>'
        f'<th style="{th_style};text-align:center" onclick="sortTable(6,\'num\')" data-col="6">Edge{sort_arrow}</th>'
        f'<th style="{th_style};text-align:center" onclick="sortTable(7,\'num\')" data-col="7">Rating{sort_arrow}</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )

    footer = (
        f'<div style="text-align:center;padding:8px;color:#9ca3af;font-size:11px">'
        f"GBSV NBA {_esc(MODEL_VERSION)} &middot; Sorted by edge descending</div>"
    )

    script = (
        "<script>"
        "var _sortCol=-1,_sortAsc=true;"
        "function sortTable(col,type){"
        "var tb=document.querySelector('#slate tbody');"
        "var rows=Array.from(tb.rows);"
        "if(_sortCol===col){_sortAsc=!_sortAsc}else{_sortCol=col;_sortAsc=type==='num'?false:true}"
        "rows.sort(function(a,b){"
        "var av=a.cells[col].textContent.trim(),bv=b.cells[col].textContent.trim();"
        "if(type==='num'){av=parseFloat(av)||0;bv=parseFloat(bv)||0;return _sortAsc?av-bv:bv-av}"
        "return _sortAsc?av.localeCompare(bv):bv.localeCompare(av);"
        "});"
        "rows.forEach(function(r,i){tb.appendChild(r);r.style.background=i%2===0?'#ffffff':'#f8f9fa'});"
        "document.querySelectorAll('#slate thead th .sort-arrow').forEach(function(s,i){"
        "s.textContent=i===col?(_sortAsc?'\\u25B2':'\\u25BC'):'\\u25B2\\u25BC';"
        "});"
        "}"
        "function applyFilters(){"
        "var m=document.getElementById('fMatchup').value,"
        "s=document.getElementById('fSeg').value,"
        "k=document.getElementById('fMkt').value,"
        "e=parseFloat(document.getElementById('fEdge').value)||0;"
        "var rows=document.querySelectorAll('#slate tbody tr');"
        "var vis=0;"
        "rows.forEach(function(r){"
        "var show=true;"
        "if(m&&r.dataset.matchup!==m)show=false;"
        "if(s&&r.dataset.seg!==s)show=false;"
        "if(k&&r.dataset.mkt!==k)show=false;"
        "if(e&&parseFloat(r.dataset.edge)<e)show=false;"
        "r.style.display=show?'':'none';"
        "if(show){r.style.background=vis%2===0?'#ffffff':'#f8f9fa';vis++;}"
        "});"
        "}"
        "function resetFilters(){"
        "document.getElementById('fMatchup').value='';"
        "document.getElementById('fSeg').value='';"
        "document.getElementById('fMkt').value='';"
        "document.getElementById('fEdge').value='';"
        "applyFilters();"
        "}"
        "</script>"
    )

    return header + filter_bar + table + odds_section + footer + script
