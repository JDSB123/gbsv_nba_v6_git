"""Build a single consolidated HTML slate from Azure API output.

Generates one self-contained HTML file with:
  - Prediction picks table with game cards
  - Data source status (honest reality)
  - Per-game odds capture detail
  - Model pipeline summary
"""

import datetime
import json
import pathlib
import zoneinfo

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "predictions_raw.json"
OUT = ROOT / "nba_picks_slate_livesync.html"

CDT = zoneinfo.ZoneInfo("America/Chicago")
now_cdt = datetime.datetime.now(CDT)
ts_label = now_cdt.strftime("%B %d, %Y  %I:%M %p CDT")

data = json.loads(RAW.read_text())
preds = data["predictions"]
freshness = data["freshness"]

# Sort by best_edge descending
preds.sort(key=lambda p: p.get("best_edge", 0), reverse=True)


# ── Helpers ──────────────────────────────────────────────────────
def edge_class(e):
    if e >= 5:
        return "hot"
    if e >= 3:
        return "warm"
    return "cold"


def pick_rows(p):
    rows = []
    for mkey, label in [
        ("fg_spread", "Spread"),
        ("fg_total", "Total"),
        ("fg_moneyline", "ML"),
        ("h1_spread", "1H Spread"),
        ("h1_total", "1H Total"),
        ("h1_moneyline", "1H ML"),
    ]:
        m = p["markets"].get(mkey, {})
        pick = m.get("pick")
        if not pick:
            continue
        edge = m.get("edge", 0)
        act = m.get("actionable", False)
        rat = m.get("rationale") or ""
        rows.append(f"""
        <tr class="{"actionable" if act else "monitoring"}">
            <td>{label}</td>
            <td><strong>{pick}</strong></td>
            <td class="{edge_class(edge)}">{edge:.1f}</td>
            <td>{"✅" if act else "👀"}</td>
            <td class="rationale">{rat}</td>
        </tr>""")
    return "\n".join(rows)


# ── Build game cards ─────────────────────────────────────────────
game_cards = []
for p in preds:
    ct_raw = p.get("commence_time", "")
    try:
        ct = (
            datetime.datetime.fromisoformat(ct_raw)
            .replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            .astimezone(CDT)
        )
        ct_str = ct.strftime("%a %b %d, %I:%M %p CDT")
    except Exception:
        ct_str = ct_raw

    scores = p.get("predicted_scores", {})
    fg = scores.get("full_game", {})
    h1 = scores.get("first_half", {})
    odds = p.get("odds", {})
    cons = odds.get("consensus", {})
    books = odds.get("book_count", "?")
    captured = odds.get("captured_at", "")
    try:
        cap_dt = datetime.datetime.fromisoformat(captured.replace("Z", "+00:00")).astimezone(CDT)
        captured_str = cap_dt.strftime("%I:%M %p CDT")
    except Exception:
        captured_str = captured

    game_cards.append(f"""
    <div class="game-card">
        <div class="matchup">
            <span class="away">{p["away_team"]}</span>
            <span class="at">@</span>
            <span class="home">{p["home_team"]}</span>
            <span class="time">{ct_str}</span>
        </div>
        <div class="scores-strip">
            FG: {fg.get("home", "?")}-{fg.get("away", "?")} &nbsp;|&nbsp;
            1H: {h1.get("home", "?")}-{h1.get("away", "?")} &nbsp;|&nbsp;
            Consensus Spread: {cons.get("spread", "?")} &nbsp; Total: {cons.get("total", "?")} &nbsp;|&nbsp;
            {books} books @ {captured_str}
        </div>
        <table class="picks">
            <thead><tr><th>Market</th><th>Pick</th><th>Edge</th><th>Act</th><th>Rationale</th></tr></thead>
            <tbody>{pick_rows(p)}</tbody>
        </table>
    </div>""")

actionable_count = sum(
    1 for p in preds for m in p["markets"].values() if isinstance(m, dict) and m.get("actionable")
)

# ── Count 1H availability ────────────────────────────────────────
games_with_1h = sum(
    1
    for p in preds
    if any(p["markets"].get(k, {}).get("pick") for k in ("h1_spread", "h1_total", "h1_moneyline"))
)

# ── Per-game odds detail rows ────────────────────────────────────
odds_rows = []
for p in preds:
    odds = p.get("odds", {})
    cons = odds.get("consensus", {})
    cap = odds.get("captured_at", "")
    try:
        cap_cdt = datetime.datetime.fromisoformat(cap.replace("Z", "+00:00")).astimezone(CDT)
        cap_str = cap_cdt.strftime("%I:%M %p CDT")
    except Exception:
        cap_str = cap
    odds_rows.append(f"""
    <tr>
        <td>{p["away_team"]} @ {p["home_team"]}</td>
        <td>{odds.get("book_count", "?")}</td>
        <td>{cap_str}</td>
        <td>{cons.get("spread", "?")}</td>
        <td>{cons.get("total", "?")}</td>
    </tr>""")

# ── Data source status ───────────────────────────────────────────
data_sources = [
    ("✅", "Full-Game Odds", "The Odds API v4 — us + us2 regions — 12 bookmaker typical"),
    (
        "✅" if games_with_1h > 0 else "⚠️",
        "1st-Half Odds",
        (
            f"{games_with_1h}/{len(preds)} games have 1H lines"
            if games_with_1h
            else f"0/{len(preds)} games — polling active but no sportsbooks posting 1H lines currently"
        ),
    ),
    ("✅", "Team Season Stats", "32 teams via Basketball API v1 (Mega plan, 150K/day)"),
    ("✅", "Player Game Stats", "Box scores via Basketball API v1 — 613+ games, 13K+ stat rows"),
    ("✅", "Elo / H2H", "Rolling Elo ratings + head-to-head history"),
    ("✅", "Schedule / Rest", "Days-rest, B2B, travel distance"),
    ("✅", "Venue / Streak", "Home/away splits, win/loss streaks"),
    ("❌", "Injuries", "No API source available — features imputed as neutral"),
    (
        "❌",
        "Referees",
        "No API source available — features imputed as neutral",
    ),
    ("⚠️", "Player Props", "3 of 13 prop features active (pts_count, DD_count, TD_count)"),
    (
        "❌",
        "Sharp/Square Market",
        "All market-microstructure features pruned during training (insufficient data)",
    ),
]

ds_rows = "\n".join(
    f'    <tr><td style="text-align:center;font-size:1.1rem">{icon}</td>'
    f"<td><strong>{name}</strong></td><td>{detail}</td></tr>"
    for icon, name, detail in data_sources
)

# ── Assemble single consolidated HTML ────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GBSV NBA Slate — {now_cdt.strftime("%b %d, %Y")}</title>
<style>
:root {{ --bg:#0d1117; --card:#161b22; --border:#30363d; --txt:#c9d1d9;
         --hot:#f85149; --warm:#d29922; --cold:#58a6ff; --green:#3fb950; --head:#58a6ff; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--txt); font-family:'Segoe UI',system-ui,sans-serif;
        padding:1rem; max-width:1000px; margin:auto; }}
h1 {{ font-size:1.4rem; margin-bottom:.25rem; color:#fff; }}
h2 {{ font-size:1.05rem; color:var(--head); margin:1.25rem 0 .5rem; }}
.meta {{ color:#8b949e; font-size:.85rem; margin-bottom:1rem; }}
.summary {{ background:var(--card); padding:.75rem 1rem; border-radius:8px;
            margin-bottom:1.25rem; border:1px solid var(--border); }}
.game-card {{ background:var(--card); border:1px solid var(--border);
              border-radius:8px; padding:1rem; margin-bottom:1rem; }}
.matchup {{ font-size:1.15rem; font-weight:600; margin-bottom:.4rem; }}
.away {{ color:var(--cold); }} .home {{ color:var(--green); }}
.at {{ color:#8b949e; margin:0 .3rem; }}
.time {{ float:right; color:#8b949e; font-size:.85rem; font-weight:400; }}
.scores-strip {{ font-size:.8rem; color:#8b949e; margin-bottom:.6rem; }}
table {{ width:100%; border-collapse:collapse; font-size:.85rem; margin:.5rem 0; }}
table.picks th {{ text-align:left; border-bottom:1px solid var(--border);
                  padding:4px 6px; color:#8b949e; font-weight:500; }}
table.picks td {{ padding:4px 6px; border-bottom:1px solid #21262d; }}
.actionable td:first-child {{ border-left:3px solid var(--green); }}
.monitoring td:first-child {{ border-left:3px solid #484f58; }}
.hot {{ color:var(--hot); font-weight:700; }}
.warm {{ color:var(--warm); font-weight:600; }}
.cold {{ color:var(--cold); }}
.rationale {{ font-size:.78rem; color:#8b949e; max-width:340px; }}
.section {{ background:var(--card); border:1px solid var(--border);
            border-radius:8px; padding:1rem; margin-bottom:1rem; }}
th {{ text-align:left; border-bottom:1px solid var(--border); padding:5px 8px; color:#8b949e; }}
td {{ padding:5px 8px; border-bottom:1px solid #21262d; }}
details {{ margin-top:1.25rem; }}
summary {{ font-size:1.1rem; font-weight:600; color:#fff; cursor:pointer; padding:.5rem 0; }}
code {{ background:#21262d; padding:2px 5px; border-radius:3px; font-size:.82rem; }}
.legend {{ font-size:.8rem; color:#8b949e; margin-top:.4rem; }}
</style></head><body>

<h1>🏀 GBSV NBA Prediction Slate</h1>
<p class="meta">Generated {ts_label} &nbsp;|&nbsp; Model v6.6.0 &nbsp;|&nbsp;
   Source: Azure Container App</p>

<div class="summary">
    <strong>{len(preds)}</strong> games &nbsp;|&nbsp;
    <strong>{actionable_count}</strong> actionable picks &nbsp;|&nbsp;
    Freshness: {freshness["status"]} (youngest {freshness["freshest_age_minutes"]:.1f} min)
</div>

{"".join(game_cards)}

<!-- ── Data Source Status ──────────────────────────────── -->
<details open>
<summary>🔍 Data Source Status</summary>
<div class="section">
<table>
<thead><tr><th style="width:40px">Status</th><th>Source</th><th>Details</th></tr></thead>
<tbody>
{ds_rows}
</tbody>
</table>
<p class="legend">✅ Active &nbsp; ⚠️ Partial/Intermittent &nbsp; ❌ Unavailable</p>
</div>
</details>

<!-- ── Per-Game Odds Capture ──────────────────────────── -->
<details>
<summary>📊 Per-Game Odds Capture</summary>
<div class="section">
<table>
<thead><tr><th>Game</th><th>Books</th><th>Captured (CDT)</th><th>Spread</th><th>Total</th></tr></thead>
<tbody>
{"".join(odds_rows)}
</tbody>
</table>
</div>
</details>

<!-- ── Model Pipeline ─────────────────────────────────── -->
<details>
<summary>⚙️ Model Pipeline</summary>
<div class="section">
<p><strong>Architecture:</strong> XGBoost base models (6 targets) → LightGBM + Ridge stacking → Elo-calibrated ML probability</p>
<p><strong>Features:</strong> ~109 features from 11 categories (team season, recent form, schedule, player, elo/h2h, venue, props, injuries, derived, interactions)</p>
<p><strong>Consensus:</strong> Average line across all available bookmakers</p>
<p><strong>Thresholds:</strong> Spread ≥ 4.0 pts &nbsp;|&nbsp; Total ≥ 3.0 pts &nbsp;|&nbsp; ML ≥ 3.0% EV</p>
<p><strong>Blend alpha:</strong> 0.55 (market shrinkage)</p>
</div>
</details>

<p class="meta" style="margin-top:1.5rem;text-align:center;">
    GBSV v6.6.0 — {ts_label} — For informational purposes only
</p>
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"✅ Consolidated slate → {OUT.name}")
print(f"📅 {ts_label}")
print(f"🏀 {len(preds)} games, {actionable_count} actionable picks")
