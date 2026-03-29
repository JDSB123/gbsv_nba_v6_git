"""Tests for Teams notification helper functions and builders."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.notifications.teams import (
    MIN_EDGE,
    Pick,
    _american_to_prob,
    _consensus_line,
    _consensus_price,
    _edge_color,
    _fire_count,
    _fire_emojis,
    _fmt_time_cst,
    _prob_to_american,
    _team_record,
    build_html_slate,
    build_slate_csv,
    build_teams_card,
    build_teams_text,
    extract_picks,
    _pick_row,
    _odds_source_block,
    _esc,
    _edge_css_color,
    _confidence_badge,
    _segment_pill,
    _pick_side_border,
)


# ── _fire_emojis ─────────────────────────────────────────────────


class TestFireEmojis:
    def test_min_edge(self):
        assert _fire_emojis(1.0) == " 🔥"

    def test_mid_edge(self):
        assert _fire_emojis(3.5) == " 🔥🔥"

    def test_high_edge(self):
        assert _fire_emojis(5.0) == " 🔥🔥🔥"

    def test_very_high(self):
        assert _fire_emojis(7.0) == " 🔥🔥🔥🔥"

    def test_extreme(self):
        assert _fire_emojis(9.0) == " 🔥🔥🔥🔥🔥"

    def test_between_thresholds(self):
        assert _fire_emojis(4.9) == " 🔥🔥"
        assert _fire_emojis(6.9) == " 🔥🔥🔥"


# ── _edge_color ──────────────────────────────────────────────────


class TestEdgeColor:
    def test_low(self):
        assert _edge_color(2.0) == "Accent"

    def test_mid(self):
        assert _edge_color(4.0) == "Good"

    def test_high(self):
        assert _edge_color(7.0) == "Attention"


# ── _fmt_time_cst ────────────────────────────────────────────────


class TestFmtTimeCst:
    def test_none_returns_tbd(self):
        assert _fmt_time_cst(None) == "TBD"

    def test_utc_datetime(self):
        dt = datetime(2024, 3, 15, 0, 30, tzinfo=UTC)  # midnight:30 UTC
        result = _fmt_time_cst(dt)
        assert "CT" in result
        assert "2024-03-14" in result  # Central is UTC-5/6

    def test_naive_datetime(self):
        dt = datetime(2024, 3, 15, 18, 0)  # naive, treated as UTC
        result = _fmt_time_cst(dt)
        assert "CT" in result
        assert "PM" in result


# ── _team_record ─────────────────────────────────────────────────


class TestTeamRecord:
    def test_from_season_stats_list(self):
        stats = SimpleNamespace(wins=42, losses=18)
        team = SimpleNamespace(season_stats=[stats])
        assert _team_record(team) == "42-18"

    def test_from_season_stats_single(self):
        stats = SimpleNamespace(wins=30, losses=30)
        team = SimpleNamespace(season_stats=stats)
        assert _team_record(team) == "30-30"

    def test_from_team_attrs(self):
        team = SimpleNamespace(season_stats=None, wins=50, losses=10)
        assert _team_record(team) == "50-10"

    def test_no_record(self):
        team = SimpleNamespace(season_stats=None)
        assert _team_record(team) == ""


# ── _fire_count ──────────────────────────────────────────────────


class TestFireCount:
    def test_levels(self):
        assert _fire_count(1.0) == 1
        assert _fire_count(3.5) == 2
        assert _fire_count(5.0) == 3
        assert _fire_count(7.0) == 4
        assert _fire_count(9.0) == 5


# ── _prob_to_american ────────────────────────────────────────────


class TestProbToAmerican:
    def test_favorite(self):
        result = _prob_to_american(0.75)
        assert result.startswith("-")

    def test_underdog(self):
        result = _prob_to_american(0.3)
        assert result.startswith("+")

    def test_boundary_low(self):
        assert _prob_to_american(0.005) == ""

    def test_boundary_high(self):
        assert _prob_to_american(0.995) == ""

    def test_even(self):
        result = _prob_to_american(0.5)
        assert result  # should produce a value


# ── _american_to_prob ────────────────────────────────────────────


class TestAmericanToProb:
    def test_empty(self):
        assert _american_to_prob("") is None

    def test_favorite(self):
        p = _american_to_prob("-200")
        assert p is not None
        assert abs(p - 2 / 3) < 0.01

    def test_underdog(self):
        p = _american_to_prob("+200")
        assert p is not None
        assert abs(p - 1 / 3) < 0.01

    def test_even(self):
        p = _american_to_prob("+100")
        assert p == 0.5

    def test_invalid(self):
        assert _american_to_prob("foo") is None

    def test_zero(self):
        p = _american_to_prob("0")
        assert p == 0.5


# ── _consensus_price ─────────────────────────────────────────────


class TestConsensusPrice:
    def test_average(self):
        books = {
            "fanduel": {"spread_price": -110},
            "draftkings": {"spread_price": -108},
        }
        result = _consensus_price(books, "spread_price")
        assert result  # should be ~"-109"

    def test_empty(self):
        assert _consensus_price({}, "spread_price") == ""

    def test_missing_key(self):
        books = {"fanduel": {"total_price": -110}}
        assert _consensus_price(books, "spread_price") == ""


# ── _consensus_line ──────────────────────────────────────────────


class TestConsensusLine:
    def test_average(self):
        books = {
            "fanduel": {"spread": -3.5},
            "draftkings": {"spread": -4.0},
        }
        result = _consensus_line(books, "spread")
        assert result is not None
        assert abs(result - (-3.8)) < 0.05

    def test_empty(self):
        assert _consensus_line({}, "spread") is None


# ── extract_picks ────────────────────────────────────────────────


def _make_pred(**overrides):
    """Build a simple prediction namespace."""
    defaults = dict(
        fg_spread=0,
        fg_total=230,
        fg_home_ml_prob=0.5,
        h1_spread=0,
        h1_total=115,
        predicted_home_fg=0,
        predicted_away_fg=0,
        predicted_home_1h=0,
        predicted_away_1h=0,
        opening_spread=None,
        opening_total=None,
        odds_sourced=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_game():
    """Build a simple game namespace."""
    return SimpleNamespace(
        home_team=SimpleNamespace(name="Celtics", season_stats=None),
        away_team=SimpleNamespace(name="Heat", season_stats=None),
        home_team_id=1,
        away_team_id=2,
        commence_time=datetime(2024, 3, 15, 23, 0),
    )


class TestExtractPicks:
    def test_no_edge_no_picks(self):
        pred = _make_pred(fg_spread=0, fg_total=230, opening_total=230, opening_spread=-0.0)
        picks = extract_picks(pred, _make_game())
        # Spread edge is 0, total edge is 0 → only ML could fire
        spread_picks = [p for p in picks if p.market_type == "SPREAD"]
        total_picks = [p for p in picks if p.market_type == "TOTAL"]
        assert len(spread_picks) == 0
        assert len(total_picks) == 0

    def test_spread_pick_with_market_line(self):
        # Model says home by 8 (fg_spread=8), market line=-3 → edge=5
        pred = _make_pred(fg_spread=8, opening_spread=-3.0)
        picks = extract_picks(pred, _make_game())
        spread_picks = [p for p in picks if p.market_type == "SPREAD" and p.segment == "FG"]
        assert len(spread_picks) == 1
        assert spread_picks[0].edge >= MIN_EDGE

    def test_total_pick_with_market_line(self):
        # Model says 240, market says 230 → edge = 10
        pred = _make_pred(fg_total=240, opening_total=230.0)
        picks = extract_picks(pred, _make_game())
        total_picks = [p for p in picks if p.market_type == "TOTAL" and p.segment == "FG"]
        assert len(total_picks) == 1
        assert "OVER" in total_picks[0].label
        assert total_picks[0].edge == 10.0

    def test_total_under(self):
        pred = _make_pred(fg_total=218, opening_total=228.0)
        picks = extract_picks(pred, _make_game())
        total_picks = [p for p in picks if p.market_type == "TOTAL" and p.segment == "FG"]
        assert len(total_picks) == 1
        assert "UNDER" in total_picks[0].label

    def test_ml_pick_with_odds(self):
        # High home win prob with favorable odds → ML pick
        pred = _make_pred(fg_spread=5, fg_home_ml_prob=0.80)
        picks = extract_picks(pred, _make_game(), odds_map={"FG_ML_HOME": "-150"})
        ml_picks = [p for p in picks if p.market_type == "ML"]
        assert len(ml_picks) == 1
        assert "Celtics ML" in ml_picks[0].label

    def test_no_market_total_uses_avg(self):
        # fg_total = 240, no opening_total → diff from _NBA_AVG_TOTAL=230 = 10
        pred = _make_pred(fg_total=240)
        picks = extract_picks(pred, _make_game())
        total_picks = [p for p in picks if p.market_type == "TOTAL" and p.segment == "FG"]
        assert len(total_picks) == 1

    def test_1h_spread_with_odds_sourced(self):
        # 1H model spread = 6, opening_h1_spread = -1 → edge = 5
        odds_sourced = {"opening_h1_spread": -1.0, "opening_h1_total": None, "books": {}}
        pred = _make_pred(h1_spread=6, odds_sourced=odds_sourced)
        picks = extract_picks(pred, _make_game())
        h1_spread = [p for p in picks if p.segment == "1H" and p.market_type == "SPREAD"]
        assert len(h1_spread) == 1

    def test_1h_total_with_odds_sourced(self):
        odds_sourced = {"opening_h1_spread": None, "opening_h1_total": 110.0, "books": {}}
        pred = _make_pred(h1_total=120, odds_sourced=odds_sourced)
        picks = extract_picks(pred, _make_game())
        h1_total = [p for p in picks if p.segment == "1H" and p.market_type == "TOTAL"]
        assert len(h1_total) == 1
        assert "OVER" in h1_total[0].label

    def test_pick_has_required_fields(self):
        pred = _make_pred(fg_spread=10, opening_spread=-3.0, fg_total=240, opening_total=228.0)
        picks = extract_picks(pred, _make_game())
        for pick in picks:
            assert isinstance(pick, Pick)
            assert pick.matchup == "Heat @ Celtics"
            assert pick.edge > 0
            assert pick.confidence >= 1


# ── build_teams_card ─────────────────────────────────────────────


class TestBuildTeamsCard:
    def test_returns_card_structure(self):
        pred = _make_pred(fg_spread=10, opening_spread=-3.0, fg_total=240, opening_total=228.0)
        game = _make_game()
        card = build_teams_card([(pred, game)], max_games=5)
        assert card["type"] == "message"
        assert len(card["attachments"]) == 1
        content = card["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert len(content["body"]) > 0

    def test_with_download_urls(self):
        pred = _make_pred(fg_spread=10, opening_spread=-3.0)
        game = _make_game()
        card = build_teams_card(
            [(pred, game)],
            max_games=5,
            download_url="https://example.com/slate",
            csv_download_url="https://example.com/slate.csv",
        )
        content = card["attachments"][0]["content"]
        assert "actions" in content
        assert len(content["actions"]) == 2


# ── build_slate_csv ──────────────────────────────────────────────


class TestBuildSlateCsv:
    def test_csv_has_header_and_rows(self):
        pred = _make_pred(fg_spread=10, opening_spread=-3.0, fg_total=245, opening_total=228.0)
        game = _make_game()
        csv_str = build_slate_csv([(pred, game)])
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2  # header + at least one pick row
        assert "Time (CT)" in lines[0]
        assert "Edge" in lines[0]

    def test_csv_with_odds_map_tuple(self):
        pred = _make_pred(fg_spread=10, opening_spread=-3.0)
        game = _make_game()
        csv_str = build_slate_csv([(pred, game, {"FG_SPREAD": "-110"})])
        assert "-110" in csv_str


# ── _pick_row ────────────────────────────────────────────────────


class TestPickRow:
    def test_returns_column_set(self):
        pick = Pick(
            label="Celtics -3.5",
            edge=5.0,
            time_cst="6:30 PM CT",
            matchup="Heat @ Celtics",
            segment="FG",
            market_type="SPREAD",
            market_line="-3.5",
            model_scores="Celtics 112, Heat 108",
            home_record="42-18",
            away_record="28-32",
            confidence=3,
            odds="-110",
            rationale="Model: Celtics by 8",
        )
        result = _pick_row(pick)
        assert result["type"] == "ColumnSet"
        assert len(result["columns"]) == 2

    def test_includes_records_in_matchup(self):
        pick = Pick(
            label="Over 224.5",
            edge=3.5,
            time_cst="7:00 PM CT",
            matchup="Heat @ Celtics",
            segment="FG",
            market_type="TOTAL",
            market_line="224.5",
            model_scores="Celtics 116, Heat 112",
            home_record="42-18",
            away_record="28-32",
            confidence=2,
        )
        result = _pick_row(pick)
        left_col = result["columns"][0]
        # Second text block should have matchup with records
        matchup_text = left_col["items"][1]["text"]
        assert "42-18" in matchup_text
        assert "28-32" in matchup_text


# ── _odds_source_block ──────────────────────────────────────────


class TestOddsSourceBlock:
    def test_no_data(self):
        assert _odds_source_block(None) == []
        assert _odds_source_block({}) == []
        assert _odds_source_block({"books": {}}) == []

    def test_with_books(self):
        data = {
            "books": {
                "fanduel": {"spread": -3.5, "spread_price": -110, "total": 224.5, "total_price": -108, "home_ml": -150},
            },
            "captured_at": "2024-03-15T12:00:00Z",
        }
        blocks = _odds_source_block(data)
        assert len(blocks) == 1
        assert "fanduel" in blocks[0]["text"].lower()


# ── HTML helpers ─────────────────────────────────────────────────


class TestHtmlHelpers:
    def test_esc(self):
        assert _esc("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"

    def test_edge_css_color_levels(self):
        assert "#16a34a" in _edge_css_color(8.0)  # green
        assert "#ca8a04" in _edge_css_color(5.5)  # amber
        assert "#2563eb" in _edge_css_color(3.5)  # blue
        assert "#6b7280" in _edge_css_color(1.0)  # gray

    def test_confidence_badge(self):
        badge = _confidence_badge(4)
        assert "🔥" in badge
        assert "span" in badge

    def test_segment_pill_fg(self):
        pill = _segment_pill("FG")
        assert "FG" in pill
        assert "span" in pill

    def test_segment_pill_1h(self):
        pill = _segment_pill("1H")
        assert "1H" in pill

    def test_pick_side_border_over(self):
        pick = Pick("OVER 224.5", 5.0, "7PM CT", "Heat @ Celtics", "FG", "TOTAL",
                     "224.5", "Celtics 116, Heat 112", "", "", 3)
        assert _pick_side_border(pick) == "#16a34a"

    def test_pick_side_border_under(self):
        pick = Pick("UNDER 224.5", 5.0, "7PM CT", "Heat @ Celtics", "FG", "TOTAL",
                     "224.5", "Celtics 106, Heat 108", "", "", 3)
        assert _pick_side_border(pick) == "#2563eb"

    def test_pick_side_border_home(self):
        pick = Pick("Celtics -3.5", 5.0, "7PM CT", "Heat @ Celtics", "FG", "SPREAD",
                     "-3.5", "Celtics 112, Heat 108", "", "", 3)
        assert _pick_side_border(pick) == "#16a34a"

    def test_pick_side_border_away(self):
        pick = Pick("Heat +3.5", 5.0, "7PM CT", "Heat @ Celtics", "FG", "SPREAD",
                     "+3.5", "Celtics 108, Heat 112", "", "", 3)
        assert _pick_side_border(pick) == "#2563eb"


# ── build_html_slate ─────────────────────────────────────────────


class TestBuildHtmlSlate:
    def test_returns_html(self):
        pred = _make_pred(fg_spread=10, opening_spread=-3.0, fg_total=240, opening_total=228.0)
        game = _make_game()
        html = build_html_slate([(pred, game)])
        assert "<html" in html or "<table" in html
        assert "Celtics" in html

    def test_empty_picks(self):
        pred = _make_pred()
        game = _make_game()
        html = build_html_slate([(pred, game)])
        assert isinstance(html, str)


# ── build_teams_text ─────────────────────────────────────────────


class TestBuildTeamsText:
    def test_returns_text(self):
        pred = _make_pred(fg_spread=5, fg_total=220, fg_home_ml_prob=0.65)
        game = _make_game()
        text = build_teams_text([(pred, game)], max_games=5)
        assert "NBA Predictions Update" in text
        assert "Celtics" in text
        assert "Heat" in text

    def test_respects_max_games(self):
        pred = _make_pred(fg_spread=5, fg_total=220, fg_home_ml_prob=0.65)
        game = _make_game()
        text = build_teams_text([(pred, game)] * 10, max_games=3)
        assert "Games: 3" in text


# ── send_alert ───────────────────────────────────────────────────


class TestSendAlert:
    @pytest.mark.anyio
    async def test_no_webhook_configured(self):
        from unittest.mock import patch

        from src.notifications.teams import send_alert

        with patch("src.config.get_settings") as mock_s:
            mock_s.return_value = SimpleNamespace(teams_webhook_url=None)
            # Should not raise
            await send_alert("Test", "Test message")

    @pytest.mark.anyio
    async def test_send_success(self):
        import httpx
        from unittest.mock import patch

        from src.notifications.teams import send_alert

        with patch("src.config.get_settings") as mock_s:
            mock_s.return_value = SimpleNamespace(teams_webhook_url="https://webhook.example.com")
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client
                await send_alert("Test Alert", "Something happened", "error")
