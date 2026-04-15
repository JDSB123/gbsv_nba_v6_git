"""Tests for src.notifications._cards — Adaptive Card builder."""

from src.notifications._cards import _odds_source_block, _pick_row
from src.notifications._helpers import Pick


def _make_pick(**kwargs):
    defaults = dict(
        label="Lakers -5.0",
        edge=6.5,
        confidence=4,
        segment="FG",
        market_type="SPREAD",
        matchup="Celtics @ Lakers",
        time_cst="07:00 PM",
        market_line="(-110)",
        model_scores="112-110",
        odds="-110",
        home_record="40-20",
        away_record="30-30",
        rationale="Model favours home team",
    )
    defaults.update(kwargs)
    return Pick(**defaults)


class TestPickRow:
    def test_returns_column_set(self):
        row = _pick_row(_make_pick())
        assert row["type"] == "ColumnSet"

    def test_contains_label(self):
        row = _pick_row(_make_pick(label="Hawks +3.0"))
        # The label appears in the text blocks nested in columns
        text_blocks = [
            item
            for col in row["columns"]
            for item in col.get("items", [])
            if item.get("type") == "TextBlock"
        ]
        texts = " ".join(tb.get("text", "") for tb in text_blocks)
        assert "Hawks +3.0" in texts

    def test_odds_tag_included(self):
        row = _pick_row(_make_pick(odds="+150"))
        texts = " ".join(
            item.get("text", "") for col in row["columns"] for item in col.get("items", [])
        )
        assert "(+150)" in texts

    def test_no_odds_tag_when_empty(self):
        row = _pick_row(_make_pick(odds=""))
        texts = " ".join(
            item.get("text", "") for col in row["columns"] for item in col.get("items", [])
        )
        assert "()" not in texts

    def test_edge_displayed(self):
        row = _pick_row(_make_pick(edge=7.3))
        edge_col = row["columns"][-1]
        edge_text = edge_col["items"][0]["text"]
        assert "7.3" in edge_text

    def test_rationale_included(self):
        row = _pick_row(_make_pick(rationale="Strong model edge"))
        all_items = [item for col in row["columns"] for item in col.get("items", [])]
        texts = " ".join(item.get("text", "") for item in all_items)
        assert "Strong model edge" in texts

    def test_no_rationale_block_when_empty(self):
        row = _pick_row(_make_pick(rationale=""))
        items = row["columns"][0]["items"]
        # With no rationale, should have 3 items (label, matchup, detail)
        assert len(items) == 3

    def test_records_in_matchup(self):
        row = _pick_row(_make_pick(home_record="50-10", away_record="25-35"))
        texts = " ".join(
            item.get("text", "") for col in row["columns"] for item in col.get("items", [])
        )
        assert "50-10" in texts
        assert "25-35" in texts


class TestOddsSourceBlock:
    def test_empty_input(self):
        assert _odds_source_block(None) == []
        assert _odds_source_block({}) == []
        assert _odds_source_block({"books": {}}) == []

    def test_single_book(self):
        sourced = {
            "books": {
                "dk": {"spread": -5.0, "spread_price": -110, "total": 220.5, "total_price": -110}
            }
        }
        blocks = _odds_source_block(sourced)
        assert len(blocks) == 1
        assert "dk" in blocks[0]["text"].lower()
        assert "-5.0" in blocks[0]["text"]
        assert "220.5" in blocks[0]["text"]

    def test_multiple_books(self):
        sourced = {
            "books": {
                "dk": {"spread": -3.0},
                "fd": {"spread": -3.5},
            }
        }
        blocks = _odds_source_block(sourced)
        assert len(blocks) == 1
        assert "dk" in blocks[0]["text"].lower()
        assert "fd" in blocks[0]["text"].lower()

    def test_1h_lines(self):
        sourced = {
            "books": {
                "bm": {
                    "spread_h1": -1.5,
                    "spread_h1_price": -115,
                    "total_h1": 112.5,
                    "total_h1_price": -110,
                }
            }
        }
        blocks = _odds_source_block(sourced)
        text = blocks[0]["text"]
        assert "1H Sprd" in text
        assert "1H O/U" in text

    def test_captured_at_timestamp(self):
        sourced = {
            "books": {"dk": {"spread": -2.0}},
            "captured_at": "2025-03-15T20:00:00+00:00",
        }
        blocks = _odds_source_block(sourced)
        assert "CT)" in blocks[0]["text"]

    def test_ml_shown(self):
        sourced = {"books": {"dk": {"home_ml": -150}}}
        blocks = _odds_source_block(sourced)
        assert "ML -150" in blocks[0]["text"]
