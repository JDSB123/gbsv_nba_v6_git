"""Tests for seasons module – remaining untested functions."""

from datetime import date

import pytest

from src.data.seasons import (
    format_season,
    parse_season,
    season_bounds,
)

# ── format_season ──────────────────────────────────────────────

def test_format_season():
    assert format_season(2024) == "2024-2025"
    assert format_season(2025) == "2025-2026"


def test_format_season_edge():
    assert format_season(2000) == "2000-2001"


# ── parse_season ───────────────────────────────────────────────

def test_parse_season_valid():
    start, end = parse_season("2024-2025")
    assert start == 2024
    assert end == 2025


def test_parse_season_invalid_gap():
    with pytest.raises(ValueError, match="Invalid NBA season"):
        parse_season("2024-2026")


def test_parse_season_invalid_reversed():
    with pytest.raises(ValueError, match="Invalid NBA season"):
        parse_season("2025-2024")


# ── season_bounds ──────────────────────────────────────────────

def test_season_bounds():
    start, end = season_bounds("2024-2025")
    assert start == date(2024, 10, 1)
    assert end == date(2025, 6, 30)


def test_season_bounds_different_year():
    start, end = season_bounds("2025-2026")
    assert start == date(2025, 10, 1)
    assert end == date(2026, 6, 30)
