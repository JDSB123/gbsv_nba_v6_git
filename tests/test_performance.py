"""Tests for performance grading, stats, accuracy, CLV, and dashboard endpoints."""

from types import SimpleNamespace

import pytest

from src.api.routes.performance import (
    EDGE_THRESHOLDS,
    GradedPick,
    _build_stats,
    _clv_summary,
    _grade_1h_winner,
    _grade_game,
    _grade_ml,
    _grade_spread_ats,
    _grade_total,
    _pct_class,
    _recent_results,
    _Record,
    _roi_class,
    _score_accuracy,
)

# ── _grade_spread_ats ────────────────────────────────────────


def test_grade_spread_ats_home_covers():
    # Home team wins by more than the spread
    assert _grade_spread_ats(edge=3.0, actual_home=110, actual_away=95, market_spread=-5.0) == "W"


def test_grade_spread_ats_home_fails_to_cover():
    # Home edge positive, but home doesn't cover
    assert _grade_spread_ats(edge=3.0, actual_home=102, actual_away=100, market_spread=-5.0) == "L"


def test_grade_spread_ats_away_covers():
    # Negative edge = bet away; away actually covers
    assert _grade_spread_ats(edge=-3.0, actual_home=100, actual_away=105, market_spread=-5.0) == "W"


def test_grade_spread_ats_away_fails():
    # Negative edge = bet away; but home covers
    assert _grade_spread_ats(edge=-3.0, actual_home=110, actual_away=100, market_spread=-5.0) == "L"


def test_grade_spread_ats_push():
    # actual_margin + market_spread == 0 → push
    assert _grade_spread_ats(edge=3.0, actual_home=105, actual_away=100, market_spread=-5.0) == "P"


# ── _grade_total ─────────────────────────────────────────────


def test_grade_total_over_wins():
    assert _grade_total(predicted_total=230, line=220, actual_home=115, actual_away=110) == "W"


def test_grade_total_over_loses():
    assert _grade_total(predicted_total=230, line=220, actual_home=105, actual_away=110) == "L"


def test_grade_total_under_wins():
    assert _grade_total(predicted_total=210, line=220, actual_home=105, actual_away=110) == "W"


def test_grade_total_under_loses():
    assert _grade_total(predicted_total=210, line=220, actual_home=115, actual_away=110) == "L"


def test_grade_total_push():
    assert _grade_total(predicted_total=230, line=220, actual_home=110, actual_away=110) == "P"


# ── _grade_ml ────────────────────────────────────────────────


def test_grade_ml_home_wins():
    assert _grade_ml(fg_spread=5.0, actual_home=110, actual_away=105) == "W"


def test_grade_ml_home_loses():
    assert _grade_ml(fg_spread=5.0, actual_home=100, actual_away=110) == "L"


def test_grade_ml_away_wins():
    assert _grade_ml(fg_spread=-5.0, actual_home=100, actual_away=110) == "W"


def test_grade_ml_away_loses():
    assert _grade_ml(fg_spread=-5.0, actual_home=110, actual_away=100) == "L"


def test_grade_ml_push():
    assert _grade_ml(fg_spread=5.0, actual_home=105, actual_away=105) == "P"


# ── _grade_1h_winner ────────────────────────────────────────


def test_grade_1h_winner_home_wins():
    assert _grade_1h_winner(h1_spread=4.0, actual_home_1h=55, actual_away_1h=50) == "W"


def test_grade_1h_winner_home_loses():
    assert _grade_1h_winner(h1_spread=4.0, actual_home_1h=50, actual_away_1h=55) == "L"


def test_grade_1h_winner_away_wins():
    assert _grade_1h_winner(h1_spread=-4.0, actual_home_1h=50, actual_away_1h=55) == "W"


def test_grade_1h_winner_away_loses():
    assert _grade_1h_winner(h1_spread=-4.0, actual_home_1h=55, actual_away_1h=50) == "L"


def test_grade_1h_winner_push():
    assert _grade_1h_winner(h1_spread=4.0, actual_home_1h=55, actual_away_1h=55) == "P"


# ── _Record ──────────────────────────────────────────────────


def test_record_add_win():
    r = _Record()
    r.add("W")
    assert r.wins == 1 and r.losses == 0 and r.pushes == 0


def test_record_add_loss():
    r = _Record()
    r.add("L")
    assert r.losses == 1


def test_record_add_push():
    r = _Record()
    r.add("P")
    assert r.pushes == 1


def test_record_total_excludes_pushes():
    r = _Record(wins=3, losses=2, pushes=1)
    assert r.total == 5


def test_record_win_pct():
    r = _Record(wins=6, losses=4)
    assert r.win_pct == 60.0


def test_record_win_pct_none_when_empty():
    r = _Record()
    assert r.win_pct is None


def test_record_roi():
    r = _Record(wins=6, losses=4)
    # profit = 6 * (100/110 * 100) - 4 * 100 = 545.45… - 400 = 145.45
    # wagered = 10 * 100 = 1000; ROI ~14.5%
    roi = r.roi
    assert roi is not None
    assert abs(roi - 14.5) < 1.0


def test_record_roi_none_when_empty():
    r = _Record()
    assert r.roi is None


def test_record_to_dict():
    r = _Record(wins=3, losses=2, pushes=1)
    d = r.to_dict()
    assert d["record"] == "3-2-1"
    assert d["picks"] == 6
    assert d["win_pct"] == 60.0
    assert d["roi_pct"] is not None


def test_record_to_dict_no_pushes():
    r = _Record(wins=3, losses=2)
    d = r.to_dict()
    assert d["record"] == "3-2"


# ── _build_stats ─────────────────────────────────────────────


def test_build_stats_overall():
    graded = [
        GradedPick("FG", "SPREAD", 3.5, "W"),
        GradedPick("FG", "SPREAD", 5.0, "L"),
        GradedPick("FG", "TOTAL", 4.0, "W"),
        GradedPick("1H", "SPREAD", 3.0, "P"),
    ]
    stats = _build_stats(graded)
    assert stats["overall"]["record"] == "2-1-1"
    assert "FG_SPREAD" in stats["by_market"]
    assert "FG_TOTAL" in stats["by_market"]
    assert "1H_SPREAD" in stats["by_market"]
    # By threshold
    for t in EDGE_THRESHOLDS:
        assert f"{t}+" in stats["by_edge_threshold"]


def test_build_stats_empty():
    stats = _build_stats([])
    assert stats["overall"]["picks"] == 0


# ── _score_accuracy ──────────────────────────────────────────


def _make_pred_game(hp=100, ap=95, ha=102, aa=98, hp1h=50, ap1h=48, ha1h=52, aa1h=49):
    pred = SimpleNamespace(
        predicted_home_fg=hp,
        predicted_away_fg=ap,
        predicted_home_1h=hp1h,
        predicted_away_1h=ap1h,
    )
    game = SimpleNamespace(
        home_score_fg=ha,
        away_score_fg=aa,
        home_score_1h=ha1h,
        away_score_1h=aa1h,
    )
    return pred, game


def test_score_accuracy_basic():
    rows = [_make_pred_game()]
    acc = _score_accuracy(rows)
    assert "score_mae" in acc
    assert "spread_mae" in acc
    assert "total_mae" in acc
    assert "h1_score_mae" in acc
    assert acc["score_mae"] >= 0


def test_score_accuracy_perfect():
    rows = [_make_pred_game(hp=100, ap=90, ha=100, aa=90)]
    acc = _score_accuracy(rows)
    assert acc["score_mae"] == 0.0
    assert acc["spread_mae"] == 0.0
    assert acc["total_mae"] == 0.0


def test_score_accuracy_no_1h():
    pred = SimpleNamespace(
        predicted_home_fg=100, predicted_away_fg=90,
        predicted_home_1h=50, predicted_away_1h=45,
    )
    game = SimpleNamespace(
        home_score_fg=102, away_score_fg=88,
        home_score_1h=None, away_score_1h=None,
    )
    acc = _score_accuracy([(pred, game)])
    assert "h1_score_mae" not in acc


def test_score_accuracy_empty():
    assert _score_accuracy([]) == {}


# ── _clv_summary ─────────────────────────────────────────────


def test_clv_summary_with_data():
    rows = [
        (SimpleNamespace(clv_spread=1.5, clv_total=-0.5), None),
        (SimpleNamespace(clv_spread=-0.5, clv_total=1.0), None),
        (SimpleNamespace(clv_spread=2.0, clv_total=0.5), None),
    ]
    clv = _clv_summary(rows)
    assert "spread" in clv
    assert "total" in clv
    assert clv["spread"]["sample_size"] == 3
    assert clv["spread"]["positive_pct"] > 0
    assert clv["total"]["sample_size"] == 3


def test_clv_summary_empty():
    rows = [(SimpleNamespace(clv_spread=None, clv_total=None), None)]
    clv = _clv_summary(rows)
    assert clv == {}


def test_clv_summary_no_rows():
    assert _clv_summary([]) == {}


# ── _recent_results ──────────────────────────────────────────


def test_recent_results_returns_last_n():
    graded = [GradedPick("FG", "SPREAD", 3.0, "W", "A @ B", "A ATS") for _ in range(100)]
    recent = _recent_results(graded, limit=10)
    assert len(recent) == 10
    assert recent[0]["matchup"] == "A @ B"


def test_recent_results_less_than_limit():
    graded = [GradedPick("FG", "ML", 5.0, "L", "X @ Y", "X ML")]
    recent = _recent_results(graded, limit=50)
    assert len(recent) == 1


# ── _pct_class / _roi_class ─────────────────────────────────


def test_pct_class():
    assert _pct_class(55.0) == "positive"
    assert _pct_class(47.0) == "negative"
    assert _pct_class(50.0) == ""
    assert _pct_class(None) == ""


def test_roi_class():
    assert _roi_class(5.0) == "positive"
    assert _roi_class(-5.0) == "negative"
    assert _roi_class(0.0) == ""
    assert _roi_class(None) == ""


# ── _grade_game (integration) ────────────────────────────────


def _fake_game(home_fg=110, away_fg=100, home_1h=55, away_1h=50):
    return SimpleNamespace(
        home_score_fg=home_fg,
        away_score_fg=away_fg,
        home_score_1h=home_1h,
        away_score_1h=away_1h,
        home_team_id=1,
        away_team_id=2,
        home_team=SimpleNamespace(name="Celtics"),
        away_team=SimpleNamespace(name="Heat"),
    )


def _fake_pred(
    fg_spread=8.0,
    fg_total=232.0,
    h1_spread=4.0,
    h1_total=120.0,
    opening_spread=-5.0,
    opening_total=220.0,
):
    return SimpleNamespace(
        fg_spread=fg_spread,
        fg_total=fg_total,
        h1_spread=h1_spread,
        h1_total=h1_total,
        opening_spread=opening_spread,
        opening_total=opening_total,
    )


def test_grade_game_produces_picks():
    game = _fake_game()
    pred = _fake_pred()
    picks = _grade_game(pred, game)
    assert len(picks) > 0
    segments = {p.segment for p in picks}
    markets = {p.market for p in picks}
    assert "FG" in segments
    assert "SPREAD" in markets


def test_grade_game_no_opening_spread():
    game = _fake_game()
    pred = _fake_pred(opening_spread=None)
    picks = _grade_game(pred, game)
    # Should still have ML pick and total pick
    markets = {p.market for p in picks}
    assert "TOTAL" in markets


def test_grade_game_no_opening_total_high_model():
    game = _fake_game()
    # fg_total far from NBA avg → should produce FG TOTAL via avg fallback
    pred = _fake_pred(opening_total=None, fg_total=240.0)
    picks = _grade_game(pred, game)
    assert any(p.segment == "FG" and p.market == "TOTAL" for p in picks)


def test_grade_game_no_opening_total_near_avg():
    game = _fake_game()
    # fg_total close to NBA avg → no FG TOTAL pick
    pred = _fake_pred(opening_total=None, fg_total=222.0)
    picks = _grade_game(pred, game)
    assert not any(p.segment == "FG" and p.market == "TOTAL" for p in picks)


def test_grade_game_1h_picks():
    game = _fake_game()
    pred = _fake_pred(h1_spread=5.0, h1_total=120.0)
    picks = _grade_game(pred, game)
    h1_picks = [p for p in picks if p.segment == "1H"]
    assert len(h1_picks) >= 1


def test_grade_game_no_1h_scores():
    game = _fake_game(home_1h=None, away_1h=None)
    pred = _fake_pred()
    picks = _grade_game(pred, game)
    assert not any(p.segment == "1H" for p in picks)


def test_grade_game_small_edge_filtered():
    game = _fake_game()
    # opening_spread + fg_spread = edge only 1.0 (below MIN_EDGE=2.0)
    pred = _fake_pred(fg_spread=1.0, opening_spread=-2.0, h1_spread=0.5, fg_total=222.0, h1_total=111.0, opening_total=221.5)
    picks = _grade_game(pred, game)
    # Small edges should be filtered
    fg_spread_picks = [p for p in picks if p.segment == "FG" and p.market == "SPREAD"]
    assert len(fg_spread_picks) == 0


# ── API endpoint tests ──────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_performance_endpoint_returns_200():
    """Performance endpoint should return 200 even with no DB data."""
    from unittest.mock import AsyncMock, MagicMock

    from httpx import ASGITransport, AsyncClient

    from src.api.main import app
    from src.db.session import get_db

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accumulating"
        assert data["games_graded"] == 0
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_performance_dashboard_returns_html():
    from unittest.mock import AsyncMock, MagicMock

    from httpx import ASGITransport, AsyncClient

    from src.api.main import app
    from src.db.session import get_db

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/performance/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "GBSV Performance Dashboard" in resp.text
    finally:
        app.dependency_overrides.clear()


def _make_pred_game_row():
    """Create a (pred, game) tuple mimicking SQLAlchemy row for endpoint tests."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    pred = SimpleNamespace(
        predicted_home_fg=110.0,
        predicted_away_fg=105.0,
        predicted_home_1h=55.0,
        predicted_away_1h=52.0,
        fg_spread=5.0,
        fg_total=215.0,
        h1_spread=3.0,
        h1_total=107.0,
        fg_home_ml_prob=0.65,
        h1_home_ml_prob=0.60,
        opening_spread=-3.5,
        opening_total=220.0,
        opening_h1_spread=None,
        opening_h1_total=None,
        clv_spread=1.0,
        clv_total=-0.5,
        predicted_at=datetime(2024, 12, 1, 15, 0, tzinfo=UTC),
        game_id=1,
        odds_sourced=None,
    )
    game = SimpleNamespace(
        id=1,
        home_team=SimpleNamespace(name="Lakers"),
        away_team=SimpleNamespace(name="Celtics"),
        home_score_fg=112,
        away_score_fg=108,
        home_score_1h=56,
        away_score_1h=50,
        status="FT",
        commence_time=datetime(2024, 12, 1, 19, 0, tzinfo=UTC),
    )
    return pred, game


@pytest.mark.anyio
async def test_performance_endpoint_with_data():
    """Performance endpoint with actual game data returns ok status."""
    from unittest.mock import AsyncMock, MagicMock

    from httpx import ASGITransport, AsyncClient

    from src.api.main import app
    from src.db.session import get_db

    row = _make_pred_game_row()

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["games_graded"] >= 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_performance_dashboard_with_data():
    """Dashboard with actual game data returns full HTML."""
    from unittest.mock import AsyncMock, MagicMock

    from httpx import ASGITransport, AsyncClient

    from src.api.main import app
    from src.db.session import get_db

    row = _make_pred_game_row()

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/performance/dashboard")
        assert resp.status_code == 200
        assert "Overview" in resp.text
        assert "Performance by Market" in resp.text
    finally:
        app.dependency_overrides.clear()
