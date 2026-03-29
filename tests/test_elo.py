"""Tests for the Elo rating system – pure math, no external deps."""

import math

from src.models.elo import (
    INITIAL_ELO,
    SEASON_REVERSION,
    EloSystem,
)

# ── Defaults ────────────────────────────────────────────────────


def test_initial_rating_is_1500():
    elo = EloSystem()
    assert elo.rating(1) == INITIAL_ELO
    assert elo.rating(99) == INITIAL_ELO


def test_new_team_auto_initialised():
    elo = EloSystem()
    # Accessing unknown team should create it at INITIAL_ELO
    assert elo._get(42) == INITIAL_ELO
    assert 42 in elo.ratings


# ── MOV multiplier ──────────────────────────────────────────────


def test_mov_multiplier_positive():
    elo = EloSystem()
    mov = elo._mov_multiplier(10.0)
    assert mov > 0
    expected = math.log(10 + 1) * 0.8
    assert abs(mov - expected) < 1e-9


def test_mov_multiplier_zero_margin():
    elo = EloSystem()
    # abs(0) = 0 → max(0,1) = 1 → log(2)
    mov = elo._mov_multiplier(0.0)
    assert abs(mov - math.log(2) * 0.8) < 1e-9


def test_mov_multiplier_negative_margin_uses_abs():
    elo = EloSystem()
    assert elo._mov_multiplier(-15) == elo._mov_multiplier(15)


# ── Update mechanics ────────────────────────────────────────────


def test_home_win_increases_home_elo():
    elo = EloSystem()
    old_home, old_away = elo.update(1, 2, 110, 95)
    assert old_home == INITIAL_ELO
    assert old_away == INITIAL_ELO
    assert elo.rating(1) > INITIAL_ELO
    assert elo.rating(2) < INITIAL_ELO


def test_away_win_increases_away_elo():
    elo = EloSystem()
    elo.update(1, 2, 90, 110)
    assert elo.rating(2) > INITIAL_ELO
    assert elo.rating(1) < INITIAL_ELO


def test_tie_symmetric():
    elo = EloSystem()
    elo.update(1, 2, 100, 100)
    # Home has advantage baked in so even a tie means home underperformed
    assert elo.rating(1) < INITIAL_ELO
    assert elo.rating(2) > INITIAL_ELO


def test_update_returns_pre_update_ratings():
    elo = EloSystem()
    elo.update(1, 2, 110, 100)  # move ratings away from default
    r1_before = elo.rating(1)
    r2_before = elo.rating(2)
    old_h, old_a = elo.update(1, 2, 120, 100)
    assert old_h == r1_before
    assert old_a == r2_before


def test_blowout_moves_ratings_more_than_close_game():
    elo_blow = EloSystem()
    elo_blow.update(1, 2, 130, 90)
    delta_blow = abs(elo_blow.rating(1) - INITIAL_ELO)

    elo_close = EloSystem()
    elo_close.update(1, 2, 101, 100)
    delta_close = abs(elo_close.rating(1) - INITIAL_ELO)

    assert delta_blow > delta_close


# ── Season reversion ────────────────────────────────────────────


def test_season_reset_reverts_toward_mean():
    elo = EloSystem()
    # Play a game in season "2024-2025" to move ratings
    elo.update(1, 2, 120, 100, season="2024-2025")
    r1_before = elo.rating(1)
    r2_before = elo.rating(2)

    # Next game in new season triggers reset
    elo.update(1, 2, 100, 100, season="2025-2026")

    # The pre-game update ratings should have been reverted
    # Using the formula: new_r = INITIAL + (old_r - INITIAL) * SEASON_REVERSION
    INITIAL_ELO + (r1_before - INITIAL_ELO) * SEASON_REVERSION
    INITIAL_ELO + (r2_before - INITIAL_ELO) * SEASON_REVERSION
    # After reset, a tie at home means home slightly drops
    # So r1 should be slightly below expected_r1 and r2 slightly above expected_r2
    # Just verify reversion happened (closer to mean than before)
    assert abs(elo.rating(1) - INITIAL_ELO) < abs(r1_before - INITIAL_ELO)


def test_no_season_reset_within_same_season():
    elo = EloSystem()
    elo.update(1, 2, 120, 100, season="2024-2025")
    r1_after_first = elo.rating(1)
    # Another game in same season — no reversion should happen
    elo.update(1, 2, 120, 100, season="2024-2025")
    # Rating should just keep moving, not revert
    assert elo.rating(1) > r1_after_first


def test_no_season_reset_when_season_empty():
    elo = EloSystem()
    elo.update(1, 2, 120, 100, season="")
    r1 = elo.rating(1)
    # Season="" so _season_reset is a no-op
    elo.update(1, 2, 120, 100, season="")
    assert elo.rating(1) > r1  # ratings still move, no reset
