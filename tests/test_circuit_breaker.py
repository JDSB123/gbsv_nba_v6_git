"""Tests for the circuit breaker – pure logic, no external deps."""

import time
from unittest.mock import patch

from src.data.circuit_breaker import CircuitBreaker

# ── Construction ────────────────────────────────────────────────

def test_initial_state_is_closed():
    cb = CircuitBreaker("test", threshold=3, cooldown_seconds=60)
    assert not cb.is_open
    assert not cb.should_skip()
    assert cb._consecutive_failures == 0


# ── Recording successes ────────────────────────────────────────

def test_record_success_resets_failures():
    cb = CircuitBreaker("test", threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb._consecutive_failures == 2
    cb.record_success()
    assert cb._consecutive_failures == 0
    assert not cb.is_open


# ── Opening the circuit ────────────────────────────────────────

def test_opens_after_threshold_consecutive_failures():
    cb = CircuitBreaker("test", threshold=3, cooldown_seconds=60)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open
    assert cb.should_skip()


def test_stays_closed_below_threshold():
    cb = CircuitBreaker("test", threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert not cb.is_open
    assert not cb.should_skip()


# ── Cooldown / half-open ───────────────────────────────────────

def test_half_open_after_cooldown_expires():
    cb = CircuitBreaker("test", threshold=2, cooldown_seconds=1)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open

    # Simulate time passing past cooldown
    with patch.object(time, "monotonic", return_value=time.monotonic() + 2):
        assert not cb.is_open  # half-open: allows retry
        assert not cb.should_skip()


def test_re_opens_if_failure_after_half_open():
    cb = CircuitBreaker("test", threshold=1, cooldown_seconds=0)
    cb.record_failure()
    # Cooldown is 0 so circuit is already half-open
    assert not cb.is_open
    # Another failure re-triggers (counter still above threshold)
    cb.record_failure()
    # _opened_at was already set; re-records don't double-set
    assert cb._consecutive_failures == 2


# ── Reset after success on half-open ───────────────────────────

def test_success_resets_after_open():
    cb = CircuitBreaker("test", threshold=2, cooldown_seconds=0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert not cb.is_open
    assert cb._consecutive_failures == 0


# ── Shared instances ───────────────────────────────────────────

def test_shared_instances_exist():
    from src.data.circuit_breaker import basketball_api_breaker, odds_api_breaker

    assert basketball_api_breaker.name == "basketball_api"
    assert odds_api_breaker.name == "odds_api"
    assert basketball_api_breaker.threshold == 5
    assert odds_api_breaker.threshold == 5
