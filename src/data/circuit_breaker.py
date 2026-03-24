"""Lightweight circuit breaker for external API calls.

Tracks consecutive failures per named service.  After ``threshold``
consecutive failures the circuit opens and calls are skipped for
``cooldown_seconds``.  A successful call resets the counter.
"""

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, name: str, threshold: int = 5, cooldown_seconds: int = 600) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        # Half-open: allow a single attempt after cooldown
        return elapsed < self.cooldown_seconds

    def record_success(self) -> None:
        if self._consecutive_failures > 0:
            logger.info("Circuit breaker [%s] reset after success", self.name)
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker [%s] OPEN after %d consecutive failures (cooldown %ds)",
                self.name,
                self._consecutive_failures,
                self.cooldown_seconds,
            )

    def should_skip(self) -> bool:
        if self.is_open:
            logger.warning(
                "Circuit breaker [%s] is open, skipping call", self.name
            )
            return True
        return False


# Shared instances for each external API
basketball_api_breaker = CircuitBreaker("basketball_api")
odds_api_breaker = CircuitBreaker("odds_api")
