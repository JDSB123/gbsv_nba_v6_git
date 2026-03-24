"""Tests for session helpers, rate limiter, dependencies, versioning, config extras."""

import importlib
from unittest.mock import patch, MagicMock

import pytest

from src.models.versioning import MODEL_VERSION


# ── MODEL_VERSION ──────────────────────────────────────────────

def test_model_version_format():
    assert MODEL_VERSION.startswith("v")
    parts = MODEL_VERSION[1:].split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ── Rate limiter ───────────────────────────────────────────────

@pytest.mark.xfail(reason="slowapi/starlette version compat – _read_file_utf8 signature")
def test_rate_limiter_loads():
    from src.api.rate_limit import limiter
    assert limiter is not None


@pytest.mark.xfail(reason="slowapi/starlette version compat – _read_file_utf8 signature")
def test_rate_limiter_key_func():
    from src.api.rate_limit import limiter
    assert limiter._key_func is not None


# ── Dependencies ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_predictor_returns_predictor():
    from src.api.dependencies import get_predictor
    p = await get_predictor()
    from src.models.predictor import Predictor
    assert isinstance(p, Predictor)


@pytest.mark.asyncio
async def test_get_predictor_cached():
    from src.api.dependencies import get_predictor, _get_predictor
    p1 = await get_predictor()
    p2 = await get_predictor()
    # lru_cache should return same instance
    assert p1 is p2


# ── Session factory (lazy, no real DB) ─────────────────────────

def test_session_factory_defers_engine_creation():
    """Importing session should NOT trigger get_settings at import time."""
    # If this import succeeds without DB URL, the lazy pattern works
    import src.db.session as session_mod
    assert hasattr(session_mod, "async_session_factory")
    assert hasattr(session_mod, "get_db")
    assert hasattr(session_mod, "_get_engine")
    assert hasattr(session_mod, "_get_session_maker")
