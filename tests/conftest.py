import os

# Force test environment for all tests — prevents validation from
# requiring real API keys and ensures .env.test is loaded if present.
os.environ.setdefault("APP_ENV", "test")


import pytest  # noqa: E402

from src.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the lru_cache on get_settings between tests so env
    changes from monkeypatch take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
