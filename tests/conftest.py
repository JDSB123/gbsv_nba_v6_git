import os

# Force test environment for all tests so secret validation stays disabled.
os.environ.setdefault("APP_ENV", "test")
# Disable API-key auth in tests; individual tests that verify auth behaviour
# set the key explicitly via monkeypatch or mock.
os.environ.pop("API_KEY", None)


import pytest  # noqa: E402

from src.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the lru_cache on get_settings between tests so env
    changes from monkeypatch take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
