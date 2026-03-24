"""Rate limiter setup.

Patches Starlette Config._read_file to use utf-8 encoding before slowapi
import, working around Windows cp1252 encoding issues with .env files.
"""

import starlette.config


def _read_file_utf8(self, file_name):  # type: ignore[no-untyped-def]
    try:
        with open(file_name, encoding="utf-8") as f:
            return dict(
                line.strip().split("=", 1)
                for line in f
                if line.strip() and not line.startswith("#") and "=" in line
            )
    except FileNotFoundError:
        return {}


# Patch before slowapi imports Config
starlette.config.Config._read_file = _read_file_utf8  # type: ignore[assignment]

from slowapi import Limiter  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
