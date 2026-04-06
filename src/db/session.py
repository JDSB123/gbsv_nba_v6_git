import re
from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings


@lru_cache
def _get_engine():
    """Create the async engine lazily on first use (avoids import-time crash)."""
    settings = get_settings()
    # Strip ?ssl= / ?sslmode= from URL — asyncpg requires ssl via connect_args,
    # not as URL query parameters, otherwise it raises TypeError.
    raw_url = settings.database_url
    clean_url = re.sub(r"[?&]ssl(?:mode)?=[^&]*", "", raw_url).rstrip("?&")
    return create_async_engine(
        clean_url,
        echo=(settings.app_env == "development"),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=60,
        pool_recycle=7200,
        pool_pre_ping=True,
        connect_args={"ssl": settings.db_ssl},
    )


@lru_cache
def _get_session_maker():
    return async_sessionmaker(_get_engine(), class_=AsyncSession, expire_on_commit=False)


def async_session_factory():
    """Lazy session factory — creates engine/session-maker on first call."""
    return _get_session_maker()()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
