"""Shared test fixtures for w2v_common tests.

Provides a real postgres engine (via DATABASE_URL) plus fakeredis fixtures
so tests exercise actual SQL and Redis behavior instead of mocks.
"""

from __future__ import annotations

import pytest
from sqlalchemy import JSON, Integer, String, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tests.ephemeral_postgres import (
    resolve_test_database_url,
    start_session_database,
    stop_session_database,
)
from w2v_common.cache.redis_client import RedisClient, get_redis_client
from w2v_common.testing import inject_fakeredis

# Re-export so existing imports from conftest keep working.
__all__ = ["inject_fakeredis"]

# The test DB: DATABASE_URL if set (CI sets it to its ephemeral postgres); else ONE
# ephemeral docker postgres:18 for the whole SESSION on a random loopback port
# (tests/ephemeral_postgres.py), started in the xdist controller's pytest_configure --
# before workers spawn and before test modules import w2v_common.database.connection
# (which reads DATABASE_URL at import) -- and removed by its own ID in
# pytest_unconfigure (+ atexit, SIGTERM/SIGHUP); else the session stops with a
# one-line message. There is NO host:5432 default: on a dev box it silently hits
# whatever runs there.
_DB_ENV_VARS = ("DATABASE_URL",)
_DB_NAME = "common_ci"


def pytest_configure(config: pytest.Config) -> None:
    """Start the ONE session database in the controller (no-op in xdist workers)."""
    start_session_database(
        env_vars=_DB_ENV_VARS, export_var="DATABASE_URL", dbname=_DB_NAME, config=config
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Controller only: remove the session container by its own ID."""
    stop_session_database(config)


def _test_database_url() -> str:
    return resolve_test_database_url(
        env_vars=_DB_ENV_VARS, export_var="DATABASE_URL", dbname=_DB_NAME
    )


# ---------------------------------------------------------------------------
# Minimal test model for session tests
# ---------------------------------------------------------------------------


class TestBase(DeclarativeBase):
    pass


class SampleItem(TestBase):
    __tablename__ = "test_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


# ---------------------------------------------------------------------------
# Database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_engine():
    """Per-test postgres engine with TRUNCATE-based isolation."""
    engine = create_async_engine(_test_database_url(), echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)
        if TestBase.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(TestBase.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session_factory(test_engine):
    """Real async_sessionmaker bound to the test engine."""
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def test_session(test_session_factory):
    """Real AsyncSession for a single test."""
    async with test_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Redis fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fakeredis_client():
    """RedisClient backed by fakeredis (real commands, in-memory)."""
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client


@pytest.fixture(autouse=True)
def _reset_redis_global():
    """Reset the global Redis client between tests."""
    import w2v_common.cache.redis_client as mod

    mod._redis_client = None
    get_redis_client.cache_clear()
    yield
    mod._redis_client = None
    get_redis_client.cache_clear()
