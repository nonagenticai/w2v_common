"""Tests for w2v_common.database.session.

Uses a real Postgres (psycopg3) engine via the ``DATABASE_URL`` conftest
fixture to verify that get_db() actually commits on success and rolls back
on exception, rather than just checking that mock methods were called.
(No SQLite/aiosqlite anywhere — those helpers were removed in 0.8.0.)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

import w2v_common.database.session as session_mod
from w2v_common.database.session import (
    close_session_maker,
    get_db,
    get_db_session,
    get_session_maker,
)

from .conftest import SampleItem


@pytest.fixture(autouse=True)
def _reset_session_maker():
    """Reset the global session maker between tests."""
    session_mod._async_session_maker = None
    yield
    session_mod._async_session_maker = None


# ---------------------------------------------------------------------------
# get_session_maker
# ---------------------------------------------------------------------------


class TestGetSessionMaker:
    """Tests for get_session_maker."""

    @patch("w2v_common.database.session.get_engine")
    async def test_creates_real_session_maker(self, mock_get_engine, test_engine):
        """get_session_maker() returns a real async_sessionmaker that can produce sessions."""
        mock_get_engine.return_value = test_engine

        maker = await get_session_maker()

        from sqlalchemy.ext.asyncio import async_sessionmaker

        assert isinstance(maker, async_sessionmaker)

        # Verify the maker can actually create a usable session
        async with maker() as session:
            session.add(SampleItem(name="maker-test"))
            await session.commit()

        # Verify the item was persisted
        async with maker() as session:
            result = await session.execute(
                select(SampleItem).where(SampleItem.name == "maker-test")
            )
            item = result.scalar_one_or_none()
            assert item is not None
            assert item.name == "maker-test"

    @patch("w2v_common.database.session.get_engine")
    async def test_returns_cached_session_maker(self, mock_get_engine, test_engine):
        mock_get_engine.return_value = test_engine

        maker1 = await get_session_maker()
        maker2 = await get_session_maker()
        assert maker1 is maker2
        mock_get_engine.assert_called_once()


# ---------------------------------------------------------------------------
# get_db (FastAPI dependency)
# ---------------------------------------------------------------------------


class TestGetDb:
    """Tests for get_db FastAPI dependency using real database."""

    @patch("w2v_common.database.session.get_session_maker")
    async def test_commits_on_success(self, mock_get_maker, test_session_factory, test_engine):
        """get_db() commits the transaction when the generator exits normally."""
        mock_get_maker.return_value = test_session_factory

        gen = get_db()
        session = await gen.__anext__()

        # Insert a row inside the generator
        session.add(SampleItem(name="commit-test", data={"key": "value"}))
        await session.flush()

        # Exit normally → should commit
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

        # Verify the row was actually committed to the database
        async with test_session_factory() as verify_session:
            result = await verify_session.execute(
                select(SampleItem).where(SampleItem.name == "commit-test")
            )
            item = result.scalar_one_or_none()
            assert item is not None
            assert item.name == "commit-test"
            assert item.data == {"key": "value"}

    @patch("w2v_common.database.session.get_session_maker")
    async def test_rolls_back_on_exception(self, mock_get_maker, test_session_factory, test_engine):
        """get_db() rolls back when an exception is thrown into the generator."""
        mock_get_maker.return_value = test_session_factory

        gen = get_db()
        session = await gen.__anext__()

        # Insert a row, then throw an exception
        session.add(SampleItem(name="rollback-test"))
        await session.flush()

        with pytest.raises(ValueError, match="test error"):
            await gen.athrow(ValueError("test error"))

        # Verify the row was NOT committed (rolled back)
        async with test_session_factory() as verify_session:
            result = await verify_session.execute(
                select(SampleItem).where(SampleItem.name == "rollback-test")
            )
            item = result.scalar_one_or_none()
            assert item is None


# ---------------------------------------------------------------------------
# get_db_session (context manager)
# ---------------------------------------------------------------------------


class TestGetDbSession:
    """Tests for get_db_session context manager using real database."""

    @patch("w2v_common.database.session.get_session_maker")
    async def test_commits_on_success(self, mock_get_maker, test_session_factory, test_engine):
        """get_db_session() commits when the context exits normally."""
        mock_get_maker.return_value = test_session_factory

        async with get_db_session() as session:
            session.add(SampleItem(name="ctx-commit", data={"nested": [1, 2, 3]}))
            await session.flush()

        # Verify persisted
        async with test_session_factory() as verify_session:
            result = await verify_session.execute(
                select(SampleItem).where(SampleItem.name == "ctx-commit")
            )
            item = result.scalar_one_or_none()
            assert item is not None
            assert item.data == {"nested": [1, 2, 3]}

    @patch("w2v_common.database.session.get_session_maker")
    async def test_rolls_back_on_exception(self, mock_get_maker, test_session_factory, test_engine):
        """get_db_session() rolls back when an exception is raised."""
        mock_get_maker.return_value = test_session_factory

        with pytest.raises(ValueError, match="ctx error"):
            async with get_db_session() as session:
                session.add(SampleItem(name="ctx-rollback"))
                await session.flush()
                raise ValueError("ctx error")

        # Verify NOT persisted
        async with test_session_factory() as verify_session:
            result = await verify_session.execute(
                select(SampleItem).where(SampleItem.name == "ctx-rollback")
            )
            assert result.scalar_one_or_none() is None

    @patch("w2v_common.database.session.get_session_maker")
    async def test_multiple_inserts_committed(self, mock_get_maker, test_session_factory):
        """Multiple inserts within a single session are all committed."""
        mock_get_maker.return_value = test_session_factory

        async with get_db_session() as session:
            session.add(SampleItem(name="multi-1"))
            session.add(SampleItem(name="multi-2"))
            session.add(SampleItem(name="multi-3"))

        async with test_session_factory() as verify_session:
            result = await verify_session.execute(
                select(SampleItem).where(SampleItem.name.like("multi-%"))
            )
            items = result.scalars().all()
            assert len(items) == 3
            names = {item.name for item in items}
            assert names == {"multi-1", "multi-2", "multi-3"}


# ---------------------------------------------------------------------------
# close_session_maker
# ---------------------------------------------------------------------------


class TestCloseSessionMaker:
    """Tests for close_session_maker."""

    async def test_resets_session_maker(self):
        session_mod._async_session_maker = AsyncMock()
        await close_session_maker()
        assert session_mod._async_session_maker is None

    async def test_close_when_none(self):
        await close_session_maker()
