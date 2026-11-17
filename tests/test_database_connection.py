"""Tests for w2v_common.database.connection."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.pool import NullPool

import w2v_common.database.connection as conn_mod
from w2v_common.database.connection import (
    _ensure_database_exists,
    close_engine,
    create_engine_with_retry,
    get_engine,
    health_check,
)


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the global engine between tests."""
    conn_mod._engine = None
    yield
    conn_mod._engine = None


def _make_mock_engine(execute_side_effect=None):
    """Create a mock engine with properly configured begin() context manager."""
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    if execute_side_effect:
        mock_conn.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        mock_conn.execute = AsyncMock()

    @asynccontextmanager
    async def mock_begin():
        yield mock_conn

    mock_engine.begin = mock_begin
    mock_engine.dispose = AsyncMock()
    return mock_engine


class TestModuleLevelConfig:
    """Tests for module-level configuration parsing from DATABASE_URL.

    Module-level constants are burned in at first import. CI sets
    DATABASE_URL to a real postgres URL (per cluster standard "no SQLite
    anywhere"), so these assertions verify the parser produced sensible
    output rather than pinning specific host/user/db names that would
    drift between local-dev and CI.
    """

    def test_db_host_parsed_from_default_url(self):
        assert isinstance(conn_mod.DB_HOST, str) and conn_mod.DB_HOST

    def test_db_port_parsed_from_default_url(self):
        assert isinstance(conn_mod.DB_PORT, int) and conn_mod.DB_PORT > 0

    def test_db_name_parsed_from_default_url(self):
        assert isinstance(conn_mod.DB_NAME, str) and conn_mod.DB_NAME

    def test_db_user_parsed_from_default_url(self):
        assert isinstance(conn_mod.DB_USER, str) and conn_mod.DB_USER

    def test_pool_sizes_have_sensible_defaults(self):
        assert conn_mod.DB_POOL_MIN_SIZE == 2
        assert conn_mod.DB_POOL_MAX_SIZE == 10
        assert conn_mod.DB_POOL_TIMEOUT == 30
        assert conn_mod.DB_POOL_RECYCLE == 3600

    def test_admin_db_defaults_to_postgres(self):
        assert conn_mod.DB_ADMIN_DB == "postgres"


class TestEnsureDatabaseExists:
    """Tests for _ensure_database_exists."""

    async def test_invalid_db_name(self):
        """Test that invalid database names are rejected."""
        original = conn_mod.DB_NAME
        try:
            conn_mod.DB_NAME = "bad-name!"
            with pytest.raises(ValueError, match="invalid characters"):
                await _ensure_database_exists()
        finally:
            conn_mod.DB_NAME = original

    @patch("w2v_common.database.connection.create_engine")
    async def test_database_already_exists(self, mock_create_engine):
        """Test that existing database is detected."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1  # database exists
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()
        mock_create_engine.return_value = mock_engine

        await _ensure_database_exists()
        # Should have queried pg_database
        assert mock_conn.execute.call_count == 1

    @patch("w2v_common.database.connection.create_engine")
    async def test_database_created(self, mock_create_engine):
        """Test that database is created when it doesn't exist."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None  # database doesn't exist
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()
        mock_create_engine.return_value = mock_engine

        await _ensure_database_exists()
        # SELECT, CREATE DATABASE, then REVOKE CONNECT ... FROM PUBLIC. The revoke is
        # not optional: Postgres has no event trigger for CREATE DATABASE and a new
        # database does not inherit template1's ACL, so a bare CREATE is born open to
        # PUBLIC and every login role on the cluster can connect.
        assert mock_conn.execute.call_count == 3
        statements = " ".join(str(call.args[0]) for call in mock_conn.execute.call_args_list)
        assert "CREATE DATABASE" in statements
        assert "REVOKE CONNECT" in statements
        assert "FROM PUBLIC" in statements

    @patch("w2v_common.database.connection.create_engine")
    async def test_database_create_race_condition(self, mock_create_engine):
        """Test handling of race condition where DB is created between check and create."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None  # database doesn't exist

        call_count = 0

        def execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_result
            # Second call is CREATE DATABASE - raise "already exists"
            raise ProgrammingError("", {}, Exception("database already exists"))

        mock_conn.execute.side_effect = execute_side_effect
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        mock_engine.dispose = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Should not raise
        await _ensure_database_exists()

    @patch("w2v_common.database.connection.asyncio.sleep", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.create_engine")
    async def test_connection_retry_on_operational_error(self, mock_create_engine, mock_sleep):
        """Test retry on OperationalError."""
        mock_create_engine.side_effect = OperationalError("", {}, Exception("conn refused"))

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await _ensure_database_exists(max_retries=2)

    @patch("w2v_common.database.connection.create_engine")
    async def test_unexpected_error_propagated(self, mock_create_engine):
        """Test that unexpected errors are propagated."""
        mock_create_engine.side_effect = RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            await _ensure_database_exists()


class TestCreateEngineWithRetry:
    """Tests for create_engine_with_retry."""

    @patch("w2v_common.database.connection.create_async_engine")
    async def test_success_on_first_attempt(self, mock_create):
        """Test successful engine creation."""
        mock_engine = _make_mock_engine()
        mock_create.return_value = mock_engine

        engine = await create_engine_with_retry(max_retries=1)
        assert engine is mock_engine

    @patch("w2v_common.database.connection.asyncio.sleep", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.create_async_engine")
    async def test_retry_on_timeout_error(self, mock_create, mock_sleep):
        """Test retry on TimeoutError."""
        mock_create.side_effect = TimeoutError("timed out")

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await create_engine_with_retry(max_retries=2, retry_delay=0.01)

    @patch("w2v_common.database.connection.asyncio.sleep", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.create_async_engine")
    async def test_retry_on_generic_exception(self, mock_create, mock_sleep):
        """Test retry on unexpected RuntimeError."""
        mock_create.side_effect = RuntimeError("something wrong")

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await create_engine_with_retry(max_retries=2, retry_delay=0.01)

    @patch("w2v_common.database.connection._ensure_database_exists", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.asyncio.sleep", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.create_async_engine")
    async def test_database_does_not_exist_triggers_creation(
        self, mock_create, mock_sleep, mock_ensure_db
    ):
        """Test that 'database does not exist' triggers creation."""
        error = OperationalError("", {}, Exception('database "mydb" does not exist'))
        mock_create.side_effect = error
        mock_ensure_db.side_effect = ConnectionError("still can't create")

        with pytest.raises(ConnectionError):
            await create_engine_with_retry(max_retries=2, retry_delay=0.01)
        mock_ensure_db.assert_called()

    @patch("w2v_common.database.connection.asyncio.sleep", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.create_async_engine")
    async def test_retry_on_operational_error(self, mock_create, mock_sleep):
        """Test retry on OperationalError (non 'does not exist')."""
        error = OperationalError("", {}, Exception("connection refused"))
        mock_create.side_effect = error

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await create_engine_with_retry(max_retries=2, retry_delay=0.01)

    @patch("w2v_common.database.connection._ensure_database_exists", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.asyncio.sleep", new_callable=AsyncMock)
    @patch("w2v_common.database.connection.create_async_engine")
    async def test_db_creation_then_retry_succeeds(self, mock_create, mock_sleep, mock_ensure_db):
        """Test successful recovery after database creation."""
        error = OperationalError("", {}, Exception('database "mydb" does not exist'))
        mock_engine = _make_mock_engine()
        # First call raises, second (after DB creation) works
        mock_create.side_effect = [error, mock_engine]
        # ensure_database_exists succeeds but engine re-creation also needs to work
        # Actually the code creates a new engine inside the except block
        # Let's just make ensure_db succeed and the inner create_async_engine also succeed
        mock_ensure_db.return_value = None

        # The code in the except block calls create_async_engine again
        # Since mock_create.side_effect is a list, second call returns mock_engine
        engine = await create_engine_with_retry(max_retries=2, retry_delay=0.01)
        assert engine is mock_engine


class TestGetEngine:
    """Tests for get_engine."""

    @patch("w2v_common.database.connection.create_engine_with_retry")
    async def test_creates_engine_on_first_call(self, mock_create):
        mock_engine = AsyncMock()
        mock_create.return_value = mock_engine

        engine = await get_engine()
        assert engine is mock_engine
        mock_create.assert_called_once()

    @patch("w2v_common.database.connection.create_engine_with_retry")
    async def test_returns_cached_engine(self, mock_create):
        mock_engine = AsyncMock()
        mock_create.return_value = mock_engine

        engine1 = await get_engine()
        engine2 = await get_engine()
        assert engine1 is engine2
        mock_create.assert_called_once()


class TestHealthCheck:
    """Tests for health_check."""

    @patch("w2v_common.database.connection.get_engine")
    async def test_healthy(self, mock_get_engine):
        mock_engine = _make_mock_engine()
        mock_get_engine.return_value = mock_engine

        result = await health_check()
        assert result is True

    @patch("w2v_common.database.connection.get_engine")
    async def test_unhealthy(self, mock_get_engine):
        mock_get_engine.side_effect = ConnectionError("connection failed")

        result = await health_check()
        assert result is False


class TestCloseEngine:
    """Tests for close_engine."""

    async def test_close_when_engine_exists(self):
        mock_engine = AsyncMock()
        conn_mod._engine = mock_engine
        await close_engine()
        mock_engine.dispose.assert_called_once()
        assert conn_mod._engine is None

    async def test_close_when_no_engine(self):
        # Should not raise
        await close_engine()


# ---------------------------------------------------------------------------
# Lazy-init race regression
# ---------------------------------------------------------------------------


async def test_get_engine_concurrent_first_call_singleton(monkeypatch):
    """Regression: 50 concurrent first-callers must all share ONE engine.

    Pre-fix, the lazy-init in `get_engine()` had an `await` between the
    `_engine is None` check and the assignment — so every coroutine that
    arrived during the await would also see `None` and spin up its own
    engine, leaking the discarded pools' connections.
    """
    import asyncio as _asyncio
    from unittest.mock import AsyncMock

    from w2v_common.database import connection as conn_mod

    # Reset module state.
    conn_mod._engine = None
    conn_mod._engine_init_lock = None

    # Stub the slow creator with a function that sleeps and tags each call,
    # so we can detect duplicate construction.
    creation_count = 0

    async def _slow_create():
        nonlocal creation_count
        creation_count += 1
        await _asyncio.sleep(0.05)
        return f"engine-instance-{creation_count}"

    monkeypatch.setattr(conn_mod, "create_engine_with_retry", AsyncMock(side_effect=_slow_create))

    # Fire 50 concurrent first-callers.
    results = await _asyncio.gather(*(conn_mod.get_engine() for _ in range(50)))

    # All 50 must resolve to the SAME engine, and the creator must have
    # been invoked exactly once.
    assert all(r == results[0] for r in results), "concurrent callers got different engines"
    assert creation_count == 1, f"expected 1 engine construction, got {creation_count}"

    # Cleanup so this test doesn't leak state into the next.
    conn_mod._engine = None
    conn_mod._engine_init_lock = None


class TestJsonbNonFiniteSerializer:
    """Regression: a non-finite float must not reach Postgres as an invalid token.

    Without ``json_serializer`` on the engine, SQLAlchemy falls back to a bare
    ``json.dumps`` (``allow_nan=True``) and a JSONB bind renders
    ``{"score": -Infinity, "n": NaN}``. Postgres answers ``invalid input syntax
    for type json ... Token "-Infinity" is invalid``: the row never writes and
    the transaction aborts. Asserted at the REAL boundary — the configured
    engine's own dialect, through ``JSONB().bind_processor``.
    """

    @staticmethod
    def _strict_loads(payload: str):
        """json.loads that REJECTS NaN/Infinity, the way Postgres does."""

        def _reject(token: str):
            raise ValueError(f'Token "{token}" is invalid')

        return json.loads(payload, parse_constant=_reject)

    def _bind(self, value):
        engine = create_engine(
            "postgresql+psycopg://u:p@localhost:5432/db",  # pragma: allowlist secret
            poolclass=NullPool,
            **{
                key: val
                for key, val in conn_mod._build_pool_config().items()
                if key == "json_serializer"
            },
        )
        try:
            return JSONB().bind_processor(engine.dialect)(value)
        finally:
            engine.dispose()

    def test_pool_config_declares_a_json_serializer(self):
        assert callable(conn_mod._build_pool_config().get("json_serializer"))

    def test_non_finite_floats_bind_as_json_null(self):
        rendered = self._bind({"score": float("-inf"), "n": float("nan"), "up": float("inf")})
        assert "Infinity" not in rendered
        assert "NaN" not in rendered
        assert self._strict_loads(rendered) == {"score": None, "n": None, "up": None}

    def test_non_finite_floats_nested_in_containers_bind_as_json_null(self):
        rendered = self._bind({"m": [{"lra": float("nan")}, 1.5], "t": (float("-inf"), "ok")})
        assert self._strict_loads(rendered) == {"m": [{"lra": None}, 1.5], "t": [None, "ok"]}

    def test_finite_payload_is_unchanged(self):
        payload = {"a": 1, "b": 2.5, "c": "x", "d": None, "e": True, "f": [1, 2]}
        assert self._strict_loads(self._bind(payload)) == payload


class TestEnsureDatabaseExistsReportsTheRealError:
    """Regression: a connect timeout must not be swallowed into ``Last error: None``.

    ``TimeoutError``/``ConnectionError``/``ConnectionRefusedError`` are all
    ``OSError`` subclasses, so a leading bare ``OSError`` handler made the loud
    branch unreachable and ``_try_ensure_database_once`` returned False without
    the caller ever assigning ``last_error`` — the terminal message carried no
    diagnosis at all.
    """

    def test_oserror_subclass_relationships_hold(self):
        # The premise of the ordering fix, asserted by execution rather than belief.
        assert issubclass(TimeoutError, OSError)
        assert issubclass(ConnectionError, OSError)
        assert issubclass(ConnectionRefusedError, ConnectionError)

    async def test_connect_timeout_is_named_in_the_terminal_error(self):
        with (
            patch.object(
                conn_mod, "create_engine", side_effect=TimeoutError("connect timed out after 5s")
            ),
            pytest.raises(ConnectionError) as excinfo,
        ):
            await _ensure_database_exists(max_retries=1)
        message = str(excinfo.value)
        assert "Last error: None" not in message
        assert "connect timed out after 5s" in message

    async def test_connection_refused_still_retries_quietly(self):
        with (
            patch.object(conn_mod, "create_engine", side_effect=ConnectionRefusedError("refused")),
            pytest.raises(ConnectionError),
        ):
            await _ensure_database_exists(max_retries=1)
