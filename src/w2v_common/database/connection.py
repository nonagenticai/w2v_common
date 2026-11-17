"""Database connection module with connection pooling and initialization.

All configuration is read from environment variables.
"""

import asyncio
import logging
import os
import re
from urllib.parse import urlparse, urlunparse

from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

from w2v_common.json_safe import json_serializer

logger = logging.getLogger(__name__)

#: There is deliberately NO default DSN. The old fallback was a localhost:5432
#: SUPERUSER DSN: a superuser BYPASSES row-level security, so a missing or typo'd
#: ``DATABASE_URL`` degraded silently into a connection with no tenant isolation, and
#: on a dev box it reached whatever happened to listen on 5432. ``config/base.Settings``
#: has no default either; both paths now fail loud.
DATABASE_URL = os.getenv("DATABASE_URL", "")

#: True when nothing configured a DSN (unset or empty).
_DATABASE_URL_IS_UNSET = not DATABASE_URL


def _assert_dsn_is_configured() -> None:
    """Refuse to CONNECT without a configured ``DATABASE_URL``.

    Checked lazily at engine creation rather than at import: this module is imported by
    every service in the fleet, including in contexts that never open a connection, so
    an import-time raise would be a far blunter instrument than the defect warrants.
    """
    if _DATABASE_URL_IS_UNSET:
        raise RuntimeError(
            "DATABASE_URL is not set. w2v_common has no default DSN (a localhost "
            "superuser fallback would bypass row-level security); set DATABASE_URL to "
            "the service's own database."
        )


parsed_url = urlparse(DATABASE_URL)
DB_HOST = parsed_url.hostname or "localhost"
DB_PORT = parsed_url.port or 5432
DB_NAME = parsed_url.path.lstrip("/") if parsed_url.path else "postgres"
DB_USER = parsed_url.username or "postgres"
DB_PASSWORD = parsed_url.password or ""

DB_POOL_MIN_SIZE = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))
DB_POOL_TIMEOUT = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("POSTGRES_POOL_RECYCLE", "3600"))
DB_ADMIN_DB = os.getenv("POSTGRES_ADMIN_DB", "postgres")

_engine: AsyncEngine | None = None
# Serialises lazy initialisation in `get_engine()` — without it, two
# concurrent first-callers both see `_engine is None`, both invoke
# `create_engine_with_retry()`, and the slower one overwrites the
# faster one's engine — leaking the discarded pool's connections.
_engine_init_lock: asyncio.Lock | None = None


def _validate_db_name() -> None:
    """Validate the database name uses only safe characters."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", DB_NAME):
        raise ValueError(
            f"Database name '{DB_NAME}' contains invalid characters; "
            f"only alphanumeric and underscore are supported."
        )


def _build_admin_url() -> str:
    """Build the admin database URL from DATABASE_URL."""
    admin_url_parts = list(urlparse(DATABASE_URL))
    admin_url_parts[2] = f"/{DB_ADMIN_DB}"
    return urlunparse(admin_url_parts)


def _create_or_check_database(conn: Connection) -> None:
    """Check existence and create the target database if missing.

    Returns silently when the database exists or has been created.
    """
    result = conn.execute(
        text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
        {"db_name": DB_NAME},
    )
    if result.scalar() is not None:
        logger.info("Database '%s' already exists", DB_NAME)
        return
    try:
        conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
        # ⚠️ Born LOCKED. Postgres has NO event trigger for CREATE DATABASE and a new
        # database does NOT inherit template1's ACL, so a bare CREATE DATABASE is born
        # open to PUBLIC — every login role on the cluster can connect. That is the same
        # mechanism that left four service databases readable for 15 days in July 2026.
        # scripts/host-configs/master/postgres/provision-service-db.sh is the supported
        # path and does this too; this branch exists for first-boot/dev, so it must not
        # be the one that creates an unlocked database.
        conn.execute(text(f"REVOKE CONNECT ON DATABASE {DB_NAME} FROM PUBLIC"))
        logger.info("Database '%s' created successfully (CONNECT revoked from PUBLIC)", DB_NAME)
    except ProgrammingError as e:
        if "already exists" in str(e).lower():
            return
        raise


def _dispose_engine_quietly(engine: Engine | None) -> None:
    """Dispose of an engine without raising on cleanup failures."""
    if engine is None:
        return
    try:
        engine.dispose()
    except OSError, ConnectionError, TimeoutError, RuntimeError:
        logger.debug("Failed to dispose temporary engine")


async def _try_ensure_database_once(sync_admin_url: str) -> bool:
    """Run a single attempt to ensure the database exists.

    Returns True on success. Raises on unexpected errors. Returns False on
    transient connection failures so the caller can retry.
    """
    engine = None
    try:
        engine = create_engine(sync_admin_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
        with engine.connect() as conn:
            _create_or_check_database(conn)
        return True
    # ⚠️ ORDER MATTERS. TimeoutError, ConnectionError and ConnectionRefusedError are
    # ALL OSError subclasses, so a leading bare `OSError` handler makes the loud
    # branch below unreachable for two of its three types: a genuine connect
    # timeout returned False silently, `_ensure_database_exists` never assigned
    # `last_error`, and the terminal message printed "Last error: None" — a
    # diagnosis-free failure. Specific/loud first, broad/quiet-retry last.
    except OperationalError, ConnectionRefusedError:
        # Nothing is listening yet — ordinary cold-start, retry quietly.
        return False
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        logger.error("Unexpected error ensuring database exists: %s", e, exc_info=True)
        raise
    except OSError:
        return False
    finally:
        _dispose_engine_quietly(engine)


async def _ensure_database_exists(max_retries: int = 3) -> None:
    """Ensure the target database exists, creating it if necessary."""
    _validate_db_name()
    sync_admin_url = _build_admin_url()

    logger.info(
        "Ensuring PostgreSQL database '%s' exists using admin database '%s' at %s:%s",
        DB_NAME,
        DB_ADMIN_DB,
        DB_HOST,
        DB_PORT,
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if await _try_ensure_database_once(sync_admin_url):
                return
        except (OperationalError, OSError, ConnectionRefusedError) as e:
            last_error = e
        if attempt < max_retries:
            await asyncio.sleep(attempt * 2)

    raise ConnectionError(
        f"Failed to connect to admin database at {DB_HOST}:{DB_PORT} "
        f"after {max_retries} attempts. Last error: {last_error}"
    )


def _build_pool_config() -> dict[str, object]:
    """Build the keyword arguments for :func:`create_async_engine`.

    ``json_serializer`` is NOT optional. Without it SQLAlchemy falls back to a
    bare ``json.dumps`` (``allow_nan=True``), so a ``-inf``/``NaN`` anywhere
    inside a JSONB value is emitted as the tokens ``-Infinity``/``NaN``, which
    Postgres rejects with ``invalid input syntax for type json``. The row never
    writes and the transaction aborts — the exact mechanism behind a silent
    audio track (``loudnorm`` -> ``-inf``) wedging a completion write and
    triggering a re-render of already-delivered output.
    """
    return {
        "poolclass": AsyncAdaptedQueuePool,
        "pool_size": DB_POOL_MIN_SIZE,
        "max_overflow": DB_POOL_MAX_SIZE - DB_POOL_MIN_SIZE,
        "pool_timeout": DB_POOL_TIMEOUT,
        "pool_recycle": DB_POOL_RECYCLE,
        "pool_pre_ping": True,
        "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
        "json_serializer": json_serializer,
    }


async def create_engine_with_retry(max_retries: int = 5, retry_delay: float = 2.0) -> AsyncEngine:
    """Create database engine with connection retry logic.

    Args:
        max_retries: Maximum number of connection attempts.
        retry_delay: Initial delay between retries (doubles each attempt).

    Returns:
        A connected AsyncEngine instance.

    Raises:
        ConnectionError: If all retry attempts fail.
    """
    _assert_dsn_is_configured()
    last_error = None
    current_delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            engine = create_async_engine(DATABASE_URL, **_build_pool_config())
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Successfully created engine on attempt %d", attempt)
            return engine

        except (OperationalError, ProgrammingError) as e:
            logger.warning(
                "Database connection attempt %d/%d failed: %s",
                attempt,
                max_retries,
                e,
            )
            error_msg = str(e).lower()
            if "database" in error_msg and (
                "does not exist" in error_msg or "no such database" in error_msg
            ):
                try:
                    await _ensure_database_exists()
                    engine = create_async_engine(DATABASE_URL, **_build_pool_config())
                    async with engine.begin() as conn:
                        await conn.execute(text("SELECT 1"))
                    return engine
                except (
                    OSError,
                    ConnectionError,
                    RuntimeError,
                    OperationalError,
                    ProgrammingError,
                ) as create_err:
                    last_error = create_err
            else:
                last_error = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

        except (TimeoutError, OSError, ConnectionRefusedError) as e:
            logger.warning(
                "Database connection attempt %d/%d failed (network): %s",
                attempt,
                max_retries,
                e,
            )
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

        except (ConnectionError, RuntimeError, ValueError, TypeError, KeyError) as e:
            logger.warning(
                "Database connection attempt %d/%d failed (unexpected): %s",
                attempt,
                max_retries,
                e,
            )
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

    raise ConnectionError(
        f"Failed to connect to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} "
        f"after {max_retries} attempts. Last error: {last_error}"
    ) from last_error


async def get_engine() -> AsyncEngine:
    """Get or create the global database engine instance.

    Thread-safe within a single asyncio event loop: the lazy-init path
    is serialised behind ``_engine_init_lock`` so concurrent
    first-callers don't each spin up a separate engine and leak the
    discarded pool's connections.
    """
    global _engine, _engine_init_lock
    if _engine is not None:
        return _engine
    if _engine_init_lock is None:
        # Lazy-construct the lock too — `asyncio.Lock()` must be created
        # from within a running event loop, so it can't be a module-level
        # initializer.
        _engine_init_lock = asyncio.Lock()
    async with _engine_init_lock:
        # Double-check pattern: another coroutine may have populated
        # `_engine` while we were waiting for the lock.
        if _engine is None:
            _engine = await create_engine_with_retry()
    return _engine


async def health_check() -> bool:
    """Perform a health check on the database connection."""
    try:
        engine = await get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except (
        OSError,
        ConnectionError,
        TimeoutError,
        RuntimeError,
        OperationalError,
        ProgrammingError,
    ) as e:
        logger.error("Database health check failed: %s", e, exc_info=True)
        return False


async def close_engine() -> None:
    """Close the global database engine."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine closed successfully")
