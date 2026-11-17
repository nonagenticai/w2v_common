"""Database session management module."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..auth.identity_context import get_tenant_id_from_context
from .connection import get_engine
from .rls import bind_session_tenant

logger = logging.getLogger(__name__)

_async_session_maker: async_sessionmaker[AsyncSession] | None = None
# Serialises lazy initialisation in `get_session_maker()`. Without it,
# concurrent first-callers each `await get_engine()` and then construct
# a separate sessionmaker — same race shape as the engine init.
_session_maker_init_lock: asyncio.Lock | None = None


async def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create the global async session maker.

    Concurrent first-callers are serialised via ``_session_maker_init_lock``
    so only one ``async_sessionmaker`` instance is ever bound to the
    process — same race pattern as ``get_engine()``.
    """
    global _async_session_maker, _session_maker_init_lock
    if _async_session_maker is not None:
        return _async_session_maker
    if _session_maker_init_lock is None:
        _session_maker_init_lock = asyncio.Lock()
    async with _session_maker_init_lock:
        # Double-check inside the lock — another coroutine may have
        # finished initialisation while we were waiting.
        if _async_session_maker is None:
            engine = await get_engine()
            _async_session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
    return _async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency for database sessions.

    When ``TENANT_RLS_MODE=enforce``, binds the per-request ``app.current_tenant``
    GUC from the identity ContextVar (set by ``AuthMiddleware``) so RLS scopes every
    query. No-op at the default ``off`` — zero behaviour change for non-enforcing
    services. (Background paths with no request context fail-closed unless they bind
    the tenant explicitly via :func:`w2v_common.database.rls.bind_session_tenant`.)
    """
    session_maker = await get_session_maker()
    async with session_maker() as session:
        try:
            await bind_session_tenant(session, get_tenant_id_from_context())
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """Context manager for manual database session management.

    Binds ``app.current_tenant`` from the identity ContextVar under
    ``TENANT_RLS_MODE=enforce`` (no-op otherwise) — same as :func:`get_db`. A
    background task that runs outside a request context (empty ContextVar) leaves the
    GUC unset → RLS fail-closes; such tasks must call
    :func:`w2v_common.database.rls.bind_session_tenant` with the known tenant.
    """
    session_maker = await get_session_maker()
    async with session_maker() as session:
        try:
            await bind_session_tenant(session, get_tenant_id_from_context())
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_session_maker() -> None:
    """Reset the global session maker instance."""
    global _async_session_maker
    _async_session_maker = None
