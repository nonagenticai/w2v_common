"""Tests for `w2v_common.database.rls.install_rls_tenant_rebind`.

Exercises the ``after_begin`` GUC re-bind against a real Postgres session
(the fleet standard — no sqlite). The listener is process-global once
installed; it is inert unless ``TENANT_RLS_MODE=enforce`` and a tenant is in
context, so leaving it registered across tests is benign.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
)
from w2v_common.database.rls import install_rls_tenant_rebind


@pytest.fixture(autouse=True)
def _clean_identity():
    clear_identity_context()
    yield
    clear_identity_context()


def test_install_is_idempotent() -> None:
    # Two calls must not raise nor double-register.
    install_rls_tenant_rebind()
    install_rls_tenant_rebind()


async def test_rebind_binds_tenant_under_enforce(monkeypatch, test_session_factory) -> None:
    install_rls_tenant_rebind()
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000abc", license_id="lic", subject_id="sub"
    )

    async with test_session_factory() as session:
        # First statement opens the transaction → after_begin fires → GUC bound.
        await session.execute(text("SELECT 1"))
        bound = (
            await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
        ).scalar()
    assert bound == "0195f48b-0000-7000-8000-000000000abc"


async def test_rebind_noop_when_not_enforcing(monkeypatch, test_session_factory) -> None:
    install_rls_tenant_rebind()
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000cd1", license_id="lic", subject_id="sub"
    )

    async with test_session_factory() as session:
        await session.execute(text("SELECT 1"))
        bound = (
            await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
        ).scalar()
    # Never bound (mode off) → the custom GUC is unset → reads back as NULL.
    assert bound is None


async def test_rebind_binds_when_mode_env_is_unset(monkeypatch, test_session_factory) -> None:
    """FAIL-CLOSED: no ``TENANT_RLS_MODE`` in the environment still re-binds."""
    install_rls_tenant_rebind()
    monkeypatch.delenv("TENANT_RLS_MODE", raising=False)
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000de5", license_id="lic", subject_id="sub"
    )

    async with test_session_factory() as session:
        await session.execute(text("SELECT 1"))
        bound = (
            await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
        ).scalar()
    assert bound == "0195f48b-0000-7000-8000-000000000de5"


async def test_rebind_noop_when_no_tenant(monkeypatch, test_session_factory) -> None:
    install_rls_tenant_rebind()
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    # No tenant in context.
    async with test_session_factory() as session:
        await session.execute(text("SELECT 1"))
        bound = (
            await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
        ).scalar()
    # No tenant in context → listener returns early, GUC stays unset → NULL.
    assert bound is None
