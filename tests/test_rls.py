"""Tests for the generic RLS DDL helpers (w2v_common.database.rls)."""

from __future__ import annotations

import uuid

import sqlalchemy as sa

from w2v_common.database import rls


def _metadata() -> sa.MetaData:
    md = sa.MetaData()
    sa.Table(  # strict: NOT NULL tenant_id
        "jobs",
        md,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Uuid, nullable=False),
    )
    sa.Table(  # global-or-own: NULLABLE tenant_id (shared templates)
        "templates",
        md,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Uuid, nullable=True),
    )
    sa.Table(  # strict: NOT NULL varchar tenant_id (every w2v service uses varchar)
        "vjobs",
        md,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.String(255), nullable=False),
    )
    sa.Table(  # no tenant_id → untouched
        "global_config",
        md,
        sa.Column("id", sa.Integer, primary_key=True),
    )
    return md


def test_tenant_table_names_only_tenant_scoped() -> None:
    assert set(rls.tenant_table_names(_metadata())) == {"jobs", "templates", "vjobs"}


def test_uuid_tenant_casts_guc_but_varchar_does_not() -> None:
    stmts = rls.upgrade_statements(_metadata())
    # uuid column → cast ::uuid
    assert any(
        "CREATE POLICY tenant_isolation ON jobs" in s
        and "tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid" in s
        for s in stmts
    )
    # varchar column → text compare, NO ::uuid cast (the bug youtube_upload caught)
    assert any(
        "CREATE POLICY tenant_isolation ON vjobs" in s
        and "tenant_id = NULLIF(current_setting('app.current_tenant', true), '')" in s
        and "::uuid" not in s
        for s in stmts
    )


def test_strict_table_gets_single_equality_policy() -> None:
    stmts = rls.upgrade_statements(_metadata())
    jobs = [s for s in stmts if " jobs " in f" {s} " or s.endswith("jobs")]
    assert any("ENABLE ROW LEVEL SECURITY" in s and "jobs" in s for s in stmts)
    assert any(
        "CREATE POLICY tenant_isolation ON jobs" in s
        and "tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid" in s
        and "WITH CHECK" in s
        for s in stmts
    )
    # strict table must NOT get the global-read disjunct
    assert not any("ON jobs FOR SELECT" in s and "tenant_id IS NULL" in s for s in stmts)
    assert jobs  # sanity


def test_nullable_table_gets_global_or_own_policies() -> None:
    stmts = rls.upgrade_statements(_metadata())
    assert any(
        "tenant_isolation_read ON templates FOR SELECT" in s and "tenant_id IS NULL" in s
        for s in stmts
    )
    assert any("tenant_isolation_ins ON templates FOR INSERT" in s for s in stmts)
    assert any("tenant_isolation_upd ON templates FOR UPDATE" in s for s in stmts)
    assert any("tenant_isolation_del ON templates FOR DELETE" in s for s in stmts)


def test_no_force_anywhere() -> None:
    assert all("FORCE ROW LEVEL SECURITY" not in s for s in rls.upgrade_statements(_metadata()))


def test_downgrade_drops_all_policies_and_disables() -> None:
    stmts = rls.downgrade_statements(_metadata())
    assert any("DISABLE ROW LEVEL SECURITY" in s and "jobs" in s for s in stmts)
    assert any("DISABLE ROW LEVEL SECURITY" in s and "templates" in s for s in stmts)
    # idempotent drops for both strict + global policy names
    assert any("DROP POLICY IF EXISTS tenant_isolation ON jobs" in s for s in stmts)
    assert any("DROP POLICY IF EXISTS tenant_isolation_read ON templates" in s for s in stmts)


def test_tenant_rls_mode_defaults_to_enforce(monkeypatch) -> None:
    """Unset env ⇒ ``enforce``. A missing ConfigMap key must not disable isolation."""
    monkeypatch.delenv("TENANT_RLS_MODE", raising=False)
    assert rls.tenant_rls_mode() == "enforce"
    monkeypatch.setenv("TENANT_RLS_MODE", "ENFORCE")
    assert rls.tenant_rls_mode() == "enforce"


def test_tenant_rls_mode_off_is_an_explicit_opt_out(monkeypatch) -> None:
    """``off`` stays reachable — local dev against a DB with no policies."""
    monkeypatch.setenv("TENANT_RLS_MODE", " OFF ")
    assert rls.tenant_rls_mode() == "off"


async def test_bind_session_tenant_binds_when_mode_unset(monkeypatch, test_session) -> None:
    """FAIL-CLOSED: with ``TENANT_RLS_MODE`` unset the GUC is bound, not skipped."""
    monkeypatch.delenv("TENANT_RLS_MODE", raising=False)
    await rls.bind_session_tenant(test_session, uuid.UUID("0195f48b-0000-7000-8000-0000000000f1"))
    bound = (
        await test_session.execute(sa.text("SELECT current_setting('app.current_tenant', true)"))
    ).scalar()
    assert bound == "0195f48b-0000-7000-8000-0000000000f1"


async def test_bind_session_tenant_skips_on_explicit_off(monkeypatch, test_session) -> None:
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    await rls.bind_session_tenant(test_session, uuid.UUID("0195f48b-0000-7000-8000-0000000000f0"))
    bound = (
        await test_session.execute(sa.text("SELECT current_setting('app.current_tenant', true)"))
    ).scalar()
    assert bound is None
