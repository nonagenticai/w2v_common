"""Generic Postgres Row-Level-Security (RLS) helpers — Phase D fleet-wide backstop.

The orchestrator shipped a bespoke ``rls.py`` first; this is the reusable,
metadata-driven version for the rest of the w2v fleet. Each service's tables
carry ``tenant_id`` **directly** (no FK-join chains like the orchestrator's
run-tree), so policy generation is uniform and can be derived from the service's
SQLAlchemy ``MetaData``:

  * every table with a ``tenant_id`` column gets RLS ``ENABLE`` + a
    ``tenant_isolation`` policy keyed on the per-request GUC ``app.current_tenant``;
  * a **NOT NULL** ``tenant_id`` → strict equality (read+write scoped to the GUC);
  * a **NULLABLE** ``tenant_id`` → "global-or-own": rows with ``tenant_id IS NULL``
    are global/shared (e.g. built-in templates), readable by every tenant; writes
    are still GUC-scoped (only the owner/platform seeds a global row, out of band).

Design invariants (match the orchestrator's proven posture):
  * **ENABLE, never FORCE.** Every w2v service DB is owned by the non-superuser
    ``w2v_service`` and its bypass/back-office paths rely on owner-bypass, which
    holds only under ENABLE-not-FORCE. The runtime isolates because it connects as
    a *non-owner* restricted role (per-service ``w2v_<svc>_app``), which is always
    bound under ENABLE. (Applying FORCE would 0-row the owner and break it.)
  * **Fail-closed GUC.** ``NULLIF(current_setting('app.current_tenant', true), '')``
    — ``current_setting(..., true)`` is missing-ok (unset → NULL); ``NULLIF(…, '')``
    folds the empty-string a ``set_config(..., is_local=true)`` leaves on a pooled
    PgBouncer connection back to NULL, so the predicate is simply false → 0 rows
    (instead of ``''::uuid`` raising a 500).
  * **Single source.** The migration (prod) and the ``create_all`` ``after_create``
    event (tests/dev) both render from these helpers, so they can never drift.
  * **Inert until cutover.** Policies change nothing while the app connects as the
    owner ``w2v_service``; they bind only once the runtime switches to the restricted
    role + ``TENANT_RLS_MODE=enforce`` (atomic, per-service gitops cutover). That
    mode is the DEFAULT (fail-closed); ``off`` is an explicit local-dev opt-out.
"""

from __future__ import annotations

import os
import uuid

from sqlalchemy import DDL, Connection, MetaData, String, event, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, SessionTransaction

TENANT_COLUMN = "tenant_id"
POLICY_NAME = "tenant_isolation"
_GLOBAL_READ = f"{POLICY_NAME}_read"
_GLOBAL_INS = f"{POLICY_NAME}_ins"
_GLOBAL_UPD = f"{POLICY_NAME}_upd"
_GLOBAL_DEL = f"{POLICY_NAME}_del"

# The transaction-scoped GUC the request/background paths bind, as TEXT. The
# predicate casts to match the column type: the orchestrator stores tenant_id as
# ``uuid`` (cast ``::uuid``), every other w2v service stores it as ``varchar``
# (compare text-to-text — casting would raise ``operator does not exist:
# character varying = uuid``). Factored once so migration + test event never drift.
_GUC_TEXT = "NULLIF(current_setting('app.current_tenant', true), '')"

_ALL_POLICY_NAMES = (POLICY_NAME, _GLOBAL_READ, _GLOBAL_INS, _GLOBAL_UPD, _GLOBAL_DEL)


def tenant_table_names(metadata: MetaData) -> list[str]:
    """Names of every table in ``metadata`` carrying a ``tenant_id`` column."""
    return [t.name for t in metadata.sorted_tables if TENANT_COLUMN in t.columns]


def _tenant_column(metadata: MetaData, table_name: str):  # noqa: ANN202 (sqlalchemy Column)
    return metadata.tables[table_name].columns[TENANT_COLUMN]


def _is_nullable(metadata: MetaData, table_name: str) -> bool:
    return bool(_tenant_column(metadata, table_name).nullable)


def _predicate(metadata: MetaData, table_name: str) -> str:
    # varchar/text tenant_id → compare text-to-text; uuid → cast the GUC to uuid.
    col = _tenant_column(metadata, table_name)
    guc = _GUC_TEXT if isinstance(col.type, String) else f"{_GUC_TEXT}::uuid"
    return f"{TENANT_COLUMN} = {guc}"


def _strict_statements(table: str, predicate: str) -> list[str]:
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}",
        f"CREATE POLICY {POLICY_NAME} ON {table} USING ({predicate}) WITH CHECK ({predicate})",
    ]


def _global_statements(table: str, predicate: str) -> list[str]:
    # Nullable tenant_id ⇒ NULL rows are shared/global (read by all), writes scoped.
    read = f"{TENANT_COLUMN} IS NULL OR {predicate}"
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"DROP POLICY IF EXISTS {_GLOBAL_READ} ON {table}",
        f"DROP POLICY IF EXISTS {_GLOBAL_INS} ON {table}",
        f"DROP POLICY IF EXISTS {_GLOBAL_UPD} ON {table}",
        f"DROP POLICY IF EXISTS {_GLOBAL_DEL} ON {table}",
        f"CREATE POLICY {_GLOBAL_READ} ON {table} FOR SELECT USING ({read})",
        f"CREATE POLICY {_GLOBAL_INS} ON {table} FOR INSERT WITH CHECK ({predicate})",
        f"CREATE POLICY {_GLOBAL_UPD} ON {table} FOR UPDATE "
        f"USING ({predicate}) WITH CHECK ({predicate})",
        f"CREATE POLICY {_GLOBAL_DEL} ON {table} FOR DELETE USING ({predicate})",
    ]


def upgrade_statements(metadata: MetaData) -> list[str]:
    """DDL applying RLS to every ``tenant_id`` table (use in a migration ``upgrade()``)."""
    out: list[str] = []
    for name in tenant_table_names(metadata):
        predicate = _predicate(metadata, name)
        out += (
            _global_statements(name, predicate)
            if _is_nullable(metadata, name)
            else _strict_statements(name, predicate)
        )
    return out


def downgrade_statements(metadata: MetaData) -> list[str]:
    """Reverse of :func:`upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for name in tenant_table_names(metadata):
        for policy in _ALL_POLICY_NAMES:
            out.append(f"DROP POLICY IF EXISTS {policy} ON {name}")
        out.append(f"ALTER TABLE {name} DISABLE ROW LEVEL SECURITY")
    return out


def register_rls_ddl(metadata: MetaData) -> None:
    """Apply the SAME policies on ``create_all`` (tests/dev build schema without alembic).

    Registers a single ``after_create`` listener on ``metadata`` — it runs once all
    tables exist, so every ``tenant_id`` table gets its policy. Idempotent across
    repeated ``create_all`` (``DROP POLICY IF EXISTS`` precedes each ``CREATE``).
    """

    def _apply(_target: MetaData, connection: Connection, **_kw: object) -> None:
        for stmt in upgrade_statements(metadata):
            connection.execute(DDL(stmt))

    event.listen(metadata, "after_create", _apply)


#: Fail-CLOSED default. A missing / typo'd ``TENANT_RLS_MODE`` key in a
#: service's ConfigMap used to silently mean "no tenant binding at all", so a
#: brand-new service, a forgotten overlay, or a misspelled key shipped with
#: isolation OFF and nothing anywhere said so. Every one of the 14 live w2v
#: ConfigMaps already carries ``enforce``, so flipping the default is inert in
#: prod and only changes what an *unconfigured* deployment does — which is
#: exactly the case that must not fail open.
DEFAULT_TENANT_RLS_MODE = "enforce"
#: The only values ``TENANT_RLS_MODE`` may take. Anything else raises rather than
#: degrading to "off" — see :func:`tenant_rls_mode`.
_TENANT_RLS_MODES = frozenset({"enforce", "off"})


def tenant_rls_mode() -> str:
    """Current enforcement mode from ``TENANT_RLS_MODE`` env (``off``|``enforce``).

    Defaults to ``enforce`` (fail-closed): isolation must be opted OUT of
    explicitly (``TENANT_RLS_MODE=off``, for local dev against a DB with no
    policies), never opted IN to by remembering a ConfigMap key.
    """
    raw = os.getenv("TENANT_RLS_MODE", DEFAULT_TENANT_RLS_MODE).strip().lower()
    if raw not in _TENANT_RLS_MODES:
        # ⚠️ FAIL LOUD. Every consumer tests ``!= "enforce"`` and silently returns, so a
        # typo or a plausible synonym ("true", "on", "1", "enforced") reads as isolation
        # ON to a human and means OFF to the code — tenant GUC never bound, RLS inert,
        # no error and no log line. The fail-closed default only ever covered a MISSING
        # key; a wrong-VALUED one was the dangerous case and was unguarded.
        raise ValueError(
            f"TENANT_RLS_MODE={raw!r} is not recognised — expected one of "
            f"{sorted(_TENANT_RLS_MODES)}. Refusing to guess: an unrecognised value "
            f"would silently DISABLE tenant isolation."
        )
    return raw


async def bind_session_tenant(
    session: AsyncSession, tenant_id: uuid.UUID | None, *, mode: str | None = None
) -> None:
    """Bind ``app.current_tenant`` for the session's current transaction when enforcing.

    Transaction-scoped (``set_config(..., is_local=true)``) → PgBouncer-safe. A no-op
    unless ``mode`` (default: :func:`tenant_rls_mode`) is ``enforce``. ``tenant_id``
    is a real ``uuid.UUID``; ``None`` means no tenant is bound, and the GUC is then
    left unset → RLS fail-closes to 0 rows (deliberate). It is NEVER filled with a
    placeholder: binding a sentinel would open every policy to the sentinel's rows.
    """
    if (mode or tenant_rls_mode()) != "enforce" or tenant_id is None:
        return
    await session.execute(
        text("SELECT set_config('app.current_tenant', :tenant, true)"),
        {"tenant": str(tenant_id)},
    )


_rebind_installed = False


def install_rls_tenant_rebind() -> None:
    """Re-bind ``app.current_tenant`` at the start of EVERY transaction (process-wide).

    :func:`bind_session_tenant` / the ``get_db_session`` auto-bind set the GUC
    transaction-scoped (``set_config(..., is_local=true)``, PgBouncer-safe),
    which is correct only while a session opens exactly one transaction. The day
    a request handler or background task COMMITs mid-flight and then issues
    another RLS-gated statement in the same session, that later statement runs in
    a fresh, UNBOUND transaction → under ``enforce`` the GUC reads back as ``''``
    → ``NULLIF`` → NULL → the isolation predicate is false → 0 rows (a spurious
    404, or an ``InsufficientPrivilege`` on write). This ``after_begin`` listener
    re-applies the tenant from the identity ContextVar on every ``BEGIN`` so any
    number of intra-request/background commits self-heal.

    Defense in depth: inert unless ``TENANT_RLS_MODE=enforce`` **and** a tenant
    is in context, so request handlers (already bound by ``AuthMiddleware``) and
    no-tenant background loops stay no-ops. Idempotent — safe to call once at app
    startup; a second call is a no-op. Scope is the calling process only.

    Consolidates the per-service ``database/rls_events.py`` clone shipped in
    analysis / content / orchestrator / html_render / video_production /
    app_inspector / ai_content / bdd_playwright.
    """
    global _rebind_installed
    if _rebind_installed:
        return

    # Deferred import: identity_context pulls in auth machinery; keep rls.py
    # importable by migrations/tooling that never touch the request path.
    from w2v_common.auth.identity_context import get_tenant_id_from_context

    @event.listens_for(Session, "after_begin")
    def _rebind_tenant(
        session: Session, transaction: SessionTransaction, connection: Connection
    ) -> None:
        if tenant_rls_mode() != "enforce":
            return
        tenant = get_tenant_id_from_context()
        if tenant is None:
            return
        connection.exec_driver_sql(
            "SELECT set_config('app.current_tenant', %s, true)", (str(tenant),)
        )

    _rebind_installed = True
