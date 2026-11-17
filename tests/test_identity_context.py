"""Tests for w2v_common.auth.identity_context."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from w2v_common.auth.identity_context import (
    caller_token_event_hook,
    clear_identity_context,
    get_actor_id_from_context,
    get_caller_license_token,
    get_correlation_id_from_context,
    get_identity_headers,
    get_license_id_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
    identity_event_hook,
    is_tenant_vouched,
    set_correlation_context,
    set_identity_context,
)
from w2v_common.discovery import service_url
from w2v_common.tenant import MissingTenantError, TenantSource

# A metered peer (worker_management) is the ONLY destination shape that gets the
# effective-tenant side-channel; example.invalid is a plain backend.
METERED_URL = service_url("worker_management") + "/workers/x/message"


def test_defaults_to_empty_strings():
    clear_identity_context()
    # CONTRACT CHANGE (0.67.0): the tenant ContextVar is ``uuid.UUID | None``.
    # It used to default to "" — an empty string is itself a sentinel, and it is
    # what the cost bus turned into the nil-UUID. ``None`` means UNBOUND.
    assert get_tenant_id_from_context() is None
    assert get_license_id_from_context() == ""
    assert get_subject_id_from_context() == ""
    # CONTRACT CHANGE: get_identity_headers no longer emits a BLANK X-Tenant-Id.
    # A blank tenant is tenant-less traffic, which the product-owner ruling
    # forbids; the caller is the only place it can be fixed, so it fails there.
    with pytest.raises(MissingTenantError):
        get_identity_headers()


def test_set_and_get_roundtrip():
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000001", license_id="lic-1", subject_id="user-42"
    )
    assert get_tenant_id_from_context() == uuid.UUID("0195f48b-0000-7000-8000-000000000001")
    assert get_license_id_from_context() == "lic-1"
    assert get_subject_id_from_context() == "user-42"
    assert get_identity_headers() == {
        "X-Tenant-Id": "0195f48b-0000-7000-8000-000000000001",
        "X-License-Id": "lic-1",
        "X-Subject-Id": "user-42",
        "X-Correlation-Id": "",
        "X-Stage": "",
    }
    clear_identity_context()


def test_none_values_become_empty_strings():
    """The four OPAQUE identity values still empty-string on None.

    Only the tenant is mandatory: a CronJob has no license and a
    service-account token has no subject, so those absences are legitimate and
    stay representable. An absent TENANT is not, so this asserts the split.
    """
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000001", license_id=None, subject_id=None
    )
    assert get_identity_headers() == {
        "X-Tenant-Id": "0195f48b-0000-7000-8000-000000000001",
        "X-License-Id": "",
        "X-Subject-Id": "",
        "X-Correlation-Id": "",
        "X-Stage": "",
    }
    set_identity_context(tenant_id=None, license_id=None, subject_id=None)
    with pytest.raises(MissingTenantError):
        get_identity_headers()
    clear_identity_context()


def test_clear_resets_all():
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000001", license_id="lic-1", subject_id="s1"
    )
    clear_identity_context()
    assert get_tenant_id_from_context() is None
    assert get_license_id_from_context() == ""
    assert get_subject_id_from_context() == ""
    assert get_correlation_id_from_context() == ""
    assert get_stage_from_context() == ""
    with pytest.raises(MissingTenantError):
        get_identity_headers()


@pytest.mark.asyncio
async def test_concurrent_tasks_have_isolated_contextvars():
    """Two concurrent asyncio tasks must see their own identity."""
    saw_a: list[dict[str, str]] = []
    saw_b: list[dict[str, str]] = []

    async def task_a():
        set_identity_context("0195f48b-0000-7000-8000-0000000001aa", "lic-a", "subj-a")
        await asyncio.sleep(0.05)
        saw_a.append(get_identity_headers())

    async def task_b():
        set_identity_context("0195f48b-0000-7000-8000-0000000001bb", "lic-b", "subj-b")
        await asyncio.sleep(0.05)
        saw_b.append(get_identity_headers())

    await asyncio.gather(task_a(), task_b())

    assert saw_a == [
        {
            "X-Tenant-Id": "0195f48b-0000-7000-8000-0000000001aa",
            "X-License-Id": "lic-a",
            "X-Subject-Id": "subj-a",
            "X-Correlation-Id": "",
            "X-Stage": "",
        }
    ]
    assert saw_b == [
        {
            "X-Tenant-Id": "0195f48b-0000-7000-8000-0000000001bb",
            "X-License-Id": "lic-b",
            "X-Subject-Id": "subj-b",
            "X-Correlation-Id": "",
            "X-Stage": "",
        }
    ]


@pytest.mark.asyncio
async def test_identity_event_hook_sets_headers_when_context_populated():
    set_identity_context("0195f48b-0000-7000-8000-000000000001", "lic-1", "u1")
    request = httpx.Request("GET", "https://example.invalid/x")
    await identity_event_hook(request)
    assert request.headers["x-tenant-id"] == "0195f48b-0000-7000-8000-000000000001"
    assert request.headers["x-license-id"] == "lic-1"
    assert request.headers["x-subject-id"] == "u1"
    clear_identity_context()


@pytest.mark.asyncio
async def test_identity_event_hook_skips_empty_values():
    """Empty license/subject ContextVars must NOT clobber pre-set headers.

    CONTRACT CHANGE: the pre-set X-Tenant-Id must now be a service-account
    ``Bearer`` form. The old fixture used the literal ``"preset"`` — a bare,
    non-UUID, non-credential value — and asserted the hook left it alone. That
    is precisely the tenant-less-call-goes-out-silently hole: nothing about
    ``"preset"`` identifies a tenant. A real SA Bearer is the one case that is
    legitimately attributable with no bound tenant.
    """
    clear_identity_context()
    request = httpx.Request(
        "GET",
        "https://example.invalid/x",
        headers={"X-Tenant-Id": "Bearer svc-tok"},
    )
    await identity_event_hook(request)
    assert request.headers["x-tenant-id"] == "Bearer svc-tok"
    assert "x-license-id" not in request.headers
    assert "x-subject-id" not in request.headers


@pytest.mark.asyncio
async def test_identity_event_hook_refuses_a_tenantless_uncredentialed_call():
    """No bound tenant AND no SA token => the call must not go out at all."""
    clear_identity_context()
    request = httpx.Request("GET", "https://example.invalid/x")
    with pytest.raises(MissingTenantError):
        await identity_event_hook(request)


@pytest.mark.asyncio
async def test_identity_event_hook_preserves_bearer_x_tenant_id():
    """When request already has X-Tenant-Id: Bearer <token>, the hook
    must NOT overwrite it (constructor's service-account auth survives).

    This is the canonical w2v S2S path: BaseServiceClient sets
    X-Tenant-Id: Bearer <service-license-token> in its constructor.
    Downstream services require the Bearer form to introspect; bare UUID
    falls to the trusted-network path which is closed."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000500",
        license_id="lic",
        subject_id="sub",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    try:
        request = httpx.Request(
            "GET",
            METERED_URL,
            headers={
                "Authorization": "Bearer svc-tok",
                "X-Tenant-Id": "Bearer svc-tok",
            },
        )
        await identity_event_hook(request)
        # X-Tenant-Id Bearer survives
        assert request.headers["x-tenant-id"] == "Bearer svc-tok"
        # Authorization is never touched by this hook
        assert request.headers["authorization"] == "Bearer svc-tok"
        # X-License-Id / X-Subject-Id are stamped
        assert request.headers["x-license-id"] == "lic"
        assert request.headers["x-subject-id"] == "sub"
        # ...and the effective-tenant side-channel carries the REAL tenant even
        # though the Bearer X-Tenant-Id (service-account auth) is preserved. This
        # is what threads a background run's tenant downstream for attribution.
        assert request.headers["x-effective-tenant-id"] == "0195f48b-0000-7000-8000-000000000500"
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_identity_event_hook_stamps_effective_tenant_from_context():
    """The effective-tenant header follows the DESTINATION, not the call site.

    CONTRACT CHANGE: it used to be stamped unconditionally, on every hop, for
    any provenance. It is a delegation credential — the receiver bills the named
    tenant — so it is now emitted ONLY on a hop to a metered peer and ONLY for a
    vouched provenance.
    """
    from w2v_common.auth.identity_context import EFFECTIVE_TENANT_HEADER

    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000e99",
        license_id="",
        subject_id="",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    try:
        # Metered destination + service-account Bearer => delegated hop.
        req = httpx.Request("GET", METERED_URL, headers={"X-Tenant-Id": "Bearer svc-tok"})
        await identity_event_hook(req)
        assert req.headers[EFFECTIVE_TENANT_HEADER] == "0195f48b-0000-7000-8000-000000000e99"

        # Same tenant, PLAIN backend: the tenant goes verbatim, no delegation.
        plain = httpx.Request("GET", "https://example.invalid/x")
        await identity_event_hook(plain)
        assert plain.headers["x-tenant-id"] == "0195f48b-0000-7000-8000-000000000e99"
        assert EFFECTIVE_TENANT_HEADER not in plain.headers
    finally:
        clear_identity_context()

    set_identity_context(tenant_id=None, license_id=None, subject_id=None)
    try:
        req2 = httpx.Request("GET", METERED_URL, headers={"X-Tenant-Id": "Bearer svc-tok"})
        await identity_event_hook(req2)
        assert EFFECTIVE_TENANT_HEADER not in req2.headers
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_identity_event_hook_fills_bare_uuid_when_no_bearer():
    """When X-Tenant-Id is absent or non-Bearer, fill with bare UUID
    from ContextVar (legacy callers without a service-account license
    token still get a tenant-id stamped for downstream introspection
    fallback)."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000d2d",
        license_id="lic",
        subject_id="sub",
    )
    try:
        # No X-Tenant-Id at all
        req = httpx.Request("GET", "https://example.invalid/x")
        await identity_event_hook(req)
        assert req.headers["x-tenant-id"] == "0195f48b-0000-7000-8000-000000000d2d"

        # X-Tenant-Id present but not Bearer (legacy bare-uuid form)
        req2 = httpx.Request(
            "GET",
            "https://example.invalid/x",
            headers={"X-Tenant-Id": "preset-bare-uuid"},
        )
        await identity_event_hook(req2)
        # Bare-uuid form gets overwritten by ContextVar
        assert req2.headers["x-tenant-id"] == "0195f48b-0000-7000-8000-000000000d2d"
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_identity_event_hook_does_not_touch_authorization():
    """identity_event_hook is UUID-only — it must NEVER touch Authorization,
    even when caller_license_token is set in the ContextVar. Authorization
    forwarding is the exclusive job of caller_token_event_hook."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000700",
        license_id="l",
        subject_id="s",
        caller_license_token="caller-tok",
    )
    try:
        request = httpx.Request(
            "GET",
            "https://example.invalid/x",
            headers={"Authorization": "Bearer svc-tok"},
        )
        await identity_event_hook(request)
        assert request.headers["authorization"] == "Bearer svc-tok"
    finally:
        clear_identity_context()


def test_caller_license_token_set_and_get():
    """caller_license_token ContextVar round-trips like the others."""
    clear_identity_context()
    assert get_caller_license_token() == ""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000700",
        license_id="l",
        subject_id="s",
        caller_license_token="caller-tok",
    )
    assert get_caller_license_token() == "caller-tok"
    clear_identity_context()
    assert get_caller_license_token() == ""


def test_caller_license_token_none_becomes_empty():
    """None caller_license_token must coerce to empty string (header-safe)."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000700",
        license_id="l",
        subject_id="s",
        caller_license_token=None,
    )
    assert get_caller_license_token() == ""
    clear_identity_context()


def test_caller_license_token_default_kwarg_omitted():
    """Existing call sites that don't pass caller_license_token still work."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000700", license_id="l", subject_id="s"
    )
    assert get_caller_license_token() == ""
    clear_identity_context()


@pytest.mark.asyncio
async def test_caller_token_hook_overrides_when_set():
    """When ContextVar carries a caller's license token, the dedicated hook
    stamps it on Authorization AND X-Tenant-Id (overriding constructor
    service-account defaults)."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000700",
        license_id="l",
        subject_id="s",
        caller_license_token="caller-tok",
    )
    try:
        request = httpx.Request(
            "GET",
            "https://example.invalid/x",
            headers={
                "Authorization": "Bearer svc-tok",
                "X-Tenant-Id": "Bearer svc-tok",
            },
        )
        await caller_token_event_hook(request)
        assert request.headers["authorization"] == "Bearer caller-tok"
        assert request.headers["x-tenant-id"] == "Bearer caller-tok"
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_caller_token_hook_noop_when_unset():
    """When no caller token in ContextVar, caller_token_event_hook leaves
    Authorization + X-Tenant-Id untouched (constructor service-account
    defaults survive — required for CronJobs / startup tasks / Keycloak-JWT
    requests / trusted-network paths)."""
    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-000000000d1d",
        license_id="lic",
        subject_id="sub",
        caller_license_token=None,
    )
    try:
        request = httpx.Request(
            "GET",
            "https://example.invalid/x",
            headers={
                "Authorization": "Bearer svc-tok",
                "X-Tenant-Id": "Bearer svc-tok",
            },
        )
        await caller_token_event_hook(request)
        assert request.headers["authorization"] == "Bearer svc-tok"
        assert request.headers["x-tenant-id"] == "Bearer svc-tok"
    finally:
        clear_identity_context()


def test_correlation_context_set_and_get():
    """correlation_id + stage round-trip and appear in the header dict."""
    clear_identity_context()
    assert get_correlation_id_from_context() == ""
    assert get_stage_from_context() == ""
    # A tenant must be bound before the header dict can be built at all.
    set_identity_context("0195f48b-0000-7000-8000-000000000001", "lic-1", "u1")
    set_correlation_context("run-123", "narration")
    assert get_correlation_id_from_context() == "run-123"
    assert get_stage_from_context() == "narration"
    headers = get_identity_headers()
    assert headers["X-Correlation-Id"] == "run-123"
    assert headers["X-Stage"] == "narration"
    clear_identity_context()
    assert get_correlation_id_from_context() == ""
    assert get_stage_from_context() == ""


@pytest.mark.asyncio
async def test_identity_event_hook_stamps_correlation_when_set():
    """The outbound hook forwards X-Correlation-Id + X-Stage when present."""
    set_identity_context("0195f48b-0000-7000-8000-000000000001", "lic-1", "u1")
    set_correlation_context("run-abc", "script_generation")
    try:
        request = httpx.Request("GET", "https://example.invalid/x")
        await identity_event_hook(request)
        assert request.headers["x-correlation-id"] == "run-abc"
        assert request.headers["x-stage"] == "script_generation"
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_identity_event_hook_skips_empty_correlation():
    """Empty correlation ContextVars must NOT add headers (opt-in)."""
    clear_identity_context()
    set_identity_context("0195f48b-0000-7000-8000-000000000001", "lic-1", "u1")
    try:
        request = httpx.Request("GET", "https://example.invalid/x")
        await identity_event_hook(request)
        assert "x-correlation-id" not in request.headers
        assert "x-stage" not in request.headers
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_caller_token_hook_isolation_across_tasks():
    """Caller-token override must be task-local — two concurrent requests
    using the same cached client must each see their own caller token."""
    saw_a: list[str] = []
    saw_b: list[str] = []

    async def task_a():
        set_identity_context(
            "0195f48b-0000-7000-8000-00000000000a", "la", "sa", caller_license_token="tok-a"
        )
        await asyncio.sleep(0.05)
        req = httpx.Request("GET", "https://x.invalid/")
        await caller_token_event_hook(req)
        saw_a.append(req.headers["authorization"])

    async def task_b():
        set_identity_context(
            "0195f48b-0000-7000-8000-00000000000b", "lb", "sb", caller_license_token="tok-b"
        )
        await asyncio.sleep(0.05)
        req = httpx.Request("GET", "https://x.invalid/")
        await caller_token_event_hook(req)
        saw_b.append(req.headers["authorization"])

    await asyncio.gather(task_a(), task_b())
    assert saw_a == ["Bearer tok-a"]
    assert saw_b == ["Bearer tok-b"]


# ---------------------------------------------------------------------------
# Provenance survives a same-tenant re-seed, and never crosses tenants.
#
# ``set_identity_context`` used to do ``_tenant_source_var.set(tenant_source)``
# unconditionally, so ``tenant_source=None`` meant BOTH "there is no provenance"
# and "I am not talking about provenance" — and the second silently destroyed
# the first. ``asyncio.create_task`` copies the caller's context, so a detached
# background job inherits the request's provenance; a re-seed that only wanted
# to rebind the tenant wiped it, and the job's next metered call failed closed
# with "provenance is 'unrecorded', which is not vouched". Four repos shipped
# per-caller patches before the default itself was fixed.
#
# The rule now: OMITTED preserves across the SAME tenant and drops across a
# DIFFERENT one, because provenance is what unlocks a tenant's BYO credentials
# and bills them. An explicit value — including an explicit None — always wins.
# ---------------------------------------------------------------------------

_T_A = uuid.uuid4()
_T_B = uuid.uuid4()


def _seed_vouched(tenant: uuid.UUID) -> None:
    set_identity_context(tenant, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN)


def test_omitting_source_preserves_provenance_for_the_same_tenant() -> None:
    """The background-task case: rebinding my own tenant must not disarm me."""
    clear_identity_context()
    _seed_vouched(_T_A)

    set_identity_context(_T_A, None, None)

    assert get_tenant_source_from_context() is TenantSource.LICENSE_TOKEN
    assert is_tenant_vouched() is True


def test_omitting_source_drops_provenance_when_the_tenant_changes() -> None:
    """The reaper case, and the reason a bare "preserve on omit" would be a HOLE.

    A loop walking rows for many tenants must never carry tenant A's proof onto
    tenant B — that is precisely "let an unauthenticated caller spend an
    arbitrary tenant's credentials", which the outbound guard exists to stop.
    """
    clear_identity_context()
    _seed_vouched(_T_A)

    set_identity_context(_T_B, None, None)

    assert get_tenant_source_from_context() is None
    assert is_tenant_vouched() is False


def test_an_explicit_none_still_clears_even_for_the_same_tenant() -> None:
    """Explicit beats implicit: this is how a caller says "drop what was here"."""
    clear_identity_context()
    _seed_vouched(_T_A)

    set_identity_context(_T_A, None, None, tenant_source=None)

    assert get_tenant_source_from_context() is None


def test_an_explicit_source_overrides_the_inherited_one() -> None:
    clear_identity_context()
    _seed_vouched(_T_A)

    set_identity_context(_T_A, None, None, tenant_source=TenantSource.UNVOUCHED_HEADER)

    assert get_tenant_source_from_context() is TenantSource.UNVOUCHED_HEADER
    assert is_tenant_vouched() is False


def test_omitting_source_with_no_prior_context_is_still_unvouched() -> None:
    """Absence must not INVENT provenance — nothing to preserve means None."""
    clear_identity_context()

    set_identity_context(_T_A, None, None)

    assert get_tenant_source_from_context() is None
    assert is_tenant_vouched() is False


def test_unbinding_the_tenant_drops_provenance() -> None:
    """tenant_id=None means "no tenant"; a dangling source would outlive it."""
    clear_identity_context()
    _seed_vouched(_T_A)

    set_identity_context(None, None, None)

    assert get_tenant_source_from_context() is None


async def test_a_detached_task_keeps_provenance_across_its_own_reseed() -> None:
    """End-to-end shape of the live bug, exercised the way production hits it."""
    clear_identity_context()
    _seed_vouched(_T_A)

    seen: list[bool] = []

    async def _bg() -> None:
        # The background entrypoint rebinds its tenant, exactly as the four
        # patched repos do, but without naming a source.
        set_identity_context(_T_A, None, None)
        seen.append(is_tenant_vouched())

    await asyncio.create_task(_bg())

    assert seen == [True], "a detached task must still be able to make a metered call"


# ---------------------------------------------------------------------------
# actor_id — WHO ACTED (JWT `sub`), carried alongside subject_id (WHO PAYS).
# ---------------------------------------------------------------------------


def test_actor_defaults_to_empty_and_roundtrips() -> None:
    """Empty default, exactly like license_id/subject_id — absence is legal."""
    clear_identity_context()
    assert get_actor_id_from_context() == ""

    set_identity_context(_T_A, "lic-1", "pays", actor_id="acted")
    assert get_actor_id_from_context() == "acted"
    # ...and it did NOT displace the subject. billing carries both; collapsing
    # them would bill the actor instead of the licence holder.
    assert get_subject_id_from_context() == "pays"
    clear_identity_context()


def test_omitting_the_actor_clears_it_rather_than_inheriting() -> None:
    """A re-seed that names no actor records NO actor.

    Deliberately UNLIKE `tenant_source`, which is preserved on omission because
    dropping it turns working metered calls into hard failures. An actor has no
    such gate: the safe direction for a money-ATTRIBUTION field is a NULL
    column, never a stale human's id stamped onto work they did not do.
    """
    set_identity_context(_T_A, None, "pays", actor_id="acted")
    set_identity_context(_T_A, None, "pays")
    assert get_actor_id_from_context() == ""
    clear_identity_context()


def test_clear_resets_the_actor() -> None:
    set_identity_context(_T_A, None, "pays", actor_id="acted")
    clear_identity_context()
    assert get_actor_id_from_context() == ""


def test_a_uuid_actor_goes_out_without_any_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0.86.0: X-Actor-Id is forwarded whenever a UUID actor is bound.

    It was opt-in (AUTH_ATTRIBUTION_FORWARD_HEADERS) from 0.82.0; the owner
    ruling of 2026-10-07 makes the actor mandatory on every cost event, which a
    downstream hop can only honour if the caller forwards it.
    """
    monkeypatch.delenv("AUTH_ATTRIBUTION_FORWARD_HEADERS", raising=False)
    actor = "0195f48b-2222-7000-8000-00000000cafe"
    set_identity_context(_T_A, "lic-1", "pays", actor_id=actor)
    try:
        assert get_identity_headers()["X-Actor-Id"] == actor
    finally:
        clear_identity_context()


def test_only_a_uuid_actor_is_stamped_on_outbound_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate wire change (ATTRIBUTION-PLAN "THE CONTRACT", 2026-09-28).

    This test used to pin that NO X-Actor-Id went out, "so the day someone does
    want S2S propagation it is a deliberate decision with a failing test". That
    decision is now made (opt-in in 0.82.0, unconditional since 0.86.0):
    X-Actor-Id is forwarded, but the contract is UUID-only, so a non-UUID lane
    value (here "acted") stays off the wire.
    """
    monkeypatch.delenv("AUTH_ATTRIBUTION_FORWARD_HEADERS", raising=False)
    set_identity_context(_T_A, "lic-1", "pays", actor_id="acted")
    assert "X-Actor-Id" not in get_identity_headers()
    assert "acted" not in get_identity_headers().values()
    actor = "0195f48b-2222-7000-8000-00000000cafe"
    set_identity_context(_T_A, "lic-1", "pays", actor_id=actor)
    assert get_identity_headers()["X-Actor-Id"] == actor
    clear_identity_context()
