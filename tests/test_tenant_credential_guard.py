"""A credential must never be accepted as a tenant identifier.

A 2026-08-27 fleet audit found live credentials sitting in ``tenant_id``
columns: ``Bearer `` + 43 urlsafe characters (opaque 32-byte keys) in
``tool_registry.tools`` and ``analysis_service.analysis_jobs`` /
``analysis_usage``. Nothing rejected them on the way in.

:func:`w2v_common.tenant.parse_tenant_id` is the one funnel every write path
reaches — ``AuthMiddleware._tenant_or_unauthorized`` (every auth lane),
``set_identity_context`` (the ContextVar that feeds RLS and every outbound
hop), the idempotency cache key, the audit logger, and ``KagScrapeClient`` —
so the guard lives there and nowhere else.

Two properties are asserted here, and both matter:

1. the credential shapes RAISE (never coerced, truncated, prefix-stripped or
   defaulted); and
2. the exception says only the SHAPE — echoing the value would write the
   credential into logs, which is the same leak in a new place.
"""

from __future__ import annotations

import uuid

import pytest

from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_tenant_id_from_context,
    set_identity_context,
)
from w2v_common.tenant import (
    MAX_TENANT_ID_LENGTH,
    InvalidTenantIdError,
    TenantError,
    parse_tenant_id,
)

# The guard keys on the "Bearer " PREFIX, not on what follows it, so the value
# is deliberately short and low-entropy: a 43-char urlsafe literal here is
# exactly what the fleet secret-scan's `nonagentic-licenses-token` rule detects
# (43 chars at entropy >= 4.35), and a fabricated fixture must not train the
# fleet to allowlist that shape.
SECRET = "not-a-real-licence-token-fixture"
BEARER_TENANT = f"Bearer {SECRET}"
# A JWT's first segment is the base64 of ``{"alg":...``, hence the eyJ prefix.
# Also fabricated: the signature segment is the literal bytes ``sig``.
JWT_TENANT = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ.c2ln"  # pragma: allowlist secret
)
OVERLONG_TENANT = "a" * (MAX_TENANT_ID_LENGTH + 1)

VALID_TENANT = "8382f0b3-c082-4f97-8d47-120ce5a18e05"


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        (BEARER_TENANT, "bearer-prefix"),
        (BEARER_TENANT.lower(), "bearer-prefix"),
        (BEARER_TENANT.upper(), "bearer-prefix"),
        ("bEaReR abc123", "bearer-prefix"),
        (f"   {BEARER_TENANT}   ", "bearer-prefix"),
        (JWT_TENANT, "jwt-prefix"),
        (OVERLONG_TENANT, "over-max-length"),
    ],
)
def test_credential_shaped_tenant_is_rejected(value: str, shape: str) -> None:
    """Each credential shape raises, and the message names the shape."""
    with pytest.raises(InvalidTenantIdError) as excinfo:
        parse_tenant_id(value, source="X-Tenant-Id header")
    message = str(excinfo.value)
    assert "credential-shaped tenant identifier rejected" in message
    assert f"shape={shape}" in message
    # ``source`` still tells the operator WHICH lane produced it.
    assert "X-Tenant-Id header" in message


def test_rejection_is_catchable_as_the_existing_base_error() -> None:
    """No new hierarchy: existing ``except TenantError`` handlers still catch it.

    Every consumer seam (audit, middleware, idempotency) catches ``TenantError``.
    If the guard raised something outside that tree it would escape as a 500
    instead of the lane's own 401 / drop-the-field behaviour.
    """
    with pytest.raises(TenantError):
        parse_tenant_id(BEARER_TENANT)
    assert issubclass(InvalidTenantIdError, TenantError)


@pytest.mark.parametrize("value", [BEARER_TENANT, JWT_TENANT, OVERLONG_TENANT])
def test_exception_never_echoes_the_offending_value(value: str) -> None:
    """⛔ The message must not carry the credential into the log.

    Asserted three ways because a partial echo leaks just as badly as a full
    one: not the whole value, not the secret body, and not any 12-char window
    of it.
    """
    with pytest.raises(InvalidTenantIdError) as excinfo:
        parse_tenant_id(value)
    message = str(excinfo.value)
    assert value not in message
    assert SECRET not in message
    for start in range(0, len(value) - 12):
        assert value[start : start + 12] not in message, (
            f"exception echoed a {12}-char window of the offending value"
        )


def test_exception_repr_and_args_also_stay_clean() -> None:
    """A traceback renders ``args``/``repr``, not only ``str`` — check those too."""
    with pytest.raises(InvalidTenantIdError) as excinfo:
        parse_tenant_id(BEARER_TENANT)
    assert SECRET not in repr(excinfo.value)
    assert all(SECRET not in str(arg) for arg in excinfo.value.args)


def test_a_normal_uuid_still_passes_untouched() -> None:
    """The guard must not cost a legitimate tenant anything."""
    assert parse_tenant_id(VALID_TENANT) == uuid.UUID(VALID_TENANT)
    as_uuid = uuid.UUID(VALID_TENANT)
    assert parse_tenant_id(as_uuid) is as_uuid
    assert parse_tenant_id(f"  {VALID_TENANT}  ") == uuid.UUID(VALID_TENANT)
    assert parse_tenant_id(VALID_TENANT.upper()) == uuid.UUID(VALID_TENANT)


def test_uuid_is_far_below_the_length_ceiling() -> None:
    """36 < 64: no legitimate tenant id can trip the length rule."""
    assert len(VALID_TENANT) == 36
    assert MAX_TENANT_ID_LENGTH == 64


def test_length_ceiling_is_strictly_greater_than_max() -> None:
    """Boundary: 64 chars is NOT the credential lane, 65 is.

    A 64-char non-UUID is still rejected — but as "not a UUID", i.e. by the
    pre-existing rule, not by this guard. Pinning that keeps a future edit from
    silently turning the ceiling into ``>=``.
    """
    at_limit = "a" * MAX_TENANT_ID_LENGTH
    with pytest.raises(InvalidTenantIdError) as excinfo:
        parse_tenant_id(at_limit)
    assert "credential-shaped" not in str(excinfo.value)
    assert "not a UUID" in str(excinfo.value)


def test_guard_covers_the_identity_contextvar_write_path() -> None:
    """The funnel is real: the guard fires through ``set_identity_context``.

    This is the seam that feeds the RLS ``app.current_tenant`` GUC and every
    outbound hop, so proving it here proves the write path, not just the
    helper.
    """
    clear_identity_context()
    try:
        with pytest.raises(InvalidTenantIdError):
            set_identity_context(BEARER_TENANT, "lic", "sub")
        # Fail LOUD means nothing was bound — not a coerced or default tenant.
        assert get_tenant_id_from_context() is None

        set_identity_context(VALID_TENANT, "lic", "sub")
        assert get_tenant_id_from_context() == uuid.UUID(VALID_TENANT)
    finally:
        clear_identity_context()
