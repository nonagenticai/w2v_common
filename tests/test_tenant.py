"""Tests for w2v_common.tenant.parse_tenant_id.

The module docstring rules that a ``tenant_id`` may NEVER hold a sentinel, and
names the all-zeros UUID as the specific sentinel that caused the damage. Until
0.68.0 ``parse_tenant_id`` did not enforce that: the nil uuid is syntactically
valid, so BOTH return paths let it through, and downstream it was bound to
``request.state.tenant_id``, the identity ContextVar, the ``app.current_tenant``
RLS GUC and outbound S2S headers before any route ran.

These tests pin the enforcement at that one seam. They are written so that
deleting the guard turns them red, not so that they merely exercise the call.
"""

from __future__ import annotations

import uuid

import pytest

from w2v_common.tenant import (
    InvalidTenantIdError,
    MissingTenantError,
    TenantError,
    parse_tenant_id,
    require_tenant_id,
)

NIL = "00000000-0000-0000-0000-000000000000"
REAL = "3f2a1b4c-5d6e-4f70-8192-a3b4c5d6e7f8"


def test_nil_uuid_object_is_refused() -> None:
    """The first return path: an already-constructed nil ``uuid.UUID``.

    This is the path video_compiler_service's auth lanes took, and the reason
    ``POST /api/thumbnails`` answered 200 for a request ``/api/compile`` 400s.
    """
    with pytest.raises(InvalidTenantIdError) as exc:
        parse_tenant_id(uuid.UUID(int=0), source="JWT claim")
    assert "nil uuid" in str(exc.value)
    assert "JWT claim" in str(exc.value)


def test_nil_uuid_string_is_refused() -> None:
    """The second return path: the sentinel as text, which ``uuid.UUID`` parses."""
    with pytest.raises(InvalidTenantIdError) as exc:
        parse_tenant_id(NIL, source="X-Tenant-Id header")
    assert "nil uuid" in str(exc.value)
    assert "X-Tenant-Id header" in str(exc.value)


@pytest.mark.parametrize(
    "spelling",
    [
        NIL,
        NIL.upper(),
        f"  {NIL}  ",
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
    ],
    ids=["plain", "upper", "padded", "undashed", "braced", "urn"],
)
def test_every_spelling_of_the_sentinel_is_refused(spelling: str) -> None:
    """``uuid.UUID`` normalises all of these to nil, so a string check would miss them."""
    with pytest.raises(InvalidTenantIdError):
        parse_tenant_id(spelling)


@pytest.mark.parametrize(
    "prefix",
    ["urn:uuid:", "URN:UUID:", "Urn:UUID:", "urn:UUID:", "URN:uuid:", "uRn:UuId:"],
    ids=["lower", "upper", "title", "upper-nid", "upper-scheme", "mixed"],
)
def test_every_case_of_the_urn_prefix_refuses_the_sentinel_as_nil(prefix: str) -> None:
    """Not merely "is refused" — refused for the RIGHT reason.

    ``uuid.UUID`` strips only the lower-case ``urn:uuid:``, so before this fix
    every other casing failed to parse at all and never reached the nil check:
    the caller was told ``tenant id is not a UUID (len=45)``, pointing an
    operator at a malformed header when the real cause is the banned nil
    sentinel. RFC 8141 §2.1/§3.1 make the scheme and the NID case-insensitive,
    so all six spellings are the same urn naming the same value.

    Asserting on ``"nil uuid"`` is what makes this test able to fail: the
    ``urn`` case in ``test_every_spelling_of_the_sentinel_is_refused`` above
    checks only the exception TYPE, which the wrong-reason refusal also
    satisfies, so it passes vacuously for the upper-case spellings.
    """
    with pytest.raises(InvalidTenantIdError) as exc:
        parse_tenant_id(prefix + NIL, source="X-Tenant-Id header")
    assert "nil uuid" in str(exc.value)
    assert "not a UUID" not in str(exc.value)


@pytest.mark.parametrize(
    "prefix",
    ["urn:uuid:", "URN:UUID:", "Urn:UUID:", "urn:UUID:", "URN:uuid:", "uRn:UuId:"],
    ids=["lower", "upper", "title", "upper-nid", "upper-scheme", "mixed"],
)
def test_every_case_of_the_urn_prefix_parses_a_real_tenant(prefix: str) -> None:
    """The other half: one tenant, six URN spellings, ONE value.

    The lower-case form was already accepted and is pinned by consumers, so this
    extends an existing invariant to casing rather than admitting a new shape.
    """
    assert parse_tenant_id(prefix + REAL) == uuid.UUID(REAL)


def test_a_urn_prefix_is_not_a_licence_to_be_malformed() -> None:
    """Normalising the prefix must not soften what follows it."""
    for bad in ["URN:UUID:not-a-uuid", "URN:UUID:", "URNX:UUID:" + REAL]:
        with pytest.raises(InvalidTenantIdError) as exc:
            parse_tenant_id(bad)
        assert "not a UUID" in str(exc.value)


def test_a_real_tenant_still_parses() -> None:
    """The guard must reject the sentinel by VALUE, not reject uuids in general."""
    assert parse_tenant_id(REAL) == uuid.UUID(REAL)
    assert parse_tenant_id(uuid.UUID(REAL)) == uuid.UUID(REAL)


def test_a_tenant_one_bit_from_nil_still_parses() -> None:
    """Discriminates 'rejects nil' from 'rejects anything zero-ish'."""
    almost = uuid.UUID(int=1)
    assert parse_tenant_id(almost) == almost
    assert parse_tenant_id(str(almost)) == almost


def test_none_still_raises_missing_not_invalid() -> None:
    """The absent case must keep its own class — consumers branch on it."""
    with pytest.raises(MissingTenantError):
        parse_tenant_id(None)
    assert not isinstance(MissingTenantError("x"), InvalidTenantIdError)


def test_empty_string_still_raises_missing() -> None:
    with pytest.raises(MissingTenantError):
        parse_tenant_id("   ")


def test_a_slug_still_raises_invalid_without_echoing_the_value() -> None:
    """The non-UUID branch still rejects a slug, including its no-echo rule.

    A ``Bearer …`` value is deliberately NOT the fixture here any more: it is a
    credential SHAPE, and ``tenant._credential_shape`` now names it explicitly
    (its own tests pin the no-echo rule for that path). This test pins the
    plain-slug path that remains the incidental UUID rejection.
    """
    # NOT named `secret`: detect-secrets' Secret Keyword plugin keys on the
    # VARIABLE NAME, not the value, so `secret = "..."` trips the baseline gate
    # on a string that is deliberately fake. Renaming is right; adding it to
    # .secrets.baseline would enroll a non-secret there permanently.
    bogus_value = "n8n-dev-legacy-slug"
    with pytest.raises(InvalidTenantIdError) as exc:
        parse_tenant_id(bogus_value, source="X-Tenant-Id header")
    assert bogus_value not in str(exc.value)
    assert "not a UUID" in str(exc.value)


def test_nil_refusal_is_catchable_as_the_base_error() -> None:
    """No new exception class: a consumer catching TenantError already handles this."""
    with pytest.raises(TenantError):
        parse_tenant_id(NIL)


def test_require_tenant_id_still_rejects_none_only() -> None:
    """Documents today's behaviour of the sibling helper — see the report.

    ``require_tenant_id`` takes an ALREADY-parsed uuid and does not re-check the
    value, so a nil constructed outside ``parse_tenant_id`` passes here. That is
    out of scope for this change and is reported, not silently altered.
    """
    with pytest.raises(MissingTenantError):
        require_tenant_id(None)
    assert require_tenant_id(uuid.UUID(REAL)) == uuid.UUID(REAL)
