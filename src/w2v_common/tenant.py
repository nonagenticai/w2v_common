"""Tenant identity primitives — the one place a tenant id becomes a ``uuid.UUID``.

Product-owner ruling (2026-08-26): **a ``tenant_id`` may NEVER hold a
sentinel.** Every row carries a real registered tenant, forwarded from the
actual requestor (human user or claude user).

Before this module, ``w2v_common`` violated that in the most damaging way
available: when the requestor's tenant was missing it SUBSTITUTED the
all-zeros UUID (``00000000-0000-0000-0000-000000000000``). That value is a
*valid* UUID, so it passed every type check and every dirty-row audit, and it
is registered in ``licenses.tenants`` as ``default-anon``, so it also looked
legitimate — while silently mis-attributing cost.

The replacement is not a better sentinel. It is failing loud:

* :class:`MissingTenantError` — no tenant at all where one is mandatory.
* :class:`InvalidTenantIdError` — a tenant that is present but is not a UUID
  (a legacy slug like ``n8n-dev``, a truncated value, a ``Bearer …`` header
  that leaked into the wrong field).

Both subclass :class:`TenantError` (a ``ValueError``), so a caller that wants
to treat "no usable tenant" as one condition can catch the base.

This module deliberately has **no dependencies on ``auth``, ``config`` or
``metering``** so every layer can import it without a cycle.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

__all__ = [
    "MAX_TENANT_ID_LENGTH",
    "VOUCHED_TENANT_SOURCES",
    "InvalidTenantIdError",
    "MissingTenantError",
    "TenantError",
    "TenantSource",
    "UnvouchedTenantError",
    "parse_tenant_id",
    "require_tenant_id",
]


class TenantError(ValueError):
    """Base for "this request has no usable tenant identity"."""


class MissingTenantError(TenantError):
    """No tenant id was supplied where a real, registered tenant is mandatory.

    Raised INSTEAD OF substituting a default. There is no fallback tenant by
    design — see the module docstring.
    """


class InvalidTenantIdError(TenantError):
    """A tenant id was supplied but is not a usable tenant.

    Covers the legacy slug shapes (``n8n-dev``, ``anonymous``, ``unknown``),
    anything else that would have been written into a ``uuid`` column as a
    coerced or truncated value, and the nil uuid — which IS a syntactically
    valid UUID but is the banned sentinel this module exists to remove.
    """


#: The all-zeros UUID, i.e. the ``default-anon`` sentinel described in the
#: module docstring. It is a *valid* UUID and it is a *registered* tenant, so
#: neither a type check nor a foreign key will ever reject it. It can only be
#: rejected BY VALUE, and this module is the one seam where every tenant id
#: becomes a ``uuid.UUID``, so this is where that happens.
_NIL_TENANT_ID = uuid.UUID(int=0)

#: The RFC 8141 URN prefix for a UUID, in the only casing ``uuid.UUID`` strips.
_URN_UUID_PREFIX = "urn:uuid:"


def _normalise_urn_prefix(text: str) -> str:
    """Lower-case a leading ``urn:uuid:`` so ``uuid.UUID`` will strip it.

    ``uuid.UUID`` removes the prefix by literal, lower-case substring match, so
    ``urn:uuid:<v>`` parses but ``URN:UUID:<v>`` raises "badly formed
    hexadecimal UUID string". RFC 8141 §2.1 ("NIDs are case insensitive") and
    §3.1 (URN-equivalence lower-cases both the ``urn`` scheme and the NID before
    comparing) make those two the SAME urn naming the SAME uuid, so giving them
    different outcomes is a defect in this parser, not a distinction worth
    keeping. (RFC 4122 §3 is sometimes cited for this; it only says the *hex
    digits* are case-insensitive on input — the NID rule comes from RFC 8141.)

    This does not widen the set of tenant VALUES this function accepts. The
    lower-case URN spelling of a real tenant is already accepted — consumers
    pin it, e.g. youtube_upload_service's "one tenant, four spellings, ONE
    value" test — so every value reachable through the upper-case form was
    already reachable through the lower-case one, and normalises to the same
    ``uuid.UUID``. What it fixes is that ``URN:UUID:<nil>`` was refused as
    "not a UUID (len=45)" instead of as the nil sentinel, which points an
    operator at a malformed header when the real cause is the banned default.

    Only an exact case-insensitive ``urn:uuid:`` prefix is touched; anything
    else is returned unchanged and still has to satisfy ``uuid.UUID``.
    """
    if text[: len(_URN_UUID_PREFIX)].lower() == _URN_UUID_PREFIX:
        return _URN_UUID_PREFIX + text[len(_URN_UUID_PREFIX) :]
    return text


def _reject_nil(parsed: uuid.UUID, source: str) -> uuid.UUID:
    """Return ``parsed`` unless it is the nil uuid, which is never a tenant.

    Unlike the surrounding ``ValueError`` branch, naming the offending value
    here is safe and much more debuggable: the nil uuid is a known constant, so
    the message cannot leak a caller-supplied secret.
    """
    if parsed == _NIL_TENANT_ID:
        raise InvalidTenantIdError(
            f"{source}: tenant id is the nil uuid "
            f"(00000000-0000-0000-0000-000000000000), which is never a real tenant"
        )
    return parsed


#: Longest a legitimate tenant identifier may be. A UUID is 36 characters, so
#: 64 is generous headroom that nothing legitimate can reach — while every
#: credential shape found in ``tenant_id`` columns by the 2026-08-27 fleet
#: audit (``Bearer `` + 43 urlsafe chars = 50, JWTs in the hundreds) exceeds a
#: UUID and is caught by the prefix rules or by this ceiling.
MAX_TENANT_ID_LENGTH = 64


def _credential_shape(text: str) -> str | None:
    """Name the credential SHAPE of ``text``, or ``None`` when it has none.

    ⛔ Returns a fixed shape LABEL and never any part of ``text`` itself. The
    whole point of this guard is that the offending value is a live credential,
    so it must not reach an exception message, a log line, or a traceback —
    that would be the same leak in a new place.
    """
    if text[:7].lower() == "bearer ":
        return "bearer-prefix"
    if text.startswith("eyJ"):
        return "jwt-prefix"
    if len(text) > MAX_TENANT_ID_LENGTH:
        return "over-max-length"
    return None


def parse_tenant_id(value: object, *, source: str = "tenant_id") -> uuid.UUID:
    """Coerce ``value`` to a :class:`uuid.UUID`, or raise.

    ``source`` names the origin (``"license introspection"``, ``"JWT claim"``,
    ``"X-Tenant-Id header"``) and appears in the exception message so an
    operator can tell WHICH lane produced an unusable tenant.

    Never returns a sentinel and never returns ``None``. In particular the nil
    uuid is refused on BOTH paths — as an already-constructed :class:`uuid.UUID`
    and as text — with :class:`InvalidTenantIdError`.
    """
    if isinstance(value, uuid.UUID):
        return _reject_nil(value, source)
    if value is None:
        raise MissingTenantError(f"{source}: no tenant id present")
    if not isinstance(value, str):
        raise InvalidTenantIdError(f"{source}: tenant id has type {type(value).__name__}, not str")
    text = value.strip()
    if not text:
        raise MissingTenantError(f"{source}: tenant id is empty")
    # Credential-shape guard. ``uuid.UUID(text)`` below already rejects these
    # today, but only INCIDENTALLY, and the resulting message ("not a UUID")
    # reads like a slug typo rather than a credential that leaked into an
    # identity field. Naming the shape explicitly, ABOVE the parse, does two
    # things the incidental rejection cannot: an operator triaging the log sees
    # a leak attempt for what it is, and the rule survives any future loosening
    # of the UUID requirement (the columns the fleet audit found credentials in
    # — ``tool_registry.tools``, ``analysis_service.analysis_jobs`` — are TEXT,
    # not uuid). Raises; never coerces, truncates, strips the prefix, or
    # substitutes a default.
    shape = _credential_shape(text)
    if shape is not None:
        raise InvalidTenantIdError(
            f"{source}: credential-shaped tenant identifier rejected "
            f"(len={len(text)}, shape={shape})"
        )
    try:
        parsed = uuid.UUID(_normalise_urn_prefix(text))
    except ValueError as exc:
        # Deliberately does NOT echo the value: an X-Tenant-Id field can carry
        # a ``Bearer <license token>`` and logging it would leak the token.
        raise InvalidTenantIdError(f"{source}: tenant id is not a UUID (len={len(text)})") from exc
    # Checked on the PARSED value, not the text: ``uuid.UUID`` accepts the nil
    # sentinel in braced, urn and undashed spellings too, and every one of them
    # normalises to this same object. A string comparison would miss them.
    return _reject_nil(parsed, source)


def require_tenant_id(value: uuid.UUID | None, *, source: str = "tenant_id") -> uuid.UUID:
    """Assert a already-parsed tenant is present. Raises :class:`MissingTenantError`."""
    if value is None:
        raise MissingTenantError(f"{source}: no tenant bound to this request")
    return value


class TenantSource(StrEnum):
    """HOW a request's tenant was established — the provenance ladder.

    Copied from kag's ``src/auth/tenant.py`` (``SOURCE_*`` constants,
    ``VOUCHED_TENANT_SOURCES``), which is the reference implementation for the
    fleet. This is not decoration: it is the input to
    :func:`w2v_common.auth.outbound.build_downstream_tenant_headers`, which may
    only DELEGATE a tenant to a metered / license-introspecting peer (unlocking
    that tenant's BYO provider credentials and billing it) when the provenance
    is one this service can actually stand behind.
    """

    #: ``X-Tenant-Id: Bearer <license-token>``, introspected to a real tenant.
    #: Self-authenticating; the strongest rung.
    LICENSE_TOKEN = "license-token"  # noqa: S105 — a provenance label, not a credential  # nosec B105

    #: A bare tenant vouched for by an allowlisted service identity (kag's WS-4
    #: on-behalf-of). w2v has no vouching lane of its own yet — the member
    #: exists so the ladder matches kag's and a future lane has a name to set.
    VOUCHED_ONBEHALF = "vouched-onbehalf"

    #: A Keycloak-JWT ``tenant_id`` claim. **w2v's one addition to kag's
    #: ladder**, because kag has no browser-JWT lane and w2v does (the SPA hits
    #: the orchestrator with a portal-realm token). It is VOUCHED: the claim is
    #: carried inside a signature this service verified against the realm's
    #: JWKS, so it is not attacker-settable in the way a bare header is. Mapping
    #: it to ``UNVOUCHED_HEADER`` instead would refuse every browser-initiated
    #: run at the first metered hop.
    JWT_CLAIM = "jwt-claim"

    #: A bare ``X-Tenant-Id`` INJECTED by the platform's ``license-authz``
    #: ext-authz gate, i.e. the licence was introspected by the gate at the
    #: waypoint and this service is reading the gate's verdict rather than the
    #: credential. Recognised only when the whole injected set is present
    #: (``x-tenant-id`` as a bare UUID *plus* ``x-license-id`` *plus*
    #: ``x-subject-id``) and only when the operator has explicitly opted in via
    #: ``AUTH_TRUST_MESH_LICENSE_IDENTITY`` — the headers themselves are
    #: forgeable by any in-mesh peer (the platform's header-strip filter is
    #: north-south only), so the lane is OFF by default and additionally
    #: peer-IP restricted and optionally HMAC-gated. See
    #: ``w2v_common.auth.middleware._dispatch_mesh_vouched`` for the full
    #: threat statement. VOUCHED: when the lane is enabled the operator has
    #: asserted that only the gate can reach this port, and without vouching it
    #: every gated request would fail at its first metered hop.
    MESH_VOUCHED = "mesh-vouched"

    #: A tenant resolved from the caller's ACTIVE ``tenant_memberships`` row by
    #: the orchestrator's membership middleware (P2 tenancy): the Keycloak JWT
    #: proves WHO the caller is (``sub``, signature verified against the realm
    #: JWKS), and w2v's own membership table — not a token claim — says which
    #: tenant that subject belongs to. VOUCHED: both halves are things this
    #: service verified itself; refusing to delegate it would fail every
    #: browser-initiated run at its first metered hop (``UnvouchedTenantError``)
    #: once ``AUTH_JWT_TENANT_REQUIRED=false`` is live on the orchestrator.
    #: Set by the APP (``set_identity_context(..., tenant_source=MEMBERSHIP)``);
    #: :class:`~w2v_common.auth.middleware.AuthMiddleware` never sets it.
    MEMBERSHIP = "membership"

    #: A tenant resolved from a w2v tenant API key (``Bearer w2v_sk_...``) by
    #: the verifier the app registered via
    #: :func:`w2v_common.auth.middleware.register_api_key_verifier` — the key's
    #: hash was matched against the tenant's own ``tenant_api_keys`` row.
    #: VOUCHED: self-authenticating, exactly like a licence token; the key was
    #: minted for that tenant and only that tenant.
    TENANT_API_KEY = "tenant-api-key"  # nosec B105  # pragma: allowlist secret — a label

    #: A tenant carried in the ``tenant`` claim of an intra-cell token (P7,
    #: ``X-W2V-Cell-Token``): an EdDSA JWT signed by the CALLING w2v service's
    #: own key, verified here against the cell keyring (kid bound to issuer,
    #: aud = this service, deployment id, 60 s lifetime, jti replay-checked).
    #: VOUCHED: the caller authenticated the tenant upstream and signed it for
    #: this hop; refusing to delegate it would fail every run at its first
    #: metered hop after the cell lane replaces the licence delegation lane.
    CELL_JWT = "cell-jwt"

    #: A bare ``X-Tenant-Id`` with no credential behind it. Any peer that can
    #: reach this service can set it, so it is accepted as a DATA SCOPE but is
    #: never delegated as an IDENTITY TO SPEND AS. This is what w2v's
    #: trusted-network lane produces: the network vouches for the CALLER, never
    #: for the TENANT.
    UNVOUCHED_HEADER = "unvouched-header"

    #: A service's own configured default tenant, used when no header arrived.
    #: **Unreachable in w2v today** and kept only so the ladder is complete:
    #: ``default_tenant_id`` / ``UNATTRIBUTED_TENANT_ID`` were deleted, so there
    #: is no default to fall back to. Not vouched, for the same reason as kag's.
    TRUSTED_NETWORK_DEFAULT = "trusted-network-default"

    #: Dev escape hatch (auth middleware not installed). Must never reach a
    #: billing path.
    AUTH_DISABLED = "auth-disabled"


#: Provenances this service may DELEGATE (i.e. send ``X-Effective-Tenant-Id``
#: for). Everything absent from this set names a tenant that is ACCEPTED — no
#: caller is rejected at the door — but that will not be vouched for downstream.
VOUCHED_TENANT_SOURCES: frozenset[TenantSource] = frozenset(
    {
        TenantSource.LICENSE_TOKEN,
        TenantSource.VOUCHED_ONBEHALF,
        TenantSource.JWT_CLAIM,
        TenantSource.MESH_VOUCHED,
        TenantSource.MEMBERSHIP,
        TenantSource.TENANT_API_KEY,
        TenantSource.CELL_JWT,
    }
)


class UnvouchedTenantError(TenantError):
    """A tenant was asked to be DELEGATED but its provenance is not vouched.

    Delegation is a privilege escalation: ``X-Effective-Tenant-Id`` converts a
    header into the authority to spend another tenant's LLM credentials and bill
    them. Raising here is the whole point — kag returns 403 TENANT_VOUCH_REQUIRED
    at the same seam. Never downgrade this to "drop the header and continue":
    the hop would then silently bill the SERVICE-ACCOUNT tenant for a customer
    run, which is the mis-attribution bug this whole change exists to remove.
    """
