"""Per-request identity ContextVars for cross-async-context propagation.

Worker_management's canonical pattern (see
``worker_management/src/api/identity_context.py``) lifted into w2v_common
so every w2v service has one shared module to import.

The auth middleware stashes identity on ``request.state`` for code paths
that have direct access to the FastAPI ``Request``. The ContextVars below
are for paths that DON'T — typically anything reached via cached/shared
``httpx.AsyncClient`` instances where baking identity into
``default_headers`` at construction time would attribute every subsequent
request to whoever happened to warm the cache first.

Pattern:
    1. ``AuthMiddleware`` calls ``set_identity_context(...)`` after license
       introspect, BEFORE ``await call_next(request)``.
    2. Anywhere downstream — including code reached via long await chains
       — ``get_*_from_context()`` returns the right value.
    3. Cached httpx clients use an ``event_hooks={"request": [hook]}``
       handler that calls ``get_identity_headers()`` per outbound request.

ContextVars are task-local in asyncio, so two concurrent requests using
the same cached client see their own identity.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar

import httpx
from prometheus_client import Counter

from ..tenant import (
    VOUCHED_TENANT_SOURCES,
    MissingTenantError,
    TenantSource,
    UnvouchedTenantError,
    parse_tenant_id,
)
from .outbound import EFFECTIVE_TENANT_HEADER, is_metered_destination

logger = logging.getLogger(__name__)

#: Wire header naming the fr project a request works for (ATTRIBUTION-PLAN
#: "THE CONTRACT"). A bare UUID; OMITTED when absent — never a placeholder.
PROJECT_ID_HEADER = "X-Project-Id"
#: Wire header naming the tenant-member who acted. Same UUID/omission rules as
#: :data:`PROJECT_ID_HEADER`, plus one more: it is an ASSERTION about somebody
#: else's identity, so a receiver honours it only from a trusted (delegating)
#: caller — see :func:`bind_attribution_from_headers`.
ACTOR_ID_HEADER = "X-Actor-Id"
#: Wire header carrying the W2V project (orchestrator ``projects.id``) a request
#: works for, between w2v services. ⛔ NOT the fr project: w2v projects and fr
#: projects are separate concepts (owner ruling 2026-10-07), so this travels on
#: its own header and its own ContextVar and never touches ``X-Project-Id``
#: binding. A bare canonical UUID; omitted when unbound.
W2V_PROJECT_ID_HEADER = "X-W2V-Project-Id"
#: Wire header carrying the W2V JOB (the orchestrator run id) a request works
#: for, between w2v services. Same shape as :data:`W2V_PROJECT_ID_HEADER`: its
#: own header, its own ContextVar, flag-independent, omitted when unbound. The
#: cost bus reads it as the default ``metadata.job_id`` of a cost event, so a
#: downstream emit (tts, analyzer, compiler, html-render, image-gen) names the
#: run that caused it without each service threading the id by hand.
W2V_JOB_ID_HEADER = "X-W2V-Job-Id"
#: A job id is opaque (a run uuid today) but it rides a header and lands in
#: billing metadata, so it is bounded to a conservative token alphabet: no
#: whitespace, no control characters, no header/CSV/log-injection surface.
_W2V_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

# The tenant ContextVar is a real ``uuid.UUID | None``. It used to be a
# ``str`` defaulting to ``""`` — an empty string is itself a sentinel: it made
# "no tenant bound" indistinguishable from "tenant bound to nothing", it
# rendered as a blank ``X-Tenant-Id`` header downstream, and it was the value
# the cost bus turned into the nil-UUID. ``None`` means UNBOUND and every
# reader must handle it explicitly (see ``get_tenant_id_from_context``).
_tenant_id_var: ContextVar[uuid.UUID | None] = ContextVar("tenant_id", default=None)
# HOW the bound tenant was obtained. Without this, rule 4 (never delegate an
# unvouched tenant) is UNENFORCEABLE: the outbound hook can see a tenant but
# not whether anyone proved it. ``None`` means "no lane recorded a provenance"
# and is treated exactly like an unvouched one — a caller that never stated
# where its tenant came from is precisely the caller that must not delegate.
_tenant_source_var: ContextVar[TenantSource | None] = ContextVar("tenant_source", default=None)
# The remaining identity vars are NOT tenant ids and keep their empty-string
# defaults — they are opaque strings whose absence is legitimate (a CronJob
# has no license, a service-account token has no subject).
_license_id_var: ContextVar[str] = ContextVar("license_id", default="")
_subject_id_var: ContextVar[str] = ContextVar("subject_id", default="")
# ATTRIBUTION-ONLY subject (0.93.0, transitional): the ``X-Subject-Id`` a cell
# caller sent on a token WITHOUT a signed ``subject`` claim (a 0.92.x peer).
# Deliberately NOT ``_subject_id_var``: that one is read by authz
# (tts_gateway's WORKER_SUBJECT_IDS gate, orchestrator ``SubjectIdDep``,
# ``require_license``), and a header any allowed cell caller can set must never
# decide access. It is forwarded as ``X-Subject-Id`` (legacy continuity) and
# readable via ``get_attribution_subject_id_from_context`` for billing, but it
# is NEVER signed into an onward cell token. Cleared on every (re-)seed.
# TODO(C-del): delete with the header fallback in AuthMiddleware._dispatch_cell_token.
_unsigned_subject_var: ContextVar[str] = ContextVar("unsigned_subject_id", default="")
# WHO ACTED, as opposed to WHO PAYS. billing's alembic 0021 ("DECISION 1 —
# IDENTITY GRAIN: CARRY BOTH") splits the two: ``subject_id`` is the licences
# Subject UUID that is BILLED, ``actor_id`` is the acting principal — the
# Keycloak JWT ``sub`` of the human who made the request. It is NULLABLE and
# NEVER DEFAULTED on the receiving side, so an empty value here must reach the
# cost bus as an ABSENT key, never as ``""`` and never as the string "None":
# billing's ``_parse_uuid`` would turn either into None anyway, but only after
# the row has been written as if attribution had been attempted.
#
# Set on the Keycloak JWT lane ONLY (``AuthMiddleware``). The license-token and
# trusted-network lanes leave it empty: a license token names a subject, not a
# human actor, and any pod inside the trusted CIDRs can assert any header, so
# an actor claimed there would be attribution we cannot back up.
_actor_id_var: ContextVar[str] = ContextVar("actor_id", default="")
# True while this execution context serves an INBOUND request (set by
# AuthMiddleware). Gates the SERVICE_ACTOR_ID fallback: a service may name
# itself as actor only for work it ORIGINATES, never to paper over a caller
# that arrived without an actor. See resolve_actor_id().
_inbound_request_var: ContextVar[bool] = ContextVar("inbound_request", default=False)
#: Inbound request header a TENANT-provider call carries (``w2v_common.providers``).
PROVIDER_HOP_HEADER = "X-W2V-Provider-Hop"
# True while this context serves an inbound request that arrived WITH
# ``X-W2V-Provider-Hop`` — i.e. a tenant's registered provider URL pointed back
# at a w2v service. ``resolve_provider_slot`` refuses there (loop guard). Set by
# AuthMiddleware next to ``mark_inbound_request``; reset by clear_identity_context.
_provider_hop_var: ContextVar[bool] = ContextVar("provider_hop", default=False)
# WHICH fr PROJECT the work belongs to. billing reads it as the top-level
# ``project_id`` of a cost event (``billing_cost_events.project_id``), nullable
# and never defaulted there — so, exactly like the actor, ``""`` means ABSENT
# and must reach the wire as an omitted key, never as ``""`` or a placeholder.
#
# Bound ONLY by :func:`bind_attribution_from_headers` (strict, canonical,
# non-nil UUID). Unlike the actor it is honoured from ANY authenticated caller:
# a project names a scope WITHIN the tenant the request already authenticated
# as, not a principal, and billing independently refuses a project id that is
# itself a tenant id (``_resolve_project_id``).
_project_id_var: ContextVar[str] = ContextVar("project_id", default="")
# WHICH W2V PROJECT (orchestrator ``projects.id``) the work belongs to. Bound by
# the orchestrator's run dispatcher (``set_w2v_project_context``) and, on every
# downstream w2v hop, from the inbound ``X-W2V-Project-Id`` header by
# ``bind_attribution_from_headers``. Read by ``WMClient`` as the X-Project-Id of
# an LLM call. ``""`` means unbound. Preserved across a re-seed of the SAME
# tenant (a project belongs to exactly one tenant), cleared on a tenant change.
_w2v_project_id_var: ContextVar[str] = ContextVar("w2v_project_id", default="")
# WHICH W2V JOB (orchestrator run id) the work belongs to. Bound by the
# orchestrator around each stage (``set_w2v_job_context``) and, on every
# downstream w2v hop, from the inbound ``X-W2V-Job-Id`` header. Same lifecycle
# as the w2v project: kept across a same-tenant re-seed, cleared on a tenant
# change. ``""`` means unbound.
_w2v_job_id_var: ContextVar[str] = ContextVar("w2v_job_id", default="")
# The caller's raw license token (preserved on request.state.license_token by
# AuthMiddleware). Read ONLY by ``caller_token_event_hook`` — which is wired
# only on clients that talk to services authorizing per *customer* tenant
# rather than per service-account (currently artifact_service). All other
# clients (WMClient et al) keep using ``identity_event_hook`` and therefore
# their constructor-baked service-account ``Authorization`` survives.
_caller_license_token_var: ContextVar[str] = ContextVar("caller_license_token", default="")
# Cost-correlation (fleet-wide X-Correlation-Id + optional X-Stage). Lets a
# caller attribute downstream LLM cost events back to one operation — for w2v,
# the orchestrator sets correlation_id=run_id and stage=DAG stage around each
# worker_management call. Stamped outbound by ``identity_event_hook``; read by
# worker_management + inference-router onto their cost events. Empty default
# keeps every outbound request header-safe and opt-in.
_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
_stage_var: ContextVar[str] = ContextVar("stage", default="")

# ``EFFECTIVE_TENANT_HEADER`` and the metered-destination predicate live in
# ``auth.outbound`` (which owns the outbound POLICY); re-exported here because
# every existing importer reads them off this module.


def set_correlation_context(
    correlation_id: str | None,
    stage: str | None = None,
) -> None:
    """Populate per-request cost-correlation ContextVars.

    Orthogonal to identity — set by the caller around an operation whose LLM
    cost it wants to attribute (w2v: correlation_id=run_id, stage=DAG stage).
    Empty values mean "no correlation" (preserves today's behavior).
    """
    _correlation_id_var.set(correlation_id or "")
    _stage_var.set(stage or "")


class _Keep:
    """Sentinel for "the caller did not mention provenance".

    ``tenant_source=None`` used to carry TWO incompatible meanings — "there is
    no provenance" and "I am not talking about provenance" — and the second one
    silently destroyed the first. See ``set_identity_context``.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover — debugging aid only
        return "<KEEP_TENANT_SOURCE>"


#: Default for ``set_identity_context(tenant_source=...)``. Exported so a caller
#: can state "preserve" explicitly where that reads better than omission.
KEEP_TENANT_SOURCE = _Keep()


def set_identity_context(
    tenant_id: uuid.UUID | str | None,
    license_id: str | None,
    subject_id: str | None,
    caller_license_token: str | None = None,
    tenant_source: TenantSource | _Keep | None = KEEP_TENANT_SOURCE,
    *,
    actor_id: str | None = None,
) -> None:
    """Populate per-request identity ContextVars.

    Called by ``AuthMiddleware`` once per request, after license-token
    introspection.

    ``tenant_id`` accepts a ``uuid.UUID`` or a UUID-shaped string and is
    stored as a ``uuid.UUID``. A non-UUID string raises
    :class:`w2v_common.tenant.InvalidTenantIdError` HERE rather than being
    stored and mis-attributed later. ``None`` means "this request has no
    tenant" (CronJob, startup task) and leaves the var UNBOUND — it is NOT
    coerced to a placeholder.

    The other identity values are opaque strings; ``None`` becomes ``""`` so
    downstream callers can use them directly as header values.

    ``caller_license_token`` is the caller's RAW license token. It is
    consumed ONLY by ``caller_token_event_hook`` (currently wired only on
    ArtifactClient). Empty when there's no inbound bearer-authenticated
    request (Keycloak JWT path, trusted-network path, CronJobs, startup
    tasks) — the constructor's service-account default survives in that
    case.

    ``actor_id`` is the ACTING PRINCIPAL (the Keycloak JWT ``sub``), which
    billing carries alongside — never instead of — ``subject_id``: the subject
    is WHO PAYS, the actor is WHO ACTED. Keyword-only with a ``None`` default,
    and it follows the plain-string convention of ``license_id``/``subject_id``
    rather than the ``KEEP_TENANT_SOURCE`` one: omitting it CLEARS the var.
    That is deliberate for a money-attribution field — a re-seed that does not
    name an actor yields "no actor recorded" (billing stores NULL), never a
    stale human's id inherited onto work they did not do. Provenance is
    preserved on omission because dropping it turns working calls into hard
    failures; an absent actor costs nothing but a NULL column.

    ``tenant_source`` records HOW the tenant was established (see
    :class:`w2v_common.tenant.TenantSource`). Every ``AuthMiddleware`` lane
    passes it explicitly.

    OMITTING it means "I am not talking about provenance", NOT "there is none":

    * Re-seeding the **same** tenant PRESERVES the bound provenance. This is the
      background-task case: ``asyncio.create_task`` copies the caller's context,
      so a detached job inherits the request's provenance, and a re-seed that
      only wants to rebind the tenant must not silently throw it away.
    * Re-seeding a **different** tenant DROPS it to ``None``. Provenance is
      bound to a tenant — it is what unlocks that tenant's BYO credentials and
      bills them — so a reaper walking rows for many tenants can never inherit
      one tenant's proof and spend as another.

    Passing an explicit value always wins, including an explicit ``None``, which
    is how a caller states "drop whatever was here".

    ⚠️ This did NOT used to be true, and the difference caused live outages.
    ``_tenant_source_var.set(tenant_source)`` ran unconditionally, so a
    background task that re-seeded its own tenant without naming a source
    DESTROYED the provenance it had inherited, and its next metered call failed
    closed with "provenance is 'unrecorded', which is not vouched" — surfacing
    on the DESTINATION's url, which made it read as a bug in kag or
    modality-gateway. The docstring here used to claim absence "degrades to the
    safe end of the ladder, never the permissive one". That was true of security
    and false of availability: it converted working requests into hard failures,
    and four separate repos shipped per-caller patches for it before the default
    itself was fixed.
    """
    new_tenant = (
        None if tenant_id is None else parse_tenant_id(tenant_id, source="set_identity_context")
    )
    if isinstance(tenant_source, _Keep):
        # OMITTED. Provenance is bound to a TENANT — it records that THIS tenant
        # was established by a credential we verified, and it is what unlocks
        # that tenant's BYO credentials and bills them. So it may be preserved
        # across a re-seed of the SAME tenant, and must NEVER follow the context
        # onto a different one.
        previous_tenant = _tenant_id_var.get()
        previous_source = _tenant_source_var.get()
        resolved = (
            previous_source
            if previous_tenant is not None and previous_tenant == new_tenant
            else None
        )
    else:
        # STATED. An explicit value always wins, including an explicit ``None``,
        # which is how a caller says "drop whatever was here".
        resolved = tenant_source

    if _tenant_id_var.get() != new_tenant:
        # A w2v project belongs to one tenant; never let it follow the context
        # onto another. Same-tenant re-seeds (background tasks) keep it. The
        # w2v job follows the same rule for the same reason.
        _w2v_project_id_var.set("")
        _w2v_job_id_var.set("")
    _tenant_id_var.set(new_tenant)
    _license_id_var.set(license_id or "")
    _subject_id_var.set(subject_id or "")
    _unsigned_subject_var.set("")
    _actor_id_var.set(actor_id or "")
    # Cleared on EVERY (re-)seed, like the actor and for the same reason: a
    # project is bound to the request that named it, so a re-seed — possibly
    # for a different tenant — must never inherit one. The middleware re-binds
    # it AFTER this call via ``bind_attribution_from_headers``.
    _project_id_var.set("")
    _caller_license_token_var.set(caller_license_token or "")
    _tenant_source_var.set(resolved)


def clear_identity_context() -> None:
    """Reset all identity ContextVars to UNSET (tenant) / empty (the rest).

    Useful in tests and in middleware that wants to scope identity to a
    single request without leaking into pytest's shared event loop.
    """
    _tenant_id_var.set(None)
    _tenant_source_var.set(None)
    _license_id_var.set("")
    _subject_id_var.set("")
    _unsigned_subject_var.set("")
    _actor_id_var.set("")
    _inbound_request_var.set(False)
    _provider_hop_var.set(False)
    _project_id_var.set("")
    _w2v_project_id_var.set("")
    _w2v_job_id_var.set("")
    _caller_license_token_var.set("")
    _correlation_id_var.set("")
    _stage_var.set("")


def get_tenant_id_from_context() -> uuid.UUID | None:
    """The requestor's real tenant, or ``None`` when no tenant is bound.

    CONTRACT: ``None`` means UNBOUND — there is no requestor tenant on this
    execution context (a CronJob, a startup task, an unauthenticated path).
    Callers must branch on it. They must NOT substitute a default: doing so
    is exactly the nil-UUID bug this replaced. A caller for whom an absent
    tenant is a bug should use
    ``w2v_common.tenant.require_tenant_id(get_tenant_id_from_context())``,
    which raises :class:`w2v_common.tenant.MissingTenantError`.
    """
    return _tenant_id_var.get()


def get_tenant_source_from_context() -> TenantSource | None:
    """How the bound tenant was obtained, or ``None`` when no lane said.

    ``None`` is NOT a neutral value: every delegation check treats it as
    unvouched. See :data:`w2v_common.tenant.VOUCHED_TENANT_SOURCES`.
    """
    return _tenant_source_var.get()


def is_tenant_vouched() -> bool:
    """True when the bound tenant may be DELEGATED to a metered peer."""
    source = _tenant_source_var.get()
    return source is not None and source in VOUCHED_TENANT_SOURCES


def get_license_id_from_context() -> str:
    return _license_id_var.get()


def get_subject_id_from_context() -> str:
    """The AUTHENTICATED subject, or ``""``. Safe for authorization.

    Since 0.93.0 this never returns a subject a cell caller asserted only by
    header (an unsigned pre-0.93.0 cell token): see
    :func:`get_attribution_subject_id_from_context`.
    """
    return _subject_id_var.get()


def bind_unsigned_attribution_subject(subject_id: str | None) -> None:
    """Bind an ATTRIBUTION-ONLY subject (cell header fallback; transitional).

    Called by ``AuthMiddleware`` AFTER ``set_identity_context`` on the cell lane
    when the token carries no signed ``subject`` claim. Never visible through
    :func:`get_subject_id_from_context` (authz), never signed onward.
    TODO(C-del): remove with the fallback.
    """
    _unsigned_subject_var.set(subject_id or "")


def get_attribution_subject_id_from_context() -> str:
    """The subject to BILL / log: the authenticated one, else the unsigned
    cell-header one (0.93.0 transitional fallback), else ``""``.

    ⛔ Never use this for an authorization decision — use
    :func:`get_subject_id_from_context`.
    """
    return _subject_id_var.get() or _unsigned_subject_var.get()


def get_actor_id_from_context() -> str:
    """The ACTING PRINCIPAL (Keycloak JWT ``sub``), or ``""`` when none.

    ``""`` means "no actor recorded on this execution context" — a
    license-token call, a trusted-network hop, a CronJob, a background task.
    Readers must treat it as ABSENT (omit the field) and must NEVER substitute
    ``subject_id``: the two answer different questions (who acted vs who pays)
    and billing stores both.
    """
    return _actor_id_var.get()


def set_w2v_project_context(project_id: uuid.UUID | str | None) -> None:
    """Bind (or with ``None`` clear) the W2V project for this execution context.

    Strict: a non-canonical or nil UUID raises ``ValueError`` — a caller that
    holds a bad project id has a bug, and silently dropping it would bill the
    tenant-level project instead.
    """
    if project_id is None:
        _w2v_project_id_var.set("")
        return
    canonical = parse_attribution_uuid(str(project_id))
    if canonical is None:
        raise ValueError(f"not a canonical non-nil w2v project uuid: {project_id!r}")
    _w2v_project_id_var.set(canonical)


def get_w2v_project_id_from_context() -> str:
    """The bound W2V project id (canonical UUID string), or ``""`` when unbound."""
    return _w2v_project_id_var.get()


def parse_w2v_job_id(raw: str | None) -> str | None:
    """Strictly parse a w2v job id; ``None`` for anything else.

    Opaque but bounded: 1-128 chars of ``[A-Za-z0-9._:-]``, starting
    alphanumeric. Surrounding whitespace is stripped; nothing else is coerced.
    """
    value = (raw or "").strip()
    if not value or not _W2V_JOB_ID_RE.match(value):
        return None
    return value


def set_w2v_job_context(job_id: uuid.UUID | str | None) -> None:
    """Bind (or with ``None`` clear) the W2V job (run id) for this execution context.

    Strict, like :func:`set_w2v_project_context`: a malformed id raises
    ``ValueError`` — a caller holding a bad job id has a bug, and silently
    dropping it would emit cost rows with no job.
    """
    if job_id is None:
        _w2v_job_id_var.set("")
        return
    parsed = parse_w2v_job_id(str(job_id))
    if parsed is None:
        raise ValueError(f"not a valid w2v job id: {job_id!r}")
    _w2v_job_id_var.set(parsed)


def get_w2v_job_id_from_context() -> str:
    """The bound W2V job id, or ``""`` when unbound."""
    return _w2v_job_id_var.get()


def get_project_id_from_context() -> str:
    """The fr project this execution context works for, or ``""`` when none.

    ``""`` means ABSENT: readers omit the field, they never substitute a
    default (there is no "default project").
    """
    return _project_id_var.get()


def _header(headers: Mapping[str, str], name: str) -> str:
    """Case-tolerant header read for Starlette ``Headers`` and plain dicts."""
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return (value or "").strip()


def tenant_level_project_id(tenant_id: uuid.UUID | str | None) -> str | None:
    """The tenant-level project id for a PROJECT-LESS app (owner ruling 2026-10-07).

    The tenant uuid's 32 hex digits REVERSED, re-formatted 8-4-4-4-12. Reversal
    is deterministic, needs no lookup, and never equals the tenant id itself
    (billing refuses a project id that is a tenant id) unless the hex is a
    palindrome. It is an involution: applying it twice returns the tenant.

    ``None`` for a missing, malformed or nil tenant — a project built from a
    fabricated tenant books spend to a project nobody owns, which is worse than
    sending none. Mirrors merqi marketing's helper of the same name.
    """
    if tenant_id is None:
        return None
    if isinstance(tenant_id, uuid.UUID):
        parsed = tenant_id
    else:
        try:
            parsed = uuid.UUID(tenant_id.strip())
        except ValueError:
            return None
    if parsed.int == 0:
        return None
    rev = parsed.hex[::-1]
    return f"{rev[:8]}-{rev[8:12]}-{rev[12:16]}-{rev[16:20]}-{rev[20:]}"


def parse_attribution_uuid(raw: str | None) -> str | None:
    """Strictly parse an attribution UUID; ``None`` for anything else.

    Strict means the CANONICAL 36-char hyphenated form only (any case, returned
    lower-cased). ``uuid.UUID`` alone also accepts ``{...}``, ``urn:uuid:...``
    and bare 32-hex forms; those are refused so every hop compares one spelling.
    The nil UUID is refused too: it is the placeholder the contract forbids.
    """
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    canonical = str(parsed)
    if canonical != value.lower() or parsed.int == 0:
        return None
    return canonical


#: Env var naming THIS service's own Keycloak service-account ``sub`` (a UUID).
#: Owner ruling 2026-10-07: the actor is the human's sub when a human acted,
#: otherwise the CALLING service's own service-account sub — never a sentinel.
SERVICE_ACTOR_ID_ENV = "SERVICE_ACTOR_ID"

_warned_bad_service_actor: set[str] = set()


def get_service_actor_id() -> str | None:
    """``SERVICE_ACTOR_ID`` as a canonical non-nil UUID, else ``None``.

    Read at call time. A set-but-malformed value is a deployment bug: it is
    refused (never shipped) and WARNs once per distinct value.
    """
    raw = os.environ.get(SERVICE_ACTOR_ID_ENV, "").strip()
    if not raw:
        return None
    parsed = parse_attribution_uuid(raw)
    if parsed is None and raw not in _warned_bad_service_actor:
        _warned_bad_service_actor.add(raw)
        logger.warning(
            "%s is set but is not a canonical non-nil UUID; ignoring it", SERVICE_ACTOR_ID_ENV
        )
    return parsed


#: Reasons an actor could not be resolved (``w2v_common_actor_missing_total``):
#: ``inbound_without_actor`` — serving a request whose caller sent no valid
#: actor; the service must NOT substitute itself (that would hide the caller's
#: gap from billing coverage and router enforcement).
#: ``no_service_actor`` — service-originated work and no valid SERVICE_ACTOR_ID.
ACTOR_MISSING_INBOUND = "inbound_without_actor"
ACTOR_MISSING_NO_SERVICE_ACTOR = "no_service_actor"

#: Actor could not be resolved. ``service`` is the emitting ``service_key`` on
#: cost events and ``outbound`` on an outbound call (the scrape target names
#: the pod). Pre-initialised per reason is impossible (service is dynamic).
actor_missing_total = Counter(
    "w2v_common_actor_missing_total",
    "Actor could not be resolved (cost event or outbound call), by reason.",
    ["service", "reason"],
)


def record_actor_missing(service: str, reason: str) -> None:
    """Best-effort observation — must never break a request or an emit."""
    try:
        actor_missing_total.labels(service=service, reason=reason).inc()
    except Exception:  # noqa: BLE001 — telemetry must never fail a request
        logger.debug("actor_missing_metric_failed service=%s reason=%s", service, reason)


def mark_inbound_request() -> None:
    """Mark this execution context as serving an inbound request.

    Called by ``AuthMiddleware`` for every non-public request; reset by
    ``clear_identity_context``. Services with their own inbound middleware
    must call it too, or their requests will be treated as service-originated.
    """
    _inbound_request_var.set(True)


def is_inbound_request() -> bool:
    return _inbound_request_var.get()


def mark_provider_hop(arrived_with_hop_header: bool) -> None:
    """Record whether the inbound request carried ``X-W2V-Provider-Hop``.

    Called by ``AuthMiddleware`` for every non-public request (services with
    their own inbound middleware call it too). ``True`` makes
    ``w2v_common.providers.resolve_provider_slot`` refuse: a tenant provider URL
    that leads back into w2v must not spend that tenant's slot again.
    """
    _provider_hop_var.set(arrived_with_hop_header)


def is_provider_hop() -> bool:
    """True when the inbound request arrived through a tenant-provider hop."""
    return _provider_hop_var.get()


def resolve_actor_id() -> tuple[str | None, str | None]:
    """``(actor_id, None)`` or ``(None, missing_reason)``.

    1. A bound canonical UUID actor (human or trusted caller) always wins.
    2. Serving an inbound request without one -> absent,
       ``inbound_without_actor``. The service's own id is NOT used.
    3. Service-originated work (no inbound request) -> ``SERVICE_ACTOR_ID``.
    4. Neither -> absent, ``no_service_actor``.
    """
    bound = parse_attribution_uuid(_actor_id_var.get())
    if bound:
        return bound, None
    if _inbound_request_var.get():
        return None, ACTOR_MISSING_INBOUND
    own = get_service_actor_id()
    if own:
        return own, None
    return None, ACTOR_MISSING_NO_SERVICE_ACTOR


def bind_attribution_from_headers(headers: Mapping[str, str], *, trusted: bool) -> None:
    """Bind ``X-Project-Id`` / ``X-Actor-Id`` / ``X-W2V-*`` from an inbound request.

    Call AFTER ``set_identity_context`` (which clears the project and sets the
    lane's own actor), inside the same per-request scope, so
    ``clear_identity_context`` in the lane's ``finally`` resets both.

    * ``X-Project-Id`` — bound whenever it is a valid UUID. An invalid value is
      ignored (DEBUG line), never coerced.
    * ``X-Actor-Id`` — bound ONLY when ``trusted`` is True. An actor is an
      assertion about WHO ACTED; from an untrusted caller it is attribution we
      cannot back up, so it is ignored and whatever actor the lane itself bound
      (e.g. the JWT ``sub``) stands. There is deliberately NO default for
      ``trusted``: every call site must state its trust decision.
    """
    raw_project = _header(headers, PROJECT_ID_HEADER)
    if raw_project:
        project_id = parse_attribution_uuid(raw_project)
        if project_id is None:
            logger.debug("attribution: ignoring non-UUID %s", PROJECT_ID_HEADER)
        else:
            _project_id_var.set(project_id)

    # The W2V project rides on its own header and var; honoured from any
    # authenticated caller for the same reason as the fr project (a scope
    # within the already-authenticated tenant, not a principal).
    raw_w2v_project = _header(headers, W2V_PROJECT_ID_HEADER)
    if raw_w2v_project:
        w2v_project = parse_attribution_uuid(raw_w2v_project)
        if w2v_project is None:
            logger.debug("attribution: ignoring non-UUID %s", W2V_PROJECT_ID_HEADER)
        else:
            _w2v_project_id_var.set(w2v_project)

    # The W2V job (run id) rides the same way, for the same reason.
    raw_w2v_job = _header(headers, W2V_JOB_ID_HEADER)
    if raw_w2v_job:
        w2v_job = parse_w2v_job_id(raw_w2v_job)
        if w2v_job is None:
            logger.debug("attribution: ignoring malformed %s", W2V_JOB_ID_HEADER)
        else:
            _w2v_job_id_var.set(w2v_job)

    raw_actor = _header(headers, ACTOR_ID_HEADER)
    if not raw_actor:
        return
    if not trusted:
        logger.debug("attribution: ignoring %s from an untrusted caller", ACTOR_ID_HEADER)
        return
    actor_id = parse_attribution_uuid(raw_actor)
    if actor_id is None:
        logger.debug("attribution: ignoring non-UUID %s", ACTOR_ID_HEADER)
        return
    _actor_id_var.set(actor_id)


def _attribution_forwarding_enabled() -> bool:
    """AUTH_ATTRIBUTION_FORWARD_HEADERS, read at call time (default OFF).

    Gates ``X-Project-Id`` ONLY. ``X-Actor-Id`` is no longer behind this flag:
    owner ruling (2026-10-07) is that every cost event carries a real actor, and
    a downstream hop can only attribute the human who acted if the caller tells
    it. Forwarding the actor is safe without an opt-in because the RECEIVER
    decides whether to believe it — ``bind_attribution_from_headers`` honours an
    inbound ``X-Actor-Id`` only on a trusted lane.
    """
    from w2v_common.config import AuthSettings

    return bool(AuthSettings().attribution_forward_headers)


def _attribution_headers() -> dict[str, str]:
    """``X-Project-Id`` / ``X-Actor-Id`` for the bound values; absent keys omitted.

    * ``X-Actor-Id`` — ALWAYS sent when resolvable (see
      :func:`resolve_actor_id`): the bound UUID actor (the human wins); for
      SERVICE-ORIGINATED work only, this service's OWN ``SERVICE_ACTOR_ID``
      (0.88.0 — 0.87.0 also used it while serving an actor-less inbound
      request, which hid the caller's gap). Otherwise absent, never "", and
      counted.
    * ``X-Project-Id`` — still opt-in, see :func:`_attribution_forwarding_enabled`.
    """
    out: dict[str, str] = {}
    if _attribution_forwarding_enabled():
        project_id = _project_id_var.get()
        if project_id:
            out[PROJECT_ID_HEADER] = project_id
    # The actor var can hold a lane-bound value that is not a UUID (the mesh
    # lane stores the gate's header verbatim). The wire contract is UUID-only,
    # so anything else stays on this hop's cost events and is NOT forwarded.
    # ⚠️ A mesh-lane actor is only as trustworthy as its vouch: x-actor-id is in
    # MESH_VOUCHED_HEADERS (HMAC-covered), but arming that lane to forward it
    # REQUIRES AUTH_MESH_VOUCH_SECRET — unset, any in-mesh peer can forge it.
    actor_id, missing = resolve_actor_id()
    if actor_id:
        out[ACTOR_ID_HEADER] = actor_id
    elif missing:
        record_actor_missing("outbound", missing)
    return out


def get_caller_license_token() -> str:
    return _caller_license_token_var.get()


def get_correlation_id_from_context() -> str:
    return _correlation_id_var.get()


def get_stage_from_context() -> str:
    return _stage_var.get()


def get_identity_headers() -> dict[str, str]:
    """Return the canonical X-* identity header dict for outbound calls.

    Raises :class:`w2v_common.tenant.MissingTenantError` when no tenant is
    bound.

    ⭐ CORRECTED. This used to return ``{"X-Tenant-Id": ""}`` when unbound and
    the docstring called that "harmless … on requests that already carry an
    Authorization header". It was not harmless, on two counts. (a) The premise
    was wrong about this code path: these headers are merged onto S2S calls
    whose auth rides in ``X-Tenant-Id`` itself, not in ``Authorization``, so
    `license-authz` was never the thing validating them. (b) Even where a
    request did authenticate, a blank tenant is exactly the tenant-less traffic
    the product-owner ruling forbids: the receiver either 401s with a confusing
    "no valid tenant" or — on a lane that tolerated blanks — attributed the work
    to nobody. Failing here names the real defect at the CALLER, which is the
    only place it can be fixed.

    The other four values keep their empty-string defaults: they are opaque
    audit strings whose absence is legitimate (a CronJob has no license, a
    service-account token has no subject). Only the tenant is mandatory.
    """
    tenant = _tenant_id_var.get()
    if tenant is None:
        raise MissingTenantError(
            "get_identity_headers(): no tenant bound to this execution context; "
            "an outbound call may not carry a blank X-Tenant-Id"
        )
    return {
        "X-Tenant-Id": str(tenant),
        "X-License-Id": _license_id_var.get(),
        "X-Subject-Id": _subject_id_var.get() or _unsigned_subject_var.get(),
        "X-Correlation-Id": _correlation_id_var.get(),
        "X-Stage": _stage_var.get(),
        # Attribution (X-Project-Id / X-Actor-Id): OMITTED when unbound — the
        # contract forbids an empty or placeholder value on the wire.
        **_attribution_headers(),
    }


def _apply_outbound_tenant(request: httpx.Request) -> None:
    """Stamp the tenant identity for ONE outbound hop. Fails closed.

    Split out of ``identity_event_hook`` so each branch can ``return`` without
    skipping the cost-correlation headers the hook stamps afterwards — an
    earlier inline version early-returned on the plain-backend branch and
    silently dropped X-Correlation-Id / X-Stage on every non-metered hop, which
    is the whole per-run cost attribution chain.
    """
    tenant = _tenant_id_var.get()
    existing = request.headers.get("X-Tenant-Id", "")
    service_credentialed = existing.lower().startswith("bearer ")

    if tenant is None:
        # ⭐ FAIL CLOSED. This used to be ``if tenant is not None:`` with no
        # else, so a tenant-less S2S call went out silently and unattributed —
        # the outbound twin of the nil-UUID substitution removed on the inbound
        # side.
        #
        # The ONE tolerated case is a client whose constructor already baked a
        # ``X-Tenant-Id: Bearer <service-account token>``: that request IS
        # authenticated and IS attributable (to the SA tenant), it simply acts
        # for nobody else. A CronJob or startup task calling a peer on its own
        # credentials is legitimate and must keep working.
        #
        # ⛔ EXCEPT inside an inbound request. ``AuthMiddleware`` marks every
        # non-public request inbound, and every inbound lane binds a tenant or
        # 401s — save the tenant-less JWT lane (``AUTH_JWT_TENANT_REQUIRED=
        # false``), whose principal's tenant is not yet resolved. Letting that
        # request's outbound calls ride the SA Bearer would act — and bill — as
        # the SERVICE-ACCOUNT tenant on a user's behalf. Fail closed until the
        # app's membership middleware binds the real tenant. Service-originated
        # work (CronJob, startup task, no inbound request) is unaffected: the
        # inbound flag is False there.
        if service_credentialed and not _inbound_request_var.get():
            return
        if service_credentialed:
            raise MissingTenantError(
                f"outbound {request.method} {request.url}: this inbound request is "
                "authenticated but has no tenant bound; refusing to act as the "
                "service-account tenant on its behalf — resolve the caller's tenant first"
            )
        raise MissingTenantError(
            f"outbound {request.method} {request.url}: no tenant bound to this execution "
            "context and no service-account license token on the request; every message "
            "must carry the requestor's tenant_id and there is no default"
        )

    tenant_id = str(tenant)
    if not service_credentialed:
        # Plain hop (or a client with no SA token): the caller's tenant
        # verbatim. Nothing is delegated, so provenance does not gate it.
        request.headers["X-Tenant-Id"] = tenant_id
        return

    if not is_metered_destination(str(request.url)):
        # The client baked in a service-account Bearer, but the callee is a
        # plain backend that does not read X-Effective-Tenant-Id. Preserve the
        # token (it is how that peer authenticates us) and send NO delegation
        # header — there is nothing to delegate, so nothing to gate. Keying
        # this on the DESTINATION rather than the call site is kag's
        # ``_is_license_introspecting_gateway`` rule: its predecessor chose by
        # call site and stamped an SA token onto every customer's request to a
        # plain backend for months.
        return

    # DELEGATED HOP: the request authenticates as this SERVICE and
    # X-Effective-Tenant-Id names a DIFFERENT tenant to act — and bill — as.
    # Receivers honour it for subjects in their AUTH_DELEGATING_SUBJECTS, and
    # the shared w2v SA subject IS on those lists, so this is live, not inert.
    # It may therefore only be sent for a tenant this service can vouch for.
    source = _tenant_source_var.get()
    if source is None or source not in VOUCHED_TENANT_SOURCES:
        raise UnvouchedTenantError(
            f"outbound {request.method} {request.url}: refusing to delegate tenant "
            f"{tenant_id} — provenance is {source.value if source else 'unrecorded'!r}, "
            "which is not vouched. A bare X-Tenant-Id is settable by any peer that can "
            "reach this service, so delegating it would let an unauthenticated caller "
            "spend an arbitrary tenant's credentials."
        )
    request.headers[EFFECTIVE_TENANT_HEADER] = tenant_id


class NoTenantForLicenceError(MissingTenantError):
    """A tenant-x-app licence was needed but no tenant is bound to this context.

    Raised by :func:`resolve_licence_tenant` (``WM_AUTH_MODE=tenant_licence``).
    There is deliberately NO fallback to the service-account licence: under the
    owner ruling of 2026-10-15 there are no app-to-app licences, only tenant-x-app
    licences, so a call with no tenant has nobody to bill and must not be made.
    Subclasses :class:`MissingTenantError` so existing handlers still catch it.
    """


def resolve_licence_tenant() -> str:
    """The END tenant a tenant-x-app licence must be minted for, as a canonical UUID.

    Same tenant + provenance rule ``identity_event_hook`` applies before it
    DELEGATES a tenant: the bound tenant, and only when its provenance is
    vouched (``VOUCHED_TENANT_SOURCES``). Minting a licence for a tenant spends
    that tenant's licence exactly as delegation did, so the bar is identical.

    It is the END tenant by construction: in a chain such as merqi -> w2v -> WM
    the inbound lane binds the tenant the caller acts for (licence token of that
    tenant, or a delegating subject's vouched ``X-Effective-Tenant-Id``), never
    the calling app's own service tenant.

    Raises :class:`NoTenantForLicenceError` (no tenant) or
    :class:`UnvouchedTenantError` (tenant bound but not vouched).
    """
    tenant = _tenant_id_var.get()
    if tenant is None:
        raise NoTenantForLicenceError(
            "tenant_licence: no tenant bound to this execution context; a WM call needs "
            "the end tenant's workers licence and there is no service-account fallback"
        )
    source = _tenant_source_var.get()
    if source is None or source not in VOUCHED_TENANT_SOURCES:
        raise UnvouchedTenantError(
            f"tenant_licence: refusing to mint a licence for tenant {tenant} — provenance "
            f"is {source.value if source else 'unrecorded'!r}, which is not vouched"
        )
    return str(tenant)


def _apply_tenant_licence(request: httpx.Request) -> None:
    """Tenant step for a hop that carries the END tenant's own licence.

    The client put ``X-Tenant-Id: Bearer <tenant-x-app licence>`` on the request.
    That licence already names the tenant, so nothing is delegated:
    ``X-Effective-Tenant-Id`` is removed if anything set it. A request without
    the Bearer is a programming error and fails loud rather than going out on
    whatever credential the pooled client happens to hold.
    """
    existing = request.headers.get("X-Tenant-Id", "")
    if not existing.lower().startswith("bearer ") or not existing[7:].strip():
        raise NoTenantForLicenceError(
            f"outbound {request.method} {request.url}: tenant_licence hop without a "
            "minted tenant licence on X-Tenant-Id"
        )
    if EFFECTIVE_TENANT_HEADER in request.headers:
        del request.headers[EFFECTIVE_TENANT_HEADER]


async def identity_event_hook(request: httpx.Request) -> None:
    """httpx ``event_hooks={'request': [...]}`` handler.

    Stamps X-License-Id / X-Subject-Id from ContextVars onto outbound
    requests for audit/per-user logging downstream. Does NOT touch
    Authorization.

    For X-Tenant-Id: if the caller's constructor already set a Bearer
    form (the canonical w2v S2S license-token path), preserve it so
    downstream services can introspect to the SERVICE-account tenant.
    Only fill in the bare-UUID form from ContextVar when the header
    is absent or non-Bearer (legacy callers without a service-account
    license token).

    Use ``caller_token_event_hook`` (in addition to this one) on clients
    that must forward the inbound caller's license token verbatim
    (artifact_service, where artifacts are stored under the introspected
    ``subject_id`` and we need symmetric upload/download).
    """
    _stamp_identity(request, _apply_outbound_tenant, stamp_license_id=True)


async def tenant_licence_event_hook(request: httpx.Request) -> None:
    """``identity_event_hook`` for hops authenticated by the END tenant's licence.

    Identical stamping (licence id, subject, correlation, stage, attribution)
    except the tenant step: the request's ``X-Tenant-Id: Bearer`` is the
    tenant-x-app licence minted for the bound tenant, so ``X-Effective-Tenant-Id``
    is never sent. Used by ``WMClient``'s inherited WM client when
    ``WM_AUTH_MODE=tenant_licence``.

    ``X-License-Id`` is NOT stamped from the inbound context: the client sets it
    to the MINTED licence's id, so attribution names the credential that
    authenticated the call.
    """
    _stamp_identity(request, _apply_tenant_licence, stamp_license_id=False)


def _stamp_identity(
    request: httpx.Request,
    tenant_step: Callable[[httpx.Request], None],
    *,
    stamp_license_id: bool,
) -> None:
    if stamp_license_id:
        license_id = _license_id_var.get()
        if license_id:
            request.headers["X-License-Id"] = license_id

    # The attribution subject (authenticated, else the transitional unsigned
    # cell-header one) keeps the legacy header chain intact; a 0.93.0+ receiver
    # ignores the header whenever the cell token signs a subject claim.
    subject_id = _subject_id_var.get() or _unsigned_subject_var.get()
    if subject_id:
        request.headers["X-Subject-Id"] = subject_id

    tenant_step(request)

    # Forward cost-correlation so worker_management (and, downstream of it,
    # inference-router) can stamp it onto the LLM cost event. Only set when
    # present — never clobber with empty values.
    correlation_id = _correlation_id_var.get()
    if correlation_id:
        request.headers["X-Correlation-Id"] = correlation_id

    stage = _stage_var.get()
    if stage:
        request.headers["X-Stage"] = stage

    # Attribution: forward the project/actor this context carries. Only set
    # when bound — an absent value is an absent header, never "". A value the
    # CALLER already put on the request wins (WMClient sets X-Project-Id to the
    # W2V project explicitly; the fr project must not overwrite it).
    for name, value in _attribution_headers().items():
        if name not in request.headers:
            request.headers[name] = value

    # The W2V project propagates to every downstream w2v hop, flag-independent:
    # it is w2v-internal routing of w2v's own data, not an fr attribution claim.
    w2v_project = _w2v_project_id_var.get()
    if w2v_project and W2V_PROJECT_ID_HEADER not in request.headers:
        request.headers[W2V_PROJECT_ID_HEADER] = w2v_project
    w2v_job = _w2v_job_id_var.get()
    if w2v_job and W2V_JOB_ID_HEADER not in request.headers:
        request.headers[W2V_JOB_ID_HEADER] = w2v_job

    # P7: an in-cell destination also gets a fresh cell token (dual-send with
    # the legacy headers above until the receivers flip to enforce).
    _apply_cell_token(request)


def _apply_cell_token(request: httpx.Request) -> None:
    """Stamp a FRESH ``X-W2V-Cell-Token`` on an in-cell request; strip it elsewhere.

    Minted per send from this execution context — tenant, w2v job (run), human
    actor, w2v project — so nothing holds a token: a retry, a poll loop or a 4 h
    run re-mints on every request. A no-op when this service has no signing key
    (every out-of-cell consumer of w2v_common). Any pre-existing cell token is
    removed first, so a header copied from an inbound request can never be
    replayed onward or leak out of the cell. The actor is the HUMAN only
    (``SERVICE_ACTOR_ID`` never enters a cell token; ``iss`` names the service).
    """
    from .cell_jwt import (
        CELL_TOKEN_HEADER,
        cell_audience_for_url,
        cell_signer_configured,
        mint_cell_token,
        record_cell_unminted,
    )

    if CELL_TOKEN_HEADER in request.headers:
        del request.headers[CELL_TOKEN_HEADER]
    tenant = _tenant_id_var.get()
    if tenant is None:
        return
    aud = cell_audience_for_url(str(request.url))
    # A half-configured signer raises here — only on IN-CELL calls, so a
    # broken key never takes down this service's out-of-cell traffic.
    if aud is None or not cell_signer_configured():
        return
    if not is_tenant_vouched():
        # The receiver records a cell-token tenant as CELL_JWT, which is
        # VOUCHED: signing a tenant this service could not itself delegate (a
        # reaper that dropped provenance, a bare-header lane) would launder it
        # into one the receiver may spend at a metered peer. No token; the
        # legacy headers above still go out (dual: unchanged; enforce: 401).
        record_cell_unminted("unvouched_tenant")
        logger.warning(
            "cell token not minted for %s %s: tenant provenance is not vouched",
            request.method,
            aud,
        )
        return
    request.headers[CELL_TOKEN_HEADER] = mint_cell_token(
        aud,
        tenant=tenant,
        run=_w2v_job_id_var.get() or None,
        actor=parse_attribution_uuid(_actor_id_var.get()),
        project=_w2v_project_id_var.get() or None,
        # Signed attribution (0.93.0). Only the AUTHENTICATED subject: an
        # unsigned header subject (``_unsigned_subject_var``) is never
        # laundered into a signature the next hop would trust for authz.
        subject_id=_signable_attribution("subject", _subject_id_var.get()),
        license_id=_signable_attribution("license", _license_id_var.get()),
    )


def _signable_attribution(name: str, value: str) -> str | None:
    """``value`` if it may ride a cell token, else ``None`` (+ a warning).

    An unexpected shape must not fail the outbound request: the claim goes out
    as ``null`` (no attribution) rather than the request going out not at all.
    """
    from .cell_jwt import is_valid_attribution_value

    if not value:
        return None
    if is_valid_attribution_value(value):
        return value
    logger.warning("cell token %s claim omitted: bound value is not a valid opaque id", name)
    return None


async def cell_token_event_hook(request: httpx.Request) -> None:
    """Stand-alone cell-token hook for clients that do not use ``identity_event_hook``."""
    _apply_cell_token(request)


def _apply_service_cell_token(request: httpx.Request, audience: str | None) -> None:
    """Shared body of the async and sync service hooks (identical host rules)."""
    from .cell_jwt import (
        CELL_TOKEN_HEADER,
        cell_audience_for_url,
        cell_signer_configured,
        mint_service_cell_token,
        record_cell_unminted,
    )

    if _tenant_id_var.get() is not None:
        record_cell_unminted("service_hook_tenant_bound")
        logger.warning(
            "service-scope cell token not minted for %s %s: a tenant is bound",
            request.method,
            request.url.host,
        )
        return
    if CELL_TOKEN_HEADER in request.headers:
        del request.headers[CELL_TOKEN_HEADER]
    url_aud = cell_audience_for_url(str(request.url))
    if url_aud is None or (audience is not None and url_aud != audience):
        record_cell_unminted("service_hook_host_mismatch")
        logger.warning(
            "service-scope cell token not minted for %s %s: host is not the in-cell endpoint of %s",
            request.method,
            request.url.host,
            audience or "any in-cell peer",
        )
        return
    if not cell_signer_configured():
        return
    request.headers[CELL_TOKEN_HEADER] = mint_service_cell_token(url_aud)


def service_cell_token_hook_sync(
    audience: str | None = None,
) -> Callable[[httpx.Request], None]:
    """Synchronous twin of :func:`service_cell_token_hook` for ``httpx.Client``.

    ``httpx.Client`` calls its request hooks without awaiting them — an async
    hook there would return an un-awaited coroutine and sign nothing. Same
    rules, same metrics: tenant bound → untouched; host not the (pinned)
    in-cell endpoint → stripped, no mint; no signing key → no-op.
    """

    def _hook(request: httpx.Request) -> None:
        _apply_service_cell_token(request, audience)

    return _hook


def service_cell_token_hook(
    audience: str | None = None,
) -> Callable[[httpx.Request], Awaitable[None]]:
    """httpx request hook signing EVERY send with a ``scope=service`` cell token.

    For non-web workers that act for NO tenant (a tts-gateway worker polling
    ``POST /api/tts/claim``): no request ContextVars are needed. Install it
    explicitly on that client — service scope is never implicit; the tenant
    hook (``identity_event_hook``) never downgrades a call that lost its
    tenant into one of these.

    ``audience``: the callee's cell identity. ``None`` resolves it from the
    URL like the tenant hook (``cell_audience_for_url`` — in-cell peers only).
    An explicit ``audience`` PINS the client to that callee: a token is minted
    only when the request's host:port IS that audience's in-cell endpoint
    (the same discovery match the tenant hook uses). Every other host — the
    open internet, a non-discovery alias, another in-cell peer, the second hop
    of a ``follow_redirects`` chain — gets its ``X-W2V-Cell-Token`` stripped and
    no mint, counted as
    ``w2v_cell_token_unminted_total{reason="service_hook_host_mismatch"}``:
    a service token that leaves its callee is replayable against the claim
    route for its whole TTL.

    No signing key → no-op (a half-configured signer raises, as on the tenant
    path). If a TENANT is bound in this context the request is left untouched
    and ``w2v_cell_token_unminted_total{reason="service_hook_tenant_bound"}``
    counts it: a tenant's work must never travel as the service.
    """

    async def _hook(request: httpx.Request) -> None:
        _apply_service_cell_token(request, audience)

    return _hook


async def caller_token_event_hook(request: httpx.Request) -> None:
    """Override Authorization + X-Tenant-Id with the caller's license token.

    Use ONLY on httpx clients that talk to services authorizing per
    *customer* tenant rather than per service-account — currently
    artifact_service, where artifacts are stored under the introspected
    ``subject_id`` and we need symmetric upload/download for the same
    caller.

    No-op when ``_caller_license_token_var`` is empty (CronJob, startup
    task, Keycloak-JWT request, trusted-network path) — the constructor's
    service-account ``Authorization`` survives in that case so background
    tasks keep working on service-account credentials.
    """
    caller = _caller_license_token_var.get()
    if caller:
        bearer = f"Bearer {caller}"
        request.headers["Authorization"] = bearer
        request.headers["X-Tenant-Id"] = bearer
