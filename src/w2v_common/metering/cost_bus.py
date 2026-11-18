"""Shared cost-bus emitter for w2v services.

A single source of truth for emitting cost events to the cluster cost-bus
Redis stream, following the minimum-viable emitter shape in
``metering_service/docs/EMITTER_GUIDE.md``.

Historically every w2v service rolled its own ``metering.py`` with its own
``DEFAULT_STREAM`` constant. When the cost-bus stream was renamed
``metering:costs`` -> ``improvement-loop:costs`` on 2026-06-24, services that
hardcoded the old name silently routed every cost event to a now-dead stream,
so their events stopped reaching billing. Centralising the canonical stream
name (and the emit logic) here means the next rename is a one-line change.

Two public emit entry points:

* :func:`emit_cost_event` — an awaitable coroutine; never raises. Use it when
  you manage the task yourself or want to await the XADD (e.g. in tests).
* :func:`schedule_cost_event` — fire-and-forget that **retains a strong
  reference** to the spawned task until it completes. ``asyncio`` only holds a
  weak reference to tasks, so the bare ``asyncio.create_task(emit(...))``
  pattern several services used could be garbage-collected mid-flight and
  silently drop the cost event. Prefer this from request handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import traceback
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from prometheus_client import Counter, Gauge

from w2v_common.auth.identity_context import (
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
    parse_attribution_uuid,
    record_actor_missing,
    resolve_actor_id,
    tenant_level_project_id,
)
from w2v_common.json_safe import finite_json_safe
from w2v_common.observability.logging import redis_endpoint
from w2v_common.tenant import parse_tenant_id

if TYPE_CHECKING:
    from collections.abc import Mapping

    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Canonical cost-bus stream. History: metering:costs -> improvement-loop:costs
# (2026-06-24) -> billing:costs (billing-bus rename). This is the single place
# that name lives. Override per-deployment via the ``METERING_STREAM`` env var.
CANONICAL_COST_STREAM = "billing:costs"

#: WHO FUNDS a metered call — owner-decided taxonomy (2026-10-07), mirroring
#: billing-service ``services/funding.py`` ``FUNDING_VALUES`` verbatim:
#:
#: * ``local_gpu``         — self-hosted platform compute (pgx, mlx, kokoro, ...).
#: * ``platform_external`` — an external vendor billed the PLATFORM's key.
#: * ``tenant_byo``        — the tenant's own provider key; the vendor bills them.
#:
#: Emitted as a TOP-LEVEL wire field (billing's ``cost_flusher`` reads
#: ``c.get("funding")``). Absent -> the key is OMITTED and billing derives it
#: from provider / pricing basis / ``usage_extra.byo``. There is deliberately no
#: default: a guessed funding is a lie billing would store verbatim, because an
#: explicit value is honoured and never re-derived.
Funding = Literal["local_gpu", "platform_external", "tenant_byo"]
FUNDING_VALUES: frozenset[str] = frozenset(get_args(Funding))


class InvalidFundingError(ValueError):
    """``funding`` was given but is not one of :data:`FUNDING_VALUES`.

    Raised out of the emit call (before the never-raises swallow), like a bad
    tenant: billing would store an out-of-set value as NULL and count it, so a
    typo here would silently turn an asserted funding into "unknown".
    """


def _validate_funding(funding: str | None) -> Funding | None:
    if funding is None:
        return None
    if funding not in FUNDING_VALUES:
        msg = f"funding={funding!r} is not one of {sorted(FUNDING_VALUES)}"
        raise InvalidFundingError(msg)
    return cast("Funding", funding)


class InvalidActorIdError(ValueError):
    """An explicit ``actor_id`` is not a canonical, non-nil UUID (or was given twice).

    ``actor_id`` is WHO ACTED: the verified Keycloak ``sub`` of the human who
    initiated the work. billing's ``cost_flusher`` parses it with ``_parse_uuid``
    and stores a malformed value as NULL (counted on
    ``billing_actor_malformed_total``), so a non-UUID actor is attribution that
    silently evaporates three hops away. Refused here, before the swallow.
    """


class InvalidEventKeyError(ValueError):
    """A caller-supplied ``event_key`` cannot serve as a per-call identity.

    billing's ``derive_event_key`` honours a producer key verbatim and dedupes on
    it GLOBALLY (``processed_cost_event_keys`` primary key, ``VARCHAR(64)``), so a
    placeholder, blank or over-long key either merges distinct billable calls or
    is condensed out of recognition. Refused here, before the swallow.
    """


#: billing ``_parsers.PLACEHOLDER_VALUES`` verbatim: a key spelled like one of
#: these is ignored by billing's rung 1 (it falls through to a derived key), so a
#: caller that sends it believes it has idempotency and does not.
_EVENT_KEY_PLACEHOLDERS: frozenset[str] = frozenset(
    {"unknown", "unspecified", "none", "null", "nil", "n/a", "na", "-", "undefined", "default"}
)

#: Width of billing's ``event_key`` column and inbox primary key.
_EVENT_KEY_MAX_LEN = 64

#: A key is printable ASCII with no whitespace: it is compared byte-for-byte by
#: the receiver, so an invisible difference would be a different identity.
_EVENT_KEY_RE = re.compile(r"^[\x21-\x7e]+$")

#: uuid5 namespace for :func:`cost_event_key`. Fixed forever: changing it changes
#: every derived key, so a re-emit after the change would bill twice.
_EVENT_KEY_NAMESPACE = uuid.UUID("0199e4a2-6b1c-7d3e-8f40-5a6b7c8d9e01")


def cost_event_key(*parts: object) -> str:
    """Deterministic ``event_key`` for one physical metered call.

    Use this when the call can be RE-EMITTED after a process restart (a job
    step retried by a queue, an outbox replay): derive the key from what
    identifies the physical call, e.g.
    ``cost_event_key(service_key, cost_source, job_id, step, attempt)``. The same
    parts always yield the same key, so billing's inbox drops the re-emit; a new
    ``attempt`` (a genuinely new provider call) yields a new key and bills.

    ⛔ billing dedupes on the key GLOBALLY, across every service and tenant.
    Include ``service_key`` (and the step) in the parts, or two services keying
    on the same job id would collapse into ONE charge — silent under-billing.
    Never derive the key from the call's CONTENT (tokens, cost): identical
    payloads are distinct calls (billing ``derive_event_key`` docstring).

    Returns a canonical uuid5 string (36 chars, fits billing's ``VARCHAR(64)``).

    :raises InvalidEventKeyError: no parts, or a part that is ``None``/blank.
    """
    if not parts:
        raise InvalidEventKeyError("cost_event_key needs at least one part")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        text = "" if part is None else str(part).strip()
        if not text:
            raise InvalidEventKeyError(f"cost_event_key part {index} is empty")
        rendered.append(text)
    # "\x1f" (unit separator) cannot occur in a sane part, so the join is
    # unambiguous: ("a|b", "c") and ("a", "b|c") never collide.
    return str(uuid.uuid5(_EVENT_KEY_NAMESPACE, "\x1f".join(rendered)))


def _validate_event_key(event_key: str | None) -> str | None:
    if event_key is None:
        return None
    if not isinstance(event_key, str):
        msg = f"event_key has type {type(event_key).__name__}, not str"
        raise InvalidEventKeyError(msg)
    if (
        not _EVENT_KEY_RE.match(event_key)
        or event_key.lower() in _EVENT_KEY_PLACEHOLDERS
        or len(event_key) > _EVENT_KEY_MAX_LEN
    ):
        msg = (
            f"event_key (len={len(event_key)}) must be 1-{_EVENT_KEY_MAX_LEN} printable "
            "non-whitespace ASCII characters and not a placeholder word"
        )
        raise InvalidEventKeyError(msg)
    return event_key


def _validate_actor_id(
    actor_id: str | None, metadata_extra: Mapping[str, Any] | None
) -> str | None:
    """Canonicalise an explicit ``actor_id`` kwarg.

    ``None`` -> not stated (ambient resolution applies). ``""`` -> stated as "no
    person acted" (platform/agent work): the key is omitted AND the ambient actor
    is suppressed. Anything else must be a canonical non-nil UUID.
    """
    if actor_id is None:
        return None
    if metadata_extra is not None and "actor_id" in metadata_extra:
        raise InvalidActorIdError("actor_id given both as a kwarg and in metadata_extra")
    if not isinstance(actor_id, str):
        msg = f"actor_id has type {type(actor_id).__name__}, not str"
        raise InvalidActorIdError(msg)
    if actor_id == "":
        return ""
    parsed = parse_attribution_uuid(actor_id)
    if parsed is None:
        msg = f"actor_id (len={len(actor_id)}) is not a canonical non-nil UUID"
        raise InvalidActorIdError(msg)
    return parsed


# REMOVED 2026-08-26 — ``UNATTRIBUTED_TENANT_ID``, the nil-UUID sentinel this
# module SUBSTITUTED whenever the requestor's tenant was missing. It was the
# worst possible shape: a valid UUID (so it passed every type check and every
# dirty-row audit) that is registered in ``licenses.tenants`` as
# ``default-anon`` (so it looked legitimate) while silently mis-attributing
# cost. Product-owner ruling: a tenant_id may NEVER hold a sentinel. An emit
# with no real tenant is now a ``MissingTenantError``, not a nil-UUID row.

# ⛔ This set MIRRORS the receiver's whitelist verbatim — billing_service
# ``src/services/_parsers.py``, the metadata allowlist loop that builds the
# ``metadata`` JSON column. The receiver landed FIRST on purpose: a producer
# that ships a key before the receiver knows it has that key dropped SILENTLY,
# three hops away, in a service we cannot see from here — and the row still
# looks correct. So this set must never grow ahead of that loop; widen the
# receiver first, then mirror it here.
#
# Byte-for-byte the same frozenset as ``kag_common.metering``'s
# ``_METADATA_PASSTHROUGH_KEYS``. The two libraries emit onto the SAME canonical
# cost bus, so a divergence between them is a divergence in what billing can
# attribute. Change both or neither.
#
# Everything outside this set is dropped locally (with a DEBUG line naming it)
# rather than shipped, so a typo'd key fails at the emitter instead of
# evaporating in a consumer we cannot see from here.
_METADATA_PASSTHROUGH_KEYS: frozenset[str] = frozenset(
    {
        "schema_id",
        "job_id",
        "fallback_used",
        "stage",
        "worker_name",
        "caller_client_id",
        # WHO ACTED (the Keycloak JWT ``sub``), carried ALONGSIDE the top-level
        # ``subject_id``, which stays WHO PAYS. Receiver landed first, as the
        # rule above requires: billing alembic 0021 adds the column and
        # ``_parse_uuid`` reads it (malformed/empty -> NULL). It is nullable and
        # never defaulted there, so an absent actor is a first-class state.
        #
        # ⚠️ ``kag_common.metering`` has NOT been widened yet, so the two sets
        # are no longer byte-identical. That is a DIVERGENCE, not a break:
        # billing accepts events with or without the key, so kag's events simply
        # carry no actor. Mirror this line into ``kag_common`` to close it.
        "actor_id",
    }
)

# Default stream trim bound (approximate) — matches the prior per-service value.
_DEFAULT_MAXLEN = 100_000

# Strong references to in-flight fire-and-forget emit tasks. Without this set
# the event loop's weak reference is the only one, so the task can be GC'd
# before the XADD completes. ``add_done_callback`` discards on completion.
_background_tasks: set[asyncio.Task[None]] = set()

# Bounded producer-side retry (FIX B1). billing-redis is a single-replica /
# Recreate / worker1-pinned store; during a pod roll, AOF cold-replay, or a
# worker1 blip every XADD fails ONCE and the event is dropped (warning only). A
# short bounded retry on connection/timeout errors rides out that brief window.
# Deliberately NOT a queue or persistence layer — that is over-engineering for a
# zero-tenant dev bus (see the no-queue note in the module/release docs). The
# backoffs (one entry per retry) are env-tunable with a safe small default;
# ``METERING_EMIT_RETRY_BACKOFFS`` is a comma-separated seconds list.
_DEFAULT_EMIT_RETRY_BACKOFFS = (0.1, 0.3)

#: ``health_check_interval`` for the metering client, in seconds. redis-py pings
#: a pooled connection that has been idle longer than this BEFORE handing it to
#: a command, so a transport the peer closed during an idle window is discovered
#: and replaced instead of surfacing as the bare asyncio ``RuntimeError`` below.
_HEALTH_CHECK_INTERVAL_SECONDS = 30

#: Emit outcome counter — the ONLY success signal this module has ever had.
#: Before it, ``cost_event*`` appeared in the live logs exclusively on failure,
#: so a success:failure ratio was not computable from any telemetry the fleet
#: collected. Deliberately a metric and NOT a log line: this fires once per
#: billable request across ~19 services, so an INFO line per emit would be a
#: hot-path log flood. ``service_key`` is the emitting service's own routing
#: slug (1-3 distinct values per process). ``outcome`` has FOUR values, not
#: two: ``ok`` when the XADD landed; ``failed`` when every attempt (including
#: the bounded retry) raised and the event was dropped with a warning;
#: ``dropped_disabled`` when metering is DISARMED in this process — a ``None``
#: ``redis_conn`` reached :func:`emit_cost_event`, so nothing was ever sent;
#: and :data:`DROPPED_NO_LOOP` when :func:`schedule_cost_event` found no
#: running event loop, so the emit was never even entered.
#: The last two are the ones that are NOT transport outcomes.
#: ``dropped_disabled`` means the process is billing nothing at all and pairs
#: with ``w2v_metering_enabled == 0`` below; ``dropped_no_loop`` means ONE
#: call site is in a sync (loop-free) frame while the rest of the process is
#: fine. ⚠️ They are mutually exclusive by construction: with no loop the
#: coroutine is never created, so a disarmed AND loop-free call is counted
#: once, as ``dropped_no_loop``.
cost_events_total = Counter(
    "w2v_cost_events_total",
    "Cost-bus emit attempts by emitting service and terminal outcome.",
    ["service_key", "outcome"],
)


#: Is metering ARMED in this process? ``init_metering_redis`` returns ``None``
#: when ``METERING_REDIS_URL`` is empty (or the connect fails), and every
#: consumer then guards its emits with ``if app.state.metering_redis:`` — code
#: this module cannot see. The result is a service that bills NOTHING and looks
#: perfectly healthy: one startup warning, then silence forever.
#:
#: Deliberately a **Gauge**, not a dropped-events Counter. A counter reads 0 on
#: an idle service, which is indistinguishable from a healthy one — exactly the
#: failure signature here. ``w2v_metering_enabled == 0`` alerts with no traffic
#: at all. Set to 0 at import time so the series exists from the FIRST scrape:
#: an absent series is indistinguishable from "the library was never loaded"
#: (the same trap documented on ``w2v_cost_event_unattributed_total`` below).
metering_enabled = Gauge(
    "w2v_metering_enabled",
    "1 when this process holds a metering Redis client, 0 when cost events are dropped.",
)
metering_enabled.set(0)


def _set_metering_enabled(value: int) -> None:
    """Best-effort arm/disarm observation — must never break boot."""
    try:
        metering_enabled.set(value)
    except Exception:  # noqa: BLE001
        logger.debug("metering_enabled_metric_failed value=%s", value, exc_info=True)


def _record_key_source(service_key: str, source: str) -> None:
    """Best-effort event_key provenance observation — must never break an emit."""
    try:
        cost_event_key_source_total.labels(service_key=service_key, source=source).inc()
    except Exception:  # noqa: BLE001 — telemetry must never fail a billable request
        logger.debug("cost_event_key_source_metric_failed source=%s", source)


def _record_outcome(service_key: str, outcome: str) -> None:
    """Best-effort outcome observation — must never break an emit."""
    try:
        cost_events_total.labels(service_key=service_key, outcome=outcome).inc()
    except Exception:  # noqa: BLE001
        logger.debug("cost_event_metric_failed outcome=%s", outcome, exc_info=True)


#: The attribution fields billing's contract (``EMITTER_GUIDE.md``, "The event
#: shape (canonical)") marks REQUIRED alongside ``tenant_id``. Each is a UUID at
#: the receiver: ``_parse_uuid`` maps an empty or malformed value to **NULL**
#: without an exception, a metric or a log line, so an emit that omits one
#: produces a row that is money nobody can attribute to a licence.
#:
#: Measured 2026-09-16 on the live billing database: 94,615
#: ``billing_metering_events`` rows in 14 days across 13 tenants with NULL
#: ``app_id``/``subject_id``, plus 1,819 ``billing_cost_events``. The cause on
#: this side was that these three were declared ``= ""`` here — a default that
#: says "attribution is optional" to every one of ~19 consumers.
ATTRIBUTION_FIELDS: tuple[str, ...] = ("license_id", "subject_id", "app_id")

#: Env flag that ESCALATES a missing attribution field from a warning + counter
#: to a raise. ``METERING_*`` + ``os.getenv`` is this module's settings idiom
#: (``METERING_STREAM``, ``METERING_EMIT_RETRY_BACKOFFS``); there is no
#: ``MeteringSettings`` Pydantic class to hang it off.
#:
#: ⛔ Defaults OFF, and that default is deliberate and load-bearing: a cost
#: event is revenue, so recording it unattributed is strictly better than
#: dropping it. This is NOT the ``tenant_id`` treatment — see
#: :func:`_check_attribution`.
_REQUIRE_ATTRIBUTION_ENV = "METERING_REQUIRE_ATTRIBUTION"

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

#: Unattributed-emit counter, labelled by the field that was missing.
#:
#: ⚠️ A labelled Counter exports NO series until a child is created, which on a
#: ``/metrics`` scrape is indistinguishable from "the library was never
#: imported". So every known field is instantiated at import time below, giving
#: a zero-valued series from the first scrape — the difference between "nothing
#: is unattributed" and "nothing is instrumented" is exactly the question this
#: metric exists to answer.
cost_event_unattributed_total = Counter(
    "w2v_cost_event_unattributed_total",
    "Cost events emitted with a billing-required attribution field missing.",
    ["field"],
)
for _attribution_field in ATTRIBUTION_FIELDS:
    cost_event_unattributed_total.labels(field=_attribution_field)

#: ``funding`` is required ONLY under ``METERING_REQUIRE_ATTRIBUTION`` (billing
#: derives it when absent). With the flag off a miss is counted on the same
#: series under ``field="funding"`` but NOT warned: every pre-0.96 consumer omits
#: it, so a WARNING would be a per-emit log flood. The counter is the per-service
#: to-do list for flipping the flag.
FUNDING_FIELD = "funding"
cost_event_unattributed_total.labels(field=FUNDING_FIELD)

#: Where an emitted ``event_key`` came from: ``caller`` (a deterministic key the
#: caller derived, e.g. :func:`cost_event_key`) or ``minted`` (the uuid7 fallback,
#: which is NOT stable across a process restart, so a re-emit after a crash bills
#: twice). Pre-initialised so "every key is minted" is a visible zero, not an
#: absent series.
EVENT_KEY_SOURCES: tuple[str, ...] = ("caller", "minted")
cost_event_key_source_total = Counter(
    "w2v_cost_event_key_source_total",
    "Cost events emitted, by where their event_key came from.",
    ["service_key", "source"],
)


#: Cost events whose ``project_id`` was NOT the bound W2V project, by why.
#:
#: ``tenant_level``: no W2V project on the context (and no explicit one), so the
#: tenant-level project (:func:`tenant_level_project_id`) was emitted instead.
#: ``no_project``: not even that could be derived, so the key was omitted.
#: A steady ``tenant_level`` rate on a service that runs under the orchestrator
#: means an inbound ``X-W2V-Project-Id`` is not reaching it. Pre-initialised for
#: the same reason as :data:`cost_event_unattributed_total`: an absent series
#: reads like "not instrumented", not "never fell back".
COST_PROJECT_FALLBACK_REASONS: tuple[str, ...] = ("tenant_level", "no_project")
cost_project_fallback_total = Counter(
    "w2v_cost_project_fallback_total",
    "Cost events emitted without the bound W2V project, by fallback reason.",
    ["reason"],
)
for _fallback_reason in COST_PROJECT_FALLBACK_REASONS:
    cost_project_fallback_total.labels(reason=_fallback_reason)


def _record_project_fallback(reason: str) -> None:
    """Best-effort fallback observation — must never break an emit."""
    try:
        cost_project_fallback_total.labels(reason=reason).inc()
    except Exception:  # noqa: BLE001 — telemetry must never fail a billable request
        logger.debug("cost_project_fallback_metric_failed reason=%s", reason)


def _resolve_project_id(project_id: str | None, tenant_id: uuid.UUID) -> str | None:
    """The cost row's ``project_id``: the CALLING APP's own (W2V) project.

    ⛔ Owner ruling 2026-10-07: a cost row's project is the calling app's own
    project. For every w2v emitter that is the W2V project (orchestrator
    ``projects.id``) — NEVER the fr project bound from an inbound
    ``X-Project-Id``. W2V and fr projects are separate concepts; this used to
    default to the fr project, so every direct w2v emit (tts, analyzer,
    compiler, html-render, image-gen) shipped ``project_id`` NULL.

    In order:

    1. An explicit ``project_id`` argument (canonical non-nil UUID). A
       malformed one is dropped (DEBUG) and the chain continues.
    2. The W2V project bound on this context (``set_w2v_project_context`` or the
       inbound ``X-W2V-Project-Id``).
    3. The tenant-level project of the event's tenant, counted
       ``reason=tenant_level``.
    4. Nothing (key omitted), counted ``reason=no_project``.
    """
    if project_id:
        explicit = parse_attribution_uuid(project_id)
        if explicit is not None:
            return explicit
        logger.debug("cost_event_project_id_dropped reason=not_a_uuid")
    bound = get_w2v_project_id_from_context()
    if bound:
        return bound
    fallback = tenant_level_project_id(tenant_id)
    _record_project_fallback("tenant_level" if fallback else "no_project")
    return fallback


class MeteringDisarmedError(RuntimeError):
    """Strict attribution is on but metering cannot emit at all (boot-time)."""


class MissingAttributionError(ValueError):
    """A billing-required attribution field was missing and escalation is ON.

    Only raised when :data:`_REQUIRE_ATTRIBUTION_ENV` is truthy. With the flag
    OFF (the default) the emit proceeds and the miss is recorded on
    :data:`cost_event_unattributed_total` instead.
    """


#: ``outcome`` label value for a cost event that was never handed to a loop.
#:
#: Deliberately a third value on the EXISTING :data:`cost_events_total` rather
#: than a new metric, because the question this counter has to answer is a
#: RATIO, not a count: "is this service dropping cost events?" is only decidable
#: against that same service's ``ok``. A separate metric puts the numerator and
#: the denominator in different series keyed by a label that is only ever
#: populated at runtime, so the two could not be joined for a service whose
#: every emit is dropped — which is exactly the failing service.
DROPPED_NO_LOOP = "dropped_no_loop"

#: Env flag that ESCALATES a no-running-loop drop from counter + WARNING to a
#: raise. Same ``METERING_*`` + ``os.getenv`` idiom as
#: :data:`_REQUIRE_ATTRIBUTION_ENV`, and OFF for the same reason, inverted: with
#: no loop there is nowhere to emit, so raising converts a lost BILLING RECORD
#: into a failed CUSTOMER REQUEST whose provider cost has already been incurred
#: upstream. That trades revenue evidence for revenue. See
#: :func:`schedule_cost_event` for the full direction argument.
#:
#: ⭐ Its real use is a CI/staging tripwire: a service that believes all of its
#: call sites reach a running loop can set this to ``true`` in its test
#: environment and turn this latent trap into a red build instead of a
#: production drop.
_REQUIRE_LOOP_ENV = "METERING_REQUIRE_LOOP"


class NoRunningLoopError(RuntimeError):
    """:func:`schedule_cost_event` was called with no running event loop.

    Only raised when :data:`_REQUIRE_LOOP_ENV` is truthy. With the flag OFF (the
    default) the drop is recorded on
    :data:`cost_events_total` as ``outcome="dropped_no_loop"``, a WARNING
    reproduces the event's identity, and the call returns ``False``.

    Subclasses ``RuntimeError`` because that is what
    ``asyncio.get_running_loop()`` itself raises for this condition, so a caller
    that already guards the sync/async boundary with ``except RuntimeError``
    keeps working.
    """


def require_attribution() -> bool:
    """Is missing-attribution escalation enabled? (``METERING_REQUIRE_ATTRIBUTION``)."""
    return os.getenv(_REQUIRE_ATTRIBUTION_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def require_loop() -> bool:
    """Is no-running-loop escalation enabled? (``METERING_REQUIRE_LOOP``)."""
    return os.getenv(_REQUIRE_LOOP_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def _missing_attribution_fields(
    *, license_id: str | None, subject_id: str | None, app_id: str | None
) -> list[str]:
    """Names of the required attribution fields that carry no value.

    ``None`` (omitted) and ``""`` (supplied empty) are BOTH unattributed: the
    receiver's ``_parse_uuid`` turns either into the same NULL column, so a
    distinction the database cannot represent is not one this check makes.
    Whitespace-only is treated the same — it is not a UUID either.
    """
    values = {"license_id": license_id, "subject_id": subject_id, "app_id": app_id}
    return [name for name in ATTRIBUTION_FIELDS if not (values[name] or "").strip()]


#: A licences registry app KEY (``w2v``, ``workers``, ``kag``): what billing's
#: ``split_asserted_app`` accepts as a non-uuid ``app_id``.
_APP_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _malformed_attribution_fields(
    *,
    license_id: str | None,
    subject_id: str | None,
    app_id: str | None,
    metadata_extra: Mapping[str, Any] | None,
) -> list[str]:
    """Present-but-not-a-UUID attribution fields (billing NULLs each one).

    Only consulted under ``METERING_REQUIRE_ATTRIBUTION``: with the flag off the
    wire is unchanged and billing's own ``_parse_uuid`` / ``actor_malformed``
    counters see it. ``actor_id`` in ``metadata_extra`` is included because it
    bypasses the kwarg's validation (``None`` there means "no actor").
    """
    values: dict[str, Any] = {"license_id": license_id, "subject_id": subject_id, "app_id": app_id}
    if metadata_extra is not None and metadata_extra.get("actor_id") is not None:
        values["actor_id"] = metadata_extra["actor_id"]
    bad = []
    for name, value in values.items():
        text = "" if value is None else str(value).strip()
        if not text or parse_attribution_uuid(text) is not None:
            continue
        # ``app_id`` is EITHER the app's uuid OR its licences registry KEY (the
        # X-App-Id slug, e.g. ``w2v``): billing ``split_asserted_app`` resolves a
        # non-uuid value through its app-key mirror (unknown keys are counted on
        # ``billing_app_key_unknown_total``). Only the key SHAPE is checked here.
        if name == "app_id" and _APP_KEY_RE.match(text):
            continue
        bad.append(name)
    return bad


def _raise_if_unattributed(
    service_key: str, cost_source: str, refused: list[str], malformed: list[str]
) -> None:
    detail = []
    if refused:
        detail.append(f"missing required attribution: {', '.join(refused)}")
    if malformed:
        detail.append(f"not a canonical UUID: {', '.join(malformed)}")
    if detail:
        raise MissingAttributionError(
            f"cost event ({service_key}/{cost_source}) has {'; '.join(detail)}"
        )


def _check_attribution(
    *,
    service_key: str,
    cost_source: str,
    license_id: str | None,
    subject_id: str | None,
    app_id: str | None,
    funding: str | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    record: bool = True,
) -> None:
    """Observe (or, when escalated, refuse) a cost event missing attribution.

    ⛔ NOT the ``tenant_id`` treatment, on purpose. ``tenant_id`` hard-fails
    because a WRONG tenant bills the wrong customer; these three only cost the
    attribution, and a dropped cost event is revenue the estate can never
    recover. So the default is: emit the row, WARN naming the field, count it.
    ``METERING_REQUIRE_ATTRIBUTION=true`` escalates to a raise for the services
    that have finished threading the headers through.

    ``record=False`` suppresses the counter/log so the synchronous pre-check in
    :func:`schedule_cost_event` cannot double-count the emit it then spawns.

    :raises MissingAttributionError: a field is missing AND escalation is on.
    """
    missing = _missing_attribution_fields(
        license_id=license_id, subject_id=subject_id, app_id=app_id
    )
    if require_attribution():
        # 0.101.0: under the flag the contract is the WHOLE attribution, not
        # just presence — ``funding`` must be stated, and every id must be a
        # canonical UUID (billing NULLs anything else without a trace).
        refused = [*missing, *([FUNDING_FIELD] if funding is None else [])]
        malformed = _malformed_attribution_fields(
            license_id=license_id,
            subject_id=subject_id,
            app_id=app_id,
            metadata_extra=metadata_extra,
        )
        _raise_if_unattributed(service_key, cost_source, refused, malformed)
        return
    if not record:
        return
    if funding is None:
        try:
            cost_event_unattributed_total.labels(field=FUNDING_FIELD).inc()
        except Exception:  # noqa: BLE001 — telemetry must never fail a billable request
            logger.debug("cost_event_unattributed_metric_failed field=funding")
    if not missing:
        return
    logger.warning(
        "cost_event_unattributed service_key=%s cost_source=%s missing=%s",
        service_key,
        cost_source,
        ",".join(missing),
    )
    for field in missing:
        try:
            cost_event_unattributed_total.labels(field=field).inc()
        except Exception:  # noqa: BLE001 — telemetry must never fail a billable request
            logger.debug("cost_event_unattributed_metric_failed field=%s", field)


#: Any ``scheme://userinfo@`` prefix. The metering DSN is
#: ``redis://<user>:<password>@host:port/db``, and an exception text that quotes
#: it (redis-py, an asyncio transport message, a wrapper library) turns
#: ``exc_info=True`` into a credential publisher: the launchd services on mini2
#: write these tracebacks to WORLD-READABLE ``-rw-r--r--`` ``*.err.log`` files,
#: and rotating the password does not unmake a line already written.
#:
#: NOT reusing ``w2v_common.clients.base._redact_for_log``: that helper is the
#: HTTP-body redactor and none of its patterns (``Bearer``, ``token=``,
#: ``api_key``, ``X-Amz-Signature``) matches URL userinfo — measured, it leaves
#: ``redis://metering:<pw>@host`` fully intact. ``net.safe_fetch._redact_url``
#: is closer but takes a bare URL, not a traceback.
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s@\"\']*@")


def _redact_dsn_userinfo(text: str) -> str:
    """Strip ``user:pass@`` userinfo from every URL-like token in ``text``."""
    if not text:
        return text
    return _URL_USERINFO_RE.sub(r"\1<redacted>@", text)


def _redacted_traceback() -> str:
    """The current exception's traceback with every URL credential stripped.

    Replaces ``exc_info=True`` at this module's failure sites. The full cause
    (exception class, message, frames) survives — only the credential is
    removed — so a failure stays as diagnosable as it was, minus the leak.
    """
    return _redact_dsn_userinfo(traceback.format_exc())


def _parse_retry_backoffs() -> tuple[float, ...]:
    """Parse ``METERING_EMIT_RETRY_BACKOFFS`` (env), else the safe default."""
    raw = os.getenv("METERING_EMIT_RETRY_BACKOFFS")
    if not raw:
        return _DEFAULT_EMIT_RETRY_BACKOFFS
    try:
        parsed = tuple(float(part) for part in raw.split(",") if part.strip())
    except Exception:  # noqa: BLE001 — see the contract above; a log line never breaks boot
        logger.warning("invalid METERING_EMIT_RETRY_BACKOFFS=%r; using default", raw)
        return _DEFAULT_EMIT_RETRY_BACKOFFS
    return parsed or _DEFAULT_EMIT_RETRY_BACKOFFS


#: Transport failures that are NOT redis-py exceptions and so used to bypass the
#: retry entirely. Measured on mini2 2026-08-28 across the three live launchd
#: services: 43 ``cost_event_emit_failed`` records, **29 of them** terminating in
#: a bare ``RuntimeError: unable to perform operation on <TCPTransport
#: closed=True ...>; the handler is closed`` — raised by asyncio, not by
#: redis-py, when a pooled connection's transport was closed under it during an
#: idle window. Those 29 cost events were dropped on the FIRST failure with no
#: retry, no dead-letter and no fallback queue, and because ``schedule_cost_event``
#: fires the emit as a detached task the loss was invisible upstream too.
#: ``ConnectionResetError`` is a subclass of ``OSError`` and is covered by it;
#: it is listed for the reader, not for the ``except``.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (RuntimeError, OSError)


def _retryable_redis_errors() -> tuple[type[BaseException], ...]:
    """Transport errors worth a bounded retry (redis-py part resolved lazily)."""
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except Exception:  # noqa: BLE001 — no redis-py: still retry the transport errors
        return _TRANSPORT_ERRORS
    return (RedisConnectionError, RedisTimeoutError, *_TRANSPORT_ERRORS)


def resolve_stream() -> str:
    """Return the cost-bus stream name (``METERING_STREAM`` env, else canonical)."""
    return os.getenv("METERING_STREAM", CANONICAL_COST_STREAM)


def _finite_json_safe(obj: object) -> object:
    """Map non-finite floats (NaN/Infinity) to ``None`` — see :mod:`w2v_common.json_safe`.

    A provider usage payload can carry such a value (a divide-by-zero ratio, a
    bad duration). ``NaN``/``Infinity`` are not valid JSON, so a strict consumer
    (Go ``encoding/json``, Rust ``serde_json``, a JS ``JSON.parse``) REJECTS the
    whole cost record -> silent billing loss, hidden by this module's
    never-raises contract. Kept as a thin alias so this module's local callers
    and the shared engine serializer can never drift apart.
    """
    return finite_json_safe(obj)


def _apply_ambient_metadata(
    fields: dict[str, str],
    metadata_extra: Mapping[str, Any] | None,
    service_key: str,
    actor_id: str | None = None,
) -> None:
    """Default ``actor_id`` / ``job_id`` from the request context when unnamed.

    ``actor_id`` is the already-validated explicit kwarg: a UUID is emitted, ``""``
    means "no person acted" (key omitted, ambient suppressed), ``None`` defers to
    ``metadata_extra`` and then the ambient chain below.
    """
    if actor_id is not None:
        if actor_id:
            fields["actor_id"] = actor_id
    # WHO ACTED, when the caller did not state it, in this order:
    #   1. the actor bound on the request context (JWT ``sub``, mesh
    #      x-actor-id, or a trusted inbound X-Actor-Id) — the human wins;
    #   2. ONLY for service-originated work (no inbound request on this
    #      context): this service's OWN sub, env ``SERVICE_ACTOR_ID``;
    #   3. otherwise the key is OMITTED ENTIRELY (not "", not "None", never a
    #      sentinel) and ``w2v_common_actor_missing_total{service,reason}``
    #      counts it. See ``identity_context.resolve_actor_id``.
    # Both sources must be canonical non-nil UUIDs; a malformed bound value is
    # treated as unbound (billing would NULL it silently anyway).
    #
    # An explicit ``metadata_extra["actor_id"]`` always wins, INCLUDING an
    # explicit ``None``, which is how a caller says "no actor for this event"
    # and suppresses the ambient one (the allowlist loop below drops None
    # values). That mirrors ``set_identity_context(tenant_source=...)``: naming
    # a field means you are talking about it.
    elif metadata_extra is None or "actor_id" not in metadata_extra:
        ambient_actor, missing = resolve_actor_id()
        if ambient_actor:
            fields["actor_id"] = ambient_actor
        elif missing:
            record_actor_missing(service_key, missing)
    # WHICH W2V JOB (run id), into ``metadata.job_id`` when the caller did not
    # name one. Same "naming it means you are talking about it" rule as the
    # actor: an explicit ``metadata_extra["job_id"]`` wins, and an explicit
    # ``None`` suppresses the ambient job.
    if metadata_extra is None or "job_id" not in metadata_extra:
        job_id = get_w2v_job_id_from_context()
        if job_id:
            fields["job_id"] = job_id


def _build_fields(
    *,
    service_key: str,
    cost_source: str,
    tenant_id: uuid.UUID,
    license_id: str | None,
    subject_id: str | None,
    app_id: str | None,
    provider: str,
    model: str,
    served_model: str | None,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    cost_usd: float,
    provider_request_id: str,
    usage_extra: dict[str, Any] | None,
    metadata_extra: Mapping[str, Any] | None = None,
    project_id: str | None = None,
    funding: Funding | None = None,
    event_key: str | None = None,
    actor_id: str | None = None,
) -> dict[str, str]:
    """Marshal a cost event into the EMITTER_GUIDE.md string-field shape."""
    # Fail-safe: a non-finite cost_usd (NaN/inf from a divide-by-zero cost
    # factor) would render as the literal string "nan"/"inf" via the .8f format
    # -> an unparseable numeric field downstream. cost_usd is advisory (metering
    # recomputes from provider_pricing), so coerce a non-finite value to 0.0
    # rather than emit a garbage figure.
    safe_cost_usd = cost_usd if math.isfinite(cost_usd) else 0.0
    # Fail-safe: sanitise non-finite floats in usage_extra to JSON null BEFORE
    # dumping, then dump with allow_nan=False so anything that slipped through
    # raises (caught by emit_cost_event's outer warning) rather than emitting
    # invalid JSON. Valid JSON always.
    safe_usage_extra = _finite_json_safe(usage_extra or {})
    fields = {
        # Producer-minted idempotency key: ONE per physical metered call.
        #
        # billing-service claims this key in an inbox table
        # (``ProcessedCostEventKey``, ``cost_flusher.py:459-493``) and drops any
        # event whose key it has already seen. An event that arrives WITHOUT the
        # field skips that claim entirely (``if p[0].get("event_key")``) and is
        # billed unconditionally — which is what every event from this module
        # used to do, because ``_build_fields`` never emitted the field.
        #
        # That is a live double-bill, not a theoretical one: ``emit_cost_event``
        # RETRIES the XADD on ``RedisTimeoutError``/``ConnectionError`` (see the
        # FIX B1 loop below). A timeout raised AFTER redis accepted the write
        # re-adds the same charge under a fresh stream message id, and the
        # message-id inbox cannot dedupe it because the id genuinely differs.
        # Billing is append-only — there is no refund, void or reversal path
        # anywhere in the estate — so the tenant simply pays twice.
        #
        # Minted HERE, in ``_build_fields``, which runs ONCE before the retry
        # loop: every attempt for one logical call therefore carries the SAME
        # key. Minting it inside the loop would make the retry undedupable again.
        #
        # 0.101.0: a CALLER-SUPPLIED deterministic key wins (validated by
        # ``_validate_event_key``; derive it with :func:`cost_event_key`). The
        # uuid7 is only the fallback: it covers the in-process retry above, but
        # NOT a re-emit after a process restart (a queue redelivering a job
        # step), which mints a fresh uuid7 and bills twice. Counted per source
        # on ``w2v_cost_event_key_source_total``.
        "event_key": event_key or str(uuid.uuid7()),
        "tenant_id": str(tenant_id),
        # ``None`` (omitted) and ``""`` (supplied empty) both land here as "".
        # The wire shape is unchanged — billing's ``_parse_uuid`` maps either to
        # NULL — so the miss is surfaced by ``_check_attribution`` above, not by
        # a new field value the receiver has never seen.
        "license_id": license_id or "",
        "subject_id": subject_id or "",
        "app_id": app_id or "",
        "service_key": service_key,
        "cost_source": cost_source,
        "provider": provider,
        "model": model,
        "served_model": served_model or model,
        "input_tokens": str(input_tokens),
        "output_tokens": str(output_tokens),
        "cost_usd": f"{safe_cost_usd:.8f}",
        "latency_ms": str(latency_ms),
        "provider_request_id": provider_request_id,
        "usage_extra": json.dumps(safe_usage_extra, sort_keys=True, allow_nan=False),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    # WHICH PROJECT, top-level (billing's ``cost_flusher`` reads
    # ``c.get("project_id")`` off the wire, NOT out of ``metadata``, so this is
    # deliberately NOT in ``_METADATA_PASSTHROUGH_KEYS``). The calling app's OWN
    # project — for w2v the W2V project, never the fr one; see
    # :func:`_resolve_project_id`. Absent -> key OMITTED, never "".
    resolved_project = _resolve_project_id(project_id, tenant_id)
    if resolved_project:
        fields["project_id"] = resolved_project
    # WHO FUNDS the call, top-level for the same reason as ``project_id``.
    # Absent -> key OMITTED (billing derives); never a default that guesses.
    if funding is not None:
        fields["funding"] = funding
    _apply_ambient_metadata(fields, metadata_extra, service_key, actor_id)
    if not metadata_extra:
        # Additive contract: with the kwarg omitted (or empty) the emitted dict
        # is byte-identical to what every existing caller ships today. That is
        # the whole property that makes this change safe to land under 19
        # consumers at once, and ``test_omitting_metadata_extra_emits_a_byte_identical_field_dict``
        # is what holds it.
        return fields
    dropped = []
    for key, value in metadata_extra.items():
        if key not in _METADATA_PASSTHROUGH_KEYS:
            dropped.append(key)
            continue
        if value is None:
            # ``str(None)`` is the truthy literal "None", which the receiver's
            # ``if c.get(k)`` would happily store as real metadata. A key with
            # no value is "not provided", not the string "None".
            continue
        fields[key] = str(value)
    if dropped:
        logger.debug(
            "cost_event_metadata_keys_dropped keys=%s allowed=%s",
            sorted(dropped),
            sorted(_METADATA_PASSTHROUGH_KEYS),
        )
    return fields


async def emit_cost_event(
    redis_conn: aioredis.Redis | None,
    *,
    service_key: str,
    cost_source: str,
    tenant_id: uuid.UUID | str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    license_id: str | None = None,
    subject_id: str | None = None,
    app_id: str | None = None,
    served_model: str | None = None,
    cost_usd: float = 0.0,
    provider_request_id: str = "",
    usage_extra: dict[str, Any] | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    stream: str | None = None,
    maxlen: int = _DEFAULT_MAXLEN,
    project_id: str | None = None,
    funding: Funding | None = None,
    event_key: str | None = None,
    actor_id: str | None = None,
) -> None:
    """XADD a cost event to the cost-bus stream.

    Field shape matches EMITTER_GUIDE.md — every value is a string at the Redis
    layer. ``cost_usd`` is advisory (metering recomputes from the centralised
    ``provider_pricing`` table). ``stream`` defaults to :func:`resolve_stream`.

    ``metadata_extra`` carries originator attribution — the keys in
    :data:`_METADATA_PASSTHROUGH_KEYS`, which billing lifts into
    ``billing_cost_events.metadata``. Values are coerced to ``str``; keys outside
    that set are dropped here (DEBUG-logged) rather than shipped to be discarded
    silently by the receiver, and a ``None`` value is skipped rather than
    stringified to the truthy literal ``"None"``. Keyword-only with a ``None``
    default: omitting it emits exactly today's field dict.

    ``project_id`` is the CALLING APP's own project — for w2v the W2V project,
    never the fr project — emitted as a TOP-LEVEL field (billing reads it there,
    not from ``metadata``). Omitted, it falls back to the bound W2V project
    (``set_w2v_project_context`` / inbound ``X-W2V-Project-Id``), then to the
    tenant-level project (counted on ``w2v_cost_project_fallback_total``).
    ``metadata.job_id`` likewise defaults to the bound W2V job
    (``set_w2v_job_context`` / inbound ``X-W2V-Job-Id``).

    ``funding`` (:data:`Funding`) says who paid for the call — ``local_gpu`` /
    ``platform_external`` / ``tenant_byo`` — as a TOP-LEVEL field. Explicit
    kwarg only: omitted, the key is not emitted and billing derives it. An
    out-of-set value raises :class:`InvalidFundingError` before the swallow.

    ``event_key`` (0.101.0) is the caller's DETERMINISTIC idempotency key for this
    physical call — derive it with :func:`cost_event_key` from what identifies
    the call (``service_key``, job id, step, attempt) so a re-emit after a
    process restart dedupes at billing. Omitted, a uuid7 is minted once per
    emit (stable across this module's own retry only). Billing dedupes on it
    globally, so it must be unique across services. Invalid (blank, placeholder
    word, whitespace, >64 chars) raises :class:`InvalidEventKeyError`.

    ``actor_id`` (0.101.0) is WHO ACTED: the verified Keycloak ``sub`` of the
    human who initiated the work, as a canonical UUID. ``""`` states that no
    person acted (platform / agent work): the key is omitted and the ambient
    actor is suppressed. ``None`` (default) keeps the ambient chain (bound actor
    -> ``SERVICE_ACTOR_ID`` for service-originated work -> omitted). A non-UUID,
    or an actor given both here and in ``metadata_extra``, raises
    :class:`InvalidActorIdError`.

    NEVER-RAISES CONTRACT, WITH ONE DELIBERATE EXCEPTION. Transport failures
    (Redis down, XADD timeout, an unserialisable usage payload) are still
    swallowed with a warning — dropping one cost event must not fail a customer
    request. A missing or non-UUID ``tenant_id`` is NOT a transport failure: it
    is a caller bug that used to be papered over with the nil-UUID sentinel, so
    it raises :class:`w2v_common.tenant.TenantError` **out of this coroutine**,
    BEFORE the swallow. It is validated first, outside the ``try``, precisely so
    the never-raises handler cannot re-hide it.

    :raises w2v_common.tenant.MissingTenantError: no tenant supplied.
    :raises w2v_common.tenant.InvalidTenantIdError: tenant is not a UUID.
    :raises InvalidFundingError: ``funding`` is not in :data:`FUNDING_VALUES`.
    :raises InvalidEventKeyError: ``event_key`` is given but unusable.
    :raises InvalidActorIdError: ``actor_id`` is given but not a canonical UUID.
    :raises MissingAttributionError: ``METERING_REQUIRE_ATTRIBUTION`` is on and
        ``license_id``/``subject_id``/``app_id``/``funding`` is missing, or an
        attribution id (incl. ``metadata_extra["actor_id"]``) is not a UUID.
        With the flag OFF (default) the event is emitted, a WARNING names a
        missing licence/subject/app field, and ``w2v_cost_event_unattributed_total``
        is incremented (``field="funding"`` counted without a warning) — a cost
        event is revenue, so an unattributed row beats a dropped one.
    """
    tenant = parse_tenant_id(tenant_id, source=f"cost event ({service_key}/{cost_source})")
    funding = _validate_funding(funding)
    event_key = _validate_event_key(event_key)
    actor_id = _validate_actor_id(actor_id, metadata_extra)
    # Outside the never-raises try, exactly like the tenant check above: the
    # swallow below must not be able to re-hide an attribution miss.
    _check_attribution(
        service_key=service_key,
        cost_source=cost_source,
        license_id=license_id,
        subject_id=subject_id,
        app_id=app_id,
        funding=funding,
        metadata_extra=metadata_extra,
    )
    if redis_conn is None:
        # Metering is disarmed in this process (empty METERING_REDIS_URL, or a
        # failed connect). Consumers already guard with
        # ``if app.state.metering_redis:``; accepting ``None`` here makes the
        # drop COUNTABLE instead of invisible. A third value on the EXISTING
        # ``outcome`` label deliberately — a second metric name would split the
        # billing dashboard away from ok/failed.
        _record_outcome(service_key, "dropped_disabled")
        return
    attempts = 0
    try:
        fields = _build_fields(
            service_key=service_key,
            cost_source=cost_source,
            tenant_id=tenant,
            license_id=license_id,
            subject_id=subject_id,
            app_id=app_id,
            provider=provider,
            model=model,
            served_model=served_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            provider_request_id=provider_request_id,
            usage_extra=usage_extra,
            metadata_extra=metadata_extra,
            project_id=project_id,
            funding=funding,
            event_key=event_key,
            actor_id=actor_id,
        )
        _record_key_source(service_key, "caller" if event_key else "minted")
        target = stream or resolve_stream()
        retryable = _retryable_redis_errors()
        # FIX B1: bounded retry on transient connection/timeout errors. The
        # trailing ``None`` sentinel marks the final attempt — re-raise so the
        # outer handler logs the existing warning (never-raises contract held).
        for backoff in (*_parse_retry_backoffs(), None):
            attempts += 1
            try:
                # redis-py stubs declare xadd's `fields` as bytes-typed; with
                # decode_responses=True the runtime accepts (and returns)
                # strings. Cast to satisfy the stub without runtime cost.
                await redis_conn.xadd(
                    target,
                    cast(dict[Any, Any], fields),
                    maxlen=maxlen,
                    approximate=True,
                )
                _record_outcome(service_key, "ok")
                return
            except retryable:
                if backoff is None:
                    raise  # retries exhausted — fall through to the warning
                await asyncio.sleep(backoff)
    except Exception as exc:  # noqa: BLE001 — fire-and-forget; never block the caller
        _record_outcome(service_key, "failed")
        # NOT ``exc_info=True``: the rendered traceback can quote the metering
        # DSN, and these lines land in world-readable log files. The redacted
        # traceback keeps the whole cause and drops only the credential.
        logger.warning(
            "cost_event_emit_failed service_key=%s cost_source=%s error=%s attempts=%d\n%s",
            service_key,
            cost_source,
            type(exc).__name__,
            attempts,
            _redacted_traceback(),
        )


def _record_no_loop_drop(
    *,
    service_key: str,
    cost_source: str,
    tenant: uuid.UUID,
    license_id: str | None,
    subject_id: str | None,
    app_id: str | None,
    provider: str,
    model: str,
    served_model: str | None,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    cost_usd: float,
    provider_request_id: str,
) -> None:
    """Make a no-running-loop drop loud: count it, then reproduce it in the log.

    Ordering is deliberate — COUNT FIRST, then log, then (only if escalation is
    on) raise. A raise that preceded the counter would mean that turning
    ``METERING_REQUIRE_LOOP`` on BLINDS the very metric that measures the
    problem it escalates, so the flag would trade one silence for another.

    The WARNING carries every field billing needs to reconstruct the row by
    hand. ``usage_extra``/``metadata_extra`` are deliberately NOT rendered: they
    are caller-controlled and unbounded, and this line is emitted once per
    dropped billable call, so inlining them turns an incident into a log flood.
    Everything the receiver actually bills on is here.

    :raises NoRunningLoopError: ``METERING_REQUIRE_LOOP`` is truthy.
    """
    _record_outcome(service_key, DROPPED_NO_LOOP)
    logger.warning(
        "cost_event_dropped_no_running_loop service_key=%s cost_source=%s "
        "tenant_id=%s license_id=%s subject_id=%s app_id=%s provider=%s "
        "model=%s served_model=%s input_tokens=%d output_tokens=%d "
        "latency_ms=%d cost_usd=%.8f provider_request_id=%s",
        service_key,
        cost_source,
        tenant,
        license_id or "",
        subject_id or "",
        app_id or "",
        provider,
        model,
        served_model or model,
        input_tokens,
        output_tokens,
        latency_ms,
        # ⚠️ KNOWN GAP, deliberately left, and it is a MANUFACTURED value rather
        # than an absent one. A non-finite ``cost_usd`` renders here as a wholly
        # plausible ``cost_usd=0.00000000`` — indistinguishable from a genuine
        # zero-cost call — in the one line this function exists to make the
        # durable record. The identical coercion in ``_build_fields`` is
        # defensible because billing recomputes the authoritative figure from
        # ``provider_pricing``; THERE IS NO RECOMPUTE HERE. It is left as-is
        # only because it is unreachable today: all 16 call sites across
        # modality_gateway / tts_backend / mlx_backend / mlx_ingestion_backend
        # pass a literal ``0.0``. Measured at origin/main, and the check is
        # cheap because 16 call sites funnel through only FIVE distinct leaf
        # calls: modality_gateway ``routers/tts_routes`` and
        # ``services/byo_cost``; tts_backend ``routers/tts_routes``; and the two
        # sync wrappers, mlx_backend and mlx_ingestion_backend
        # ``services/metering``. (2 + 1 + 2 = 5 -- an earlier revision of this
        # comment said "four" while its own parenthetical listed five, which
        # would have closed the set one site early. Count the list, not the
        # label; that miscount is the same defect this comment exists to flag.)
        # ⛔ If a caller ever computes this value, render
        # the non-finite case verbatim instead of coercing it — a "nan" a human
        # must adjudicate beats a zero they will bill.
        cost_usd if math.isfinite(cost_usd) else 0.0,
        provider_request_id,
    )
    if require_loop():
        raise NoRunningLoopError(
            f"cost event ({service_key}/{cost_source}) was dropped: "
            f"schedule_cost_event was called with no running event loop"
        )


def schedule_cost_event(
    redis_conn: aioredis.Redis | None,
    *,
    service_key: str,
    cost_source: str,
    tenant_id: uuid.UUID | str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    license_id: str | None = None,
    subject_id: str | None = None,
    app_id: str | None = None,
    served_model: str | None = None,
    cost_usd: float = 0.0,
    provider_request_id: str = "",
    usage_extra: dict[str, Any] | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    stream: str | None = None,
    maxlen: int = _DEFAULT_MAXLEN,
    project_id: str | None = None,
    funding: Funding | None = None,
    event_key: str | None = None,
    actor_id: str | None = None,
) -> bool:
    """Fire-and-forget :func:`emit_cost_event`, GC-safe.

    Spawns the emit as a background task and retains a strong reference until it
    completes, so the event can never be garbage-collected before the XADD
    lands. Returns immediately; never blocks the caller.

    ``redis_conn`` accepts ``None`` for the same reason :func:`emit_cost_event`
    does, and the two MUST agree: the whole point of accepting ``None`` is that
    a consumer can delete its ``if app.state.metering_redis:`` guard, and a
    guardless consumer on the fire-and-forget path would otherwise fail ``ty``
    here. ``None`` is forwarded verbatim into the spawned coroutine, which hits
    the disarmed early-return and records ``dropped_disabled`` exactly once —
    this function records no outcome of its own (its ``_check_attribution`` call
    is ``record=False`` precisely so nothing is double-counted).

    The tenant is validated SYNCHRONOUSLY, in the caller's frame, before the
    task is spawned — a ``TenantError`` raised inside a fire-and-forget task
    would surface only as an "exception never retrieved" warning, i.e. exactly
    the silent mis-attribution this change exists to end.

    :returns: ⭐ A POSITIVE HANDOFF MARKER. ``True`` means the emit was handed to
        a running event loop as a retained task. ``False`` means there was no
        running loop, so nothing was handed off and the event was DROPPED (it is
        counted and logged — see below). Before this return value existed, the
        two outcomes were indistinguishable at the call site: the function
        returned ``None`` and raised nothing in both cases, so a caller could
        only INFER success from the absence of an exception.

        ⚠️ ``True`` is a statement about the HANDOFF, not about delivery. The
        XADD may still fail inside the task; that terminal outcome is
        ``cost_events_total{outcome="ok"|"failed"}``. This return value
        distinguishes "the emit path ran" from "the emit path was never
        entered", which is the distinction this function previously could not
        make.

    NO-RUNNING-LOOP BEHAVIOUR (the drop, and why it stays a drop). When
    ``asyncio.get_running_loop()`` finds no loop there is nowhere to schedule
    the XADD, and this module holds no durable spool to defer it to. The event
    is therefore lost — but it is no longer lost SILENTLY:

    * ``cost_events_total{service_key, outcome="dropped_no_loop"}`` is
      incremented, on the same counter and under the same ``service_key`` as
      ``ok``/``failed``, so "this service is dropping cost events" is a ratio
      any existing dashboard on that metric already sees.
    * A **WARNING** (was: DEBUG, i.e. off in every deployed service) reproduces
      the event's full identity — tenant, provider, model, token counts,
      latency, advisory cost and ``provider_request_id`` — so the billing record
      can be reconstructed by hand from the log line alone. That log line IS the
      durable record; there is no automatic replay, and pretending otherwise
      would be worse than the drop.
    * ``METERING_REQUIRE_LOOP=true`` escalates the drop to
      :class:`NoRunningLoopError`.

    ⛔ The raise is OPT-IN and stays opt-in. Raising by default would convert a
    lost billing record into a failed customer request — and by the time this
    function is reached the provider call has already happened and already cost
    money, so the raise would forfeit the revenue as well as the evidence. This
    is the same ruling, and the same direction, that
    :func:`_check_attribution` already applies to a missing attribution field.

    :raises w2v_common.tenant.MissingTenantError: no tenant supplied.
    :raises w2v_common.tenant.InvalidTenantIdError: tenant is not a UUID.
    :raises InvalidFundingError: ``funding`` is not in :data:`FUNDING_VALUES`.
    :raises InvalidEventKeyError: ``event_key`` is given but unusable.
    :raises InvalidActorIdError: ``actor_id`` is given but not a canonical UUID.
    :raises MissingAttributionError: ``METERING_REQUIRE_ATTRIBUTION`` is on and
        ``license_id``/``subject_id``/``app_id``/``funding`` is missing, or an
        attribution id (incl. ``metadata_extra["actor_id"]``) is not a UUID.
        With the flag OFF (default) the event is emitted, a WARNING names a
        missing licence/subject/app field, and ``w2v_cost_event_unattributed_total``
        is incremented (``field="funding"`` counted without a warning) — a cost
        event is revenue, so an unattributed row beats a dropped one.
    :raises NoRunningLoopError: there is no running event loop AND
        ``METERING_REQUIRE_LOOP`` is on.
    """
    tenant = parse_tenant_id(tenant_id, source=f"cost event ({service_key}/{cost_source})")
    funding = _validate_funding(funding)
    event_key = _validate_event_key(event_key)
    actor_id = _validate_actor_id(actor_id, metadata_extra)
    # Raise-only pre-check, for the same reason the tenant is parsed here: with
    # escalation ON, a MissingAttributionError raised inside the detached task
    # would surface only as "exception never retrieved". ``record=False`` — the
    # spawned emit does the counting, so the miss is observed exactly once.
    _check_attribution(
        service_key=service_key,
        cost_source=cost_source,
        license_id=license_id,
        subject_id=subject_id,
        app_id=app_id,
        funding=funding,
        metadata_extra=metadata_extra,
        record=False,
    )
    # Resolve the loop BEFORE constructing the coroutine. The previous order
    # built it first and then had to ``coro.close()`` it on the no-loop path to
    # suppress a "coroutine was never awaited" warning. Not creating the
    # coroutine is strictly better than creating and closing it: it makes the
    # orphaned-coroutine class UNREACHABLE rather than merely handled, and it
    # removes the only step between detecting the drop and recording it.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # ⛔ Do NOT collapse this into a bare ``return False``. The recording is
        # the fix; the return value only lets the caller act on it.
        _record_no_loop_drop(
            service_key=service_key,
            cost_source=cost_source,
            tenant=tenant,
            license_id=license_id,
            subject_id=subject_id,
            app_id=app_id,
            provider=provider,
            model=model,
            served_model=served_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            provider_request_id=provider_request_id,
        )
        return False
    task = loop.create_task(
        emit_cost_event(
            redis_conn,
            service_key=service_key,
            cost_source=cost_source,
            tenant_id=tenant,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            license_id=license_id,
            subject_id=subject_id,
            app_id=app_id,
            served_model=served_model,
            cost_usd=cost_usd,
            provider_request_id=provider_request_id,
            usage_extra=usage_extra,
            metadata_extra=metadata_extra,
            stream=stream,
            maxlen=maxlen,
            project_id=project_id,
            funding=funding,
            event_key=event_key,
            actor_id=actor_id,
        )
    )
    # Retain a strong ref until the task completes so it can't be GC'd mid-XADD.
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return True


def _redis_endpoint(url: str) -> str:
    """Render a Redis DSN as ``host:port`` for logging — never any userinfo.

    THIN ALIAS, kept only as the in-module name this file's callers already use.
    The implementation moved to the PUBLIC
    ``w2v_common.observability.redis_endpoint`` in 0.67.25, because eleven
    services outside metering log ``settings.redis.url`` raw at INFO and needed
    the same redactor; a private helper is not importable as a fix. Behaviour is
    byte-identical — same ``host:port`` shape, same ``"<unparseable>"``
    fail-closed sentinel, same deliberately-wide ``except Exception`` so a
    ``MagicMock`` url from a mocked-settings test cannot abort boot from inside
    ``init_metering_redis``'s own except handler. See that function's docstring
    for the full rationale; do not re-inline it here.
    """
    return redis_endpoint(url)


async def init_metering_redis(url: str) -> aioredis.Redis | None:
    """Open the metering Redis connection. Returns None when disabled/unreachable.

    The metering stream lives on a SEPARATE Redis from any service-local cache.
    When ``url`` is empty, metering is disabled — emits silently no-op.

    Under ``METERING_REQUIRE_ATTRIBUTION`` (0.104.0) a disarmed metering is a
    BOOT failure instead: a service that has declared every cost event must be
    fully attributed cannot be allowed to drop them all (``dropped_disabled``)
    without a sound.

    :raises MeteringDisarmedError: strict mode is on and the URL is empty or the
        connection cannot be opened.
    """
    if not url:
        if require_attribution():
            raise MeteringDisarmedError(
                f"{_REQUIRE_ATTRIBUTION_ENV} is on but METERING_REDIS_URL is not set"
            )
        logger.warning("METERING_REDIS_URL not set — cost events will not be emitted")
        _set_metering_enabled(0)
        return None
    try:
        import redis.asyncio as aioredis

        # A client built with no health check, no keepalive and no timeout
        # retry is stored on ``app.state.metering_redis`` for the WHOLE process
        # lifetime, so an idle-closed transport is only discovered when the next
        # billable request tries to use it (the bare RuntimeError above).
        client = aioredis.from_url(
            url,
            decode_responses=True,
            health_check_interval=_HEALTH_CHECK_INTERVAL_SECONDS,
            socket_keepalive=True,
            retry_on_timeout=True,
        )
        # ping() returns Awaitable[bool] under decode_responses=True; cast for ty.
        await cast(Any, client.ping())
        logger.info("Metering Redis connected: %s", _redis_endpoint(url))
        _set_metering_enabled(1)
        return client
    except Exception as exc:  # noqa: BLE001 — metering must never fail boot (unless strict)
        if require_attribution():
            raise MeteringDisarmedError(
                f"{_REQUIRE_ATTRIBUTION_ENV} is on but the metering redis at "
                f"{_redis_endpoint(url)} is unreachable ({type(exc).__name__})"
            ) from None
        logger.warning(
            "metering_redis_init_failed endpoint=%s error=%s; events will be dropped\n%s",
            _redis_endpoint(url),
            type(exc).__name__,
            _redacted_traceback(),
        )
        _set_metering_enabled(0)
        return None


async def drain_pending(timeout: float = 5.0) -> None:  # noqa: ASYNC109 — bounded drain budget is the intended public knob, not an asyncio.timeout cancel scope
    """Await in-flight fire-and-forget emit tasks (FIX B2). Never raises.

    ``schedule_cost_event`` spawns emits as background tasks; on SIGTERM the
    event loop stops and any still-pending XADD is cancelled mid-flight ->
    silent cost-event loss. Calling this on a normal lifespan shutdown lets the
    in-flight emits land first. Bounded by ``timeout`` so a wedged emit can't
    block shutdown forever. Safe to call with nothing pending (returns at once).
    """
    # Copy first: tasks remove themselves from the live set via done-callback.
    pending = [task for task in set(_background_tasks) if not task.done()]
    if not pending:
        return
    try:
        await asyncio.wait(pending, timeout=timeout)
    except Exception:  # noqa: BLE001 — drain is best-effort; never block shutdown
        logger.warning("metering_drain_pending_failed\n%s", _redacted_traceback())


async def close_metering_redis(
    client: aioredis.Redis | None, *, drain_timeout: float = 5.0
) -> None:
    """Close the metering Redis connection. Safe to call with None.

    Drains in-flight fire-and-forget emits (FIX B2) BEFORE closing the client so
    a normal lifespan shutdown doesn't cancel a pending XADD. Bounded by
    ``drain_timeout`` so shutdown can never hang on a wedged emit.
    """
    await drain_pending(timeout=drain_timeout)
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001 — shutdown best-effort
        logger.warning("metering_redis_close_failed\n%s", _redacted_traceback())
