"""Resolve the RUN's provider for one slot from the orchestrator (shipped dark).

Owner goal: every model/GPU call a w2v service makes uses the provider the run
pinned for that slot — the tenant's own registered provider (URI + token + kind)
or the platform default — under the licence's ``byo_mode``. The orchestrator owns
the connection store, the pin and the policy; services ask it per call:

``POST {service_url("orchestrator")}/api/internal/provider-slots/resolve``
``{"slot": <slot>, "run_id": <X-W2V-Job-Id> | null}`` on the CELL-JWT lane only
(a fresh ``X-W2V-Cell-Token`` minted for the bound, vouched tenant; no licence, no
``X-Tenant-Id``). Answer: the orchestrator's P5 ``slot_payload`` JSON
(``src/services/providers/slot_payload.py``)::

    {"slot", "kind", "base_url", "model", "options", "secret",
     "ref": {"connection_id", "version"}, ["source"], ["pinned_ip"]}

Errors: 403 ``byo_provider_required`` → :class:`ProviderSlotRequiredError`;
409 ``provider_unavailable`` / 404 (run of another tenant) / anything else →
:class:`ProviderSlotUnavailableError`. 503 and connect failures are retried (the
orchestrator is ``replicas: 1`` and every merge rolls it).

Switch ``W2V_PROVIDER_SLOTS`` = ``off`` (default) | ``enforce``. ``off`` ⇒ no
resolve call is ever made and every client behaves byte-identically to before.

The resolved :class:`ProviderSlot` carries the DECRYPTED tenant secret as a
:class:`~pydantic.SecretStr`. It is cached in-process for 300 s keyed
``(tenant, run, slot)`` — never shared across tenants — and must never be logged,
persisted or forwarded to anything but the tenant's own provider.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, cast

import httpx
from cachetools import TTLCache
from pydantic import SecretStr

from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    cell_audience_for_url,
    cell_signer_configured,
)
from w2v_common.auth.identity_context import (
    cell_token_event_hook,
    get_tenant_id_from_context,
    is_provider_hop,
    is_tenant_vouched,
)
from w2v_common.clients import base as _retry
from w2v_common.clients.vllm_usage_report import resolve_run_id
from w2v_common.discovery import service_url
from w2v_common.observability.request_context import trace_event_hook
from w2v_common.providers.errors import (
    BYO_PROVIDER_REQUIRED,
    CELL_LANE_REQUIRED,
    RUN_NOT_FOUND,
    UNKNOWN_SLOT,
    ProviderHopRefusedError,
    ProviderSlotConfigError,
    ProviderSlotRequiredError,
    ProviderSlotUnavailableError,
)
from w2v_common.tenant import UnvouchedTenantError

logger = logging.getLogger(__name__)

#: Env switch. Read per call, so a flip needs no restart and tests can patch it.
SWITCH_ENV: Final = "W2V_PROVIDER_SLOTS"
ProviderSlotsMode = Literal["off", "enforce"]
MODE_OFF: Final = "off"
MODE_ENFORCE: Final = "enforce"

#: The orchestrator route (orchestrator PR in flight when this shipped).
RESOLVE_PATH: Final = "/api/internal/provider-slots/resolve"
ORCHESTRATOR: Final = "orchestrator"

SlotSource = Literal["tenant", "deployment"]

CACHE_TTL_SECONDS: Final = 300.0
#: Bound on distinct (tenant, run, slot) entries held at once.
CACHE_MAX_ENTRIES: Final = 2048
#: TTL for a RUN-LESS resolve (no X-W2V-Job-Id / correlation run): short, so a
#: tenant-level answer is never reused across runs for the full 300 s.
RUNLESS_CACHE_TTL_SECONDS: Final = 30.0
#: Negative cache for ``ProviderSlotUnavailableError`` (absorbs a burst, recovers fast).
NEGATIVE_CACHE_TTL_SECONDS: Final = 5.0
#: Resolve is an idempotent READ: 502/503/504 + connect failures are retried with
#: capped exponential backoff + jitter, inside ``RESOLVE_RETRY_BUDGET_S`` of sleep
#: (the single-replica orchestrator rolls on every merge).
RESOLVE_MAX_ATTEMPTS: Final = 8
RESOLVE_RETRY_DELAY_S: Final = 0.5
RESOLVE_RETRY_MAX_DELAY_S: Final = 8.0
RESOLVE_RETRY_BUDGET_S: Final = 30.0
_RETRYABLE_RESOLVE_STATUS: Final = frozenset({502, 503, 504})
RESOLVE_TIMEOUT_S: Final = 10.0

#: Patched by tests; every retry sleep goes through here.
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def provider_slots_mode(value: str | None = None) -> ProviderSlotsMode:
    """``value`` or env ``W2V_PROVIDER_SLOTS``; blank → ``off``; unknown → ``ValueError``.

    A typo must not silently mean ``off``: the operator would believe the flip happened.
    """
    raw = value if value is not None else os.environ.get(SWITCH_ENV, "")
    raw = str(raw).strip().lower() or MODE_OFF
    if raw == MODE_OFF:
        return MODE_OFF
    if raw == MODE_ENFORCE:
        return MODE_ENFORCE
    raise ValueError(f"{SWITCH_ENV}={raw!r} is not one of: {MODE_OFF}, {MODE_ENFORCE}")


def provider_slots_enforced() -> bool:
    """True when ``W2V_PROVIDER_SLOTS=enforce`` (callers must resolve the run's slot)."""
    return provider_slots_mode() == MODE_ENFORCE


# ── the resolved slot ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProviderSlotRef:
    """The immutable connection version the run pinned (tenant slots only)."""

    connection_id: uuid.UUID
    version: int


#: Keys of the orchestrator's 200 answer (#1150 ``internal_provider_slots._view``):
#: ``source`` + the P5 ``slot_payload``. ``tests/fixtures/provider_slot_contract.json``
#: pins it; a drift on either side fails ``test_contract_fixture_*``.
WIRE_KEYS: Final = frozenset(
    {"source", "slot", "kind", "base_url", "model", "options", "secret", "ref"}
)
#: Optional extras this parser understands (``pinned_ip`` is not sent by #1150).
OPTIONAL_WIRE_KEYS: Final = frozenset({"pinned_ip"})


@dataclass(frozen=True, slots=True)
class ProviderSlot:
    """One slot's provider for the current run. ``secret`` never appears in a repr."""

    slot: str
    source: SlotSource
    kind: str
    base_url: str | None = None
    model: str | None = None
    options: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    secret: SecretStr | None = None
    ref: ProviderSlotRef | None = None
    #: IP the client must connect to (DNS-rebinding pin); ``None`` = no pin.
    pinned_ip: str | None = None

    @property
    def is_tenant(self) -> bool:
        return self.source == "tenant"

    def secret_value(self) -> str | None:
        """The clear secret, or ``None`` when the connection stores none."""
        value = self.secret.get_secret_value() if self.secret is not None else None
        return value or None

    @classmethod
    def from_payload(cls, payload: object, *, expected_slot: str) -> ProviderSlot:
        """Parse the orchestrator's ``slot_payload`` JSON exactly; malformed → unavailable.

        ``source`` is optional on the wire (``slot_payload`` itself is only ever built
        for a tenant connection): explicit ``source`` wins, else a payload with a
        ``ref.connection_id`` is ``tenant`` and one without is ``deployment``.
        """

        def bad(what: str) -> ProviderSlotUnavailableError:
            return ProviderSlotUnavailableError(
                expected_slot, f"the orchestrator's provider answer is malformed ({what})"
            )

        if not isinstance(payload, Mapping):
            raise bad("not an object")
        body = cast("Mapping[str, object]", payload)
        missing = (WIRE_KEYS - {"source"}) - body.keys()
        if missing:
            raise bad("missing " + ",".join(sorted(missing)))
        slot = body.get("slot")
        if slot != expected_slot:
            raise bad("slot does not match the request")
        kind = body.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise bad("kind")
        base_url = _opt_str(body, "base_url", bad)
        model = _opt_str(body, "model", bad)
        pinned_ip = _opt_str(body, "pinned_ip", bad)
        raw_options = body.get("options")
        if raw_options is None:
            options: Mapping[str, object] = {}
        elif isinstance(raw_options, Mapping):
            options = dict(cast("Mapping[str, object]", raw_options))
        else:
            raise bad("options")
        raw_secret = body.get("secret")
        if raw_secret is not None and not isinstance(raw_secret, str):
            raise bad("secret")
        ref = _parse_ref(body.get("ref"), bad)
        source = _parse_source(body.get("source"), ref, has_secret=bool(raw_secret), bad=bad)
        return cls(
            slot=expected_slot,
            source=source,
            kind=kind.strip(),
            base_url=base_url,
            model=model,
            options=MappingProxyType(options),
            secret=SecretStr(raw_secret) if raw_secret else None,
            ref=ref,
            pinned_ip=pinned_ip,
        )


def _parse_source(
    raw: object,
    ref: ProviderSlotRef | None,
    *,
    has_secret: bool,
    bad: Callable[[str], ProviderSlotUnavailableError],
) -> SlotSource:
    if raw is None:
        source: SlotSource = "tenant" if ref is not None else "deployment"
    elif raw in ("tenant", "deployment"):
        source = cast("SlotSource", raw)
    else:
        raise bad("source")
    if source == "tenant" and ref is None:
        raise bad("a tenant slot without its connection ref")
    if source == "deployment" and (has_secret or ref is not None):
        # The platform default has no tenant credential / connection: a contract bug.
        raise bad("a deployment slot carrying a secret or connection ref")
    return source


def _opt_str(
    body: Mapping[str, object],
    key: str,
    bad: Callable[[str], ProviderSlotUnavailableError],
) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise bad(key)
    return value.strip() or None


def _parse_ref(
    raw: object, bad: Callable[[str], ProviderSlotUnavailableError]
) -> ProviderSlotRef | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise bad("ref")
    ref = cast("Mapping[str, object]", raw)
    raw_id, raw_version = ref.get("connection_id"), ref.get("version")
    if raw_id is None and raw_version is None:
        return None
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise bad("ref.version")
    try:
        connection_id = uuid.UUID(str(raw_id))
    except ValueError:
        raise bad("ref.connection_id") from None
    return ProviderSlotRef(connection_id=connection_id, version=raw_version)


# ── cache ───────────────────────────────────────────────────────────────────

CacheKey = tuple[uuid.UUID, str | None, str]
#: Run-bound answers (key ``(tenant, run, slot)``, run never ``None``).
_cache: TTLCache[CacheKey, ProviderSlot] = TTLCache(
    maxsize=CACHE_MAX_ENTRIES, ttl=CACHE_TTL_SECONDS
)
#: Run-less answers (key ``(tenant, None, slot)``) — short TTL, see above.
_runless_cache: TTLCache[CacheKey, ProviderSlot] = TTLCache(
    maxsize=CACHE_MAX_ENTRIES, ttl=RUNLESS_CACHE_TTL_SECONDS
)
#: Recent ``ProviderSlotUnavailableError`` per key.
_negative_cache: TTLCache[CacheKey, ProviderSlotUnavailableError] = TTLCache(
    maxsize=CACHE_MAX_ENTRIES, ttl=NEGATIVE_CACHE_TTL_SECONDS
)
#: Single-flight: the in-flight resolve per key (concurrent callers await it).
_inflight: dict[CacheKey, asyncio.Future[ProviderSlot]] = {}


def clear_provider_slot_cache() -> None:
    """Drop every cached slot (tests; a tenant rotating a key mid-run waits ≤300 s)."""
    _cache.clear()
    _runless_cache.clear()
    _negative_cache.clear()
    _inflight.clear()


# ── the orchestrator client ─────────────────────────────────────────────────


def _error_code(response: httpx.Response) -> tuple[str | None, str | None]:
    """``(code, message)`` from a FastAPI ``{"detail": ...}`` / flat error body."""
    try:
        body = response.json()
    except ValueError:
        return None, None
    candidates: list[object] = [body]
    if isinstance(body, Mapping):
        candidates.append(cast("Mapping[str, object]", body).get("detail"))
        candidates.append(cast("Mapping[str, object]", body).get("error"))
    for item in candidates:
        if isinstance(item, Mapping):
            mapping = cast("Mapping[str, object]", item)
            code = mapping.get("code") or mapping.get("error")
            message = mapping.get("message") or mapping.get("reason")
            if isinstance(code, str):
                return code, message if isinstance(message, str) else None
        elif isinstance(item, str) and item == BYO_PROVIDER_REQUIRED:
            return item, None
    return None, None


def _short(text: str | None, limit: int = 200) -> str:
    return " ".join((text or "").split())[:limit]


def _require_cell_lane(slot: str, url: str) -> None:
    """Fail closed unless this service can put a cell token on ``url``."""
    if cell_audience_for_url(url) is None:
        raise ProviderSlotUnavailableError(
            slot, "the orchestrator endpoint is not an in-cell peer (no cell token possible)"
        )
    try:
        signer = cell_signer_configured()
    except Exception as exc:  # noqa: BLE001 - half-configured signer: fail closed, typed
        raise ProviderSlotUnavailableError(
            slot, f"cell signing is misconfigured on this service ({type(exc).__name__})"
        ) from None
    if not signer:
        raise ProviderSlotUnavailableError(
            slot, "cell signing is not configured on this service (CELL_JWT_SIGNING_KEY)"
        )


def _parse_ok(slot: str, response: httpx.Response) -> ProviderSlot:
    try:
        payload = response.json()
    except ValueError:
        raise ProviderSlotUnavailableError(
            slot, "the orchestrator answered 200 with a non-JSON body"
        ) from None
    return ProviderSlot.from_payload(payload, expected_slot=slot)


class ProviderSlotClient:
    """Pooled client for the orchestrator resolve route (cell-JWT lane only).

    The pooled ``httpx.AsyncClient`` carries NO credential: each request gets a
    fresh cell token from the bound context (``cell_token_event_hook``) and
    nothing else identity-shaped. ``transport`` is a test seam.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = RESOLVE_TIMEOUT_S,
    ) -> None:
        self.base_url = (base_url or service_url(ORCHESTRATOR)).rstrip("/")
        self._transport = transport
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def resolve(self, slot: str, *, run_id: str | None) -> ProviderSlot:
        """One resolve (an idempotent read): 502/503/504 + connect failures retried."""
        url = f"{self.base_url}{RESOLVE_PATH}"
        _require_cell_lane(slot, url)
        body = {"slot": slot, "run_id": run_id}
        client = self._get_client()
        slept = 0.0
        attempt = 0
        while True:
            attempt += 1
            request = client.build_request("POST", url, json=body)
            await cell_token_event_hook(request)
            await trace_event_hook(request)
            if CELL_TOKEN_HEADER not in request.headers:
                # Unvouched / unbound tenants were refused before; this is defence.
                raise ProviderSlotUnavailableError(slot, "no cell token could be minted")
            failure: str
            response: httpx.Response | None = None
            try:
                response = await client.send(request)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                failure = type(exc).__name__
            except httpx.TransportError as exc:
                raise ProviderSlotUnavailableError(
                    slot, f"the orchestrator call failed ({type(exc).__name__})", transient=True
                ) from None
            else:
                if response.status_code == 200:
                    return _parse_ok(slot, response)
                if response.status_code not in _RETRYABLE_RESOLVE_STATUS:
                    raise self._status_error(slot, response)
                failure = f"HTTP {response.status_code}"
            delay = self._delay(attempt, response)
            if attempt >= RESOLVE_MAX_ATTEMPTS or slept + delay > RESOLVE_RETRY_BUDGET_S:
                raise ProviderSlotUnavailableError(
                    slot,
                    f"the orchestrator is unavailable ({failure}) after {attempt} attempts",
                    status=response.status_code if response is not None else None,
                    transient=True,
                )
            slept += delay
            logger.warning(
                "provider slot resolve %s for slot=%s (attempt %d), retrying in %.1fs",
                failure,
                slot,
                attempt,
                delay,
            )
            await _sleep(delay)

    @staticmethod
    def _delay(attempt: int, response: httpx.Response | None) -> float:
        base = min(RESOLVE_RETRY_DELAY_S * 2 ** (attempt - 1), RESOLVE_RETRY_MAX_DELAY_S)
        # Jitter for a retry backoff, never a secret. `noqa` silences ruff and
        # `nosec` silences bandit — bandit does not read `noqa`, and without the
        # second marker its B311 finding reds the pre-commit hook on every
        # commit to this repo (measured 2026-11-17: six blocked attempts).
        delay = base * (0.5 + random.random() / 2)  # noqa: S311  # nosec B311
        if response is not None:
            honoured = _retry._parse_retry_after(response.headers.get("Retry-After"))
            if honoured is not None:
                delay = max(delay, min(honoured, RESOLVE_RETRY_MAX_DELAY_S))
        return delay

    @staticmethod
    def _status_error(
        slot: str, response: httpx.Response
    ) -> ProviderSlotRequiredError | ProviderSlotUnavailableError | ProviderSlotConfigError:
        status = response.status_code
        code, message = _error_code(response)
        if status == 403 and code == BYO_PROVIDER_REQUIRED:
            return ProviderSlotRequiredError(slot, message)
        if code in (CELL_LANE_REQUIRED, UNKNOWN_SLOT):
            return ProviderSlotConfigError(slot, code, status=status)
        if status == 404:
            if code != RUN_NOT_FOUND:
                # Not the orchestrator's structured error: the route is missing (an
                # old pod mid-roll) — transient, NOT "run not found".
                return ProviderSlotUnavailableError(
                    slot,
                    "the orchestrator does not serve the resolve route (yet)",
                    status=404,
                    transient=True,
                )
            return ProviderSlotUnavailableError(
                slot, "the run was not found for this tenant", status=404
            )
        if status == 409:
            return ProviderSlotUnavailableError(
                slot, _short(message) or "the run's provider cannot be used", status=409
            )
        return ProviderSlotUnavailableError(
            slot,
            f"the orchestrator refused the resolve (HTTP {status}"
            + (f", {code}" if code else "")
            + ")",
            status=status,
            transient=status >= 500,
        )


_default_client: ProviderSlotClient | None = None


def get_provider_slot_client() -> ProviderSlotClient:
    """The process-wide resolve client (lazy)."""
    global _default_client
    if _default_client is None:
        _default_client = ProviderSlotClient()
    return _default_client


def set_provider_slot_client(client: ProviderSlotClient | None) -> None:
    """Replace the process-wide client (tests; ``None`` resets to lazy default)."""
    global _default_client
    _default_client = client


def platform_slot(slot: str) -> ProviderSlot:
    """The platform default for ``slot`` without asking anyone (no tenant bound)."""
    return ProviderSlot(slot=slot, source="deployment", kind="platform")


def current_run_id() -> str | None:
    """The run this call belongs to — same rule as ``WMClient``'s ``X-Job-Id``:
    the bound w2v job (``X-W2V-Job-Id``), else the correlation id; ``None`` if neither."""
    return resolve_run_id()


async def _resolve_shared(
    key: CacheKey, slot: str, run_id: str | None, client: ProviderSlotClient | None
) -> ProviderSlot:
    """Single-flight per key: one orchestrator call, every concurrent caller awaits it."""
    pending = _inflight.get(key)
    if pending is not None:
        return await asyncio.shield(pending)
    future: asyncio.Future[ProviderSlot] = asyncio.get_running_loop().create_future()
    _inflight[key] = future
    try:
        resolved = await (client or get_provider_slot_client()).resolve(slot, run_id=run_id)
    except BaseException as exc:
        if isinstance(exc, ProviderSlotUnavailableError):
            _negative_cache[key] = exc
        if isinstance(exc, Exception):
            future.set_exception(exc)
            future.exception()  # retrieved: no "never retrieved" warning without followers
        else:
            future.cancel()
        raise
    else:
        (_cache if run_id is not None else _runless_cache)[key] = resolved
        future.set_result(resolved)
        return resolved
    finally:
        _inflight.pop(key, None)


async def resolve_provider_slot(
    slot: str, *, client: ProviderSlotClient | None = None
) -> ProviderSlot:
    """The current run's provider for ``slot``.

    * inbound request arrived through a tenant-provider hop → :class:`ProviderHopRefusedError`;
    * NO tenant bound (cron, service-originated work) → :func:`platform_slot`, no call
      (no tenant ⇒ the platform, exactly as today);
    * bound but UNVOUCHED tenant → ``UnvouchedTenantError`` (its secret is never fetched);
    * else the orchestrator's answer: cached 300 s per (tenant, run, slot), 30 s when no
      run is bound, single-flight per key, ``ProviderSlotUnavailableError`` negative-cached 5 s.

    Callers check :func:`provider_slots_enforced` first; this function is the mechanism,
    the switch is the policy.
    """
    if not slot or not slot.strip():
        raise ValueError("slot must be a non-empty slot name")
    if is_provider_hop():
        raise ProviderHopRefusedError(slot)
    tenant = get_tenant_id_from_context()
    if tenant is None:
        return platform_slot(slot)
    if not is_tenant_vouched():
        raise UnvouchedTenantError(
            f"provider slot {slot!r}: the bound tenant's provenance is not vouched; "
            "its provider (and secret) is never resolved for it"
        )
    run_id = current_run_id()
    key: CacheKey = (tenant, run_id, slot)
    cached = (_cache if run_id is not None else _runless_cache).get(key)
    if cached is not None:
        return cached
    negative = _negative_cache.get(key)
    if negative is not None:
        raise negative
    return await _resolve_shared(key, slot, run_id, client)


__all__ = [
    "CACHE_TTL_SECONDS",
    "MODE_ENFORCE",
    "MODE_OFF",
    "RESOLVE_PATH",
    "SWITCH_ENV",
    "WIRE_KEYS",
    "ProviderSlot",
    "ProviderSlotClient",
    "ProviderSlotRef",
    "ProviderSlotsMode",
    "SlotSource",
    "clear_provider_slot_cache",
    "current_run_id",
    "get_provider_slot_client",
    "platform_slot",
    "provider_slots_enforced",
    "provider_slots_mode",
    "resolve_provider_slot",
    "set_provider_slot_client",
]
