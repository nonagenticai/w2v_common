"""Base async HTTP client for downstream microservices."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import ssl
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import httpx

from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.auth.keycloak import _looks_like_jwt
from w2v_common.auth.outbound import is_metered_destination
from w2v_common.observability.metrics import record_downstream_call
from w2v_common.observability.request_context import trace_event_hook

if TYPE_CHECKING:
    from w2v_common.auth.tenant_licence_lane import TenantLicenceLane

logger = logging.getLogger(__name__)


def _observe_downstream[T](
    func: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """Time + classify one logical ``BaseServiceClient`` request for metrics.

    Wraps the whole request (including its internal retries) so exactly ONE
    ``w2v_downstream_*`` observation is emitted per logical call, labelled by
    the callee (``self.service_name``), method, and outcome. The retry/raise
    behaviour of the wrapped method is untouched — this only observes.

    ``httpx.TimeoutException`` is a subclass of ``httpx.TransportError`` so it
    is matched first. ``BaseException`` (e.g. ``asyncio.CancelledError`` on pod
    drain) is intentionally NOT recorded — a cancelled call is not a downstream
    outcome.
    """

    @functools.wraps(func)
    async def wrapper(self: BaseServiceClient, *args: Any, **kwargs: Any) -> T:
        # `method` is the first positional arg, or a keyword — resolve both so
        # the wrapper is a transparent pass-through for any call convention.
        method = str(args[0]) if args else str(kwargs.get("method", ""))
        t0 = time.monotonic()
        try:
            result = await func(self, *args, **kwargs)
        except httpx.HTTPStatusError:
            record_downstream_call(self.service_name, method, "http_error", time.monotonic() - t0)
            raise
        except httpx.TimeoutException:
            record_downstream_call(self.service_name, method, "timeout", time.monotonic() - t0)
            raise
        except httpx.TransportError:
            record_downstream_call(
                self.service_name, method, "transport_error", time.monotonic() - t0
            )
            raise
        except Exception:
            record_downstream_call(self.service_name, method, "error", time.monotonic() - t0)
            raise
        record_downstream_call(self.service_name, method, "ok", time.monotonic() - t0)
        return result

    return wrapper


_POLL_SUCCESS_STATUSES = frozenset({"completed", "completed_with_errors", "partial"})
_POLL_FAILURE_STATUSES = frozenset({"failed", "error"})
# Statuses that legitimately mean "keep polling". Anything outside this set and the
# terminal sets is a GAP -- the client cannot tell done from working -- so it is
# logged once per status rather than silently absorbed into the deadline.
_POLL_NONTERMINAL_STATUSES = frozenset(
    {"pending", "queued", "processing", "running", "in_progress", "started", "retrying"}
)
_POLL_BACKOFF_MULTIPLIER = 1.5
_POLL_MAX_INTERVAL = 10.0

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0

# --- Rate-limit (429) retry budget -----------------------------------------
# A 429 is a SCHEDULING signal, not a fault: the downstream is healthy and is
# telling us to come back later. The 5xx budget (3 attempts, linear 1s -> 1s +
# 2s = 3s of total waiting) can never outlive a limiter whose window is 60s,
# so every 429 burst exhausted the budget inside ~5% of the window and failed
# the run. Rate limits therefore get their own, deliberately patient budget:
# 6 attempts with a linear 5s base sleeps 5 + 10 + 15 + 20 + 25 = 75s, which
# comfortably outlasts a 60s sliding window and gives a per-tenant
# concurrency slot time to free.
#
# These are module-level so a consumer can override/monkeypatch them exactly
# the way ``_MAX_RETRIES`` / ``_RETRY_DELAY`` already allow. They are read at
# call time, never captured at import time.
_RATE_LIMIT_STATUS = 429
_RATE_LIMIT_MAX_RETRIES = 6
_RATE_LIMIT_RETRY_DELAY = 5.0

# Statuses whose ``Retry-After`` header is honoured. Deliberately 429-ONLY:
# honouring it on 5xx would silently re-time every retry for ~20 consumer
# services against servers that already emit the header on 503, and 5xx
# semantics must stay byte-for-byte identical.
_RETRY_AFTER_STATUS_CODES = frozenset({_RATE_LIMIT_STATUS})

# Upper bound on a SINGLE honoured ``Retry-After``. A downstream may
# legitimately say "come back in an hour"; parking a request coroutine (and the
# DB session / job slot behind it) for that long is worse than surfacing the
# 429 and letting the caller's own stage-level retry take over. 60s is one full
# limiter window -- long enough to clear any real queue, and small enough that
# the cumulative budget below can still afford one such wait plus a follow-up.
#
# Deliberately kept BELOW ``_RETRY_SLEEP_BUDGET``: a per-sleep cap larger than
# the total budget could never actually be slept, which would make this
# constant dead.
_RETRY_AFTER_MAX_DELAY = 60.0

# Upper bound on the CUMULATIVE time one logical ``_request`` / ``_request_raw``
# call may spend asleep between attempts. ``_RETRY_AFTER_MAX_DELAY`` bounds a
# SINGLE sleep; without a total, a downstream repeating ``Retry-After: 120``
# parks one call for 5 x 120 = 600s -- five times the SHORTEST job lease in the
# fleet (``video_compiler_service.job_lease_seconds`` = 120s; tts_gateway and
# html_render_service both use 180s), so the caller's job gets reclaimed out
# from under a coroutine that is still running.
#
# 75s is chosen because it is EXACTLY what the default 429 ladder already
# spends (5 + 10 + 15 + 20 + 25), so this budget can never truncate existing
# behaviour -- it only binds when a server-supplied ``Retry-After`` inflates
# the ladder. It comfortably outlasts a 60s sliding limiter window (+25%) and
# leaves 45s of the 120s lease for the six requests themselves, whose own
# duration is bounded separately by ``_clamp_timeout``.
_RETRY_SLEEP_BUDGET = 75.0


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an RFC 9110 ``Retry-After`` header into a delay in seconds.

    Both forms are accepted: delta-seconds (``"5"``) and HTTP-date
    (``"Wed, 21 Oct 2026 07:28:00 GMT"``). The result is clamped to
    ``_RETRY_AFTER_MAX_DELAY``; it may be NEGATIVE (a date in the past), and
    flooring is the CALLER's job -- see ``_retry_sleep_seconds``, which floors
    at the default backoff so the header can never make us more aggressive
    than we would have been without it.

    Returns ``None`` when the header is absent, empty, or unparseable so the
    caller falls back to its own backoff. This function NEVER raises -- a
    malformed header from a downstream service must not be able to break the
    retry path. Every operation below is inside a guard, because hostile input
    reaches all of them:

    * ``int(raw)`` raises ``ValueError`` on non-digits AND on more than 4300
      digits (CPython's ``int_max_str_digits``).
    * ``float(int(raw))`` raises ``OverflowError`` -- NOT a ``ValueError`` --
      for a large-but-under-4300-digit integer such as 400 nines. This was the
      escape hatch that let an attacker-controlled header raise out of the
      retry path into ~20 consumer services.
    * ``parsedate_to_datetime`` raises ``ValueError`` for out-of-range years,
      out-of-range day/time fields, offsets beyond +/-24h, and non-ASCII junk,
      and ``TypeError`` for a non-string.
    * subtracting a NAIVE datetime from an aware one raises ``TypeError``, and
      datetime arithmetic can raise ``OverflowError`` at the type's bounds.
    """
    if not value:
        return None
    raw = value.strip()
    try:
        seconds = float(int(raw))
    except ValueError, OverflowError:
        # Not delta-seconds; try the HTTP-date form.
        try:
            when = parsedate_to_datetime(raw)
            if when is None:
                return None
            if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
                # RFC 9110 HTTP-dates are always GMT; a naive result means the
                # sender omitted the zone (or sent the "-0000" unknown-zone
                # form, which yields a tzinfo whose utcoffset is None).
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except TypeError, ValueError, OverflowError:
            return None
    # No isfinite() guard: neither branch can produce NaN/inf. float(int(...))
    # overflows rather than returning inf, and timedelta.total_seconds() is
    # finite across the whole datetime range. A guard that cannot fire implies
    # coverage that does not exist.
    return min(seconds, _RETRY_AFTER_MAX_DELAY)


def _retry_budget(status_code: int) -> tuple[int, float]:
    """Return ``(max_attempts, base_delay)`` for a retryable status.

    429 gets its own longer budget; every other retryable status keeps the
    historical 5xx budget unchanged.
    """
    if status_code == _RATE_LIMIT_STATUS:
        return _RATE_LIMIT_MAX_RETRIES, _RATE_LIMIT_RETRY_DELAY
    return _MAX_RETRIES, _RETRY_DELAY


def _retry_sleep_seconds(response: httpx.Response, attempt: int) -> float:
    """How long to wait before replaying ``response``'s request.

    A server-supplied ``Retry-After`` wins when the status opts into it
    (429 only); otherwise the historical linear backoff applies.

    ``Retry-After`` may only ever make us WAIT LONGER, never shorter: the
    honoured value is floored at the backoff we would have used anyway. A
    hostile or buggy upstream answering ``Retry-After: 0`` (or a date in the
    past) would otherwise turn the 429 budget into six back-to-back requests
    with zero backoff -- strictly more aggressive than the client was before
    the header was honoured at all, and a free amplification primitive.
    """
    _, base_delay = _retry_budget(response.status_code)
    default_delay = base_delay * attempt
    if response.status_code in _RETRY_AFTER_STATUS_CODES:
        honoured = _parse_retry_after(response.headers.get("Retry-After"))
        if honoured is not None:
            return max(honoured, default_delay)
    return default_delay


def _max_attempts() -> int:
    """Outer loop bound: the largest per-status budget in force.

    Each retry branch still enforces its OWN budget, so a 5xx cannot use a
    429's extra attempts.
    """
    return max(_MAX_RETRIES, _RATE_LIMIT_MAX_RETRIES)


# RFC 9110 idempotent methods. ``POST`` and ``PATCH`` are excluded — replaying
# either after a partial-write failure can silently double-create resources.
# Callers that want POST/PATCH to participate in retries must pass an
# ``idempotency_key`` so the downstream service can dedupe replays.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

# Transport errors that PROVE the request never reached the application: no
# connection was established, so not one byte of the request was ever handed to
# a server. Replaying these is safe for ANY method — there is nothing on the
# other side that could have acted on the first attempt.
_PRE_SEND_TRANSPORT_ERRORS: tuple[type[httpx.TransportError], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

# Transport errors that LOOK like a connection reset but are AMBIGUOUS: they are
# raised on an ESTABLISHED connection, after the request bytes have already gone
# out. The dominant benign cause is a stale keepalive after a downstream pod
# restart — the server closed an idle connection and never saw the request. But
# the exact same exception is raised when the server received the COMPLETE
# request, began working (and billing), and *then* died before writing a
# response byte. httpx cannot tell those apart, and neither can we.
#
# This tuple used to be merged into the one above under the comment "the request
# never reached the application". That premise is false, and it is demonstrable:
# a server that reads the whole body and then closes the socket without
# answering makes httpx raise ``RemoteProtocolError`` — see
# ``tests/unit/test_client_retry_metered_replay.py::test_remote_protocol_error_
# is_raised_after_the_server_received_the_whole_body``. Replaying a bare POST on
# that error re-buys the work.
_AMBIGUOUS_TRANSPORT_ERRORS: tuple[type[httpx.TransportError], ...] = (
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)

_CONNECTION_RESET_ERRORS: tuple[type[httpx.TransportError], ...] = (
    *_PRE_SEND_TRANSPORT_ERRORS,
    *_AMBIGUOUS_TRANSPORT_ERRORS,
)

# Deliberately EXCLUDED from both: ``ReadTimeout``/``WriteTimeout`` (the server
# may be mid-process — replaying a non-idempotent request could double-act) and
# ``LocalProtocolError`` (a bug on our side, not a transient).


def _may_replay_transport_error(
    exc: BaseException,
    *,
    retryable: bool,
    destination: str,
) -> bool:
    """Whether ``exc`` may be replayed for this call.

    ``retryable`` is the method/idempotency-key gate already computed by the
    caller: True for an RFC 9110 idempotent method, or for any method carrying
    an ``Idempotency-Key`` the downstream can dedupe on. When it is True the
    replay is safe by construction and the error class does not matter.

    When it is False the request is a bare POST/PATCH with nothing to dedupe on,
    so the ONLY errors we may replay are the ones that prove the server never
    saw it. An ambiguous error (see ``_AMBIGUOUS_TRANSPORT_ERRORS``) may be
    replayed against a plain backend — re-doing free work is merely wasteful —
    but NEVER against a metered peer, where the first attempt may already have
    been billed and the replay buys it a second time.
    """
    if retryable:
        return isinstance(exc, _CONNECTION_RESET_ERRORS)
    if isinstance(exc, _PRE_SEND_TRANSPORT_ERRORS):
        return True
    if not isinstance(exc, _AMBIGUOUS_TRANSPORT_ERRORS):
        return False
    return not is_metered_destination(destination)


#: Credential key NAMES, matched as a key-name SUFFIX (see ``_CREDENTIAL_KEY``).
#: ``access_key`` also covers ``access_key_id`` / ``AccessKeyId``: AWS treats the
#: key id as non-secret, but it is half of a credential pair and masking it costs
#: nothing — over-masking an identifier beats leaking a half-credential.
_CREDENTIAL_NAMES = (
    r"api[_-]?key|apikey|client[_-]?secret|password|passwd|secret|"
    r"access[_-]?key(?:[_-]?id)?|private[_-]?key|refresh[_-]?token|session[_-]?key|"
    r"access[_-]?token|auth[_-]?token|id[_-]?token|bearer[_-]?token"
)
#: A key whose name ENDS in a credential name: ``access_key``, but also
#: ``minio_access_key``, ``aws_secret_access_key``, ``browserstack_access_key``,
#: ``db-password``. Up to 0.93.x the rule was ``\b(access[_-]?key)``, and ``_`` is
#: a word character, so ``\b`` never fired inside ``browserstack_access_key`` —
#: the orchestrator had to install its own log filter to cover that leak.
#: ``[\w-]{0,64}?`` is the prefix, with NO leading ``\b``: bounded so the scan
#: stays linear, and unanchored so a key with a >64-char prefix still has its
#: value masked (the match just starts later in the name). The name must still
#: be the END of the key: the value rule below requires ``=``/``:`` right after
#: it, and ``_CREDENTIAL_NAME_RE`` a ``\b``, so ``secret_name`` /
#: ``password_policy`` / ``max_tokens`` are untouched.
_CREDENTIAL_KEY = rf"[\w-]{{0,64}}?(?:{_CREDENTIAL_NAMES})"

# Match ``Bearer <token>``, ``token=...``, and ``Authorization: ...`` so we
# never echo a secret-bearing fragment of an upstream error body into logs.
# Group 1 carries the prefix we want to keep; the value is replaced with
# ``[REDACTED]``. Keep this tight — over-redaction makes logs useless.
#: ``(pattern, replacement)`` — a shared template does not fit, because the
#: pydantic rule below has to keep TWO groups.
_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(Bearer)\s+[^\s\"',}\]]+"), r"\1 [REDACTED]"),
    (re.compile(r"(?i)\b(token)[\"']?\s*[=:]\s*[\"']?[^\s&\"',}\]]+"), r"\1 [REDACTED]"),
    (
        re.compile(r"(?i)\b(Authorization)\s*:\s*[^\s\"',}\]]+(?:\s+[^\s\"',}\]]+)?"),
        r"\1 [REDACTED]",
    ),
    # Credential NAMES this estate actually moves over HTTP, beyond the bearer
    # lane above. The orchestrator PUTs a tenant's BYO provider key to
    # worker_management, and none of the three patterns above matches
    # ``"api_key": "sk-..."``.
    (
        re.compile(
            rf"(?i)({_CREDENTIAL_KEY})"
            r"[\"']?\s*[=:]\s*[\"']?[^\s&\"',}\]]+"
        ),
        r"\1 [REDACTED]",
    ),
    # Presigned-URL signatures. A storage 4xx echoes the URL it refused, and the
    # signature in that query string IS the credential.
    (
        re.compile(
            r"(?i)\b(X-Amz-Signature|X-Amz-Credential|Signature|AWSAccessKeyId)="
            r"[^\s&\"',}\]]+"
        ),
        r"\1=[REDACTED]",
    ),
)

#: FastAPI/pydantic 422 bodies do not put the value next to its name. They carry
#: ``{"loc": ["body", "api_key"], "input": "sk-live-..."}`` — the credential is  # pragma: allowlist secret
#: under ``input``, an innocuous key, and every name-based pattern above misses
#: it. This is the shape the orchestrator's BYO-key PUT actually produces, i.e.
#: the leak that matters most, so it gets its own rule: when a body names a
#: credential ANYWHERE, every ``input`` value in it is redacted.
_CREDENTIAL_NAME_RE = re.compile(rf"(?i)(?:{_CREDENTIAL_KEY}\b|\btoken\b)")
_PYDANTIC_INPUT_RE = re.compile(r"(?i)(\"input\"\s*:\s*)\"[^\"]*\"")


def _redact_for_log(text: str) -> str:
    """Mask Bearer tokens / Authorization headers / token= parameters in ``text``.

    Called before any log line that embeds an upstream response body. Upstream
    services occasionally echo the bearer/license token they rejected — we
    must not capture those in log aggregators.
    """
    if not text:
        return text
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    if _CREDENTIAL_NAME_RE.search(text):
        text = _PYDANTIC_INPUT_RE.sub(r'\1"[REDACTED]"', text)
    return text


def raise_for_status_with_body(response: httpx.Response, service_name: str) -> None:
    """``raise_for_status`` that preserves the response body in the error.

    Bare ``response.raise_for_status()`` drops the body, which is the ONE thing
    that explains an upstream rejection (validation detail, provider 4xx/5xx).
    Embed the redacted, truncated body into the raised ``HTTPStatusError`` so it
    surfaces verbatim at every w2v layer instead of a blank reason. Use this for
    direct ``client.get/post`` calls that bypass ``BaseServiceClient._request``
    (which already logs+redacts the body before raising).
    """
    if response.status_code < 400:
        return
    detail = _redact_for_log(response.text[:2000])
    logger.error("%s returned %d: %s", service_name, response.status_code, detail)
    raise httpx.HTTPStatusError(
        f"{service_name} returned {response.status_code}: {detail}",
        request=response.request,
        response=response,
    )


# Caller-bounds for client construction and per-call poll overrides.
# A ``timeout=None`` from a caller is httpx's "no timeout" signal — a
# silent foot-gun in long-running orchestrators. Cap defaults at sane
# upper bounds so misconfiguration surfaces loudly rather than hanging.
_MIN_TIMEOUT = 1.0
_MAX_TIMEOUT = 600.0
_MAX_POLL_INTERVAL = 60.0
_MAX_POLL_ATTEMPTS = 500


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment at call time; fall back on missing/invalid."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-float %s=%r; using %.1f", name, raw, default)
        return default


def _clamp_timeout(value: float | None) -> float:
    ceiling = _env_float("CLIENT_MAX_TIMEOUT_S", _MAX_TIMEOUT)
    if value is None:
        raise ValueError(
            "BaseServiceClient timeout must be a positive number, not None "
            "(httpx interprets None as 'no timeout' — refusing to expose that "
            "foot-gun to callers)."
        )
    if value < _MIN_TIMEOUT:
        raise ValueError(f"BaseServiceClient timeout must be >= {_MIN_TIMEOUT}s, got {value}.")
    if value > ceiling:
        raise ValueError(
            f"BaseServiceClient timeout must be <= {ceiling}s "
            f"(server-side timeouts make longer values pointless), got {value}."
        )
    return value


def _clamp_poll_interval(value: float) -> float:
    if value <= 0:
        raise ValueError(f"BaseServiceClient poll_interval must be > 0, got {value}.")
    if value > _MAX_POLL_INTERVAL:
        raise ValueError(
            f"BaseServiceClient poll_interval must be <= {_MAX_POLL_INTERVAL}s, got {value}."
        )
    return value


def _clamp_max_poll_attempts(value: int) -> int:
    if value <= 0:
        raise ValueError(f"BaseServiceClient max_poll_attempts must be > 0, got {value}.")
    if value > _MAX_POLL_ATTEMPTS:
        raise ValueError(
            f"BaseServiceClient max_poll_attempts must be <= {_MAX_POLL_ATTEMPTS}, got {value}."
        )
    return value


class JobCancelledError(RuntimeError):
    """Raised by ``poll_job`` when the downstream job reports ``status="cancelled"``.

    Subclasses ``RuntimeError`` (the same base ``poll_job`` already raises for
    ``failed``/``error``) so existing ``except RuntimeError:`` callers keep
    catching it without any change -- this is additive, not a breaking type
    change. Callers that care can add a narrower
    ``except JobCancelledError:`` branch ahead of the generic one to treat a
    deliberate cancellation differently from an actual failure (e.g. skip
    retry/alerting for a cancellation the caller itself triggered).
    """


class JobSubmissionError(RuntimeError):
    """Raised by ``submit_job`` when a 2xx response carries no usable job ID.

    The job was **accepted** -- it is running downstream right now -- but the
    handle needed to supervise it is not in the envelope. Returning ``""`` here
    (the behaviour up to 0.67.25) made every caller poll ``<endpoint>/``, so the
    run died later as a 404/timeout that named the POLL url and never the
    submit; meanwhile the real job ran to completion unsupervised.

    Trigger shapes: a renamed or nested id field, a partial-success envelope, or
    an intermediary (istio / cloudflared) answering ``{}`` with a 2xx. A list or
    an empty body already raises inside ``response.json()``, so ``{}``-shaped
    dicts are the gap this closes.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError:`` stage
    handlers keep catching it -- additive, not a new escape type.
    """


class BaseServiceClientRedirectError(RuntimeError):
    """Raised when a downstream service returns a 3xx redirect.

    ``BaseServiceClient`` deliberately does **not** follow redirects: a
    redirect from a microservice almost always indicates a route-shape
    bug (e.g. ``POST /api/foo`` 307-ing to ``POST /api/foo/`` because the
    callee declared ``@router.post("/")`` under a ``/api/foo`` prefix).
    Following the redirect would silently mask the bug while inflating
    request count and latency. Surfacing it as an explicit error forces
    the callee to be fixed.

    Attributes:
        service_name: Logical name of the downstream service.
        method: HTTP method that produced the redirect.
        endpoint: Path component of the redirected request.
        status_code: 3xx status code returned by the service.
        location: Value of the ``Location`` response header, if any.
    """

    def __init__(
        self,
        service_name: str,
        method: str,
        endpoint: str,
        status_code: int,
        location: str | None,
    ) -> None:
        self.service_name = service_name
        self.method = method
        self.endpoint = endpoint
        self.status_code = status_code
        self.location = location
        super().__init__(
            f"{service_name}: {method} {endpoint} returned {status_code} "
            f"redirect to {location!r}; BaseServiceClient does not follow "
            f"redirects (likely a route-shape bug in the callee)."
        )


class BaseServiceClient:
    """Base async HTTP client with retry logic for downstream services."""

    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 30.0,
        license_token: str | None = None,
        ssl_context: ssl.SSLContext | bool = True,
        poll_interval: float = 2.0,
        max_poll_attempts: int = 150,
        *,
        licence_lane: TenantLicenceLane | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_name = service_name
        self.timeout = _clamp_timeout(timeout)
        self.license_token = license_token
        self.ssl_context = ssl_context
        self.poll_interval = _clamp_poll_interval(poll_interval)
        self.max_poll_attempts = _clamp_max_poll_attempts(max_poll_attempts)
        #: The out-of-cell hop's auth lane (``<HOP>_AUTH_MODE``). Only a lane in
        #: ``tenant_licence`` mode changes anything: the pooled client then carries
        #: NO baked token and every request carries the bound tenant's minted
        #: licence (``TenantLicenceLane.event_hooks``). ``None`` or a ``shared``
        #: lane = the ``license_token`` behaviour below, byte for byte.
        self.licence_lane = licence_lane
        self._client: httpx.AsyncClient | None = None

    @property
    def _tenant_licence(self) -> bool:
        return self.licence_lane is not None and self.licence_lane.tenant_licence

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it lazily if needed."""
        if self._client is None or self._client.is_closed:
            if self._tenant_licence and self.licence_lane is not None:
                lane_hooks = self.licence_lane.event_hooks()
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    verify=self.ssl_context,
                    follow_redirects=False,
                    event_hooks={
                        "request": [*lane_hooks["request"], trace_event_hook],
                        "response": lane_hooks["response"],
                    },
                )
                return self._client
            headers: dict[str, str] = {}
            if self.license_token:
                # JWT-shaped tokens (3 dot-delimited base64url segments) are
                # Keycloak bearers and travel via Authorization. Opaque
                # license tokens travel via X-Tenant-Id: Bearer <token>,
                # which is the only header AuthMiddleware reads for license
                # tokens (see w2v_common.auth.middleware._extract_license_token).
                # Sending license tokens via Authorization is a silent no-op
                # downstream and forces reliance on the trusted-network
                # fallback (closed by the 2026-04-18 incident fix).
                if _looks_like_jwt(self.license_token):
                    headers["Authorization"] = f"Bearer {self.license_token}"
                else:
                    headers["X-Tenant-Id"] = f"Bearer {self.license_token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
                verify=self.ssl_context,
                # Redirects from a microservice almost always indicate a
                # route-shape bug (POST /api/foo -> 307 /api/foo/ etc.).
                # Surface them as BaseServiceClientRedirectError instead of
                # silently following them. See ``_raise_on_redirect``.
                follow_redirects=False,
                # Forward request-scoped identity ContextVars
                # (X-Tenant-Id / X-License-Id / X-Subject-Id) on every
                # outbound call so the downstream service sees the same
                # caller identity that the inbound AuthMiddleware bound.
                # ``trace_event_hook`` mirrors that pattern for the
                # request-id ContextVar so peer-service logs stay
                # correlated under a single X-Request-Id.
                event_hooks={"request": [identity_event_hook, trace_event_hook]},
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def set_transport(self, transport: httpx.AsyncBaseTransport) -> None:
        """Test seam: replace the underlying httpx transport.

        Production callers must NOT use this -- it bypasses any header /
        auth setup that ``_get_client`` would have configured. Use only in
        tests, with ``httpx.MockTransport``, to avoid poking the private
        ``_client`` attribute (which would require ``# type: ignore``).
        """
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            transport=transport,
            follow_redirects=False,
            timeout=self.timeout,
        )

    def _raise_on_redirect(
        self,
        response: httpx.Response,
        method: str,
        endpoint: str,
    ) -> None:
        """Raise ``BaseServiceClientRedirectError`` for any 3xx response.

        ``follow_redirects=False`` means httpx returns the redirect
        response untouched. We treat any 3xx as a hard error so that
        route-shape bugs in the callee surface loudly rather than being
        masked by silent redirect following.
        """
        if 300 <= response.status_code < 400:
            location = response.headers.get("location")
            logger.error(
                "%s: %s %s returned %d redirect to %r",
                self.service_name,
                method,
                endpoint,
                response.status_code,
                location,
            )
            raise BaseServiceClientRedirectError(
                service_name=self.service_name,
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                location=location,
            )

    async def health_check(self) -> dict[str, Any]:
        """Check downstream service health.

        In tenant_licence mode the probe goes out on a bare client: a readiness
        probe has no tenant and ``/health`` is licence-exempt, so the lane's hook
        (which refuses any request without a minted licence) must not run.
        """
        try:
            if self._tenant_licence:
                async with httpx.AsyncClient(
                    base_url=self.base_url, timeout=self.timeout, verify=self.ssl_context
                ) as bare:
                    response = await bare.get("/health")
                    response.raise_for_status()
                    data = response.json()
                return {"status": "healthy", "service": self.service_name, **data}
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            data = response.json()
            return {"status": "healthy", "service": self.service_name, **data}
        except (httpx.HTTPError, OSError) as exc:
            return {"status": "unhealthy", "service": self.service_name, "error": str(exc)}

    @_observe_downstream
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
        idempotency_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: URL path relative to base_url.
            json: Optional JSON body payload.
            params: Optional query parameters.
            idempotency_key: When set, the value is sent as the
                ``Idempotency-Key`` header on every attempt and unlocks retry
                for non-idempotent methods (POST/PATCH). Without it, retries
                are restricted to RFC 9110 idempotent methods to prevent
                silent double-creates on transient upstream errors.
            extra_headers: Per-request headers sent on every attempt (e.g.
                WMClient's ``X-App-Id`` / ``X-Job-Id`` caller attribution).
                ``Idempotency-Key`` from ``idempotency_key`` wins on a clash.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP error.
            httpx.TransportError: On network failure after all retries.
            RuntimeError: If all retries exhausted with no exception captured.
        """
        retryable = method.upper() in _IDEMPOTENT_METHODS or idempotency_key is not None
        headers: dict[str, str] | None = dict(extra_headers) if extra_headers else None
        if idempotency_key:
            headers = {**(headers or {}), "Idempotency-Key": idempotency_key}
        last_error: Exception | None = None
        # Cumulative retry sleep for THIS logical call. Per-sleep clamping is
        # not enough: a downstream that keeps answering ``Retry-After: 120``
        # would otherwise park one call for 600s and outlive the caller's job
        # lease. See ``_RETRY_SLEEP_BUDGET``.
        slept_total = 0.0
        logger.debug(
            "%s: starting %s %s",
            self.service_name,
            method,
            endpoint,
        )

        for attempt in range(1, _max_attempts() + 1):
            try:
                client = await self._get_client()
                response = await client.request(
                    method,
                    endpoint,
                    json=json,
                    params=params,
                    headers=headers,
                )

                # Per-status budget: 429 (a scheduling signal) gets more
                # attempts and longer waits than a 5xx (a fault), and honours
                # the server's own ``Retry-After``. 5xx keeps the historical
                # 3-attempt / linear-1s behaviour exactly.
                status_max_attempts, _ = _retry_budget(response.status_code)
                if (
                    response.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < status_max_attempts
                    and retryable
                ):
                    delay = _retry_sleep_seconds(response, attempt)
                    if slept_total + delay <= _RETRY_SLEEP_BUDGET:
                        slept_total += delay
                        logger.warning(
                            "%s returned %d (attempt %d/%d), retrying in %.1fs",
                            self.service_name,
                            response.status_code,
                            attempt,
                            status_max_attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Budget spent: stop retrying and surface the status below
                    # rather than outliving the caller's job lease.
                    logger.warning(
                        "%s returned %d (attempt %d/%d): next wait of %.1fs would "
                        "exceed the %.1fs retry sleep budget (%.1fs already slept); "
                        "surfacing the error",
                        self.service_name,
                        response.status_code,
                        attempt,
                        status_max_attempts,
                        delay,
                        _RETRY_SLEEP_BUDGET,
                        slept_total,
                    )

                # 3xx is fatal: with follow_redirects=False the response is
                # returned untouched. Surface as BaseServiceClientRedirectError.
                self._raise_on_redirect(response, method, endpoint)

                if response.status_code >= 400:
                    # Embed the redacted body in the RAISED error, not just this
                    # log line: a bare response.raise_for_status() drops it, so a
                    # caller that cannot read this pod's logs (e.g. the
                    # post-promote E2E validator) is left with only
                    # "500 Internal Server Error" and no cause. Mirrors
                    # raise_for_status_with_body() but keeps the method+endpoint
                    # context this layer already has.
                    detail = _redact_for_log(response.text[:1000])
                    logger.error(
                        "%s: %s %s returned %d: %s",
                        self.service_name,
                        method,
                        endpoint,
                        response.status_code,
                        detail,
                    )
                    raise httpx.HTTPStatusError(
                        f"{self.service_name} {method} {endpoint} returned "
                        f"{response.status_code}: {detail}",
                        request=response.request,
                        response=response,
                    )
                logger.info(
                    "%s: %s %s completed status=%d",
                    self.service_name,
                    method,
                    endpoint,
                    response.status_code,
                )
                return response.json()

            except httpx.TransportError as exc:
                last_error = exc
                # Whether this transport error may be REPLAYED. A bare
                # POST/PATCH may only replay an error that proves the server
                # never saw the request, or -- for an ambiguous
                # reset-shaped error -- only when the destination is not a
                # metered peer. See ``_may_replay_transport_error``.
                conn_reset = _may_replay_transport_error(
                    exc, retryable=retryable, destination=self.base_url
                )
                if attempt < _MAX_RETRIES and (retryable or conn_reset):
                    logger.warning(
                        "%s transport error (attempt %d/%d, conn_reset=%s): %s",
                        self.service_name,
                        attempt,
                        _MAX_RETRIES,
                        conn_reset,
                        exc,
                    )
                    # Counted against the same per-call budget so a mixed
                    # 429 + transport-error call cannot exceed it. This branch
                    # is structurally bounded at 1s + 2s = 3s, so the gate can
                    # never bite here on its own — the accounting exists to
                    # keep the total honest, not to change this budget.
                    slept_total += _RETRY_DELAY * attempt
                    await asyncio.sleep(_RETRY_DELAY * attempt)
                    continue
                logger.error(
                    "%s: %s %s failed after %d attempts: %s",
                    self.service_name,
                    method,
                    endpoint,
                    attempt,
                    exc,
                )
                raise

        raise last_error or RuntimeError(f"{self.service_name} request failed after retries")

    @_observe_downstream
    async def _request_raw(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
        idempotency_key: str | None = None,
    ) -> httpx.Response:
        """Like _request() but returns the raw httpx.Response.

        Use for binary downloads (audio, video, images) where calling
        ``.json()`` would fail. Same retry and error-handling logic.
        Retry gate: see ``_request`` — non-idempotent methods only retry
        when an ``idempotency_key`` is provided.
        """
        retryable = method.upper() in _IDEMPOTENT_METHODS or idempotency_key is not None
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        last_error: Exception | None = None
        # Cumulative retry sleep for THIS logical call. Per-sleep clamping is
        # not enough: a downstream that keeps answering ``Retry-After: 120``
        # would otherwise park one call for 600s and outlive the caller's job
        # lease. See ``_RETRY_SLEEP_BUDGET``.
        slept_total = 0.0
        logger.debug(
            "%s: starting %s %s (raw)",
            self.service_name,
            method,
            endpoint,
        )

        for attempt in range(1, _max_attempts() + 1):
            try:
                client = await self._get_client()
                response = await client.request(
                    method,
                    endpoint,
                    json=json,
                    params=params,
                    headers=headers,
                )

                # Per-status budget: 429 (a scheduling signal) gets more
                # attempts and longer waits than a 5xx (a fault), and honours
                # the server's own ``Retry-After``. 5xx keeps the historical
                # 3-attempt / linear-1s behaviour exactly.
                status_max_attempts, _ = _retry_budget(response.status_code)
                if (
                    response.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < status_max_attempts
                    and retryable
                ):
                    delay = _retry_sleep_seconds(response, attempt)
                    if slept_total + delay <= _RETRY_SLEEP_BUDGET:
                        slept_total += delay
                        logger.warning(
                            "%s returned %d (attempt %d/%d), retrying in %.1fs",
                            self.service_name,
                            response.status_code,
                            attempt,
                            status_max_attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Budget spent: stop retrying and surface the status below
                    # rather than outliving the caller's job lease.
                    logger.warning(
                        "%s returned %d (attempt %d/%d): next wait of %.1fs would "
                        "exceed the %.1fs retry sleep budget (%.1fs already slept); "
                        "surfacing the error",
                        self.service_name,
                        response.status_code,
                        attempt,
                        status_max_attempts,
                        delay,
                        _RETRY_SLEEP_BUDGET,
                        slept_total,
                    )

                # 3xx is fatal: with follow_redirects=False the response is
                # returned untouched. Surface as BaseServiceClientRedirectError.
                self._raise_on_redirect(response, method, endpoint)

                if response.status_code >= 400:
                    # Embed the redacted body in the RAISED error, not just this
                    # log line: a bare response.raise_for_status() drops it, so a
                    # caller that cannot read this pod's logs (e.g. the
                    # post-promote E2E validator) is left with only
                    # "500 Internal Server Error" and no cause. Mirrors
                    # raise_for_status_with_body() but keeps the method+endpoint
                    # context this layer already has.
                    detail = _redact_for_log(response.text[:1000])
                    logger.error(
                        "%s: %s %s returned %d: %s",
                        self.service_name,
                        method,
                        endpoint,
                        response.status_code,
                        detail,
                    )
                    raise httpx.HTTPStatusError(
                        f"{self.service_name} {method} {endpoint} returned "
                        f"{response.status_code}: {detail}",
                        request=response.request,
                        response=response,
                    )
                logger.info(
                    "%s: %s %s completed status=%d (raw)",
                    self.service_name,
                    method,
                    endpoint,
                    response.status_code,
                )
                return response

            except httpx.TransportError as exc:
                last_error = exc
                # Shares ``_request``'s replay gate — see
                # ``_may_replay_transport_error``.
                conn_reset = _may_replay_transport_error(
                    exc, retryable=retryable, destination=self.base_url
                )
                if attempt < _MAX_RETRIES and (retryable or conn_reset):
                    logger.warning(
                        "%s transport error (attempt %d/%d, conn_reset=%s): %s",
                        self.service_name,
                        attempt,
                        _MAX_RETRIES,
                        conn_reset,
                        exc,
                    )
                    # Counted against the same per-call budget so a mixed
                    # 429 + transport-error call cannot exceed it. This branch
                    # is structurally bounded at 1s + 2s = 3s, so the gate can
                    # never bite here on its own — the accounting exists to
                    # keep the total honest, not to change this budget.
                    slept_total += _RETRY_DELAY * attempt
                    await asyncio.sleep(_RETRY_DELAY * attempt)
                    continue
                logger.error(
                    "%s: %s %s failed after %d attempts: %s",
                    self.service_name,
                    method,
                    endpoint,
                    attempt,
                    exc,
                )
                raise

        raise last_error or RuntimeError(f"{self.service_name} request failed after retries")

    async def submit_job(
        self,
        endpoint: str,
        payload: dict,
        idempotency_key: str | None = None,
    ) -> str:
        """Submit an async job and return the service job ID.

        Args:
            endpoint: API endpoint to POST the job to.
            payload: Job request payload.
            idempotency_key: Optional dedupe key. When provided, the POST is
                retried on transient 429/5xx + transport errors with the same
                ``Idempotency-Key`` header so the downstream service can
                dedupe replays. Without it, POST is one-shot.

        Returns:
            The service-assigned job ID string. Never empty -- a handle that
            cannot address a job is refused rather than returned.

        Raises:
            JobSubmissionError: If the 2xx envelope carries no non-empty
                ``job_id``. The message names the submit endpoint and the
                envelope's key shape (keys only -- values may carry secrets) so
                the failure points at the submit, not at a later poll.
        """
        logger.info("%s: submitting job to %s", self.service_name, endpoint)
        result = await self._request(
            "POST", endpoint, json=payload, idempotency_key=idempotency_key
        )
        raw = result.get("job_id") if isinstance(result, dict) else None
        # An int id is a legitimate (if unusual) handle -- it addresses a job
        # fine once interpolated into the poll path -- so coerce rather than
        # refuse. bool is an int subclass and is never a handle.
        if isinstance(raw, int) and not isinstance(raw, bool):
            raw = str(raw)
        job_id = raw.strip() if isinstance(raw, str) else ""
        if not job_id:
            shape = sorted(result) if isinstance(result, dict) else type(result).__name__
            raise JobSubmissionError(
                f"{self.service_name} POST {endpoint} returned 2xx with no usable "
                f"'job_id' (got {raw!r}); envelope keys={shape}. The job may be "
                f"RUNNING downstream but is unsupervisable without a handle."
            )
        logger.info("%s: job submitted, service_job_id=%s", self.service_name, job_id)
        return job_id

    async def poll_job(
        self,
        endpoint: str,
        job_id: str,
        interval: float | None = None,
        timeout: float = 300.0,  # noqa: ASYNC109 — why: client-level poll deadline, not asyncio.timeout; httpx requests inside the loop carry their own timeouts
        max_attempts: int | None = None,
        extra_success_statuses: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Poll a downstream job until completion with adaptive backoff.

        The initial poll interval defaults to ``self.poll_interval``
        and increases by 1.5x after each poll, capped at 10 s.

        Args:
            endpoint: Base endpoint path (job_id appended automatically).
            job_id: Service job ID to poll.
            interval: Initial seconds between poll attempts.  When *None* the
                value is read from ``self.poll_interval``.
            timeout: Maximum seconds to wait before raising TimeoutError.
            max_attempts: Maximum number of poll attempts as a safeguard.
                When *None* the value is read from ``self.max_poll_attempts``.
            extra_success_statuses: Service-specific statuses that are terminal
                and successful for THIS caller.  A service may park a finished
                job in a state the shared set cannot know about --
                ``app_inspector`` returns ``awaiting_selection`` when a job was
                submitted with ``require_selection=True``, and its ``/result``
                is readable in exactly that state.  Without declaring it here
                the poll cannot tell "parked, done" from "still working" and
                spins to ``timeout``.

        Returns:
            Completed job result dict.

        Raises:
            TimeoutError: If the job does not complete within timeout or
                max_attempts is exceeded.
            RuntimeError: If the downstream job reports failure.
            JobCancelledError: If the downstream job reports ``status="cancelled"``.
                Subclasses ``RuntimeError`` so existing failure handlers still
                catch it.
        """
        interval = self.poll_interval if interval is None else _clamp_poll_interval(interval)
        if max_attempts is None:
            max_attempts = self.max_poll_attempts
        else:
            max_attempts = _clamp_max_poll_attempts(max_attempts)
        timeout = _clamp_timeout(timeout)

        initial_interval = interval
        current_interval = interval
        warned_unknown: set[str] = set()

        start = time.monotonic()
        logger.info(
            "%s: polling job %s (timeout=%.0fs, initial_interval=%.1fs, max_attempts=%d)",
            self.service_name,
            job_id,
            timeout,
            initial_interval,
            max_attempts,
        )

        for _attempt in range(max_attempts):
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                logger.error(
                    "%s: job %s timed out after %.1fs",
                    self.service_name,
                    job_id,
                    timeout,
                )
                raise TimeoutError(f"{self.service_name} job {job_id} timed out after {timeout}s")

            try:
                result = await self._request("GET", f"{endpoint}/{job_id}")
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                # Service may be restarting -- treat as transient and retry
                # within the poll loop rather than failing the stage.
                logger.warning(
                    "%s: poll transport error for job %s, will retry: %s",
                    self.service_name,
                    job_id,
                    exc,
                )
                await asyncio.sleep(current_interval)
                current_interval = min(
                    current_interval * _POLL_BACKOFF_MULTIPLIER, _POLL_MAX_INTERVAL
                )
                continue

            status = result.get("status", "unknown")

            if status in _POLL_SUCCESS_STATUSES or (
                extra_success_statuses is not None and status in extra_success_statuses
            ):
                elapsed_ms = (time.monotonic() - start) * 1000
                logger.info(
                    "%s: job %s completed (status=%s) in %.1fms",
                    self.service_name,
                    job_id,
                    status,
                    elapsed_ms,
                )
                return result
            if status in _POLL_FAILURE_STATUSES:
                elapsed_ms = (time.monotonic() - start) * 1000
                _err = result.get("error")
                error_detail = _err if _err is not None else (result.get("message") or str(result))
                logger.error(
                    "%s: job %s failed after %.1fms -- full response: %s",
                    self.service_name,
                    job_id,
                    elapsed_ms,
                    result,
                )
                raise RuntimeError(f"{self.service_name} job {job_id} failed: {error_detail}")
            if status == "cancelled":
                elapsed_ms = (time.monotonic() - start) * 1000
                logger.warning(
                    "%s: job %s cancelled after %.1fms",
                    self.service_name,
                    job_id,
                    elapsed_ms,
                )
                raise JobCancelledError(f"{self.service_name} job {job_id} was cancelled")

            if status not in _POLL_NONTERMINAL_STATUSES and status not in warned_unknown:
                warned_unknown.add(status)
                logger.warning(
                    "%s: job %s reported status=%r, which this client treats as neither "
                    "terminal nor in-progress -- polling will continue until the deadline. "
                    "Known terminal: %s; declare a service-specific one via "
                    "extra_success_statuses.",
                    self.service_name,
                    job_id,
                    status,
                    sorted(
                        _POLL_SUCCESS_STATUSES
                        | _POLL_FAILURE_STATUSES
                        | {"cancelled"}
                        | (extra_success_statuses or frozenset())
                    ),
                )

            await asyncio.sleep(current_interval)
            # Adaptive backoff: increase interval, cap at max
            current_interval = min(current_interval * _POLL_BACKOFF_MULTIPLIER, _POLL_MAX_INTERVAL)

        raise TimeoutError(
            f"{self.service_name} job {job_id} exceeded {max_attempts} poll attempts"
        )
