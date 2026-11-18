"""Canonical Prometheus HTTP metrics for w2v services.

Promoted from the per-service ``MetricsMiddleware`` copies (orchestrator,
analysis_service, video_production_service), which each defined their own
prefixed metric name (``orchestrator_requests_total``,
``analysis_http_requests_total``, ``http_requests_total``) over an identical
``method``/``endpoint``/``status_code`` label set.

The fleet now shares a single ``w2v_http_*`` metric family. Per-service
disambiguation is the in-cluster scrape's job — it stamps a ``kubernetes_name``
label on every series (the same way the metering dashboards already slice
``http_requests_total{kubernetes_name=...}``), so no ``service`` label is
carried in-app. The ``w2v_`` prefix keeps this family distinct from the
metering service's identically-named-but-differently-labelled
``http_requests_total``.

The counters register against the prometheus_client default ``REGISTRY`` at
import; :func:`render_latest` and :data:`metrics_router` expose them on the
canonical auth-exempt ``GET /metrics`` endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

http_requests_total = Counter(
    "w2v_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "w2v_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    # Superset of the per-service bucket sets that were merged here.
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ---------------------------------------------------------------------------
# Outbound service-to-service (downstream) call metrics.
#
# The ``w2v_http_*`` family above counts INBOUND requests this service serves.
# This pair counts OUTBOUND calls this service makes to its peers, so an
# operator can answer "which downstream microservice is failing/slow" from
# metrics alone — previously invisible fleet-wide. Emitted once per logical
# ``BaseServiceClient`` request (i.e. after its internal retries resolve), via
# :func:`record_downstream_call`. ``target`` is the callee's logical
# ``service_name``; ``endpoint`` is deliberately NOT a label (per-path
# cardinality blow-up). ``outcome`` ∈ {ok, http_error, timeout,
# transport_error, error}.
# ---------------------------------------------------------------------------
downstream_calls_total = Counter(
    "w2v_downstream_calls_total",
    "Outbound service-to-service calls, by callee and outcome.",
    ["target", "method", "outcome"],
)

downstream_latency_seconds = Histogram(
    "w2v_downstream_latency_seconds",
    "Outbound service-to-service call latency in seconds (includes client retries).",
    ["target", "method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)


def record_downstream_call(target: str, method: str, outcome: str, elapsed_seconds: float) -> None:
    """Record one logical outbound call to a downstream service.

    Best-effort observation: called from ``BaseServiceClient._observe_downstream``
    from INSIDE its ``except httpx.HTTPStatusError`` / ``except httpx.TimeoutException``
    / ``except httpx.TransportError`` / ``except Exception`` handlers, immediately
    before each ``raise`` that re-propagates the real downstream error. ``method``
    is upper-cased for a stable label; ``elapsed_seconds`` is the wall time of the
    whole logical call.

    ``except Exception``, deliberately total: this used to have NO guard at all,
    which is the same defect class as ``cost_bus._redis_endpoint`` shipping
    ``except ValueError`` (w2v_common 0.67.9, fixed 0.67.10 #141) — a helper
    invoked from inside an ``except`` block that can itself raise REPLACES the
    original exception instead of letting the bare ``raise`` re-propagate it. A
    label-cardinality mismatch or a non-numeric ``elapsed_seconds`` here (e.g.
    Prometheus rejecting a bad label value) would silently swap a real
    ``httpx.HTTPStatusError``/timeout/transport failure for a ``TypeError``/
    ``ValueError`` from this metrics call — masking the actual downstream fault
    every caller up the stack is trying to see. The width is justified by what
    this returns: nothing, purely a side-effecting observation.
    """
    try:
        m = method.upper()
        downstream_calls_total.labels(target=target, method=m, outcome=outcome).inc()
        downstream_latency_seconds.labels(target=target, method=m).observe(elapsed_seconds)
    except Exception:  # noqa: BLE001
        logger.debug(
            "record_downstream_call_failed target=%r method=%r outcome=%r",
            target,
            method,
            outcome,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Modality-config resolve fail-open counter.
#
# ``ModalityConfigClient.resolve`` fails OPEN to ``ModalityRoute.platform_default()``
# on any transport error, 5xx, or unparseable body. That default means "use the
# platform provider", which is harmless for an UNCONFIGURED tenant and is NOT
# harmless for a ``byo_required`` tenant: during the fail-open window the caller
# spends PLATFORM credentials for a tenant contractually required to bring its
# own key. The window was previously visible only as a ``logger.warning``.
#
# Cardinality is bounded and tiny: 5 modalities x 3 reasons.
# ``reason`` in {transport_error, server_error, unparseable_body}.
# ---------------------------------------------------------------------------
modality_resolve_failopen_total = Counter(
    "w2v_modality_resolve_failopen_total",
    "Modality-config resolves that fell back to the platform default, by modality and reason.",
    ["modality", "reason"],
)


def record_modality_resolve_failopen(modality: str, reason: str) -> None:
    """Record one fail-open modality resolve. Best-effort, never raises.

    ``except Exception``, deliberately total, for the same reason as
    :func:`record_downstream_call`: this is a pure side-effecting observation
    called on an error path whose whole contract is to return a default. A
    label-cardinality or type error here must not convert a documented
    fail-open into an exception for the caller.
    """
    try:
        modality_resolve_failopen_total.labels(modality=modality, reason=reason).inc()
    except Exception:  # noqa: BLE001
        logger.debug(
            "record_modality_resolve_failopen_failed modality=%r reason=%r",
            modality,
            reason,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# GPU admission gate (w2v_common.admission, #217).
#
# NO ``w2v_`` prefix, deliberately: the names are fixed by the GPU admission
# contract (gitops-production tracking issue) so the fleet-wide dashboards and
# alerts can be written once against every GPU service, including ones that are
# not w2v services. Registered on the same default ``REGISTRY`` that
# :func:`render_latest` serves. ``lane`` is a code-defined lane name (bounded:
# one per GPU resource on a host), never caller input.
# ---------------------------------------------------------------------------
gpu_admission_active = Gauge(
    "gpu_admission_active",
    "Requests currently holding a GPU lane slot (work in flight).",
    ["lane"],
)

gpu_admission_waiting = Gauge(
    "gpu_admission_waiting",
    "Requests currently queued for a GPU lane slot.",
    ["lane"],
)

gpu_admission_rejected_total = Counter(
    "gpu_admission_rejected_total",
    "Requests refused by a GPU lane with 503, by reason (full|timeout).",
    ["lane", "reason"],
)

gpu_admission_wait_seconds = Histogram(
    "gpu_admission_wait_seconds",
    "Time from arrival at a GPU lane to admission (or timeout rejection).",
    ["lane"],
    buckets=(0.005, 0.05, 0.25, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

gpu_admission_service_seconds = Histogram(
    "gpu_admission_service_seconds",
    "Time a GPU lane slot was held (admission to release, i.e. the work itself).",
    ["lane"],
    buckets=(0.05, 0.25, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)


def render_latest() -> tuple[bytes, str]:
    """Return ``(payload, content_type)`` for a Prometheus scrape response."""
    return generate_latest(), CONTENT_TYPE_LATEST


metrics_router = APIRouter()


@metrics_router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    """Expose the default Prometheus registry in text exposition format."""
    payload, content_type = render_latest()
    return Response(content=payload, media_type=content_type)
