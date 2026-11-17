"""Prometheus HTTP metrics middleware.

Records request count + duration into the canonical ``w2v_http_*`` metric
family defined in :mod:`w2v_common.observability.metrics`. Promoted verbatim
from the per-service copies (orchestrator / analysis_service /
video_production_service), which were logic-identical and differed only in
the metric name they targeted.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from w2v_common.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

_KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record HTTP request count + duration in Prometheus."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Record request count and duration for every HTTP request."""
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        # BOUNDED labels only (this registry is scraped via W2V_METRICS_PORT):
        # the matched route TEMPLATE, never the raw path (which embeds run /
        # project / tenant UUIDs and is caller-chosen — one series per request);
        # unmatched paths (404s, scanners, auth refusals before routing)
        # collapse to ``<unmatched>``. The method token is caller-chosen too.
        route = request.scope.get("route")
        endpoint = getattr(route, "path_format", None) or "<unmatched>"
        method = request.method if request.method in _KNOWN_METHODS else "OTHER"
        status = str(response.status_code)

        http_requests_total.labels(method=method, endpoint=endpoint, status_code=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

        return response
