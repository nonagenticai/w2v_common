"""Observability helpers for w2v services."""

from w2v_common.observability.logging import (
    JsonFormatter,
    configure_logging,
    database_endpoint,
    redis_endpoint,
)
from w2v_common.observability.metrics import (
    downstream_calls_total,
    downstream_latency_seconds,
    http_request_duration_seconds,
    http_requests_total,
    metrics_router,
    record_downstream_call,
    render_latest,
)
from w2v_common.observability.request_context import (
    RequestIdLogFilter,
    clear_request_id,
    get_request_id,
    set_request_id,
    trace_event_hook,
)

__all__ = [
    "JsonFormatter",
    "RequestIdLogFilter",
    "clear_request_id",
    "configure_logging",
    "database_endpoint",
    "downstream_calls_total",
    "downstream_latency_seconds",
    "get_request_id",
    "http_request_duration_seconds",
    "http_requests_total",
    "metrics_router",
    "record_downstream_call",
    "redis_endpoint",
    "render_latest",
    "set_request_id",
    "trace_event_hook",
]
