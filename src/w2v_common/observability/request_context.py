"""Per-request request-id ContextVar for cross-async-context propagation.

Mirrors the identity-context pattern in ``w2v_common.auth.identity_context``
but for the inbound request id (a UUID generated or preserved by the
service's RequestId middleware). Carrying it in a ContextVar lets:

* the JSON log formatter pick it up automatically via ``RequestIdLogFilter``,
  so every log record emitted while serving the request is tagged;
* outbound ``httpx`` calls forward it via ``X-Request-Id`` so peer-service
  logs can be reconstructed into a single trace.

This is intentionally orthogonal to identity (license/tenant/subject) and
to OpenTelemetry — manual ``X-Request-Id`` propagation is the floor;
OTel is a future upgrade.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

import httpx

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: str | None) -> None:
    """Bind the inbound request id for the current async task."""
    _request_id_var.set(request_id or "")


def get_request_id() -> str:
    """Return the request id bound on the current async task, or empty string."""
    return _request_id_var.get()


def clear_request_id() -> None:
    """Reset the request-id ContextVar (test/teardown helper)."""
    _request_id_var.set("")


async def trace_event_hook(request: httpx.Request) -> None:
    """httpx ``event_hooks={'request': [...]}`` handler.

    Stamps ``X-Request-Id`` from the per-task ContextVar onto outbound
    requests so downstream services see the same trace id the inbound
    middleware bound. No-op when the var is empty (CronJobs, startup tasks,
    test fixtures) so background work keeps emitting unattributed logs
    rather than fabricating a spurious id.

    Preserves any ``X-Request-Id`` the caller set manually on the request
    (caller wins) — same shape as ``identity_event_hook``.
    """
    if "X-Request-Id" in request.headers:
        return
    request_id = _request_id_var.get()
    if request_id:
        request.headers["X-Request-Id"] = request_id


class RequestIdLogFilter(logging.Filter):
    """Inject the per-task request id onto every ``LogRecord``.

    Register once on the root logger and ``JsonFormatter`` will pick up
    the value (see ``observability.logging.JsonFormatter.format``). Empty
    values are still attached so downstream log shippers can group by
    presence; filter at the aggregator if you want to drop unattributed
    lines.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get()
        return True
