"""Centralized logging configuration driven by `LoggingSettings`.

Each w2v service used to hand-roll its own root-logger setup. This module
consumes the shared `LoggingSettings` model (env-driven via `LOG_*`) and
applies it idempotently so per-service `asgi.py` can shrink to a single call.

Mirrors the behavior of `w2v_orchestrator/src/observability/logging.py` but
hand-rolls the JSON formatter to avoid pulling in `python-json-logger`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from urllib.parse import urlsplit

from w2v_common.config.base import LoggingSettings

__all__ = ["JsonFormatter", "configure_logging", "database_endpoint", "redis_endpoint"]


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a single-line JSON string."""
        log_entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = str(record.request_id)
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = str(record.correlation_id)
        return json.dumps(log_entry, default=str)


def configure_logging(
    settings: LoggingSettings | None = None,
    *,
    json_logs: bool | None = None,
    level: str | None = None,
) -> None:
    """Configure the root logger from `LoggingSettings` (idempotent).

    Args:
        settings: Optional `LoggingSettings` instance. When `None`, an
            env-driven `LoggingSettings()` is constructed (reads `LOG_*` env
            vars per its `SettingsConfigDict`).
        json_logs: Optional override for `settings.json_logs`.
        level: Optional override for `settings.level`.

    Behavior:
        - Clears existing root-logger handlers before installing a fresh
          `StreamHandler` on stdout, so calling this twice does not stack
          handlers.
        - When JSON logging is enabled, attaches `JsonFormatter`. Otherwise
          attaches a plain `logging.Formatter` using `settings.format` and
          `settings.date_format`.
    """
    if settings is None:
        settings = LoggingSettings()

    effective_level = (level if level is not None else settings.level).upper()
    effective_json = json_logs if json_logs is not None else settings.json_logs

    root = logging.getLogger()
    root.setLevel(effective_level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(effective_level)

    if effective_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(settings.format, datefmt=settings.date_format))

    root.addHandler(handler)


def database_endpoint(url: str) -> str:
    """Render a database DSN as ``scheme://host:port/dbname`` for logging — never userinfo.

    WHY THIS IS SHARED. Nine w2v services log
    ``logger.info("Database connected: %s", settings.database.url)`` on the
    connect path, i.e. the full
    ``postgresql+psycopg://<user>:<password>@host:port/db`` DSN at INFO. In
    ``w2v_orchestrator`` that was inert only because gitops pinned it to
    ``LOG_LEVEL=WARNING``; the moment the level was raised so the service logged
    anything at all (gitops #2952) the live app password began landing in
    plaintext in every pod's stdout and in anything shipping those logs. A log
    line whose safety depends on the log LEVEL is not safe, it is merely
    unexercised — and the other eight services never had even that accident
    protecting them.

    Lifted verbatim from ``w2v_orchestrator/src/observability/logging.py``, which
    fixed it for one service. It lives here so the other eight import the fix
    instead of re-deriving it, and it is deliberately a sibling of
    ``w2v_common.metering.cost_bus._redis_endpoint`` (w2v_common #119), which is
    the identical redaction for the metering Redis DSN. Same module family, same
    shape, so the pairing is discoverable: if you are logging a DSN of any kind,
    one of these two is what you want.

    Only the ``urlsplit`` hostname, port and path are used: those components
    structurally cannot carry the ``user:pass@`` userinfo, so no redaction
    PATTERN is needed — and a pattern is exactly what tends to rot. The scheme is
    kept because ``postgresql+psycopg`` vs anything else is the driver assertion
    this fleet cares about, and it carries no secret.

    Fails CLOSED: a malformed DSN renders ``"<unparseable>"`` rather than falling
    back to the raw URL, because a redactor that reverts to the original on a
    parse error leaks precisely when it is least expected. A log line must also
    never be able to break service startup, so nothing here raises.

    ⚠️ THIS IS STEP 1 OF 2, AND THE ORDER IS LOAD-BEARING — do not "simplify" it
    by making ``DatabaseSettings.url`` a ``SecretStr`` instead. That is the
    better end state and it is explicitly DEFERRED, not rejected. Measured
    2026-09-05: twelve consumers carry
    ``def get_database_url(self) -> str: return self.database.url``, and ``ty``
    rejects that under ``SecretStr`` with ``invalid-return-type``. ``ci-cd.yaml``
    job ``bump-consumers`` gate 2 runs exactly that ``ty check`` on the relocked
    checkout and fails CLOSED (exit 20 = BLOCKED, no push), so a ``SecretStr``
    release would produce eleven BLOCKED rows, a red fan-out, and reach ZERO
    services — the 0.67.0 shape. This helper is purely ADDITIVE and cannot fail
    that gate, so it reaches all seventeen consumers on the next fan-out. Once
    those twelve accessors are gone, ``SecretStr`` becomes safe hardening and
    should be done.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return "<unparseable>"
    if not host:
        return "<unparseable>"
    where = f"{host}:{port}" if port is not None else host
    dbname = parts.path.lstrip("/")
    if dbname:
        where = f"{where}/{dbname}"
    return f"{parts.scheme}://{where}" if parts.scheme else where


def redis_endpoint(url: str) -> str:
    """Render a Redis DSN as ``host:port`` for logging — never any userinfo.

    THE SIBLING LEAK. ``database_endpoint`` above closed
    ``logger.info("Database connected: %s", settings.database.url)``. Eleven w2v
    services sit one function below that with
    ``logger.info("Redis connected: %s", settings.redis.url)`` — and a Redis DSN
    carries its password in exactly the same place
    (``redis://[<user>]:<password>@host:port/db``). Guarded-beside-unguarded: the
    two lines are usually within fifty lines of each other in the same
    ``lifespan``, so fixing only the first one is the failure mode this function
    exists to prevent.

    Only the ``urlsplit`` hostname and port are used: those components
    structurally cannot carry the ``user:pass@`` userinfo, so no redaction
    PATTERN is needed — and a pattern is exactly what tends to rot.

    Unlike ``database_endpoint`` this renders neither scheme nor path. That is
    deliberate, not an oversight: this is the EXACT rendering
    ``w2v_common.metering.cost_bus._redis_endpoint`` has emitted since #119, and
    that private helper is now a thin alias of this one. Changing the shape here
    would silently change metering's boot log line and break its tests. The
    ``redis`` vs ``rediss`` scheme is not the driver assertion this fleet cares
    about the way ``postgresql+psycopg`` is, so nothing is lost.

    Fails CLOSED: a malformed DSN renders ``"<unparseable>"`` rather than falling
    back to the raw URL, because a redactor that reverts to the original on a
    parse error leaks precisely when it is least expected.

    ⚠️ ``except Exception``, deliberately, NOT ``except ValueError`` — and this
    is the one place this function is NOT a mirror of ``database_endpoint``. This
    function's contract is "a log line must never be able to break service
    startup", and a curated exception tuple only encodes the failures its author
    thought of. The narrow form shipped in 0.67.9 and broke SIX lifespan tests in
    ``ai_content_generation_service`` the same day: ``url`` is not always a real
    ``str``, and a service whose settings are mocked hands this a ``MagicMock``,
    at which point ``urlsplit`` reaches ``if url.find(':') > 0`` and comparing a
    ``MagicMock`` to an ``int`` raises ``TypeError``. Worse, ``cost_bus``
    re-invokes this from INSIDE its own ``except`` handler, so the escape did not
    merely spoil a log line, it replaced the real connection error and aborted
    boot. The width is justified by what this returns: a string for a log line,
    nothing more.

    Purely ADDITIVE, like ``database_endpoint``: no signature or type changes, so
    ``ci-cd.yaml`` job ``bump-consumers`` gate 2 (``ty check`` on the relocked
    consumer checkout, fail-closed, exit 20 = BLOCKED) cannot block it, and it
    reaches every consumer on the next fan-out.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except Exception:  # noqa: BLE001 — see the contract above; a log line never breaks boot
        return "<unparseable>"
    if not host:
        return "<unparseable>"
    return f"{host}:{port}" if port is not None else host
