"""Tests for w2v_common.observability.structlog_bridge.add_request_id.

The processor is the structlog twin of ``RequestIdLogFilter``: it reads the
request-id ContextVar set by the shared ``RequestIdMiddleware`` and stamps it
onto every structlog event. structlog is a w2v_common optional dependency; it
is a dev dependency here so this test can render through a real chain.
"""

from __future__ import annotations

import structlog

from w2v_common.observability.request_context import clear_request_id, set_request_id
from w2v_common.observability.structlog_bridge import add_request_id


def test_processor_injects_request_id_from_contextvar() -> None:
    set_request_id("rid-123")
    try:
        event: dict = {"event": "hello"}
        out = add_request_id(None, "info", event)
        assert out["request_id"] == "rid-123"
    finally:
        clear_request_id()


def test_processor_noop_when_contextvar_empty() -> None:
    clear_request_id()
    out = add_request_id(None, "info", {"event": "hello"})
    assert "request_id" not in out


def test_explicit_request_id_wins_over_contextvar() -> None:
    set_request_id("rid-ctx")
    try:
        out = add_request_id(None, "info", {"event": "x", "request_id": "rid-bound"})
        assert out["request_id"] == "rid-bound"
    finally:
        clear_request_id()


def test_end_to_end_through_structlog_chain() -> None:
    """Nothing bound in structlog contextvars — request_id still reaches output
    purely via the ContextVar the shared middleware sets."""
    cap = structlog.testing.CapturingLogger()
    log = structlog.wrap_logger(
        cap,
        processors=[
            structlog.contextvars.merge_contextvars,
            add_request_id,
            structlog.processors.KeyValueRenderer(key_order=["event", "request_id"]),
        ],
    )
    set_request_id("rid-e2e")
    try:
        log.info("served")
    finally:
        clear_request_id()
    assert cap.calls, "expected a log call"
    rendered = cap.calls[0].args[0]
    assert "request_id='rid-e2e'" in rendered
