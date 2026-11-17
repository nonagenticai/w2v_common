"""structlog processor bridging the w2v request-id ContextVar into structlog logs.

The stdlib-logging adapter for the request id is
:class:`w2v_common.observability.request_context.RequestIdLogFilter`. This module
is its structlog twin — for services that render logs through structlog rather
than a stdlib formatter.

There is a single source of truth: the ``request_id`` ContextVar set by the
shared :class:`w2v_common.middleware.RequestIdMiddleware`. This processor reads
it and stamps ``request_id`` onto every structlog event, so a service no longer
needs a bespoke RequestId middleware that binds structlog contextvars itself
(the pattern this replaces in tts_service).

Register it in the structlog processor chain, typically right after
``structlog.contextvars.merge_contextvars``::

    import structlog
    from w2v_common.observability.structlog_bridge import add_request_id

    structlog.configure(processors=[
        structlog.contextvars.merge_contextvars,
        add_request_id,
        ...,
    ])

The processor is a plain callable with **no structlog import of its own**, so
importing it never pulls structlog into a service that does not use it. structlog
is a w2v_common *optional* dependency declared by the consuming service.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from w2v_common.observability.request_context import get_request_id


def add_request_id(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: stamp ``request_id`` from the w2v request-id ContextVar.

    Reads the request id bound by the shared ``RequestIdMiddleware`` and adds it
    to the event under ``request_id`` when non-empty. Uses ``setdefault`` so an
    explicit per-event ``request_id`` (e.g. a handler that bound its own via
    ``structlog.contextvars.bind_contextvars``) wins over the ContextVar.
    """
    request_id = get_request_id()
    if request_id:
        event_dict.setdefault("request_id", request_id)
    return event_dict
