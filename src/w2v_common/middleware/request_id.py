"""Request ID middleware for request tracing."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from w2v_common.observability.request_context import set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-Id header for request tracing.

    If the incoming request already has an X-Request-Id header, it is preserved.
    Otherwise, a new UUID is generated. The id is bound to the
    ``request_context`` ContextVar so the JSON log formatter picks it up
    (``get_request_id()``) and ``trace_event_hook`` forwards it on outbound
    S2S calls — without this binding the inbound trace id never propagates.
    (The live per-service copies of this middleware already do this; binding
    here keeps the shared copy from drifting if a service adopts the factory.)
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Extract or generate a request ID and attach to request state.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or handler.

        Returns:
            Response with X-Request-Id header set.
        """
        request_id = request.headers.get("x-request-id") or str(uuid.uuid7())
        request.state.request_id = request_id
        set_request_id(request_id)

        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
