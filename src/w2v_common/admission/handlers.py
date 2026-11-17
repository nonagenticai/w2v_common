"""FastAPI wiring for the GPU admission gate: 503 + Retry-After, quiet 499."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from w2v_common.admission.gate import ClientGone, GpuBusy

CLIENT_GONE_STATUS = 499


async def gpu_busy_response(_request: Request, exc: Exception) -> Response:
    """``GpuBusy`` -> 503, ``Retry-After: N``, ``{"error":"gpu_busy","lane":..,"retry_after":N}``."""
    if not isinstance(exc, GpuBusy):  # pragma: no cover — registered for GpuBusy only
        raise exc
    return JSONResponse(
        status_code=503,
        content={"error": "gpu_busy", "lane": exc.lane, "retry_after": exc.retry_after},
        headers={"Retry-After": str(exc.retry_after)},
    )


async def client_gone_response(_request: Request, exc: Exception) -> Response:
    """``ClientGone`` -> bodyless 499 (nginx's "client closed request"); nobody reads it."""
    if not isinstance(exc, ClientGone):  # pragma: no cover — registered for ClientGone only
        raise exc
    return Response(status_code=CLIENT_GONE_STATUS)


def install_gpu_busy_handler(app: FastAPI) -> None:
    """Register the ``GpuBusy`` (503) and ``ClientGone`` (499) handlers on ``app``."""
    app.add_exception_handler(GpuBusy, gpu_busy_response)
    app.add_exception_handler(ClientGone, client_gone_response)
