"""The GPU gate's disconnect skip must survive w2v_common's middleware stack.

``Lane.acquire`` skips a job (``ClientGone`` -> 499) when
``await request.is_disconnected()`` is true at admission. The gate's own tests
proved that only on a bare FastAPI app. In a real service the handler sits
behind ``create_app``'s canonical stack (Auth, RequestId, SecurityHeaders,
Metrics, Audit, ...). Starlette's ``BaseHTTPMiddleware`` hands the inner app a
wrapped ``receive`` whose disconnect is invisible to a non-blocking
``is_disconnected()`` probe, so behind one such middleware the skip never fired
and an abandoned job still burned the GPU.

These tests drive the app with a raw ASGI ``receive`` (as uvicorn does): the
body arrives, then — while the request waits in the gate queue — the next
``receive()`` yields ``http.disconnect``.

The fix is ``install_disconnect_probe(app)`` (probe outermost by construction); the
``*_without_probe`` test pins the Starlette behaviour that makes it necessary.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, Request, UploadFile
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from w2v_common.admission import (
    AdmissionGate,
    DisconnectProbe,
    DisconnectProbeMiddleware,
    LaneConfig,
    client_disconnected,
    install_disconnect_probe,
    install_gpu_busy_handler,
)
from w2v_common.admission.disconnect import PROBE_SCOPE_KEY
from w2v_common.asgi import create_app
from w2v_common.config import (
    APISettings,
    AuditSettings,
    AuthSettings,
    IdempotencySettings,
    MetricsSettings,
)

ROUTE = "/api/gpu-job"


class _Job(BaseModel):
    text: str


class _PassThrough(BaseHTTPMiddleware):
    """The smallest possible ``BaseHTTPMiddleware``: a control for the mechanism."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        return await call_next(request)


def _router(gate: AdmissionGate, lane: str, ran: list[str]) -> APIRouter:
    router = APIRouter()

    @router.post(ROUTE)
    async def gpu_job(job: _Job, request: Request) -> dict[str, str]:
        await gate.run(lane, ran.append, job.text, request=request)
        return {"ok": job.text}

    return router


def _service_settings() -> BaseSettings:
    class Settings(BaseSettings):
        auth: AuthSettings = Field(default_factory=AuthSettings)
        api: APISettings = Field(
            default_factory=lambda: APISettings(port=8080, host="127.0.0.1", cors_origins=[])
        )
        metrics: MetricsSettings = Field(default_factory=lambda: MetricsSettings(enabled=True))
        audit: AuditSettings = Field(default_factory=lambda: AuditSettings(enabled=True))
        idempotency: IdempotencySettings = Field(
            default_factory=lambda: IdempotencySettings(enabled=True)
        )

    return Settings()


def _bare(router: APIRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    install_gpu_busy_handler(app)
    return app


def _one_base_http(router: APIRouter) -> FastAPI:
    app = _bare(router)
    app.add_middleware(_PassThrough)
    return app


def _canonical(router: APIRouter) -> FastAPI:
    # The full canonical stack exactly as a service builds it, auth ON; the GPU
    # route is declared public so no license introspection is needed — the
    # request still traverses every middleware's wrapper.
    app = create_app(
        _service_settings(),
        routers=[router],
        auth_enabled=True,
        auth_public_endpoints={ROUTE},
    )
    install_gpu_busy_handler(app)
    return app


def _canonical_plus_service_middleware(router: APIRouter) -> FastAPI:
    # What services actually run: create_app's stack PLUS their own
    # BaseHTTPMiddleware (e.g. a RequestIdMiddleware) added afterwards.
    app = _canonical(router)
    app.add_middleware(_PassThrough)
    return app


STACKS: dict[str, Callable[[APIRouter], FastAPI]] = {
    "bare_fastapi": _bare,
    "one_base_http_middleware": _one_base_http,
    "create_app_canonical_stack": _canonical,
    "create_app_plus_service_middleware": _canonical_plus_service_middleware,
}
HIDING_STACKS = sorted(set(STACKS) - {"bare_fastapi"})


def _build(stack: str, router: APIRouter, *, probe: bool) -> FastAPI:
    app = STACKS[stack](router)
    if probe:
        install_disconnect_probe(app)
    return app


async def _drive(
    app: ASGIApp, body: bytes, disconnect: asyncio.Event
) -> tuple[asyncio.Task[None], list[Message]]:
    """Start one HTTP request against ``app`` the way uvicorn would."""
    sent_body = False

    async def receive() -> Message:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": ROUTE,
        "raw_path": ROUTE.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("203.0.113.9", 50000),
        "server": ("testserver", 80),
        "state": {},
    }

    async def _serve() -> None:
        await app(scope, receive, send)

    return asyncio.create_task(_serve()), messages


async def _wait_until(pred: Callable[[], bool], timeout: float = 5.0) -> None:  # noqa: ASYNC109 — a polling helper
    async with asyncio.timeout(timeout):
        while not pred():  # noqa: ASYNC110 — gate counters expose no event to await
            await asyncio.sleep(0.005)


async def _abandon_while_queued(app: ASGIApp, gate: AdmissionGate, lane: str) -> list[Message]:
    """Queue one request behind a held slot, hang up, then free the slot."""
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with gate.slot(lane):
            await release_holder.wait()

    held = asyncio.create_task(holder())
    await _wait_until(lambda: gate.lane(lane).active == 1)

    disconnect = asyncio.Event()
    request_task, messages = await _drive(app, b'{"text": "abandoned"}', disconnect)
    await _wait_until(lambda: gate.lane(lane).waiting == 1)

    disconnect.set()  # the client hangs up while its job is queued
    await asyncio.sleep(0.05)  # the transport has delivered http.disconnect
    release_holder.set()  # the slot frees; the queued request is admitted

    async with asyncio.timeout(5):
        await request_task
    await held
    return messages


def _new_gate() -> tuple[AdmissionGate, str]:
    lane = f"d{uuid.uuid7().hex[-12:]}"
    gate = AdmissionGate({lane: LaneConfig(max_active=1, max_waiting=4, max_wait_s=10)}, environ={})
    return gate, lane


def _status(messages: list[Message]) -> int | None:
    starts = [m for m in messages if m["type"] == "http.response.start"]
    return starts[0]["status"] if starts else None


@pytest.mark.parametrize("stack", sorted(STACKS))
async def test_client_that_leaves_while_queued_is_skipped_through_middleware(
    stack: str,
) -> None:
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _build(stack, _router(gate, lane, ran), probe=True)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == [], f"{stack}: the abandoned job ran on the GPU"
    assert _status(messages) == 499, f"{stack}: {messages}"
    assert gate.lane(lane).active == 0
    assert await gate.shutdown()


@pytest.mark.parametrize("stack", HIDING_STACKS)
async def test_base_http_middleware_hides_the_disconnect_without_probe(stack: str) -> None:
    """Why the probe exists: without it the abandoned job RUNS behind any
    ``BaseHTTPMiddleware``. If Starlette ever fixes this, this test fails and
    the probe can be reconsidered."""
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _build(stack, _router(gate, lane, ran), probe=False)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == ["abandoned"]
    assert _status(messages) == 200
    assert await gate.shutdown()


async def test_bare_app_without_probe_still_skips() -> None:
    """The fallback (``request.is_disconnected()``) keeps working with no probe."""
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _build("bare_fastapi", _router(gate, lane, ran), probe=False)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == []
    assert _status(messages) == 499
    assert await gate.shutdown()


@pytest.mark.parametrize("stack", sorted(STACKS))
async def test_connected_client_is_still_served_through_middleware(stack: str) -> None:
    """Control: the same stack, probe included, runs the job when nobody hung up."""
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _build(stack, _router(gate, lane, ran), probe=True)

    request_task, messages = await _drive(app, b'{"text": "kept"}', asyncio.Event())
    async with asyncio.timeout(5):
        await request_task

    assert ran == ["kept"]
    assert _status(messages) == 200, f"{stack}: {messages}"
    assert await gate.shutdown()


# ── DisconnectProbe: never steals from the app ────────────────────────────────


def _scripted_receive(script: list[Message], *, suspend: bool = False):
    calls = 0

    async def receive() -> Message:
        nonlocal calls
        calls += 1
        if suspend:
            await asyncio.sleep(0)
        if script:
            return script.pop(0)
        await asyncio.Event().wait()  # nothing more: block like a live server
        raise AssertionError("unreachable")

    def count() -> int:
        return calls

    return receive, count


async def test_poll_before_the_body_is_read_buffers_it_for_the_app() -> None:
    chunks: list[Message] = [
        {"type": "http.request", "body": b"ab", "more_body": True},
        {"type": "http.request", "body": b"cd", "more_body": False},
    ]
    receive, _ = _scripted_receive(list(chunks))
    probe = DisconnectProbe(receive)

    # Two polls each obtain a ready body chunk: neither is a disconnect.
    assert await probe.poll() is False
    assert await probe.poll() is False
    # The app still receives every chunk, in order (a lost chunk would block).
    async with asyncio.timeout(1):
        assert await probe.receive() == chunks[0]
        assert await probe.receive() == chunks[1]


async def test_poll_on_a_suspending_receive_consumes_nothing() -> None:
    script: list[Message] = [{"type": "http.request", "body": b"x", "more_body": False}]
    receive, _ = _scripted_receive(script, suspend=True)
    probe = DisconnectProbe(receive)

    assert await probe.poll() is False  # cancelled at the suspension point
    async with asyncio.timeout(1):
        assert await probe.receive() == {"type": "http.request", "body": b"x", "more_body": False}


async def test_poll_records_disconnect_and_app_then_sees_it() -> None:
    receive, count = _scripted_receive(
        [
            {"type": "http.request", "body": b"", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )
    probe = DisconnectProbe(receive)
    assert (await probe.receive())["type"] == "http.request"
    assert await probe.poll() is True
    assert await probe.poll() is True
    calls = count()
    assert await probe.receive() == {"type": "http.disconnect"}
    assert count() == calls  # answered from the recorded state, not the server


async def test_disconnect_seen_by_the_app_is_recorded() -> None:
    receive, count = _scripted_receive([{"type": "http.disconnect"}])
    probe = DisconnectProbe(receive)
    assert await probe.receive() == {"type": "http.disconnect"}
    calls = count()
    assert await probe.poll() is True
    assert count() == calls


async def test_poll_never_races_an_app_receive_in_flight() -> None:
    receive, count = _scripted_receive([])  # blocks forever
    probe = DisconnectProbe(receive)
    pending = asyncio.create_task(probe.receive())
    await asyncio.sleep(0.01)
    assert count() == 1
    assert await probe.poll() is False
    assert count() == 1  # the poll did not open a second receive
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


@pytest.mark.parametrize(
    "scope",
    [
        {"type": "lifespan"},
        {"type": "websocket", "path": "/ws", "headers": [], "query_string": b""},
    ],
    ids=["lifespan", "websocket"],
)
async def test_middleware_passes_non_http_scopes_through_untouched(
    scope: dict[str, Any],
) -> None:
    seen: list[tuple[Scope, Receive, Send]] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append((scope, receive, send))

    async def receive() -> Message:
        return {"type": "websocket.connect"}

    async def send(_m: Message) -> None:
        return None

    await DisconnectProbeMiddleware(inner)(scope, receive, send)
    got_scope, got_receive, got_send = seen[0]
    assert got_scope is scope
    assert got_receive is receive  # the raw channel, not a probe wrapper
    assert got_send is send
    assert PROBE_SCOPE_KEY not in got_scope


async def test_poll_between_chunks_delivers_every_chunk_once_in_order() -> None:
    chunks: list[Message] = [
        {"type": "http.request", "body": b"one-", "more_body": True},
        {"type": "http.request", "body": b"two-", "more_body": True},
        {"type": "http.request", "body": b"three", "more_body": False},
    ]
    receive, count = _scripted_receive(list(chunks))
    probe = DisconnectProbe(receive)

    got: list[Message] = []
    async with asyncio.timeout(1):
        got.append(await probe.receive())  # chunk 1
        assert await probe.poll() is False  # between chunk 1 and 2: grabs chunk 2
        assert count() == 2
        got.append(await probe.receive())  # chunk 2, from the buffer
        assert count() == 2  # served without touching the server
        got.append(await probe.receive())  # chunk 3
    assert got == chunks
    # Nothing is replayed: the next receive waits on the server.
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await probe.receive()


def _multipart(boundary: str, filename: str, payload: bytes) -> bytes:
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    return head + payload + f"\r\n--{boundary}--\r\n".encode()


def _upload_app(received: dict[str, Any]) -> FastAPI:
    app = FastAPI()
    app.add_middleware(_PassThrough)

    @app.post("/api/upload")
    async def upload(file: UploadFile) -> dict[str, int]:
        received["data"] = await file.read()
        received["name"] = file.filename
        return {"size": len(received["data"])}

    return app


class _FeedingServer:
    """A server ``receive``: a chunk already read off the socket is returned at
    once (no suspension, so a cancelled poll CAN obtain it); otherwise it waits.
    Counts how many chunks the poller task, not the app, obtained."""

    def __init__(self) -> None:
        self.ready: list[Message] = []
        self.arrived = asyncio.Event()
        self.poller: asyncio.Task[None] | None = None
        self.poller_reads = 0

    def feed(self, message: Message) -> None:
        self.ready.append(message)
        self.arrived.set()

    async def receive(self) -> Message:
        while not self.ready:
            self.arrived.clear()
            await self.arrived.wait()
        if self.poller is not None and asyncio.current_task() is self.poller:
            self.poller_reads += 1
        return self.ready.pop(0)


async def test_streamed_multipart_upload_arrives_intact_through_the_probe() -> None:
    """A real FastAPI ``UploadFile`` behind a BaseHTTPMiddleware, streamed in
    many chunks while something polls the probe the whole time."""
    payload = bytes(range(256)) * 400  # 100 KiB, every byte value
    boundary = "w2vboundary7d1"
    body = _multipart(boundary, "clip.wav", payload)
    pieces = [body[i : i + 4096] for i in range(0, len(body), 4096)]
    assert len(pieces) > 20

    received: dict[str, Any] = {}
    app = _upload_app(received)
    install_disconnect_probe(app)
    server = _FeedingServer()
    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/upload",
        "raw_path": b"/api/upload",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("203.0.113.9", 50000),
        "server": ("testserver", 80),
        "state": {},
    }

    async def serve() -> None:
        await app(scope, server.receive, send)

    stop = asyncio.Event()
    poll_results: list[bool] = []

    async def poll_until_stopped() -> None:
        # Stops on an event, never by cancellation: poll() runs an anyio
        # CancelScope that must not be asked to absorb a foreign cancel.
        while not stop.is_set():
            probe = scope.get(PROBE_SCOPE_KEY)  # set by the installed outermost layer
            if isinstance(probe, DisconnectProbe):
                poll_results.append(await probe.poll())
            await asyncio.sleep(0)

    request_task = asyncio.create_task(serve())
    server.poller = asyncio.create_task(poll_until_stopped())
    try:
        for i, piece in enumerate(pieces):
            server.feed({"type": "http.request", "body": piece, "more_body": i < len(pieces) - 1})
            await asyncio.sleep(0)
        async with asyncio.timeout(5):
            await request_task
    finally:
        stop.set()
        await asyncio.wait_for(server.poller, 5)
        request_task.cancel()

    assert poll_results and not any(poll_results)

    assert _status(messages) == 200, messages
    assert received["name"] == "clip.wav"
    assert received["data"] == payload
    assert server.poller_reads > 0, "the poller never obtained a chunk: the test proved nothing"


async def test_client_disconnected_falls_back_without_a_probe() -> None:
    class _Req:
        def __init__(self) -> None:
            self.scope: dict[str, Any] = {"type": "http"}

        async def is_disconnected(self) -> bool:
            return True

    assert await client_disconnected(_Req()) is True


@pytest.mark.parametrize("ticks", range(6))
async def test_a_cancel_landing_during_a_poll_is_never_swallowed(ticks: int) -> None:
    """A request task cancelled while it polls must stay cancelled.

    The anyio already-cancelled-``CancelScope`` poll (Starlette's own
    ``is_disconnected()`` shape) absorbed a real cancel landing on the same
    await (measured: tick 2), so a cancelled request would carry on into the GPU.
    """

    async def never() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    probe = DisconnectProbe(never)
    stop = False

    async def keep_polling() -> None:
        while not stop:
            await probe.poll()
            await asyncio.sleep(0)

    task = asyncio.create_task(keep_polling())
    for _ in range(ticks):
        await asyncio.sleep(0)
    task.cancel()
    # asyncio.wait never re-cancels, so a swallowed cancel fails fast, not hangs.
    await asyncio.wait({task}, timeout=1)
    swallowed = not task.done()
    stop = True
    await asyncio.wait({task}, timeout=1)
    assert not swallowed, "the cancel was swallowed: the polling task kept running"
    assert task.cancelled()


async def test_a_future_returning_receive_is_parked_for_the_app_not_dropped() -> None:
    loop = asyncio.get_running_loop()
    futures: list[asyncio.Future[Message]] = []

    def receive() -> asyncio.Future[Message]:
        fut: asyncio.Future[Message] = loop.create_future()
        futures.append(fut)
        return fut

    probe = DisconnectProbe(receive)
    assert await probe.poll() is False  # cannot step a Future: parked
    assert await probe.poll() is False  # parked receive pending: no second call
    assert len(futures) == 1
    futures[0].set_result({"type": "http.request", "body": b"late", "more_body": False})
    async with asyncio.timeout(1):
        assert await probe.receive() == {
            "type": "http.request",
            "body": b"late",
            "more_body": False,
        }
    assert len(futures) == 1  # the app got the parked one, not a fresh receive
