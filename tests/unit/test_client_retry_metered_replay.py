"""A bare POST to a METERED peer must not be replayed on an ambiguous reset.

``BaseServiceClient`` retries "connection reset" transport errors for ANY
method, including a bare POST with no ``Idempotency-Key``. That is safe only
for the errors which prove the request never reached the application. Three of
the six errors in the original tuple do NOT prove that: ``ReadError``,
``WriteError`` and ``RemoteProtocolError`` are raised on an ESTABLISHED
connection, after the request bytes have already gone out.

The metered peers (e.g. ``worker_management``, ``inference_router``,
``modality_gateway``, ``video_production``) bill the tenant for the work a POST
spawns, and billing is APPEND-ONLY: there is no refund, void or reversal path
anywhere in the estate. So a replayed POST to one of them is money the tenant
never gets back.

``test_remote_protocol_error_is_raised_after_the_server_received_the_whole_body``
is the load-bearing one: it proves against a REAL socket that the exception
class the client treats as "never reached the application" is raised when the
server demonstrably received the complete request.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from w2v_common.auth.outbound import is_metered_destination
from w2v_common.clients import BaseServiceClient
from w2v_common.clients.base import (
    _AMBIGUOUS_TRANSPORT_ERRORS,
    _PRE_SEND_TRANSPORT_ERRORS,
    _may_replay_transport_error,
)
from w2v_common.discovery import service_url

# Resolved through discovery so the test binds to the REAL metered host rather
# than a hand-typed one. A hand-typed host silently stops matching the moment
# discovery changes, and the test would then pass vacuously.
_METERED = service_url("video_production")
# ``content`` rather than the ``analysis`` this used to name (which in turn
# replaced ``html_render``): each of those BECAME a metered destination in turn
# — html_render in PR #174, analysis in the owner flip of 2026-09-13 — and each
# time the module-level assert below went red and made the swap a one-liner
# instead of a silent vacuous pass. Keep it.
_PLAIN = service_url("content")

assert is_metered_destination(_METERED), "_METERED must actually be a metered peer"
assert not is_metered_destination(_PLAIN), "_PLAIN must NOT be a metered peer"


def _client(transport: httpx.MockTransport, base_url: str) -> BaseServiceClient:
    client = BaseServiceClient(base_url=base_url, service_name="svc")
    client._client = httpx.AsyncClient(
        transport=transport, base_url=base_url, follow_redirects=False
    )
    return client


class TestTheAmbiguityIsReal:
    """The premise "a reset proves the server never saw it" is FALSE."""

    async def test_remote_protocol_error_is_raised_after_the_server_received_the_whole_body(
        self,
    ) -> None:
        received: list[bytes] = []

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                data += chunk
            head, _, rest = data.partition(b"\r\n\r\n")
            content_length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":")[1])
            while len(rest) < content_length:
                more = await reader.read(4096)
                if not more:
                    break
                rest += more
            # The server now holds the COMPLETE request: at this point it could
            # already have dispatched the recording and emitted the cost event.
            received.append(rest)
            # ...and then it dies before writing a single response byte.
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient() as raw:
                with pytest.raises(httpx.RemoteProtocolError) as caught:
                    await raw.post(f"http://127.0.0.1:{port}/api/record/url", json={"url": "x"})
        finally:
            server.close()
            await server.wait_closed()

        assert received == [b'{"url":"x"}'], (
            "the server must have received the whole body for this proof to hold"
        )
        # The exception the client classified as "the request never reached the
        # application" is raised on a request the application fully received.
        assert isinstance(caught.value, _AMBIGUOUS_TRANSPORT_ERRORS)
        assert not isinstance(caught.value, _PRE_SEND_TRANSPORT_ERRORS)


class TestReplayGate:
    """``_may_replay_transport_error`` unit truth table."""

    @pytest.mark.parametrize("exc_type", _PRE_SEND_TRANSPORT_ERRORS)
    def test_pre_send_errors_always_replayable(self, exc_type: type) -> None:
        # No connection was established, so nothing could have been billed.
        assert _may_replay_transport_error(exc_type("boom"), retryable=False, destination=_METERED)

    @pytest.mark.parametrize("exc_type", _AMBIGUOUS_TRANSPORT_ERRORS)
    def test_ambiguous_errors_blocked_for_bare_post_to_metered_peer(self, exc_type: type) -> None:
        assert not _may_replay_transport_error(
            exc_type("boom"), retryable=False, destination=_METERED
        )

    @pytest.mark.parametrize("exc_type", _AMBIGUOUS_TRANSPORT_ERRORS)
    def test_ambiguous_errors_still_replayable_against_a_plain_backend(
        self, exc_type: type
    ) -> None:
        # Re-doing FREE work is merely wasteful; the stale-keepalive resilience
        # this branch exists for is preserved everywhere money is not at stake.
        assert _may_replay_transport_error(exc_type("boom"), retryable=False, destination=_PLAIN)

    @pytest.mark.parametrize("exc_type", _AMBIGUOUS_TRANSPORT_ERRORS)
    def test_ambiguous_errors_replayable_to_metered_peer_when_dedupable(
        self, exc_type: type
    ) -> None:
        # ``retryable=True`` means an idempotent method or an Idempotency-Key
        # the downstream dedupes on — the replay cannot double-bill.
        assert _may_replay_transport_error(exc_type("boom"), retryable=True, destination=_METERED)

    def test_read_timeout_is_never_replayable(self) -> None:
        assert not _may_replay_transport_error(
            httpx.ReadTimeout("slow"), retryable=False, destination=_PLAIN
        )


class TestEndToEndThroughTheClient:
    """The gate is actually wired into ``_request`` / ``_request_raw``."""

    async def test_bare_post_to_metered_peer_is_not_replayed(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.RemoteProtocolError("server disconnected")

        client = _client(httpx.MockTransport(handler), _METERED)
        try:
            with pytest.raises(httpx.RemoteProtocolError):
                await client._request("POST", "/api/record/url", json={"url": "x"})
        finally:
            await client.close()

        # ONE attempt. A second would be a second recording the tenant pays for
        # and can never be refunded.
        assert len(calls) == 1

    async def test_bare_post_to_plain_backend_is_still_replayed(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 2:
                raise httpx.RemoteProtocolError("stale keepalive")
            return httpx.Response(status_code=200, content=b'{"ok":true}')

        client = _client(httpx.MockTransport(handler), _PLAIN)
        try:
            result = await client._request("POST", "/api/render", json={"x": 1})
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 2

    async def test_bare_post_to_metered_peer_still_replays_a_connect_error(
        self,
    ) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 2:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(status_code=200, content=b'{"ok":true}')

        client = _client(httpx.MockTransport(handler), _METERED)
        try:
            result = await client._request("POST", "/api/record/url", json={"url": "x"})
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 2

    async def test_request_raw_shares_the_gate(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ReadError("reset")

        client = _client(httpx.MockTransport(handler), _METERED)
        try:
            with pytest.raises(httpx.ReadError):
                await client._request_raw("POST", "/api/record/capture/screens")
        finally:
            await client.close()

        assert len(calls) == 1
