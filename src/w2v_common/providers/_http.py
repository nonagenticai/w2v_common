"""The ONLY way a tenant provider is reached: a bare, origin-locked, pinned httpx client.

Ported from the orchestrator's ``src/services/llm/in_process.py`` (``PinnedTransport``,
``_http_client``) onto plain ``httpx`` (the orchestrator uses its ``httpx2`` fork, so
the code is a port, not a verbatim lift):

* every request must stay on the tenant's ONE origin (scheme, host, port) — a
  different origin is refused before any socket opens; redirects are never followed;
* the socket goes to ONE vetted IP while the ``Host`` header and TLS SNI keep the
  name (certificate verification still checks the name): the orchestrator's
  ``pinned_ip`` when it sent one, else the host is resolved HERE, once, and every
  address must be public — loopback, RFC 1918, link-local, CGNAT/tailnet (100.64/10),
  ULA (fc00::/7), multicast and reserved space are refused (SSRF / DNS rebinding);
* NO identity hooks: no ``X-Tenant-Id``, ``X-Effective-Tenant-Id``, ``X-License-Id``,
  ``X-Subject-Id``, cell token or trace id leaves for a third party. The credential is
  the tenant's own secret, plus ``X-W2V-Provider-Hop: 1`` (loop guard);
* retries (:class:`TenantRetryTransport`): ONLY a connect failure (the request provably
  never reached the provider) and a 429 (refused, not executed; ``Retry-After``
  honoured). Never a 5xx and never a read/write timeout — the paid call may already
  be running, and re-sending it bills the tenant twice.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable, Mapping
from typing import Final
from urllib.parse import urlsplit

import httpx

from w2v_common.auth.identity_context import PROVIDER_HOP_HEADER
from w2v_common.clients import base as _retry
from w2v_common.providers.errors import ProviderSlotUnavailableError

logger = logging.getLogger(__name__)

#: Value of the loop-guard header on every tenant-provider request.
PROVIDER_HOP_VALUE: Final = "1"

#: Connect failures: attempts and linear base delay.
CONNECT_MAX_ATTEMPTS: Final = 3
CONNECT_RETRY_DELAY_S: Final = 1.0
#: 429: attempts, base delay, and the cap on an honoured ``Retry-After``.
RATE_LIMIT_MAX_ATTEMPTS: Final = 3
RATE_LIMIT_RETRY_DELAY_S: Final = 2.0
RATE_LIMIT_MAX_DELAY_S: Final = 30.0

#: Never connect here without an orchestrator pin.
_BLOCKED_NETWORKS: Final = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",  # CGNAT — the tailnet
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",  # v4-mapped: checked again as v4 below
        "64:ff9b::/96",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)

#: Test seams: the socket-level transport, the resolver and the retry sleep.
_inner_transport: httpx.AsyncBaseTransport | None = None
Resolver = Callable[[str, int], Awaitable[list[str]]]


async def _system_resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    return [str(info[4][0]) for info in infos]


_resolve: Resolver = _system_resolve
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def set_inner_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    """Swap the socket-level transport (tests: ``httpx.MockTransport`` / respx)."""
    global _inner_transport
    _inner_transport = transport


def is_blocked_address(address: str) -> bool:
    """True for any address a tenant provider must not be reached at without a pin."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip.version == net.version and ip in net for net in _BLOCKED_NETWORKS)


class PinnedTransport(httpx.AsyncBaseTransport):
    """Send every request to ONE origin, connecting to ONE vetted IP."""

    def __init__(
        self, slot: str, base_url: str, pinned_ip: str | None, inner: httpx.AsyncBaseTransport
    ) -> None:
        parts = urlsplit(base_url)
        if not parts.hostname or parts.scheme not in {"http", "https"}:
            raise ProviderSlotUnavailableError(slot, "the provider endpoint is not a valid URL")
        self._slot = slot
        self._scheme = parts.scheme
        self._host = parts.hostname.lower().rstrip(".")
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._ip = pinned_ip
        self._inner = inner

    def _same_origin(self, url: httpx.URL) -> bool:
        port = url.port or (443 if url.scheme == "https" else 80)
        return (
            url.scheme == self._scheme
            and (url.host or "").lower().rstrip(".") == self._host
            and port == self._port
        )

    async def _vetted_ip(self) -> str:
        """The orchestrator's pin, else ONE public address of the host (resolved once)."""
        if self._ip is not None:
            return self._ip
        try:
            addresses = await _resolve(self._host, self._port)
        except OSError as exc:
            raise ProviderSlotUnavailableError(
                self._slot,
                f"the provider host could not be resolved ({type(exc).__name__})",
                transient=True,
            ) from None
        if not addresses:
            raise ProviderSlotUnavailableError(self._slot, "the provider host has no address")
        if any(is_blocked_address(a) for a in addresses):
            # ANY private answer refuses: a mixed answer is how rebinding is staged.
            raise ProviderSlotUnavailableError(
                self._slot, "the provider host resolves to a non-public address"
            )
        self._ip = addresses[0]
        return self._ip

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not self._same_origin(request.url):
            raise httpx.ConnectError(
                "refused: request left the tenant provider's origin", request=request
            )
        ip = await self._vetted_ip()
        # Host was set from the ORIGINAL url when the request was built; only the
        # connect target changes. SNI keeps cert verification on the name.
        request.url = request.url.copy_with(host=ip)
        request.extensions = {**request.extensions, "sni_hostname": self._host}
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class TenantRetryTransport(httpx.AsyncBaseTransport):
    """Retries that cannot double-bill: connect failures and 429 only (module doc)."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        connect_attempt = 0
        limited_attempt = 0
        while True:
            try:
                response = await self._inner.handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                connect_attempt += 1
                if connect_attempt >= CONNECT_MAX_ATTEMPTS:
                    raise
                delay = CONNECT_RETRY_DELAY_S * connect_attempt
                logger.warning(
                    "tenant provider connect failed (%s), retrying in %.1fs",
                    type(exc).__name__,
                    delay,
                )
                await _sleep(delay)
                continue
            if response.status_code != 429:
                return response
            limited_attempt += 1
            if limited_attempt >= RATE_LIMIT_MAX_ATTEMPTS:
                return response
            delay = RATE_LIMIT_RETRY_DELAY_S * limited_attempt
            honoured = _retry._parse_retry_after(response.headers.get("Retry-After"))
            if honoured is not None:
                if honoured > RATE_LIMIT_MAX_DELAY_S:
                    return response  # the provider asks for longer than a stage may wait
                delay = max(delay, honoured)
            await response.aclose()
            logger.warning("tenant provider returned 429, retrying in %.1fs", delay)
            await _sleep(delay)

    async def aclose(self) -> None:
        await self._inner.aclose()


def tenant_headers(
    secret: str | None,
    extra: Mapping[str, str] | None = None,
    *,
    bearer: bool = True,
) -> dict[str, str]:
    """The complete header set a tenant provider receives (besides httpx's own)."""
    headers = {PROVIDER_HOP_HEADER: PROVIDER_HOP_VALUE}
    if bearer and secret:
        headers["Authorization"] = f"Bearer {secret}"
    if extra:
        headers.update(extra)
    return headers


def tenant_http_client(
    slot: str,
    base_url: str,
    pinned_ip: str | None,
    timeout: float,
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx.AsyncClient:
    """A fresh client for one tenant origin. No event hooks, by construction."""
    inner = _inner_transport or httpx.AsyncHTTPTransport()
    return httpx.AsyncClient(
        # Origin + address enforcement OUTSIDE the retries: a refusal is never retried.
        transport=PinnedTransport(slot, base_url, pinned_ip, TenantRetryTransport(inner)),
        timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
        follow_redirects=False,
        headers=dict(headers or {}),
    )


__all__ = [
    "PROVIDER_HOP_VALUE",
    "PinnedTransport",
    "TenantRetryTransport",
    "is_blocked_address",
    "set_inner_transport",
    "tenant_headers",
    "tenant_http_client",
]
