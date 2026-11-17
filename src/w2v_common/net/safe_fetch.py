"""SSRF-safe outbound HTTP for caller-supplied URLs.

Several w2v services fetch a URL that an (authenticated) tenant supplied in a
request body — image references for slide/html_render/video_compiler rendering.
A naive ``client.get(url, follow_redirects=True)`` lets that tenant point the
service at an internal address: a public host can ``30x``-redirect to
``169.254.169.254`` (cloud metadata), an in-cluster ``*.svc`` ClusterIP
(``10.0.0.0/8``), or a Tailnet ``*.internal`` host (``100.64.0.0/10`` CGNAT).

This module is the one fleet-wide guard. It is lifted from
``app_inspector_service``'s proven ``url_validation`` helper (the only service
that already did this correctly) so every URL-fetching service shares one
implementation:

- :func:`validate_public_url` — resolve the hostname and reject any address in a
  private/reserved/loopback/link-local/CGNAT/multicast range, plus known
  cloud-metadata hostnames.
- :func:`safe_get` / :func:`safe_get_sync` — GET a caller URL with redirects
  followed **manually** so every hop is re-validated (``follow_redirects=False``
  on the client send). This is pure transport hardening: it does not change
  status handling — callers still see the final response and raise as they did.

Note: validation does a synchronous ``socket.getaddrinfo``. It is a quick DNS
lookup and matches the upstream app_inspector behaviour; the fetches it guards
are already network-bound image downloads.

Known residual — httpx request event hooks
------------------------------------------
Every hop is built with :meth:`httpx.AsyncClient.build_request` and sanitised
**before** :meth:`~httpx.AsyncClient.send`, so this module governs the complete
effective header set: the caller's ``headers=``, the client's own default
headers, the client cookie jar, and the client-level ``auth=``. It cannot
govern ``event_hooks={"request": [...]}`` — httpx runs those *inside* ``send``,
after the last point this module can touch the request. A client carrying
``w2v_common.clients.base.BaseServiceClient``'s ``identity_event_hook`` would
therefore stamp ``X-Tenant-Id`` / ``X-License-Id`` / ``X-Subject-Id`` onto a
cross-origin redirect hop regardless of the rules below. No current caller
passes such a client (all pass a bare ``httpx.AsyncClient``); do not start.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import socket
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MAX_REDIRECTS = 5
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})

#: Default ceiling on a buffered response body, applied when a caller does not
#: state its own ``max_fetch_bytes``. Before this existed the default was
#: *unbounded*: ``await response.aread()`` into pod memory, so any caller that
#: had not thought about size — notably ``skill_builder``'s tenant-authored
#: ``api_call`` step, whose pod limit is 512Mi — could be pointed at an
#: arbitrarily large public file and OOM.
#:
#: 64 MiB is not an arbitrary round number: it is the ceiling
#: ``w2v_orchestrator`` independently chose for an *untrusted,
#: internet-sourced* media fetch (``visual_media._MAX_VIDEO_BYTES``), which is
#: exactly the class of fetch an unstated cap describes. It is the fleet's own
#: demonstrated risk appetite, not this module's invention.
#:
#: ⚠️ It is deliberately NOT sized for allow-listed presigned-artifact
#: downloads. Two kinds of caller legitimately exceed it, and both must state
#: their own ceiling rather than inherit this one:
#:
#: * ``video_compiler_service`` already does (``settings.max_fetch_bytes``,
#:   512 MiB, MinIO shot/clip assets);
#: * ``video_production_service`` does NOT yet — four artifact fetches
#:   (3 in ``compilation/_segment_builders.py``, 1 in ``compilation/compiler_service.py``)
#:   pass ``allow_hosts=artifact_allow_hosts()`` with no cap, and one of them
#:   pulls an unbounded source recording. Those four MUST gain an explicit
#:   ``max_fetch_bytes=`` in the same PR that re-locks this version, or a long
#:   recording will start soft-failing to a skipped segment.
#:
#: Pass ``0`` (or ``None``) to opt out and stream without a bound.
DEFAULT_MAX_FETCH_BYTES = 64 * 1024 * 1024


class SSRFError(ValueError):
    """Raised when a URL targets a blocked/internal network range."""


class ResponseTooLargeError(ValueError):
    """Raised when a fetched response body exceeds ``max_fetch_bytes``."""


# Private/reserved IPv4 networks that must never be reached by a fleet service.
_BLOCKED_IPV4_NETWORKS = [
    ipaddress.IPv4Network("127.0.0.0/8"),  # Loopback
    ipaddress.IPv4Network("10.0.0.0/8"),  # Private class A (k8s ClusterIP, pods)
    ipaddress.IPv4Network("172.16.0.0/12"),  # Private class B
    ipaddress.IPv4Network("192.168.0.0/16"),  # Private class C
    ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local (cloud metadata)
    ipaddress.IPv4Network("100.64.0.0/10"),  # CGNAT (RFC 6598 — Tailscale *.internal)
    ipaddress.IPv4Network("0.0.0.0/8"),  # "This" network
    ipaddress.IPv4Network("192.0.0.0/24"),  # IETF protocol assignments (RFC 6890)
    ipaddress.IPv4Network("192.88.99.0/24"),  # 6to4 relay anycast (RFC 7526, deprecated)
    ipaddress.IPv4Network("198.18.0.0/15"),  # Benchmarking (RFC 2544)
    ipaddress.IPv4Network("224.0.0.0/4"),  # Multicast
    ipaddress.IPv4Network("240.0.0.0/4"),  # Reserved (future use)
    ipaddress.IPv4Network("255.255.255.255/32"),  # Broadcast
]

# IPv6 prefixes blocked on the WRAPPER address itself, regardless of any IPv4
# address embedded in them. ``2002::/16`` (6to4) and ``2001::/32`` (Teredo) are
# here deliberately: both are deprecated transition mechanisms (RFC 7526 /
# RFC 4380) that no legitimate fetch target uses, and treating the wrapper as
# merely a container for its embedded v4 would let ``2002:0808:0808::1`` through
# on 8.8.8.8's good name while still reaching a 6to4 relay. ``64:ff9b::/96``
# (NAT64) reports ``is_global=True`` to :mod:`ipaddress` and must be explicit.
_BLOCKED_IPV6_NETWORKS = [
    ipaddress.IPv6Network("::/96"),  # IPv4-compatible (RFC 4291 §2.5.5.1, deprecated).
    #   Subsumes ::/128 (unspecified) and ::1/128 (loopback). ``::169.254.169.254``
    #   lives here: ipaddress reports no .ipv4_mapped/.sixtofour/.teredo for it, so
    #   the unwrap in _embedded_ipv4 never sees the metadata address.
    ipaddress.IPv6Network("::ffff:0:0:0/96"),  # IPv4-translated (RFC 2765 SIIT).
    #   The ``::ffff:0:a.b.c.d`` form — note the extra ``0:`` group. Distinct from
    #   IPv4-MAPPED ``::ffff:0:0/96`` (``::ffff:a.b.c.d``), which stays unwrapped
    #   rather than blocked wholesale because a resolver may legitimately return
    #   the mapped form of a PUBLIC address.
    ipaddress.IPv6Network("fc00::/7"),  # Unique local
    ipaddress.IPv6Network("fe80::/10"),  # Link-local
    ipaddress.IPv6Network("fec0::/10"),  # Site-local (RFC 3879, deprecated)
    ipaddress.IPv6Network("ff00::/8"),  # Multicast
    ipaddress.IPv6Network("2002::/16"),  # 6to4 (RFC 3056, deprecated by RFC 7526)
    ipaddress.IPv6Network("2001::/32"),  # Teredo (RFC 4380)
    ipaddress.IPv6Network("64:ff9b::/96"),  # NAT64 well-known prefix (is_global=True!)
    ipaddress.IPv6Network("64:ff9b:1::/48"),  # NAT64 local-use prefix (RFC 8215)
    ipaddress.IPv6Network("100::/64"),  # Discard-only (RFC 6666)
]

# NOTE: ``::ffff:0:0/96`` (IPv4-mapped) is deliberately NOT in the wholesale
# list above. A resolver may legitimately hand back the v4-mapped form of a
# PUBLIC address (``getaddrinfo`` with ``AI_V4MAPPED``), so blocking the whole
# /96 would break public fetches. Mapped addresses are instead unwrapped and
# their embedded IPv4 judged on the IPv4 block-list — see :func:`_embedded_ipv4`.

#: The IPv4-compatible prefix (RFC 4291 §2.5.5.1), used by :func:`_embedded_ipv4`
#: to derive ``::a.b.c.d``. :mod:`ipaddress` exposes an accessor for the mapped,
#: 6to4 and Teredo forms but **not** for this one, so it is computed by hand.
#: The same prefix appears in :data:`_BLOCKED_IPV6_NETWORKS` as a wholesale
#: block; the two are deliberately redundant. The wholesale entry is what
#: actually stops ``::100.64.0.1`` today, and the unwrap is the defence-in-depth
#: half: it keeps the *embedded* verdict correct — so a future reader relaxing
#: the wholesale entry (the form is deprecated, and stacks differ on whether it
#: routes) does not silently reopen the tailnet/RFC1918/loopback targets with it.
_IPV4_COMPATIBLE_NETWORK = ipaddress.IPv6Network("::/96")

# Hostnames known to expose cloud metadata services (name-based block; the IP
# block covers 169.254.169.254 already).
_BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "localhost",
    }
)

# Hostname suffixes that only ever name infrastructure inside the estate. These
# are rejected BEFORE any DNS lookup: split-horizon DNS means the same name can
# resolve publicly from one vantage point and to a tailnet/ClusterIP address
# from inside a pod, so the name itself — not just where it happens to resolve
# right now — is the thing to refuse.
_BLOCKED_HOST_SUFFIXES = (
    ".internal",  # Tailscale split-DNS estate (*.internal -> 100.64.0.0/10)
    ".local",  # mDNS / Bonjour
    ".localhost",
    ".svc",  # Kubernetes short Service form
    ".cluster.local",  # Kubernetes cluster domain
)


def _normalize_allow_hosts(allow_hosts: Iterable[str] | None) -> frozenset[str]:
    """Lower-case + strip an allow-list of trusted host (or ``host:port``) keys."""
    if not allow_hosts:
        return frozenset()
    return frozenset(h.strip().lower() for h in allow_hosts if h and h.strip())


def _normalize_host(parsed: Any) -> str:
    """Return the URL's hostname lower-cased with any trailing dot(s) stripped.

    A fully-qualified name may carry a trailing root dot — ``db.internal.``
    resolves identically to ``db.internal`` but does NOT satisfy
    ``endswith(".internal")``. Normalising once, here, keeps the suffix block
    and the allow-list from disagreeing about what the host is.
    """
    return (parsed.hostname or "").rstrip(".").lower()


def _host_in_allowlist(host: str, port: int | None, allow_hosts: frozenset[str]) -> bool:
    """Return True if the host (or ``host:port``) is an explicitly-trusted host.

    Matches either the bare hostname or the ``host:port`` form so a caller can
    allow ``blob.internal`` (any port) or ``blob.internal:8389`` (exact).
    ``host`` must already be normalised by :func:`_normalize_host`.
    """
    if not allow_hosts or not host:
        return False
    if host in allow_hosts:
        return True
    return port is not None and f"{host}:{port}" in allow_hosts


def _host_is_blocked_by_name(host: str) -> bool:
    """Return True if the hostname itself names estate infrastructure.

    Checked before resolution — see :data:`_BLOCKED_HOST_SUFFIXES`.
    """
    return host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_HOST_SUFFIXES)


def _ipv4_compatible(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Return the IPv4 address in an ``::a.b.c.d`` literal, or ``None``.

    The IPv4-COMPATIBLE form (RFC 4291 §2.5.5.1) is the one embedded form
    :mod:`ipaddress` has no accessor for — ``.ipv4_mapped``, ``.sixtofour`` and
    ``.teredo`` all return ``None`` for ``::100.64.0.1``, which is additionally
    reported as ``is_global=True`` / ``is_private=False``. This mirrors those
    accessors' contract (``None`` when the address is not of that form) so
    :func:`_embedded_ipv4` can treat all four uniformly.

    ``::`` (unspecified) and ``::1`` (loopback) sit inside ``::/96`` by the
    arithmetic but are not a wrapped IPv4 target — unwrapping them would assert
    that a fetch to ``::1`` is a fetch to ``0.0.0.1``, which is nonsense — so
    they are excluded here and left to the wrapper block-list, exactly as
    ``.ipv4_mapped`` excludes everything outside its own /96.

    Genuinely low-valued compatible addresses such as ``::0.0.0.2`` are NOT
    special-cased: they unwrap to ``0.0.0.2``, which ``0.0.0.0/8`` ("this
    network") already blocks on the IPv4 list. That is the right answer, not an
    accident — such an address is never a legitimate fetch target.
    """
    if ip not in _IPV4_COMPATIBLE_NETWORK:
        return None
    if ip.is_unspecified or ip.is_loopback:
        return None
    # Membership in ::/96 guarantees the value fits in 32 bits.
    return ipaddress.IPv4Address(int(ip))


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """Return every IPv4 address embedded in an IPv6 transition/mapping format.

    An IPv6 literal can smuggle an internal IPv4 target past a guard that only
    inspects the v6 form. Four forms carry a v4 destination:
    ``::ffff:100.64.0.1`` (IPv4-mapped), ``2002:6440:1::1`` (6to4), Teredo, and
    ``::100.64.0.1`` (IPv4-compatible, RFC 4291 §2.5.5.1 — see
    :func:`_ipv4_compatible` for why it needs its own derivation).
    """
    embedded: list[ipaddress.IPv4Address] = []
    if (mapped := ip.ipv4_mapped) is not None:
        embedded.append(mapped)
    if (sixtofour := ip.sixtofour) is not None:
        embedded.append(sixtofour)
    if (teredo := ip.teredo) is not None:
        _server, client = teredo
        embedded.append(client)
    if (compatible := _ipv4_compatible(ip)) is not None:
        embedded.append(compatible)
    return embedded


def _is_blocked_ip(addr: str) -> bool:
    """Return True if an IP address string falls within a blocked range.

    For IPv6 the verdict is **additive**: the address is blocked if the wrapper
    prefix is blocked OR any IPv4 address embedded in it is blocked. Letting the
    embedded verdict *replace* the wrapper's would be a widening — a 6to4
    address wrapping a public IPv4 would then be judged solely on that public
    IPv4 and unblock the deprecated 6to4 relay path.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in _BLOCKED_IPV4_NETWORKS)
    if isinstance(ip, ipaddress.IPv6Address):
        if any(ip in net for net in _BLOCKED_IPV6_NETWORKS):
            return True
        return any(v4 in net for v4 in _embedded_ipv4(ip) for net in _BLOCKED_IPV4_NETWORKS)
    return False


def validate_public_url(url: str, *, allow_hosts: Iterable[str] | None = None) -> str | None:
    """Validate ``url`` and return the single resolved IP to pin the connection to.

    Rejects the hostname **before resolution** if it is a known cloud-metadata
    host or carries an estate-internal suffix (``.internal``, ``.local``,
    ``.localhost``, ``.svc``, ``.cluster.local``) — split-horizon DNS means such
    a name can resolve publicly from one vantage point and to a tailnet or
    ClusterIP address from inside a pod, so the name is refused on sight.
    Otherwise resolves the hostname **exactly once** and rejects it if any
    resolved address is in a private / reserved / loopback / link-local /
    CGNAT / multicast range. The
    address returned is the one the caller must connect to — pinning it closes
    the DNS-rebinding TOCTOU window (validate-time and connect-time would
    otherwise resolve independently and could differ).

    Returns:
        The validated IP string to pin the connection to, or ``None`` when the
        host is in ``allow_hosts`` (an explicitly-trusted host is fetched with
        its original hostname and no pin — see below).

    ``allow_hosts`` is an explicit trust-list of artifact hosts (bare hostname
    or ``host:port``) that legitimately resolve into an otherwise-blocked
    internal range — e.g. presigned MinIO URLs at ``blob.internal:8389``. A URL
    whose host is in ``allow_hosts`` bypasses the IP/hostname blocks (the scheme
    is still required to be http(s)) and is fetched WITHOUT IP-pinning so the
    presigned ``Host``/SigV4 signature is preserved byte-for-byte. The allow-list
    is consulted on **every** redirect hop, so a trusted host that
    ``30x``-redirects to a *non*-allowed internal address is still rejected.

    Raises:
        ValueError: if a hostname cannot be extracted, or the scheme is not http(s).
        SSRFError: if the URL resolves to a blocked address/hostname, or DNS
            resolution fails (fail-closed — an unresolvable host is not fetched).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}: {url}")

    hostname = _normalize_host(parsed)
    if not hostname:
        raise ValueError(f"Cannot extract hostname from URL: {url}")

    if _host_in_allowlist(hostname, parsed.port, _normalize_allow_hosts(allow_hosts)):
        # Explicitly-trusted artifact host (e.g. presigned MinIO) — skip the
        # internal-range block AND the IP-pin (None). Scheme validated above.
        return None

    if _host_is_blocked_by_name(hostname):
        raise SSRFError(f"Blocked hostname: {hostname} (estate-internal / metadata name)")

    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        # Fail-closed: a host we cannot resolve here must not be handed to the
        # transport (which would resolve it independently — the TOCTOU window).
        raise SSRFError(
            f"DNS resolution failed for {hostname}: {_safe_url_for_message(url)}"
        ) from exc

    if not addrinfo:
        raise SSRFError(f"DNS returned no addresses for {hostname}: {_safe_url_for_message(url)}")

    for _family, _type, _proto, _canonname, sockaddr in addrinfo:
        addr = str(sockaddr[0])
        if _is_blocked_ip(addr):
            raise SSRFError(f"URL {_safe_url_for_message(url)} resolves to blocked address {addr}")

    # Every resolved address passed the block-list. Pin to the first — that is
    # the address the transport would try first anyway, and it is now fixed so
    # the connect step cannot re-resolve to a different (internal) address.
    return str(addrinfo[0][4][0])


def _safe_url_for_message(url: str) -> str:
    """``url`` with its credentials removed, for embedding in an error message.

    ``SSRFError`` text reaches logs, exception chains, and in places a client
    response — and the URLs this module refuses are frequently credentialed. A
    presigned S3/MinIO URL carries ``X-Amz-Signature`` in the QUERY STRING, and
    ``_pin`` shows ``user:pass@`` userinfo is an expected shape here. Embedding
    the raw URL therefore published the very secret the fetch was denied over.

    Host, port, scheme and path are kept — those are what makes the message
    actionable. Query and userinfo are dropped wholesale rather than
    pattern-matched: an allow-list of "safe" query keys is the kind of thing that
    silently stops covering a new signature parameter.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return "<unparseable url>"
    netloc = parsed.hostname or ""
    if ":" in netloc:
        netloc = f"[{netloc}]"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    redacted_query = "?<redacted>" if parsed.query else ""
    return f"{parsed.scheme}://{netloc}{parsed.path}{redacted_query}"


def _pin(url: str, ip: str | None) -> tuple[str, dict[str, str], dict[str, str]]:
    """Return ``(request_url, headers, extensions)`` for an IP-pinned request.

    When ``ip`` is set, the connection target host is rewritten to the literal
    IP while the original hostname is preserved for the ``Host`` header and for
    TLS (``sni_hostname`` drives both the ClientHello SNI and the certificate
    hostname-verification in httpx), so a pinned https fetch still verifies the
    cert against the hostname — not the IP. When ``ip`` is ``None`` (trusted
    allow-listed host) the URL is returned unchanged with no pin.
    """
    if ip is None:
        return url, {}, {}
    parsed = urlparse(url)
    host_bracket = f"[{ip}]" if ":" in ip else ip
    netloc = f"{host_bracket}:{parsed.port}" if parsed.port is not None else host_bracket
    if parsed.username:
        auth = parsed.username + (f":{parsed.password}" if parsed.password else "")
        netloc = f"{auth}@{netloc}"
    pinned_url = urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )
    # Trailing root dot stripped: RFC 6066 forbids it in the TLS SNI extension.
    hostname = _normalize_host(parsed)
    host_disp = f"[{hostname}]" if ":" in hostname else hostname
    host_header = f"{host_disp}:{parsed.port}" if parsed.port is not None else host_disp
    return pinned_url, {"Host": host_header}, {"sni_hostname": hostname}


def _validate_redirect(
    location: str | None,
    current_url: str,
    *,
    allow_hosts: Iterable[str] | None,
) -> tuple[str, str | None]:
    """Validate a redirect ``Location`` and return ``(next_url, next_ip)``.

    Re-resolves and re-pins from the redirect target's OWN hostname — never
    reuses the prior hop's IP. Raises SSRFError on a missing Location or a
    target that resolves to a blocked address.
    """
    if not location:
        raise SSRFError(
            f"Redirect with no Location header from {_safe_url_for_message(current_url)}"
        )
    next_url = urljoin(current_url, location)
    next_ip = validate_public_url(next_url, allow_hosts=allow_hosts)
    return next_url, next_ip


_BODY_HEADERS = frozenset({"content-type", "content-length", "transfer-encoding"})

# Headers permitted to SURVIVE a cross-origin redirect hop. This is deliberately
# an ALLOW-list, not a deny-list.
#
# The deciding call site is ``skill_builder``'s ``api_call`` step, which hands
# this module a header dict authored verbatim by the tenant. Against an
# attacker-chosen *name* space a deny-list can only ever enumerate the names
# somebody already thought of — it fails open on the next one, and on every
# estate header added later (``X-Caller-Authorization``, ``X-License-Key``,
# ``X-License-Id``, ``X-Subject-Id``, ``X-Tenant-Id`` …). Inverting makes the
# failure mode "a new header is dropped", which is visible and recoverable,
# instead of "a new credential is replayed to a redirector-chosen host".
#
# The cost was enumerated across all 12 fleet call sites before choosing: only
# ONE production caller depends on a custom header surviving a cross-origin hop
# (``w2v_orchestrator`` site_capture's ``User-Agent``, allowed below). The other
# header-passing caller (``video_production_service``'s signed compilation
# webhook) uses ``max_redirects=0``, so no cross-origin hop exists for it.
#
# Membership rule: content negotiation, conditional requests and ranges — never
# anything that authenticates, identifies or authorises the caller.
_CROSS_ORIGIN_SAFE_HEADERS = frozenset(
    {
        # Transport-mandatory. ``host`` is re-derived per hop (see _pin), so it
        # always describes the hop's own origin, never the previous one.
        "host",
        "connection",
        "accept-encoding",
        # Content negotiation.
        "accept",
        "accept-language",
        "user-agent",
        # Body description — only ever present when the body itself survived
        # (307/308); _drop_body_headers runs first on a body-dropping hop.
        "content-type",
        "content-length",
        "transfer-encoding",
        # Caching / conditional / ranged reads.
        "cache-control",
        "pragma",
        "range",
        "if-match",
        "if-none-match",
        "if-modified-since",
        "if-unmodified-since",
        "if-range",
        "te",
    }
)

# Headers the CALLER may never set: this module owns them because they select
# the connection target rather than describe the request. ``Host`` decides which
# vhost is reached on the validated IP — a caller-supplied one would let a
# request validated against (and pinned to, or allow-listed for) one host be
# served by a different vhost on the same address. On the pinned path _pin's
# Host wins anyway; on the un-pinned ``allow_hosts`` path (presigned MinIO)
# nothing overrode it before this, which is the vhost-confusion gap.
_CALLER_FORBIDDEN_HEADERS = frozenset({"host"})


def _redirect_method(method: str, status_code: int) -> tuple[str, bool]:
    """Return ``(next_method, keep_body)`` for a redirect, mirroring httpx/RFC 7231.

    - **307 / 308** — method and body are preserved.
    - **303** — always becomes GET (except HEAD, which stays HEAD).
    - **301 / 302** — POST becomes GET (the universal browser behaviour httpx
      also implements); every other method is preserved.

    ``keep_body`` is False whenever the method changed to GET, because the body
    must not be replayed onto a method that has no body.
    """
    if status_code in (307, 308):
        return method, True
    if status_code == 303:
        return (method, True) if method == "HEAD" else ("GET", False)
    if status_code in (301, 302) and method == "POST":
        return "GET", False
    return method, True


def _drop_caller_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop headers the caller is not allowed to set — see :data:`_CALLER_FORBIDDEN_HEADERS`."""
    return {k: v for k, v in headers.items() if k.lower() not in _CALLER_FORBIDDEN_HEADERS}


def _drop_body_headers(request: httpx.Request) -> None:
    """Drop body-describing headers in place when a redirect dropped the body.

    Without this a bodiless GET would still advertise ``Content-Type:
    application/json`` and a stale ``Content-Length``.
    """
    for name in _BODY_HEADERS:
        if name in request.headers:
            del request.headers[name]


def _apply_cross_origin_header_policy(request: httpx.Request) -> None:
    """Reduce a built request's headers in place to :data:`_CROSS_ORIGIN_SAFE_HEADERS`.

    Applied to the request **as httpx built it** rather than to the caller's
    ``headers=`` dict, because three separate sources feed the wire and only the
    built request sees all three:

    1. the caller's per-request ``headers=``;
    2. the client's own default headers (``httpx.AsyncClient(headers=...)``) —
       merged in by ``build_request`` and previously invisible to any strip
       done on the caller dict;
    3. the client's **cookie jar** — ``build_request`` materialises it into a
       ``Cookie`` header. This matters beyond the caller's own ``Cookie=``:
       a jar built from a plain dict (``httpx.AsyncClient(cookies={...})``)
       carries no domain, so ``http.cookiejar`` attaches it to *every* host.
       (Cookies harvested from ``Set-Cookie`` are domain-scoped by the jar and
       would not cross hosts on their own — the dict form is the leak.)
    """
    for name in list(request.headers.keys()):
        if name.lower() not in _CROSS_ORIGIN_SAFE_HEADERS:
            del request.headers[name]


_DEFAULT_PORTS = {"http": 80, "https": 443}

# Request kwargs that would smuggle a body or a credential past the per-hop
# guards below, because they are consumed by httpx rather than by this module.
_REJECTED_KWARGS = {
    "files": (
        "files= is a one-shot body: it would be consumed by the first hop and "
        "silently sent empty on a redirect. Read it into bytes and pass content=."
    ),
    "auth": (
        "auth= is applied by httpx per-request, so it cannot be dropped on a "
        "cross-origin redirect. Pass the credential in headers= instead."
    ),
}


def _same_origin(a: str, b: str) -> bool:
    """True if two URLs share scheme + host + port — httpx's own credential rule.

    Comparing hostname alone would keep ``Authorization`` across an
    ``https -> http`` downgrade on the same host, sending it in cleartext.
    """
    pa, pb = urlparse(a), urlparse(b)
    if pa.scheme != pb.scheme:
        return False
    if (pa.hostname or "").lower() != (pb.hostname or "").lower():
        return False
    return (pa.port or _DEFAULT_PORTS.get(pa.scheme)) == (pb.port or _DEFAULT_PORTS.get(pb.scheme))


def _check_replayable_body(content: object, data: object) -> None:
    """Reject a one-shot request body.

    A redirect is followed manually, so the body may have to be sent more than
    once. A file handle or an (async) iterator is consumed by the first hop and
    would silently send an empty body on the second — fail loudly instead.
    """
    if content is not None and not isinstance(content, bytes | str):
        raise ValueError(
            "safe_request only accepts a replayable body (bytes | str); "
            "streaming/file-like bodies cannot survive a manually-followed redirect"
        )
    if data is not None and not isinstance(data, Mapping):
        raise ValueError(
            "safe_request only accepts a replayable data= body (a Mapping); "
            "streaming/file-like bodies cannot survive a manually-followed redirect"
        )


def _reject_unguardable_kwargs(kwargs: dict[str, object]) -> None:
    """Refuse httpx kwargs this module cannot subject to its per-hop guards."""
    for name, why in _REJECTED_KWARGS.items():
        if name in kwargs:
            raise ValueError(f"safe_request does not support {name}=: {why}")


# RFC 9110 §5.6.2 `token` — the complete set of characters a method may contain.
_TCHAR = frozenset("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _validate_method(method: str) -> None:
    """Reject a method that is not a bare RFC 9110 token.

    A method containing CR/LF splits the request line and lets the caller inject
    headers of its own (``"GET\\r\\nX-Injected: 1"`` becomes a second line on the
    wire). httpx does NOT validate this — ``build_request`` stores the method
    verbatim, and h11 formats the request line with a raw
    ``b"%s %s HTTP/1.1\\r\\n" %`` interpolation.

    Today the only thing that stops it is h11's own ``Illegal method
    characters`` check, one layer below this module. That is a real defence but
    the wrong owner: it disappears under any transport that is not h11 — which
    includes ``httpx.MockTransport``, i.e. every mocked test in the fleet would
    pass a splitting method straight through and report the guard healthy. The
    method reaches this module from tenant-authored config in
    ``skill_builder``'s ``api_call`` step, which only checks that it is a
    ``str``, so the value really is attacker-shaped.
    """
    if not method or not _TCHAR.issuperset(method):
        raise ValueError(
            f"safe_request rejects a non-token HTTP method {method!r}: "
            "a method must be a bare RFC 9110 token (no whitespace, CR, LF or NUL)"
        )


def _content_length_exceeds(response: httpx.Response, cap: int | None) -> bool:
    """True if the response's declared ``Content-Length`` exceeds ``cap``."""
    if not cap:
        return False
    raw = response.headers.get("content-length")
    if raw is None:
        return False
    try:
        return int(raw) > cap
    except ValueError:
        return False  # malformed header — rely on the streaming byte counter


def _build_hop_request(
    client: httpx.Client | httpx.AsyncClient,
    method: str,
    request_url: str,
    *,
    headers: dict[str, str],
    extensions: dict[str, Any],
    content: bytes | str | None,
    json: object,
    data: Mapping[str, str] | None,
    keep_body: bool,
    cross_origin: bool,
    kwargs: dict[str, Any],
) -> httpx.Request:
    """Build one hop's request and apply this module's header policy to it.

    ``client.stream()`` is exactly ``build_request()`` + ``send(stream=True)``;
    splitting the two is what gives this module a point at which the *complete*
    effective header set — caller headers, client default headers and the
    materialised cookie jar — exists and can still be edited.
    """
    request = client.build_request(
        method,
        request_url,
        headers=headers,
        extensions=extensions,
        content=content if keep_body else None,
        json=json if keep_body else None,
        data=data if keep_body else None,
        **kwargs,
    )
    if not keep_body:
        _drop_body_headers(request)
    if cross_origin:
        _apply_cross_origin_header_policy(request)
    return request


def _hop_auth(cross_origin: bool) -> dict[str, None]:
    """Return the ``auth=`` keyword to splat into ``send`` for this hop.

    ``auth=None`` on a cross-origin hop disables the client-level ``auth=`` flow
    (httpx falls through to a no-op ``Auth()``), so a client constructed with
    ``httpx.AsyncClient(auth=...)`` cannot re-sign a request aimed at a
    redirector-chosen host. Per-call ``auth=`` is refused outright — see
    :data:`_REJECTED_KWARGS`.

    Returned as a kwargs mapping rather than a value so the same-origin case can
    omit the argument entirely instead of naming httpx's ``USE_CLIENT_DEFAULT``
    sentinel, whose type is not part of httpx's public surface.
    """
    return {"auth": None} if cross_origin else {}


#: Headers that describe the ON-THE-WIRE body and are FALSE of a decoded one.
#: ``aiter_bytes()``/``aread()`` return bytes httpx has already content-decoded, so
#: carrying these over to the materialized Response would misdescribe it.
_WIRE_ONLY_HEADERS = ("content-encoding", "content-length", "transfer-encoding")


def _materialize(response: httpx.Response, content: bytes) -> httpx.Response:
    """Rebuild a fully-read Response from streamed ``content`` (drops the live stream).

    ``content`` is ALREADY content-decoded — ``aiter_bytes()`` and ``aread()`` run
    httpx's ``Content-Encoding`` decoder as they stream. Constructing the new
    Response with the origin's headers therefore cannot include
    ``Content-Encoding``: ``httpx.Response.__init__`` eagerly ``read()``s a
    ``content=`` body through the decoder for that header, so re-attaching it makes
    httpx gunzip plaintext and raise ``DecodingError: Error -3 while decompressing
    data: incorrect header check`` on the FIRST body access.

    That is not hypothetical: it is what ``site_capture`` hit live against
    https://www.anthropic.com/news on 2026-08-10, and — because the fetch there is
    a best-effort ``except`` -> ``""`` — it degraded silently into an ungrounded
    promo script. Most of the public web negotiates gzip/br, so this was the
    common path for every ``safe_*`` caller in the fleet, not an edge case.

    ``Content-Length``/``Transfer-Encoding`` go with it: both describe the wire
    body. httpx recomputes ``Content-Length`` from the bytes actually held.
    """
    extensions = {k: v for k, v in response.extensions.items() if k != "network_stream"}
    headers = httpx.Headers(
        [(k, v) for k, v in response.headers.multi_items() if k.lower() not in _WIRE_ONLY_HEADERS]
    )
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=content,
        request=response.request,
        extensions=extensions,
    )


#: A terminal-response consumer for :func:`_safe_request_to_sink`.
#:
#: Called with ``(response, max_fetch_bytes, original_url)`` on the FINAL (non-redirect)
#: hop, after the ``Content-Length`` pre-check has already passed, with the response
#: still un-read and streaming. Whatever it returns becomes the public call's return
#: value. The loop closes the response afterwards either way.
type _BodySink[T] = Callable[[httpx.Response, int | None, str], Awaitable[T]]


async def _buffer_sink(
    response: httpx.Response, max_fetch_bytes: int | None, url: str
) -> httpx.Response:
    """Accumulate the decoded body in memory and rebuild a read Response.

    This is the historical terminal of :func:`safe_request`, lifted verbatim into a
    sink so the streaming-to-disk path can share the redirect/validation loop rather
    than re-implement it. Peak RSS is ~2x the body (the chunk list plus the joined
    copy), which is exactly why :func:`safe_get_to_path` exists for large bodies.
    """
    if not max_fetch_bytes:
        await response.aread()
        return _materialize(response, response.content)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_fetch_bytes:
            raise ResponseTooLargeError(f"Response body exceeds {max_fetch_bytes} bytes: {url}")
        chunks.append(chunk)
    return _materialize(response, b"".join(chunks))


def _open_binary_for_write(path: Path) -> BinaryIO:
    """``path.open("wb")`` behind a concrete return type.

    ``asyncio.to_thread`` erases the mode literal, so passing ``Path.open`` to it
    directly leaves the handle typed as the full text|binary union and every
    ``write(bytes)`` below becomes a type error.
    """
    return path.open("wb")


def _path_sink(dest: Path) -> _BodySink[Path]:
    """Build a sink that streams the decoded body to ``dest`` atomically.

    Constant memory: one chunk at a time, never a full copy. The bytes are written
    to a uuid7-suffixed SIBLING of ``dest`` (same directory ⇒ same filesystem, so the
    final ``os.replace`` is a true atomic rename and never degrades to a copy), then
    renamed onto ``dest`` (``Path.replace`` is ``os.replace``). ``dest`` therefore only ever appears complete — which is
    load-bearing for ``video_compiler_service``'s per-source dedupe, where one fetched
    file is the shared ``-i`` input of N shots and a torn file silently ships a short
    film rather than failing.

    The byte cap counts DECODED bytes, identically to :func:`_buffer_sink`:
    ``aiter_bytes()`` yields post-``Content-Encoding``-decoder chunks, so a compression
    bomb is measured at its expanded size. ``max_fetch_bytes`` of ``0``/``None`` means
    unlimited, again matching the buffering path.

    Writes are pushed to a worker thread (:func:`asyncio.to_thread`) so a slow or
    blocked disk cannot stall the event loop for the duration of a multi-GiB download.

    On ANY failure — cap breach, transport error, disk-full, or
    ``asyncio.CancelledError`` — the partial temp file is unlinked and the original
    exception propagates unchanged. The cleanup is ``contextlib.suppress(OSError)``d
    because an ``unlink`` failure (permissions, a disk-full that also fails the unlink)
    must never replace the real cause. ``dest`` itself is never touched on failure, so
    a failed re-fetch cannot destroy a good earlier one.
    """

    async def sink(response: httpx.Response, max_fetch_bytes: int | None, url: str) -> Path:
        tmp = dest.with_name(f"{dest.name}.part-{uuid.uuid7().hex}")
        try:
            handle = await asyncio.to_thread(_open_binary_for_write, tmp)
            try:
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if max_fetch_bytes and total > max_fetch_bytes:
                        raise ResponseTooLargeError(
                            f"Response body exceeds {max_fetch_bytes} bytes: {url}"
                        )
                    await asyncio.to_thread(handle.write, chunk)
                await asyncio.to_thread(handle.flush)
            finally:
                # Synchronous on purpose: this runs on the cancellation path too, and
                # `await`ing in a `finally` while being cancelled is not reliable.
                handle.close()
            tmp.replace(dest)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
        return dest

    return sink


async def _safe_request_to_sink[T](
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    content: bytes | str | None,
    json: object,
    data: Mapping[str, str] | None,
    max_redirects: int,
    allow_hosts: Iterable[str] | None,
    max_fetch_bytes: int | None,
    sink: _BodySink[T],
    raise_for_status: bool = False,
    kwargs: dict[str, Any],
) -> T:
    """The SSRF-safe redirect loop, parameterised by what consumes the final body.

    Every guard — ``validate_public_url`` seeding, per-hop ``_pin`` IP pinning,
    ``_validate_redirect``, the ``keep_body`` / ``cross_origin`` latches,
    ``_hop_auth`` / ``_apply_cross_origin_header_policy``, the forced-off
    ``follow_redirects``, ``allow_hosts``, and the ``_content_length_exceeds``
    pre-check — lives HERE, above the sink, so a new terminal cannot skip one.
    See :func:`safe_request` for the full public semantics.

    ``raise_for_status`` defaults to **False**, which is the historical contract of
    every buffering entry point (:func:`safe_request` and its wrappers): they RETURN
    the final response and let the caller decide, because several fleet callers
    inspect a non-2xx body on purpose. Only a terminal that writes the body
    SOMEWHERE THE CALLER CANNOT INSPECT IT opts in — :func:`safe_get_to_path`, where
    a 404 would otherwise land a few bytes of S3 ``<Error>NoSuchKey</Error>`` XML at
    ``dest`` and return successfully, handing ffmpeg a corrupt input instead of
    raising into ``video_compiler_service``'s bad-asset taxonomy.

    The check sits INSIDE the non-redirect branch, before the sink is invoked, so
    the error body is never streamed anywhere: every redirect hop has already been
    validated by the loop, and the sink never opens a file for a failed fetch.
    """
    _validate_method(method)
    _check_replayable_body(content, data)
    _reject_unguardable_kwargs(kwargs)
    kwargs.pop("follow_redirects", None)
    extra_headers = _drop_caller_headers(kwargs.pop("headers", None) or {})
    extra_ext = dict(kwargs.pop("extensions", None) or {})
    current_url = url
    keep_body = True
    cross_origin = False
    ip = validate_public_url(url, allow_hosts=allow_hosts)
    for _hop in range(max_redirects + 1):
        request_url, pin_headers, pin_ext = _pin(current_url, ip)
        request = _build_hop_request(
            client,
            method,
            request_url,
            headers={**extra_headers, **pin_headers},
            extensions={**extra_ext, **pin_ext},
            content=content,
            json=json,
            data=data,
            keep_body=keep_body,
            cross_origin=cross_origin,
            kwargs=kwargs,
        )
        response = await client.send(
            request, stream=True, follow_redirects=False, **_hop_auth(cross_origin)
        )
        try:
            if response.status_code not in _REDIRECT_STATUS:
                if raise_for_status:
                    # BEFORE the sink: an error body must never reach the terminal.
                    # httpx's own raiser, so .response/.request/message are standard
                    # and callers keep catching plain httpx.HTTPStatusError.
                    response.raise_for_status()
                if _content_length_exceeds(response, max_fetch_bytes):
                    raise ResponseTooLargeError(
                        f"Response Content-Length exceeds {max_fetch_bytes} bytes: {url}"
                    )
                return await sink(response, max_fetch_bytes, url)
            location = response.headers.get("location")
            status_code = response.status_code
        finally:
            await response.aclose()
        next_url, ip = _validate_redirect(location, current_url, allow_hosts=allow_hosts)
        method, hop_keeps_body = _redirect_method(method, status_code)
        # Latch: once a hop drops the body it is gone for good. Re-deriving it
        # per hop would let 302 (drop) -> 307 (keep) resurrect a body that was
        # deliberately discarded, and replay it to a redirector-chosen host.
        keep_body = keep_body and hop_keeps_body
        # Same latch for the header reduction: A(secret) -> B -> A must not
        # hand the credential back just because the third hop is same-origin
        # with the first — B chose that hop, and B has already seen nothing.
        cross_origin = cross_origin or not _same_origin(current_url, next_url)
        current_url = next_url
    raise SSRFError(f"Too many redirects (> {max_redirects}) fetching {_safe_url_for_message(url)}")


async def safe_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    content: bytes | str | None = None,
    json: Any = None,
    data: Mapping[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """Send ``method url`` with SSRF-safe, IP-pinned, manually-followed redirects (async).

    The initial URL and every redirect hop are resolved **once** and the
    connection is pinned to that validated IP (via :func:`_pin`), closing the
    DNS-rebinding TOCTOU window — a public host cannot rebind to an internal
    address between validation and connect. ``follow_redirects`` is forced off.

    ``allow_hosts`` (bare hostname or ``host:port``) names artifact hosts that
    legitimately resolve internal — e.g. presigned ``blob.internal`` URLs;
    those are fetched un-pinned (Host/signature preserved) and re-checked on
    every hop.

    ``max_fetch_bytes`` caps the **response** body: an oversized
    ``Content-Length`` is rejected before the body is read, and the streamed
    body is aborted the moment the running total exceeds the cap — bounding
    memory. It defaults to :data:`DEFAULT_MAX_FETCH_BYTES` (64 MiB); pass ``0``
    or ``None`` to stream without a bound. The *request* body is caller-sized
    and not capped here.

    Redirect semantics follow :func:`_redirect_method`: 307/308 preserve method
    and body; 301/302/303 downgrade POST to GET and drop the body (along with
    its ``Content-Type``/``Content-Length`` headers).

    On a hop that changes origin the request is reduced to
    :data:`_CROSS_ORIGIN_SAFE_HEADERS` and the client-level ``auth=`` is
    disabled, so no credential or identity header — the caller's, the client's
    defaults, or one materialised from the client cookie jar — is replayed to a
    redirector-chosen host. The reduction is latched: once any hop has crossed
    origin, every later hop stays reduced, so ``A -> B -> A`` cannot restore it.

    A caller-supplied ``Host`` header is refused (see
    :data:`_CALLER_FORBIDDEN_HEADERS`): the Host selects the vhost served on the
    validated address, so this module owns it.

    Raises:
        SSRFError: blocked target, blocked redirect, or too many redirects.
        ResponseTooLargeError: body exceeds ``max_fetch_bytes``.
        ValueError: malformed URL, or a non-replayable (streaming) body.
        httpx.HTTPError: transport/timeout errors propagate unchanged.
    """
    return await _safe_request_to_sink(
        client,
        method,
        url,
        content=content,
        json=json,
        data=data,
        max_redirects=max_redirects,
        allow_hosts=allow_hosts,
        max_fetch_bytes=max_fetch_bytes,
        sink=_buffer_sink,
        kwargs=kwargs,
    )


async def safe_get_to_path(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: object,
) -> Path:
    """GET ``url`` STRAIGHT TO ``dest`` — same guards as :func:`safe_get`, constant memory.

    Identical SSRF posture to :func:`safe_get`: it runs the very same redirect loop
    (:func:`_safe_request_to_sink`) with the same ``validate_public_url`` seeding,
    per-hop ``_pin`` IP pinning, ``_validate_redirect``, cross-origin header/auth
    reduction latches, forced-off ``follow_redirects``, ``allow_hosts`` and
    ``Content-Length`` pre-check. Only the TERMINAL differs: bytes go to a file
    instead of into a ``list[bytes]``.

    Why it exists: :func:`safe_get` materialises the whole body and peaks at roughly
    TWICE it (the chunk list plus the joined copy). ``video_compiler_service`` ingests
    tenant-supplied drone footage up to 4 GiB, i.e. ~8 GiB of pod RSS per concurrent
    fetch — an OOM, not a slowdown. This path holds one chunk at a time regardless of
    body size, and pushes the writes to a thread so the event loop keeps running.

    The result is published ATOMICALLY: bytes land in a uuid7-suffixed sibling temp
    file and are ``os.replace``d onto ``dest`` only once the body is complete. A
    truncated file is never visible at ``dest``, and on any failure — cap breach,
    transport error, ``asyncio.CancelledError`` — the partial is unlinked, ``dest`` is
    left untouched, and the original exception propagates unchanged.

    ``max_fetch_bytes`` counts DECODED bytes mid-stream exactly as :func:`safe_get`
    does, so a ``Content-Encoding: gzip`` bomb is measured expanded. ``0``/``None``
    means unlimited. ``dest``'s parent directory must already exist.

    ⚠️ **Status handling deliberately DIFFERS from** :func:`safe_get`. The buffering
    wrappers return the response and let the caller call ``raise_for_status()``;
    here there is no response for the caller to inspect — the body has gone to a
    file. So a non-success status raises ``httpx.HTTPStatusError`` and **nothing is
    written to** ``dest``. Without this an expired presigned MinIO URL — the exact
    degrade case this path was built for — would land a few bytes of
    ``<Error>NoSuchKey</Error>`` XML at ``dest`` and return SUCCESSFULLY, feeding
    ffmpeg a corrupt input instead of raising into the caller's bad-asset taxonomy.

    Returns:
        ``dest``, for call-site chaining.

    Raises:
        SSRFError: blocked target, blocked redirect, or too many redirects.
        ResponseTooLargeError: body exceeds ``max_fetch_bytes``.
        httpx.HTTPStatusError: the final response was not a success status. Nothing
            is written to ``dest``.
        OSError: the destination could not be written or renamed.
        httpx.HTTPError: transport/timeout errors propagate unchanged.
    """
    return await _safe_request_to_sink(
        client,
        "GET",
        url,
        content=None,
        json=None,
        data=None,
        max_redirects=max_redirects,
        allow_hosts=allow_hosts,
        max_fetch_bytes=max_fetch_bytes,
        sink=_path_sink(dest),
        raise_for_status=True,
        kwargs=kwargs,
    )


async def safe_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """GET ``url`` with SSRF-safe, IP-pinned, manually-followed redirects (async).

    Thin wrapper over :func:`safe_request` — see it for the full semantics.
    """
    return await safe_request(
        client,
        "GET",
        url,
        max_redirects=max_redirects,
        allow_hosts=allow_hosts,
        max_fetch_bytes=max_fetch_bytes,
        **kwargs,
    )


async def safe_post(
    client: httpx.AsyncClient,
    url: str,
    *,
    content: bytes | str | None = None,
    json: Any = None,
    data: Mapping[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """POST ``url`` with SSRF-safe, IP-pinned, manually-followed redirects (async).

    Thin wrapper over :func:`safe_request`. Callers sending a signed body
    (a webhook HMAC) should pass ``max_redirects=0`` — a signed request must
    never be replayed to a redirect target.
    """
    return await safe_request(
        client,
        "POST",
        url,
        content=content,
        json=json,
        data=data,
        max_redirects=max_redirects,
        allow_hosts=allow_hosts,
        max_fetch_bytes=max_fetch_bytes,
        **kwargs,
    )


def safe_request_sync(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    content: bytes | str | None = None,
    json: Any = None,
    data: Mapping[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """Send ``method url`` with SSRF-safe, IP-pinned, manual redirects (sync).

    Synchronous counterpart of :func:`safe_request` (identical pinning,
    redirect-method, cross-origin header/auth reduction and ``max_fetch_bytes``
    semantics — including the 64 MiB default), for services that fetch via a
    sync ``httpx.Client`` (typically off the event loop in
    ``asyncio.to_thread``).
    """
    _validate_method(method)
    _check_replayable_body(content, data)
    _reject_unguardable_kwargs(kwargs)
    kwargs.pop("follow_redirects", None)
    extra_headers = _drop_caller_headers(kwargs.pop("headers", None) or {})
    extra_ext = dict(kwargs.pop("extensions", None) or {})
    current_url = url
    keep_body = True
    cross_origin = False
    ip = validate_public_url(url, allow_hosts=allow_hosts)
    for _hop in range(max_redirects + 1):
        request_url, pin_headers, pin_ext = _pin(current_url, ip)
        request = _build_hop_request(
            client,
            method,
            request_url,
            headers={**extra_headers, **pin_headers},
            extensions={**extra_ext, **pin_ext},
            content=content,
            json=json,
            data=data,
            keep_body=keep_body,
            cross_origin=cross_origin,
            kwargs=kwargs,
        )
        response = client.send(
            request, stream=True, follow_redirects=False, **_hop_auth(cross_origin)
        )
        try:
            if response.status_code not in _REDIRECT_STATUS:
                if _content_length_exceeds(response, max_fetch_bytes):
                    raise ResponseTooLargeError(
                        f"Response Content-Length exceeds {max_fetch_bytes} bytes: {url}"
                    )
                if not max_fetch_bytes:
                    response.read()
                    return _materialize(response, response.content)
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_fetch_bytes:
                        raise ResponseTooLargeError(
                            f"Response body exceeds {max_fetch_bytes} bytes: {url}"
                        )
                    chunks.append(chunk)
                return _materialize(response, b"".join(chunks))
            location = response.headers.get("location")
            status_code = response.status_code
        finally:
            response.close()
        next_url, ip = _validate_redirect(location, current_url, allow_hosts=allow_hosts)
        method, hop_keeps_body = _redirect_method(method, status_code)
        # Latch: once a hop drops the body it is gone for good. Re-deriving it
        # per hop would let 302 (drop) -> 307 (keep) resurrect a body that was
        # deliberately discarded, and replay it to a redirector-chosen host.
        keep_body = keep_body and hop_keeps_body
        # Same latch for the header reduction: A(secret) -> B -> A must not
        # hand the credential back just because the third hop is same-origin
        # with the first — B chose that hop, and B has already seen nothing.
        cross_origin = cross_origin or not _same_origin(current_url, next_url)
        current_url = next_url
    raise SSRFError(f"Too many redirects (> {max_redirects}) fetching {_safe_url_for_message(url)}")


def safe_get_sync(
    client: httpx.Client,
    url: str,
    *,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """GET ``url`` with SSRF-safe, IP-pinned, manually-followed redirects (sync).

    Thin wrapper over :func:`safe_request_sync` — see it for the full semantics.
    """
    return safe_request_sync(
        client,
        "GET",
        url,
        max_redirects=max_redirects,
        allow_hosts=allow_hosts,
        max_fetch_bytes=max_fetch_bytes,
        **kwargs,
    )


def safe_post_sync(
    client: httpx.Client,
    url: str,
    *,
    content: bytes | str | None = None,
    json: Any = None,
    data: Mapping[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_hosts: Iterable[str] | None = None,
    max_fetch_bytes: int | None = DEFAULT_MAX_FETCH_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """POST ``url`` with SSRF-safe, IP-pinned, manual redirects (sync).

    Thin wrapper over :func:`safe_request_sync`. Pass ``max_redirects=0`` when
    the body carries a signature.
    """
    return safe_request_sync(
        client,
        "POST",
        url,
        content=content,
        json=json,
        data=data,
        max_redirects=max_redirects,
        allow_hosts=allow_hosts,
        max_fetch_bytes=max_fetch_bytes,
        **kwargs,
    )
