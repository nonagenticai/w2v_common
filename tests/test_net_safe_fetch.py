"""Tests for the SSRF-safe outbound fetch guard (w2v_common.net.safe_fetch)."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import cast

import httpx
import pytest

from w2v_common.net import (
    DEFAULT_MAX_FETCH_BYTES,
    ResponseTooLargeError,
    SSRFError,
    safe_get,
    safe_get_sync,
    safe_get_to_path,
    safe_post,
    safe_post_sync,
    safe_request,
    safe_request_sync,
    validate_public_url,
)
from w2v_common.net import safe_fetch as safe_fetch_module

# Hostname -> resolved IP for the hermetic DNS stub. No real DNS/network is hit.
_DNS: dict[str, str] = {
    "public.example.com": "93.184.216.34",  # public, allowed
    "cdn.example.com": "203.0.113.10",  # TEST-NET-3, treated as public/allowed
    "internal.example.com": "10.0.0.5",  # RFC1918 -> blocked
    "svc.cluster.local": "10.96.0.10",  # k8s ClusterIP -> blocked
    "svc.internal": "100.64.0.5",  # Tailscale CGNAT -> blocked
    "metadata": "169.254.169.254",  # link-local metadata -> blocked
    "blob.internal": "100.64.0.5",  # presigned-artifact host (CGNAT) -> allow-listed
}


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Resolve test hostnames deterministically; literal IPs resolve to themselves."""

    def fake_getaddrinfo(host: str, *_args: object, **_kwargs: object) -> list:
        ip = _DNS.get(host, host)  # literal IPs (e.g. "10.0.0.1") pass through
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    yield


# --------------------------------------------------------------------------- #
# validate_public_url
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",  # loopback (literal)
        "http://10.0.0.1/x",  # RFC1918 (literal)
        "http://192.168.1.1/x",  # RFC1918 (literal)
        "http://169.254.169.254/latest/meta-data",  # link-local metadata (literal)
        "http://internal.example.com/x",  # RFC1918 (DNS)
        "http://svc.cluster.local/x",  # k8s ClusterIP (DNS)
        "http://svc.internal/x",  # Tailscale CGNAT (DNS)
        "http://metadata/x",  # link-local via DNS
        "https://metadata.google.internal/x",  # blocked hostname
    ],
)
def test_validate_blocks_internal(url: str) -> None:
    with pytest.raises(SSRFError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    ["http://public.example.com/img.png", "https://cdn.example.com/a.jpg"],
)
def test_validate_allows_public(url: str) -> None:
    validate_public_url(url)  # does not raise


# --------------------------------------------------------------------------- #
# RFC 6598 CGNAT — the Tailscale tailnet. `ipaddress` reports these as
# is_private=False / is_global=True, so every guard written as
# `not (ip.is_private or ip.is_loopback)` declares the whole estate fetchable.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("addr", "what"),
    [
        ("100.64.0.7", "example CGNAT host"),
        ("100.64.0.8", "example CGNAT host"),
        ("100.64.0.9", "example CGNAT host"),
        ("100.64.0.0", "CGNAT lower bound"),
        ("100.127.255.255", "CGNAT upper bound"),
    ],
)
def test_validate_blocks_tailnet_cgnat_literals(addr: str, what: str) -> None:
    """Real estate hosts must be blocked. `ip_address(a).is_private` is False for all of these."""
    import ipaddress as _ia

    assert not _ia.ip_address(addr).is_private, f"{addr} unexpectedly is_private"
    with pytest.raises(SSRFError):
        validate_public_url(f"http://{addr}:5432/")


@pytest.mark.parametrize("addr", ["100.63.255.255", "100.128.0.0"])
def test_validate_allows_addresses_adjacent_to_cgnat(addr: str) -> None:
    """The deny is the /10 — not `startswith("100.")`.

    ``100.63.255.255`` is one below and ``100.128.0.0`` one above
    ``100.64.0.0/10``; both are genuinely public and must still fetch.
    """
    validate_public_url(f"http://{addr}/x")  # does not raise


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("http://[::ffff:100.64.0.1]/x", "IPv4-mapped CGNAT"),
        ("http://[::ffff:10.0.0.5]/x", "IPv4-mapped RFC1918"),
        ("http://[2002:6440:1::1]/x", "6to4 wrapping 100.64.0.1"),
        ("http://[2002:0808:0808::1]/x", "6to4 wrapping a PUBLIC v4 — wrapper is blocked"),
        ("http://[64:ff9b::808:808]/x", "NAT64 well-known prefix (is_global=True)"),
        ("http://[::]/x", "unspecified"),
        ("http://[::1]/x", "loopback"),
        ("http://[fe80::1]/x", "link-local"),
    ],
)
def test_validate_blocks_ipv6_wrappers_and_transitions(url: str, why: str) -> None:
    with pytest.raises(SSRFError):
        validate_public_url(url)


def test_validate_ipv6_unwrap_is_additive_not_substitutive() -> None:
    """A 6to4 address wrapping a PUBLIC IPv4 stays blocked.

    If the embedded-v4 verdict *replaced* the wrapper's, ``2002:0808:0808::1``
    (6to4 for 8.8.8.8) would be judged solely on 8.8.8.8 and become reachable —
    a widening of the allow-surface, not a fix. ``blocked`` must be
    ``wrapper_blocked OR embedded_blocked``.
    """
    import ipaddress as _ia

    # IPv6Address, not ip_address(): only the v6 branch of the union has .sixtofour
    assert _ia.IPv6Address("2002:0808:0808::1").sixtofour == _ia.IPv4Address("8.8.8.8")
    with pytest.raises(SSRFError):
        validate_public_url("http://[2002:0808:0808::1]/x")


def test_validate_allows_v4_mapped_public_address() -> None:
    """``::ffff:<public v4>`` must still fetch.

    A resolver with ``AI_V4MAPPED`` can hand back the v4-mapped form of a public
    address, so ``::ffff:0:0/96`` must NOT be blocked wholesale — the embedded
    IPv4 is what gets judged.
    """
    validate_public_url("http://[::ffff:93.184.216.34]/x")  # does not raise


# --------------------------------------------------------------------------- #
# The four IPv6-embedded-IPv4 forms, as a matrix
#
# ``_embedded_ipv4`` must recognise every form that carries an IPv4 destination.
# :mod:`ipaddress` supplies an accessor for three of them; the IPv4-COMPATIBLE
# form (RFC 4291 §2.5.5.1) has none and is derived by this module. Enumerating
# the forms as data — rather than as four hand-written cases — is what makes a
# *fifth* form obvious by its absence.
# --------------------------------------------------------------------------- #


def _v4_mapped(v4: str) -> str:
    """``::ffff:a.b.c.d`` — RFC 4291 §2.5.5.2, ``ipaddress.ipv4_mapped``."""
    return f"::ffff:{v4}"


def _v4_compatible(v4: str) -> str:
    """``::a.b.c.d`` — RFC 4291 §2.5.5.1. No ``ipaddress`` accessor exists."""
    return f"::{v4}"


def _sixtofour(v4: str) -> str:
    """``2002:<v4-hex>::1`` — RFC 3056, ``ipaddress.sixtofour``."""
    import ipaddress as _ia

    packed = int(_ia.IPv4Address(v4))
    return f"2002:{packed >> 16:04x}:{packed & 0xFFFF:04x}::1"


def _teredo(v4: str) -> str:
    """``2001:0:<server>:<flags>:<port>:<obfuscated client>`` — RFC 4380.

    The client IPv4 is stored one's-complemented, which is exactly why a guard
    cannot pattern-match the literal and must go through ``ipaddress.teredo``.
    """
    import ipaddress as _ia

    obfuscated = int(_ia.IPv4Address(v4)) ^ 0xFFFFFFFF
    server = int(_ia.IPv4Address("203.0.113.1"))  # TEST-NET-3, arbitrary
    return (
        f"2001:0:{server >> 16:04x}:{server & 0xFFFF:04x}:"
        f"0:0:{obfuscated >> 16:04x}:{obfuscated & 0xFFFF:04x}"
    )


#: form name -> (builder, wrapper-prefix-blocked?). Only the IPv4-MAPPED wrapper
#: is NOT blocked wholesale — a resolver may legitimately return it for a public
#: address — so it is the one form whose verdict rests entirely on the unwrap.
_EMBEDDED_FORMS = {
    "ipv4-mapped": (_v4_mapped, False),
    "ipv4-compatible": (_v4_compatible, True),
    "6to4": (_sixtofour, True),
    "teredo": (_teredo, True),
}

#: IPv4 class -> blocked on the IPv4 block-list?
_V4_CLASSES = {
    "public": ("8.8.8.8", False),
    "cgnat-tailnet": ("100.64.0.1", True),
    "rfc1918": ("10.0.0.5", True),
    "loopback": ("127.0.0.1", True),
}


@pytest.mark.parametrize("form_name", list(_EMBEDDED_FORMS))
@pytest.mark.parametrize("v4_name", list(_V4_CLASSES))
def test_embedded_ipv4_matrix(form_name: str, v4_name: str) -> None:
    """Every form x every IPv4 class: the unwrap sees it, and the verdict is right.

    The unwrap half is the regression this pins: on ``::100.64.0.1`` the three
    :mod:`ipaddress` accessors all return ``None``, so before
    ``_ipv4_compatible`` existed the four ``ipv4-compatible`` rows yielded an
    empty list and the address was judged purely on its (``is_global=True``)
    wrapper.

    The verdict half is the false-positive control: ``ipv4-mapped``/``public``
    is the single cell that must come out ALLOWED. A guard that blocks the whole
    matrix passes a badly-written test but breaks every ``AI_V4MAPPED`` resolver
    answer.
    """
    import ipaddress as _ia

    from w2v_common.net.safe_fetch import _embedded_ipv4, _is_blocked_ip

    builder, wrapper_blocked = _EMBEDDED_FORMS[form_name]
    v4, v4_blocked = _V4_CLASSES[v4_name]
    literal = builder(v4)

    assert _ia.IPv4Address(v4) in _embedded_ipv4(_ia.IPv6Address(literal)), (
        f"{form_name} literal {literal} did not unwrap to {v4}"
    )

    # Additive verdict: wrapper OR embedded (never embedded *replacing* wrapper).
    assert _is_blocked_ip(literal) is (wrapper_blocked or v4_blocked), (
        f"{form_name}/{v4_name} ({literal}) verdict is wrong"
    )


def test_embedded_ipv4_compatible_form_is_unwrapped() -> None:
    """RED-proof for the added branch, asserted on the real function's real output.

    Every ``ipaddress`` accessor returns ``None`` for ``::100.64.0.1`` and the
    address reports ``is_global=True`` / ``is_private=False``, so nothing in the
    stdlib flags it. Without ``_ipv4_compatible`` this returns ``[]``.
    """
    import ipaddress as _ia

    from w2v_common.net.safe_fetch import _embedded_ipv4

    ip = _ia.IPv6Address("::100.64.0.1")
    assert (ip.ipv4_mapped, ip.sixtofour, ip.teredo) == (None, None, None)
    assert (ip.is_global, ip.is_private) == (True, False)
    assert _embedded_ipv4(ip) == [_ia.IPv4Address("100.64.0.1")]


@pytest.mark.parametrize(
    ("literal", "why"),
    [
        ("::", "unspecified — int 0 would unwrap to the nonsense '0.0.0.0'"),
        ("::1", "loopback — int 1 would unwrap to the nonsense '0.0.0.1'"),
    ],
)
def test_ipv4_compatible_excludes_the_reserved_specials(literal: str, why: str) -> None:
    """``::`` and ``::1`` are inside ``::/96`` by arithmetic but are not wrapped IPv4.

    They stay blocked — by the wrapper block-list, which is where they belong.
    """
    import ipaddress as _ia

    from w2v_common.net.safe_fetch import _ipv4_compatible, _is_blocked_ip

    assert _ipv4_compatible(_ia.IPv6Address(literal)) is None
    assert _is_blocked_ip(literal) is True


def test_ipv4_compatible_low_valued_address_unwraps_to_this_network() -> None:
    """``::0.0.0.2`` is a real compatible address, not a special.

    It unwraps to ``0.0.0.2`` and is caught by ``0.0.0.0/8`` on the IPv4 list —
    the correct answer, so it is deliberately not special-cased alongside
    ``::``/``::1``. NB :mod:`ipaddress` renders it ``::2``, not in dotted form.
    """
    import ipaddress as _ia

    from w2v_common.net.safe_fetch import _ipv4_compatible

    assert _ipv4_compatible(_ia.IPv6Address("::0.0.0.2")) == _ia.IPv4Address("0.0.0.2")


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("http://[::ffff:8.8.8.8]/x", "IPv4-mapped PUBLIC — the AI_V4MAPPED resolver answer"),
        ("http://[::ffff:93.184.216.34]/x", "IPv4-mapped PUBLIC (second address)"),
        ("http://[2606:4700:4700::1111]/x", "plain public IPv6, no embedded v4 at all"),
        ("http://[2001:4860:4860::8888]/x", "plain public IPv6 (outside Teredo's 2001::/32)"),
    ],
)
def test_v6_false_positive_controls_stay_allowed(url: str, why: str) -> None:
    """The important half: widening the unwrap must not block a legitimate target.

    ``_is_blocked_ip`` is additive, so a new unwrap branch can only ever block
    MORE. These four assert it blocked nothing it should not have.
    """
    validate_public_url(url)  # does not raise


@pytest.mark.parametrize(
    ("addr", "why"),
    [
        ("192.0.0.10", "IETF protocol assignments — is_private is False for /24 outside the /29"),
        ("198.18.0.1", "benchmarking range"),
        ("192.88.99.1", "6to4 relay anycast (RFC 7526) — is_private/is_reserved are both False"),
    ],
)
def test_validate_blocks_newly_added_reserved_ipv4(addr: str, why: str) -> None:
    with pytest.raises(SSRFError):
        validate_public_url(f"http://{addr}/x")


@pytest.mark.parametrize(
    ("url", "why"),
    [
        (
            "http://[::169.254.169.254]/latest/meta-data",
            "IPv4-COMPATIBLE (::/96) metadata address — ipaddress reports ipv4_mapped=None, "
            "sixtofour=None and teredo=None for it. Stopped TWICE over: by the ::/96 prefix "
            "block, and (since _ipv4_compatible) by the embedded-v4 unwrap deriving "
            "169.254.169.254 by hand. Either alone suffices; both is the point",
        ),
        (
            "http://[::ffff:0:169.254.169.254]/latest/meta-data",
            "IPv4-TRANSLATED (::ffff:0:0:0/96, RFC 2765 SIIT) — note the extra 0: group; "
            "also invisible to .ipv4_mapped, which only recognises ::ffff:0:0/96",
        ),
        ("http://[::ffff:0:10.0.0.5]/x", "IPv4-translated wrapping RFC1918"),
        (
            "http://[fec0::1]/x",
            "site-local (RFC 3879) — deprecated but still routed by some stacks",
        ),
    ],
)
def test_validate_blocks_ipv6_prefixes_invisible_to_the_v4_unwrap(url: str, why: str) -> None:
    """Prefixes whose embedded IPv4 the :mod:`ipaddress` unwrap helpers do NOT expose.

    This is the whole reason the wrapper prefixes are blocked by CIDR rather
    than by unwrapping: for all four, every ``ipaddress`` accessor returns
    ``None``, so an implementation that trusted the stdlib accessors alone would
    call ``::169.254.169.254`` a public address. ``_ipv4_compatible`` now derives
    the first row's IPv4 by hand, but the other three (IPv4-translated x2,
    site-local) still have no unwrap at all and rest solely on the CIDR block.
    """
    with pytest.raises(SSRFError):
        validate_public_url(url)


def test_validate_v4_mapped_public_survives_the_new_v6_prefix_blocks() -> None:
    """Control: the new ``::/96`` + ``::ffff:0:0:0/96`` blocks must not swallow ``::ffff:<public>``.

    ``::ffff:93.184.216.34`` is IPv4-MAPPED (``::ffff:0:0/96``) — one group
    shorter than the IPv4-translated form blocked above, and a legitimate
    ``AI_V4MAPPED`` resolver answer. If the new prefixes were written one group
    wide, every v4-mapped public address would become unfetchable.
    """
    validate_public_url("http://[::ffff:93.184.216.34]/x")  # does not raise


@pytest.mark.parametrize("url", ["ftp://public.example.com/x", "file:///etc/passwd", "/relative"])
def test_validate_rejects_bad_scheme_or_host(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_validate_blocks_unresolvable_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-resolving host is fail-CLOSED: it must not be handed to the transport.

    If validation returned/allowed here, the transport would resolve the host
    independently at connect time — exactly the DNS-rebinding TOCTOU window the
    pinning is meant to close. So a gaierror is an SSRFError, not a pass-through.
    """

    def boom(*_a: object, **_k: object) -> list:
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(SSRFError):
        validate_public_url("http://does-not-exist.example.com/x")


# --------------------------------------------------------------------------- #
# Hostname suffix block — refused BEFORE resolution. Split-horizon DNS means
# `*.internal` / `*.svc.cluster.local` can resolve publicly from one vantage
# point and to a tailnet/ClusterIP address from inside a pod, so the NAME is
# what must be refused, not merely where it happens to point right now.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "host",
    [
        "foo.internal",
        "db.internal",
        "svc.internal",
        "svc.cluster.local",
        "kagsvc.internal",
        "backend.default.svc",
        "printer.local",
        "app.localhost",
        "localhost",
        "metadata.google.internal",
    ],
)
def test_validate_blocks_estate_hostnames_before_resolution(
    host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is rejected without a DNS lookup ever being issued."""
    calls: list[str] = []

    def counting_getaddrinfo(h: str, *_a: object, **_k: object) -> list:
        calls.append(h)
        raise AssertionError(f"DNS lookup issued for blocked host {h!r}")

    monkeypatch.setattr(socket, "getaddrinfo", counting_getaddrinfo)
    with pytest.raises(SSRFError):
        validate_public_url(f"http://{host}/x")
    assert calls == [], f"resolution must not be attempted for {host}"


@pytest.mark.parametrize(
    "host",
    ["db.internal.", "svc.cluster.local.", "localhost.", "foo.internal..."],
)
def test_validate_blocks_trailing_dot_estate_hostnames(
    host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trailing root dot must not defeat the suffix block.

    ``db.internal.`` resolves identically to ``db.internal``
    but does not satisfy ``endswith(".internal")`` without normalisation.
    """

    def boom(h: str, *_a: object, **_k: object) -> list:
        raise AssertionError(f"DNS lookup issued for blocked host {h!r}")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(SSRFError):
        validate_public_url(f"http://{host}/x")


def test_validate_allowlist_beats_suffix_block() -> None:
    """An explicitly-trusted artifact host keeps working despite the `.internal` suffix."""
    assert (
        validate_public_url("http://blob.internal:8389/b/k", allow_hosts={"blob.internal"}) is None
    )


def test_validate_allowlist_matches_trailing_dot_host() -> None:
    """Host normalisation applies to the allow-list too — no dot/no-dot mismatch."""
    assert (
        validate_public_url("http://blob.internal.:8389/b/k", allow_hosts={"blob.internal"}) is None
    )


def test_validate_returns_pinned_ip() -> None:
    """A public host validates and returns the resolved IP to pin the connection to."""
    assert validate_public_url("https://public.example.com/x") == "93.184.216.34"


def test_validate_allowlisted_host_returns_none_pin() -> None:
    """A trusted host is fetched un-pinned (None) so its Host/signature is preserved."""
    assert (
        validate_public_url(
            "http://blob.internal:8389/bucket/key?X-Amz=sig", allow_hosts={"blob.internal"}
        )
        is None
    )


# --------------------------------------------------------------------------- #
# validate_public_url — allow_hosts (presigned-artifact trust-list)
# --------------------------------------------------------------------------- #


def test_validate_blocks_internal_host_without_allowlist() -> None:
    """Baseline: the artifact host is internal and blocked when not allow-listed."""
    with pytest.raises(SSRFError):
        validate_public_url("http://blob.internal:8389/bucket/key?X-Amz=sig")


@pytest.mark.parametrize("allow", [{"blob.internal"}, {"blob.internal:8389"}])
def test_validate_allows_listed_internal_host(allow: set[str]) -> None:
    """An explicitly-trusted host (bare or host:port) bypasses the internal block."""
    validate_public_url(
        "http://blob.internal:8389/bucket/key?X-Amz=sig", allow_hosts=allow
    )  # no raise


def test_validate_allowlist_is_host_specific() -> None:
    """The allow-list trusts only the named host — a different internal host still blocks."""
    with pytest.raises(SSRFError):
        validate_public_url("http://internal.example.com/secret", allow_hosts={"blob.internal"})


def test_validate_allowlist_still_requires_http_scheme() -> None:
    """Allow-listing a host does not relax the http(s)-only scheme guard."""
    with pytest.raises(ValueError):
        validate_public_url("ftp://blob.internal:8389/x", allow_hosts={"blob.internal"})


# --------------------------------------------------------------------------- #
# safe_get (async) / safe_get_sync
# --------------------------------------------------------------------------- #


def _host_of(request: httpx.Request) -> str:
    """Route on the ``Host`` header, not the URL host.

    After IP-pinning the request URL host is the literal resolved IP; the
    original hostname lives in the ``Host`` header (and TLS SNI). Un-pinned
    (allow-listed) requests carry the hostname in both, so this works for both.
    """
    raw = request.headers.get("host") or request.url.host
    return raw.rsplit(":", 1)[0].strip("[]").lower()


def _redirect_to(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


# (host, path) -> response factory. A table rather than an if-ladder so adding a
# scenario does not push the handler past ruff's complexity ceiling.
_ROUTES: dict[tuple[str, str], Callable[[], httpx.Response]] = {
    ("public.example.com", "/ok"): lambda: httpx.Response(200, content=b"image-bytes"),
    ("cdn.example.com", "/final"): lambda: httpx.Response(200, content=b"final-bytes"),
    ("public.example.com", "/to-public"): lambda: _redirect_to("https://cdn.example.com/final"),
    ("public.example.com", "/to-internal"): lambda: _redirect_to(
        "http://internal.example.com/secret"
    ),
    # The estate read-oracle: a public host 302s to bare-metal prod postgres.
    ("public.example.com", "/to-tailnet"): lambda: _redirect_to("http://100.64.0.7:5432/"),
    ("public.example.com", "/to-tailnet-name"): lambda: _redirect_to("http://db.internal./"),
    ("public.example.com", "/loop"): lambda: _redirect_to("https://public.example.com/loop"),
    ("public.example.com", "/no-location"): lambda: httpx.Response(302),
    ("blob.internal", "/bucket/key"): lambda: httpx.Response(200, content=b"artifact-bytes"),
    # A trusted artifact host that 30x-redirects to a *different* internal host
    # must NOT inherit the allow-list trust.
    ("blob.internal", "/redirect-internal"): lambda: _redirect_to(
        "http://internal.example.com/secret"
    ),
}


def _handler(request: httpx.Request) -> httpx.Response:
    route = _ROUTES.get((_host_of(request), request.url.path))
    return route() if route else httpx.Response(404)


@pytest.mark.asyncio
async def test_safe_get_direct_ok() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/ok")
    assert resp.status_code == 200
    assert resp.content == b"image-bytes"


@pytest.mark.asyncio
async def test_safe_get_follows_public_redirect() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/to-public")
    assert resp.status_code == 200
    assert resp.content == b"final-bytes"


@pytest.mark.asyncio
async def test_safe_get_blocks_redirect_to_internal() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get(client, "https://public.example.com/to-internal")


@pytest.mark.asyncio
async def test_safe_get_blocks_initial_internal() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get(client, "http://internal.example.com/secret")


# --------------------------------------------------------------------------- #
# Redirect oracle — a guard that raises AFTER the hop already fired is not a
# guard. These assert the transport recorded ZERO requests to the tailnet host.
# --------------------------------------------------------------------------- #


def _recording_transport() -> tuple[httpx.MockTransport, list[str]]:
    """MockTransport that records the connect target of every request it sees."""
    seen: list[str] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.host}:{request.url.port or ''}")
        return _handler(request)

    return httpx.MockTransport(recording_handler), seen


@pytest.mark.parametrize("path", ["/to-tailnet", "/to-tailnet-name"])
@pytest.mark.asyncio
async def test_safe_get_blocks_redirect_to_tailnet_without_opening_a_socket(path: str) -> None:
    """A 302 to prod postgres is refused and no request is ever issued to it."""
    transport, seen = _recording_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get(client, f"https://public.example.com{path}")

    assert not any("100.64.0.7" in s for s in seen), f"socket opened to tailnet: {seen}"
    assert not any("prod-postgres" in s for s in seen), f"socket opened to tailnet: {seen}"
    # Exactly one hop happened: the public host that served the redirect.
    assert len(seen) == 1, seen


@pytest.mark.parametrize("path", ["/to-tailnet", "/to-tailnet-name"])
def test_safe_get_sync_blocks_redirect_to_tailnet_without_opening_a_socket(path: str) -> None:
    """Sync counterpart — same no-socket guarantee."""
    transport, seen = _recording_transport()
    with httpx.Client(transport=transport) as client, pytest.raises(SSRFError):
        safe_get_sync(client, f"https://public.example.com{path}")

    assert not any("100.64.0.7" in s for s in seen), f"socket opened to tailnet: {seen}"
    assert not any("prod-postgres" in s for s in seen), f"socket opened to tailnet: {seen}"
    assert len(seen) == 1, seen


@pytest.mark.asyncio
async def test_safe_get_blocked_redirect_leaks_no_response_detail() -> None:
    """The SSRFError names the refused target only — never the blocked host's response.

    `site_capture` appends the exception text to user-visible run notes, so an
    error carrying a body/header from the internal hop would itself be the leak.
    """
    transport, _seen = _recording_transport()
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError) as exc:
            await safe_get(client, "https://public.example.com/to-tailnet")
    message = str(exc.value)
    for leak in ("image-bytes", "final-bytes", "content-type", "PostgreSQL", "SFATAL"):
        assert leak not in message, f"error message leaked {leak!r}: {message}"


@pytest.mark.asyncio
async def test_safe_get_too_many_redirects() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get(client, "https://public.example.com/loop", max_redirects=2)


@pytest.mark.asyncio
async def test_safe_get_redirect_without_location() -> None:
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get(client, "https://public.example.com/no-location")


def test_safe_get_sync_follows_public_redirect() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport) as client:
        resp = safe_get_sync(client, "https://public.example.com/to-public")
    assert resp.status_code == 200
    assert resp.content == b"final-bytes"


def test_safe_get_sync_blocks_internal_redirect() -> None:
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport) as client, pytest.raises(SSRFError):
        safe_get_sync(client, "https://public.example.com/to-internal")


@pytest.mark.asyncio
async def test_safe_get_allows_listed_minio_host() -> None:
    """A presigned internal MinIO URL fetches when the host is allow-listed."""
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(
            client,
            "http://blob.internal:8389/bucket/key?X-Amz=sig",
            allow_hosts={"blob.internal:8389"},
        )
    assert resp.status_code == 200
    assert resp.content == b"artifact-bytes"


@pytest.mark.asyncio
async def test_safe_get_allowlisted_host_redirect_to_internal_still_blocked() -> None:
    """A trusted host redirecting to a non-allowed internal address is rejected."""
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get(
                client,
                "http://blob.internal:8389/redirect-internal",
                allow_hosts={"blob.internal:8389"},
            )


def test_safe_get_sync_allows_listed_minio_host() -> None:
    """Sync counterpart: presigned internal MinIO URL fetches when allow-listed."""
    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport) as client:
        resp = safe_get_sync(
            client,
            "http://blob.internal:8389/bucket/key?X-Amz=sig",
            allow_hosts={"blob.internal"},
        )
    assert resp.status_code == 200
    assert resp.content == b"artifact-bytes"


# --------------------------------------------------------------------------- #
# DNS-rebinding TOCTOU — the connection must pin the validated IP and resolve
# ONCE. (The MockTransport-only tests above never resolve DNS, so they cannot
# prove this; these do, via a rebinding getaddrinfo counter.)
# --------------------------------------------------------------------------- #


def _rebinding_resolver(counter: dict[str, int]) -> object:
    """getaddrinfo stub that returns a PUBLIC IP first, a PRIVATE IP after.

    A correct implementation resolves exactly once and connects to the public
    IP; a re-resolving (vulnerable) implementation would get the private IP at
    connect time.
    """

    def resolver(host: str, *_a: object, **_k: object) -> list:
        counter["n"] += 1
        ip = "93.184.216.34" if counter["n"] == 1 else "10.0.0.5"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    return resolver


@pytest.mark.asyncio
async def test_safe_get_pins_validated_ip_and_resolves_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = {"n": 0}
    monkeypatch.setattr(socket, "getaddrinfo", _rebinding_resolver(counter))

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url_host"] = request.url.host
        seen["host_header"] = request.headers.get("host") or ""
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://rebind.example.com/x")

    assert resp.status_code == 200
    # Resolved exactly once — a second lookup is the rebinding window.
    assert counter["n"] == 1
    # Connected to the validated (public) IP, not the hostname the transport
    # would have re-resolved (to the private IP) on its own.
    assert seen["url_host"] == "93.184.216.34"
    # Original hostname preserved for Host header + TLS SNI / cert check.
    assert seen["host_header"].rsplit(":", 1)[0].strip("[]") == "rebind.example.com"


def test_safe_get_sync_pins_validated_ip_and_resolves_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = {"n": 0}
    monkeypatch.setattr(socket, "getaddrinfo", _rebinding_resolver(counter))

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url_host"] = request.url.host
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        resp = safe_get_sync(client, "https://rebind.example.com/x")

    assert resp.status_code == 200
    assert counter["n"] == 1
    assert seen["url_host"] == "93.184.216.34"


# --------------------------------------------------------------------------- #
# max_fetch_bytes — streaming OOM cap
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_safe_get_rejects_oversized_content_length() -> None:
    """A declared Content-Length over the cap is rejected before the body is read."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 500)  # Content-Length: 500

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get(client, "https://public.example.com/ok", max_fetch_bytes=100)


def test_safe_get_sync_streaming_cap_without_content_length() -> None:
    """A chunked body with no/lying Content-Length is aborted once the running total exceeds the cap."""

    def handler(request: httpx.Request) -> httpx.Response:
        # An iterable body ⇒ chunked transfer, no Content-Length header, so only
        # the streaming byte-counter can catch it.
        return httpx.Response(200, content=iter([b"a" * 80, b"a" * 80]))

    transport = httpx.MockTransport(handler)
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(ResponseTooLargeError),
    ):
        safe_get_sync(client, "https://public.example.com/ok", max_fetch_bytes=100)


@pytest.mark.asyncio
async def test_safe_get_under_cap_returns_full_body() -> None:
    """A body within the cap returns 200 with content intact (materialized response)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"small-body")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/ok", max_fetch_bytes=1000)
    assert resp.status_code == 200
    assert resp.content == b"small-body"


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_out", [0, None], ids=["zero", "none"])
async def test_safe_get_explicit_opt_out_is_unlimited(opt_out: int | None) -> None:
    """``0``/``None`` is the explicit opt-out and still streams without a bound.

    (This test was ``..._no_cap_unlimited_by_default``. Its body only ever sent
    10 kB, so it passed either way and its NAME — "by default" — is exactly the
    claim the default cap invalidates. The unlimited path is real and still
    supported; it is now something a caller asks for, not something it gets by
    forgetting.)
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"z" * 10_000)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/ok", max_fetch_bytes=opt_out)
    assert resp.status_code == 200
    assert len(resp.content) == 10_000


# --------------------------------------------------------------------------- #
# CGNAT / tailnet block-list — the estate's own machines are NOT "public"
#
# ipaddress.ip_address("100.64.0.7").is_private is False, and so are
# is_reserved / is_link_local / is_multicast / is_unspecified. Any guard built
# from those attributes calls every machine in this estate a safe public target.
# These tests pin the explicit /10 deny so that can never regress.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://100.64.0.1/x",  # first address of the /10
        "http://100.64.0.7/x",  # bare-metal postgres PRIMARY
        "http://100.64.0.8/x",  # docker registry (worker3)
        "http://100.64.0.9/x",  # k3s control plane (master)
        "http://100.64.0.10/x",  # example CGNAT host
        "http://100.127.255.255/x",  # last address of the /10
    ],
)
def test_validate_blocks_cgnat_tailnet_literals(url: str) -> None:
    """RFC 6598 100.64.0.0/10 is this estate's Tailscale tailnet, not public space."""
    with pytest.raises(SSRFError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://100.63.255.255/x",  # the address immediately BELOW the /10
        "http://100.128.0.0/x",  # the address immediately ABOVE the /10
        "http://8.8.8.8/x",
        "http://1.1.1.1/x",
    ],
)
def test_validate_allows_addresses_bracketing_the_cgnat_block(url: str) -> None:
    """The deny is the /10 itself — not a sloppy ``startswith("100.")``."""
    validate_public_url(url)  # does not raise


@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:100.64.0.7]/x",  # IPv4-mapped
        "http://[64:ff9b::100.64.0.7]/x",  # NAT64 well-known prefix (RFC 6052)
        "http://[2002:6471:2030::]/x",  # 6to4 wrapping 100.64.0.7
    ],
)
def test_validate_blocks_ipv6_embedded_tailnet(url: str) -> None:
    """An IPv6 wrapper must not smuggle a tailnet target past the IPv4 rules.

    ``IPv6Address("::ffff:100.64.0.7").is_private`` is **False** and
    ``64:ff9b::100.64.0.7`` is even ``is_global`` — verified by execution on
    Python 3.14 — so each embedding has to be unwrapped explicitly.
    """
    with pytest.raises(SSRFError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.0.1/x",  # IETF protocol assignments (192.0.0.0/24)
        "http://198.18.0.1/x",  # benchmarking (198.18.0.0/15)
        "http://198.19.255.255/x",
    ],
)
def test_validate_blocks_special_purpose_ranges(url: str) -> None:
    with pytest.raises(SSRFError):
        validate_public_url(url)


def test_validate_still_allows_test_net() -> None:
    """TEST-NET stays allowed — it is the standard 'pretend public IP' fixture."""
    validate_public_url("http://203.0.113.10/x")  # does not raise


# --------------------------------------------------------------------------- #
# safe_post / safe_request — method + body semantics across manual redirects
# --------------------------------------------------------------------------- #


def _recording_handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler that records every request it is given."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        host = _host_of(request)
        path = request.url.path
        if host == "public.example.com" and path == "/307":
            return httpx.Response(307, headers={"location": "https://public.example.com/land"})
        if host == "public.example.com" and path == "/302":
            return httpx.Response(302, headers={"location": "https://public.example.com/land"})
        if host == "public.example.com" and path == "/303":
            return httpx.Response(303, headers={"location": "https://public.example.com/land"})
        if host == "public.example.com" and path == "/cross":
            return httpx.Response(307, headers={"location": "https://cdn.example.com/land"})
        if host == "public.example.com" and path == "/to-tailnet":
            return httpx.Response(302, headers={"location": "http://100.64.0.7/secret"})
        if path == "/land":
            return httpx.Response(200, content=b"landed")
        if path == "/echo":
            return httpx.Response(200, content=request.content)
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_safe_post_sends_body() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_post(client, "https://public.example.com/echo", content=b'{"a":1}')
    assert resp.status_code == 200
    assert resp.content == b'{"a":1}'
    assert seen[0].method == "POST"


@pytest.mark.asyncio
async def test_post_body_preserved_across_307() -> None:
    """307 preserves both method and body."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_post(
            client,
            "https://public.example.com/307",
            content=b'{"a":1}',
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200
    assert len(seen) == 2
    assert seen[1].method == "POST"
    assert seen[1].content == b'{"a":1}'
    assert seen[1].headers.get("content-type") == "application/json"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/302", "/303"])
async def test_post_downgraded_to_get_and_body_dropped(path: str) -> None:
    """301/302/303 turn a POST into a bodiless GET — including its body headers."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_post(
            client,
            f"https://public.example.com{path}",
            content=b'{"a":1}',
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200
    assert len(seen) == 2
    assert seen[1].method == "GET"
    assert seen[1].content == b""
    assert "content-type" not in {k.lower() for k in seen[1].headers}


@pytest.mark.asyncio
async def test_post_redirect_into_tailnet_is_rejected_before_hop_two() -> None:
    """A public host cannot bounce a POST into the tailnet; hop 2 is never issued."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_post(client, "https://public.example.com/to-tailnet", content=b"payload")
    assert len(seen) == 1  # only the first hop was ever sent


@pytest.mark.asyncio
async def test_auth_header_dropped_on_cross_host_redirect() -> None:
    """A signed/authorized request must not have its credentials replayed elsewhere.

    ``x-sig`` was previously asserted to SURVIVE the hop, on the reasoning that
    it is "not a credential". It is an HMAC over the body — i.e. precisely a
    credential, and precisely the kind of name a deny-list of
    ``{authorization, proxy-authorization}`` can never enumerate. Under the
    allow-list it is dropped, and this test now pins that.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_post(
            client,
            "https://public.example.com/cross",
            content=b"payload",
            headers={"authorization": "Bearer secret-token", "x-sig": "hmac"},
        )
    assert len(seen) == 2
    assert seen[0].headers.get("authorization") == "Bearer secret-token"
    assert seen[0].headers.get("x-sig") == "hmac"
    assert "authorization" not in {k.lower() for k in seen[1].headers}
    assert "x-sig" not in {k.lower() for k in seen[1].headers}


@pytest.mark.asyncio
async def test_auth_header_kept_on_same_host_redirect() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_post(
            client,
            "https://public.example.com/307",
            content=b"payload",
            headers={"authorization": "Bearer secret-token"},
        )
    assert seen[1].headers.get("authorization") == "Bearer secret-token"


@pytest.mark.asyncio
async def test_streaming_body_is_rejected() -> None:
    """A one-shot body cannot survive a manually-followed redirect — fail loudly."""

    def gen() -> Iterator[bytes]:
        yield b"chunk"

    # A caller passing a generator is exactly the mistake this guard exists for;
    # cast so the *static* type stays honest while the runtime value is wrong.
    one_shot = cast(bytes, gen())
    transport = httpx.MockTransport(_recording_handler([]))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="replayable"):
            await safe_post(client, "https://public.example.com/echo", content=one_shot)


@pytest.mark.asyncio
async def test_safe_post_max_redirects_zero_refuses_to_follow() -> None:
    """A signed webhook passes max_redirects=0 — a 30x must not be followed."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError, match="Too many redirects"):
            await safe_post(client, "https://public.example.com/307", content=b"x", max_redirects=0)
    assert len(seen) == 1


def test_safe_post_sync_sends_body() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    with httpx.Client(transport=transport) as client:
        resp = safe_post_sync(client, "https://public.example.com/echo", content=b"sync-body")
    assert resp.content == b"sync-body"
    assert seen[0].method == "POST"


def test_safe_request_sync_arbitrary_method() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_request_sync(client, "PUT", "https://public.example.com/echo", content=b"p")
    assert seen[0].method == "PUT"


# --------------------------------------------------------------------------- #
# Adversarial regressions — each of these FAILED before the fix that follows it
# --------------------------------------------------------------------------- #


def _chain_handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """302 -> 307 -> 200, so a dropped body must stay dropped for the rest of the walk."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/a":  # downgrades POST -> GET and drops the body
            return httpx.Response(302, headers={"location": "https://public.example.com/b"})
        if path == "/b":  # 307 "preserves" method+body — must NOT resurrect it
            return httpx.Response(307, headers={"location": "https://public.example.com/c"})
        return httpx.Response(200, content=b"done")

    return handler


@pytest.mark.asyncio
async def test_dropped_body_is_not_resurrected_by_a_later_307() -> None:
    """302 (drop) then 307 (keep) must not replay the discarded body.

    ``keep_body`` is latched: once a hop drops the body it is gone for the rest
    of the chain. Re-deriving it per hop sent the original POST body on hop 3 —
    as a GET, to a host the redirector chose.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_chain_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_post(
            client,
            "https://public.example.com/a",
            content=b'{"secret":1}',
            headers={"content-type": "application/json"},
        )
    assert len(seen) == 3
    assert seen[1].method == "GET"
    assert seen[2].method == "GET"
    assert seen[2].content == b""  # the body stays dropped
    assert "content-type" not in {k.lower() for k in seen[2].headers}


@pytest.mark.asyncio
async def test_form_data_body_is_dropped_on_302_like_content() -> None:
    """``data=`` is a body too — it must obey the same drop rule as ``content=``."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_post(client, "https://public.example.com/302", data={"amount": "100"})
    assert len(seen) == 2
    assert seen[1].method == "GET"
    assert seen[1].content == b""


@pytest.mark.asyncio
async def test_files_kwarg_is_rejected() -> None:
    """A file handle cannot be replayed on hop 2 — refuse it instead of sending empty."""
    transport = httpx.MockTransport(_recording_handler([]))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="files="):
            await safe_post(client, "https://public.example.com/echo", files={"f": b"x"})


@pytest.mark.asyncio
async def test_auth_kwarg_is_rejected() -> None:
    """httpx applies ``auth=`` per-request, so it cannot be dropped cross-origin."""
    transport = httpx.MockTransport(_recording_handler([]))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="auth="):
            await safe_post(client, "https://public.example.com/echo", auth=("u", "p"))


@pytest.mark.asyncio
async def test_streaming_data_mapping_required() -> None:
    transport = httpx.MockTransport(_recording_handler([]))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="replayable"):
            await safe_post(
                client, "https://public.example.com/echo", data=cast(dict, iter([b"x"]))
            )


def _downgrade_handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """https -> http on the SAME host: a different origin, so credentials must drop."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/downgrade":
            return httpx.Response(307, headers={"location": "http://public.example.com/land"})
        return httpx.Response(200, content=b"landed")

    return handler


@pytest.mark.asyncio
async def test_auth_dropped_on_https_to_http_downgrade_same_host() -> None:
    """Same hostname is NOT the same origin — an https->http hop must drop creds.

    Comparing hostname alone sent the bearer token in cleartext.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_downgrade_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_post(
            client,
            "https://public.example.com/downgrade",
            content=b"x",
            headers={"authorization": "Bearer secret-token"},
        )
    assert len(seen) == 2
    assert "authorization" not in {k.lower() for k in seen[1].headers}


# --- the sync loop had NO redirect coverage at all: inverting it failed nothing --- #


def test_sync_post_downgraded_to_get_on_302() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_post_sync(
            client,
            "https://public.example.com/302",
            content=b'{"a":1}',
            headers={"content-type": "application/json"},
        )
    assert len(seen) == 2
    assert seen[1].method == "GET"
    assert seen[1].content == b""
    assert "content-type" not in {k.lower() for k in seen[1].headers}


def test_sync_post_body_preserved_across_307() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_post_sync(client, "https://public.example.com/307", content=b'{"a":1}')
    assert seen[1].method == "POST"
    assert seen[1].content == b'{"a":1}'


def test_sync_auth_dropped_on_cross_host_redirect() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_post_sync(
            client,
            "https://public.example.com/cross",
            content=b"x",
            headers={"authorization": "Bearer t"},
        )
    assert "authorization" not in {k.lower() for k in seen[1].headers}


def test_sync_redirect_into_tailnet_rejected() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_recording_handler(seen))
    with httpx.Client(transport=transport) as client, pytest.raises(SSRFError):
        safe_post_sync(client, "https://public.example.com/to-tailnet", content=b"x")
    assert len(seen) == 1


def test_sync_dropped_body_not_resurrected_by_307() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_chain_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_post_sync(client, "https://public.example.com/a", content=b'{"secret":1}')
    assert len(seen) == 3
    assert seen[2].content == b""


# =========================================================================== #
# F5 — cross-origin header reduction (allow-list)
#
# The pre-0.55 rule stripped exactly {authorization, proxy-authorization} from
# the CALLER'S dict on a cross-origin hop. Three things were wrong with that:
#
#   1. the name space is attacker-chosen, so a deny-list is only ever as good
#      as the last name someone remembered — Cookie, X-Api-Key, X-License-Key,
#      X-Caller-Authorization and X-Tenant-Id were all replayed;
#   2. it operated on the caller's dict, so the client's OWN default headers
#      and its cookie jar were never even visible to it;
#   3. `auth=` on the client is applied inside send(), after any header edit.
#
# The tests below pin all three. Every one of them fails on the old code.
# =========================================================================== #


def _cross_origin_handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """`/cross` 307s to a DIFFERENT origin; `/same` 307s within the origin; `/land` 200s."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        host, path = _host_of(request), request.url.path
        if path == "/cross":
            return httpx.Response(307, headers={"location": "https://cdn.example.com/land"})
        if path == "/same":
            return httpx.Response(307, headers={"location": "https://public.example.com/land"})
        if host == "public.example.com" and path == "/latch":
            return httpx.Response(307, headers={"location": "https://cdn.example.com/back"})
        if host == "cdn.example.com" and path == "/back":
            # Bounces straight back to the ORIGINAL origin.
            return httpx.Response(307, headers={"location": "https://public.example.com/land"})
        return httpx.Response(200, content=b"landed")

    return handler


#: Headers that must never survive a hop to a redirector-chosen host. The last
#: five are this estate's own service-principal / tenant / license headers (see
#: ``w2v_common.auth.identity_context``); leaking them is identity forgery
#: material, not merely a credential leak.
_MUST_NOT_SURVIVE = [
    ("authorization", "Bearer keycloak-jwt"),
    ("proxy-authorization", "Basic cHJveHk6cHc="),
    ("cookie", "session=abc123"),
    ("x-api-key", "ak_live_deadbeef"),
    ("x-sig", "hmac-over-the-body"),
    ("x-caller-authorization", "Bearer original-caller-jwt"),
    ("x-license-key", "lic_secret_value"),
    ("x-tenant-id", "Bearer license-token"),
    ("x-license-id", "01890000-0000-7000-8000-000000000000"),
    ("x-subject-id", "subject-42"),
]

#: Headers that describe the REQUEST rather than the caller, and so must keep
#: working across a hop. ``user-agent`` is load-bearing: it is the one custom
#: header any current fleet call site passes into a redirect-following fetch
#: (``w2v_orchestrator`` site_capture).
_MUST_SURVIVE = [
    ("user-agent", "w2v-site-capture/1.0"),
    ("accept", "image/avif,image/webp,*/*"),
    ("accept-language", "en-GB,en;q=0.9"),
    ("cache-control", "no-cache"),
    ("range", "bytes=0-1023"),
    ("if-none-match", '"a1b2c3"'),
    ("if-modified-since", "Wed, 21 Oct 2026 07:28:00 GMT"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "value"), _MUST_NOT_SURVIVE)
async def test_credential_header_is_not_replayed_to_a_redirect_target(
    name: str, value: str
) -> None:
    """THE LEAK: each of these reached the redirector's chosen host before 0.55."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get(client, "https://public.example.com/cross", headers={name: value})
    assert len(seen) == 2
    assert _host_of(seen[0]) == "public.example.com"
    assert _host_of(seen[1]) == "cdn.example.com"
    assert seen[0].headers.get(name) == value, "must still reach the host the caller chose"
    assert name not in {k.lower() for k in seen[1].headers}, f"{name} leaked cross-origin"


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "value"), _MUST_NOT_SURVIVE)
async def test_credential_header_survives_a_same_origin_hop(name: str, value: str) -> None:
    """CONTROL: the reduction is scoped to an origin change, not to redirects in general.

    Without this the "fix" could be a blanket strip on every hop, which would
    break every authenticated caller that follows an in-origin redirect.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get(client, "https://public.example.com/same", headers={name: value})
    assert len(seen) == 2
    assert seen[1].headers.get(name) == value


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "value"), _MUST_SURVIVE)
async def test_request_describing_header_survives_a_cross_origin_hop(name: str, value: str) -> None:
    """CONTROL: an allow-list that dropped everything would be trivially "secure" and useless."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get(client, "https://public.example.com/cross", headers={name: value})
    assert len(seen) == 2
    assert seen[1].headers.get(name) == value


@pytest.mark.asyncio
async def test_client_default_headers_are_reduced_cross_origin() -> None:
    """A header set on the CLIENT leaked even for names the old deny-list did list.

    The old strip edited the caller's ``headers=`` dict. ``httpx.AsyncClient(
    headers=...)`` is merged in later, by ``build_request``, so it was never in
    that dict — an ``Authorization`` default on the client was replayed
    cross-origin despite ``authorization`` being on the deny-list.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(
        transport=transport,
        headers={"authorization": "Bearer client-default", "x-api-key": "client-default-key"},
    ) as client:
        await safe_get(client, "https://public.example.com/cross")
    assert seen[0].headers.get("authorization") == "Bearer client-default"
    assert seen[0].headers.get("x-api-key") == "client-default-key"
    assert "authorization" not in {k.lower() for k in seen[1].headers}
    assert "x-api-key" not in {k.lower() for k in seen[1].headers}


@pytest.mark.asyncio
async def test_cookie_jar_is_not_replayed_cross_origin() -> None:
    """The jar is a SECOND cookie channel, independent of any ``Cookie`` header.

    ``httpx.AsyncClient(cookies={...})`` builds a jar entry with no domain, and
    ``http.cookiejar`` then attaches it to every host — verified here by the
    first hop carrying it even though the pinned request URL is a bare IP. So
    stripping a caller-supplied ``Cookie`` header alone would not have closed
    this; the reduction has to run on the BUILT request, after the jar has been
    materialised into a header.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport, cookies={"session": "abc123"}) as client:
        await safe_get(client, "https://public.example.com/cross")
    assert seen[0].headers.get("cookie") == "session=abc123"
    assert "cookie" not in {k.lower() for k in seen[1].headers}


@pytest.mark.asyncio
async def test_cookie_jar_survives_a_same_origin_hop() -> None:
    """CONTROL: normal cookie-following behaviour within one origin is untouched."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport, cookies={"session": "abc123"}) as client:
        await safe_get(client, "https://public.example.com/same")
    assert seen[1].headers.get("cookie") == "session=abc123"


@pytest.mark.asyncio
async def test_client_level_auth_is_not_reapplied_cross_origin() -> None:
    """``auth=`` cannot be fixed by editing headers — httpx applies it inside ``send``.

    ``build_request`` leaves no ``Authorization`` behind (asserted below), so the
    header reduction never sees it. The only lever is refusing the client's auth
    flow for that hop, which is what ``_hop_auth`` does.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport, auth=("user", "pass")) as client:
        built = client.build_request("GET", "https://public.example.com/cross")
        assert "authorization" not in {k.lower() for k in built.headers}, (
            "auth= is applied at send-time; a header-only fix would be vacuous here"
        )
        await safe_get(client, "https://public.example.com/cross")
    assert seen[0].headers.get("authorization", "").startswith("Basic ")
    assert "authorization" not in {k.lower() for k in seen[1].headers}


@pytest.mark.asyncio
async def test_client_level_auth_still_signs_a_same_origin_hop() -> None:
    """CONTROL: an authenticated client must keep working across an in-origin redirect."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport, auth=("user", "pass")) as client:
        await safe_get(client, "https://public.example.com/same")
    assert seen[1].headers.get("authorization", "").startswith("Basic ")


@pytest.mark.asyncio
async def test_cross_origin_reduction_is_latched_across_a_return_hop() -> None:
    """A -> B -> A must not hand the credential back on the third hop.

    The third hop is same-origin with the first, so a per-hop (unlatched) test
    would call it safe. But B chose that hop: B can point back at A's origin and
    harvest whatever A would send itself. Once anything has crossed, it stays
    reduced.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get(
            client, "https://public.example.com/latch", headers={"x-api-key": "ak_live_deadbeef"}
        )
    assert len(seen) == 3
    assert [_host_of(r) for r in seen] == [
        "public.example.com",
        "cdn.example.com",
        "public.example.com",
    ]
    assert seen[0].headers.get("x-api-key") == "ak_live_deadbeef"
    assert "x-api-key" not in {k.lower() for k in seen[1].headers}
    assert "x-api-key" not in {k.lower() for k in seen[2].headers}, "latch released on return hop"


def test_sync_credential_header_is_not_replayed_to_a_redirect_target() -> None:
    """The sync path carries the identical rule — it is a full copy, not a thin shim."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    with httpx.Client(transport=transport, cookies={"session": "abc123"}) as client:
        safe_get_sync(
            client, "https://public.example.com/cross", headers={"x-tenant-id": "Bearer lic"}
        )
    assert len(seen) == 2
    assert seen[0].headers.get("x-tenant-id") == "Bearer lic"
    assert seen[0].headers.get("cookie") == "session=abc123"
    assert "x-tenant-id" not in {k.lower() for k in seen[1].headers}
    assert "cookie" not in {k.lower() for k in seen[1].headers}


def test_sync_cross_origin_reduction_is_latched() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_get_sync(client, "https://public.example.com/latch", headers={"x-api-key": "k"})
    assert len(seen) == 3
    assert "x-api-key" not in {k.lower() for k in seen[2].headers}


# --------------------------------------------------------------------------- #
# F5b — a caller-supplied Host header (vhost confusion)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_caller_host_header_cannot_override_the_pin() -> None:
    """``Host`` selects the vhost served on the validated address — this module owns it."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get(client, "https://public.example.com/land", headers={"Host": "svc.internal"})
    assert seen[0].headers.get("host") == "public.example.com"


@pytest.mark.asyncio
async def test_caller_host_header_cannot_override_an_allow_listed_host() -> None:
    """THE GAP: on the un-pinned ``allow_hosts`` path nothing overrode a caller ``Host``.

    Presigned MinIO URLs are deliberately fetched un-pinned (``_pin`` returns the
    URL untouched and supplies no ``Host``), so before 0.55 a caller-supplied
    ``Host`` was the only one on the wire: a request validated as
    ``blob.internal`` could be served by any other vhost on that address. The
    URL-derived Host must win.
    """
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get(
            client,
            "https://blob.internal/bucket/obj",
            allow_hosts={"blob.internal"},
            headers={"host": "svc.internal"},
        )
    assert seen[0].headers.get("host") == "blob.internal"


def test_sync_caller_host_header_cannot_override_an_allow_listed_host() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    with httpx.Client(transport=transport) as client:
        safe_get_sync(
            client,
            "https://blob.internal/bucket/obj",
            allow_hosts={"blob.internal"},
            headers={"Host": "svc.internal"},
        )
    assert seen[0].headers.get("host") == "blob.internal"


# =========================================================================== #
# F6 — a DEFAULT response size cap
#
# Before 0.55 `max_fetch_bytes` defaulted to None, i.e. `await response.aread()`
# with no bound. skill_builder's tenant-authored `api_call` step passes no cap
# and then `.json()`s the result back to the caller, on a pod limited to 512Mi.
# =========================================================================== #


_OVER_DEFAULT = DEFAULT_MAX_FETCH_BYTES + 1


def _oversized_declared_handler(request: httpx.Request) -> httpx.Response:
    """Declare a Content-Length past the default cap without allocating it."""
    return httpx.Response(200, headers=[("content-length", str(_OVER_DEFAULT))], content=b"tiny")


@pytest.mark.asyncio
async def test_default_cap_rejects_an_oversized_declared_body() -> None:
    """THE DoS: with no cap stated, this body was read into pod memory in full."""
    transport = httpx.MockTransport(_oversized_declared_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get(client, "https://public.example.com/huge")


def test_sync_default_cap_rejects_an_oversized_declared_body() -> None:
    transport = httpx.MockTransport(_oversized_declared_handler)
    with httpx.Client(transport=transport) as client, pytest.raises(ResponseTooLargeError):
        safe_get_sync(client, "https://public.example.com/huge")


@pytest.mark.asyncio
async def test_default_cap_applies_to_safe_request_the_skill_builder_entry_point() -> None:
    """``safe_request`` is the exact function skill_builder's ``api_call`` step calls."""
    transport = httpx.MockTransport(_oversized_declared_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_request(client, "POST", "https://public.example.com/huge", json={"a": 1})


@pytest.mark.asyncio
async def test_default_cap_aborts_an_undeclared_oversized_stream() -> None:
    """A body that lies about (or omits) its length is stopped by the running counter.

    The generator is unbounded: if the default cap were not enforced on the
    streaming path this test would not fail, it would never terminate. The chunk
    count proves the read was abandoned just past the cap rather than drained.
    """
    chunks_served = 0
    chunk = b"a" * (1024 * 1024)

    async def endless() -> AsyncIterator[bytes]:
        nonlocal chunks_served
        while True:
            chunks_served += 1
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=endless())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get(client, "https://public.example.com/endless")
    assert chunks_served <= (DEFAULT_MAX_FETCH_BYTES // len(chunk)) + 2


@pytest.mark.asyncio
async def test_body_under_the_default_cap_still_reads_normally() -> None:
    """CONTROL: the overwhelming majority of fleet fetches are far under 64 MiB."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"ok":true}')

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/ok")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_an_explicit_larger_cap_beats_the_default() -> None:
    """CONTROL: the presigned-artifact callers that need more can still ask for it.

    ``video_compiler_service`` states 512 MiB for MinIO shot assets; that must
    not be silently clamped to the default.
    """
    transport = httpx.MockTransport(_oversized_declared_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(
            client, "https://public.example.com/huge", max_fetch_bytes=512 * 1024 * 1024
        )
    assert resp.status_code == 200


def test_default_cap_is_the_declared_default_of_every_public_wrapper() -> None:
    """All SEVEN entry points must inherit the cap — a wrapper left at ``None`` is the hole.

    Two prior releases shipped a guard that only some of the wrappers used; this
    derives the check from the module's public surface instead of hand-listing.
    """
    import inspect

    wrappers = [
        safe_get,
        safe_get_sync,
        safe_get_to_path,
        safe_post,
        safe_post_sync,
        safe_request,
        safe_request_sync,
    ]
    defaults = {
        fn.__name__: inspect.signature(fn).parameters["max_fetch_bytes"].default for fn in wrappers
    }
    assert set(defaults.values()) == {DEFAULT_MAX_FETCH_BYTES}, defaults
    assert DEFAULT_MAX_FETCH_BYTES == 64 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Request-line integrity — the method must be a bare RFC 9110 token
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("method", "why"),
    [
        ("GET\r\nX-Injected: 1", "CRLF splits the request line and injects a header"),
        ("GET\nX-Injected: 1", "bare LF is enough on a lenient server"),
        ("GET /admin HTTP/1.1", "SP smuggles a second request-target"),
        ("GET\x00", "NUL"),
        ("GET\tX", "tab"),
        ("", "empty method"),
    ],
)
def test_non_token_method_is_rejected(method: str, why: str) -> None:
    """The method reaches this module from tenant-authored config; httpx does not validate it.

    ``build_request`` stores the method verbatim and h11 formats the request
    line with a raw ``b"%s %s HTTP/1.1\\r\\n" %`` interpolation. h11 does reject
    illegal method characters — but h11 is one layer BELOW this module and is
    absent under ``httpx.MockTransport``, so before this check every mocked test
    in the fleet would have let a splitting method through while reporting the
    guard healthy. The guarantee has to live here.
    """
    transport = httpx.MockTransport(_cross_origin_handler([]))
    with httpx.Client(transport=transport) as client, pytest.raises(ValueError) as ei:
        safe_request_sync(client, method, "https://public.example.com/land")
    assert "token" in str(ei.value)


def test_method_validation_rejects_before_any_socket_is_opened() -> None:
    """A rejected method must not produce a first hop."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    with httpx.Client(transport=transport) as client, pytest.raises(ValueError):
        safe_request_sync(client, "GET\r\nX: 1", "https://public.example.com/land")
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def test_ordinary_methods_are_accepted(method: str) -> None:
    """CONTROL: every method the fleet actually uses must still pass."""
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_cross_origin_handler(seen))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_request(client, method, "https://public.example.com/land")
    assert resp.status_code == 200
    assert seen[0].method == method


# --------------------------------------------------------------------------- #
# Content-Encoding round-trip. ``aiter_bytes()``/``aread()`` hand back bytes that
# httpx has ALREADY content-decoded. Re-attaching the origin's
# ``Content-Encoding`` to the materialized Response makes httpx decode a SECOND
# time on first body access -> ``httpx.DecodingError: Error -3 while
# decompressing data: incorrect header check``.
#
# Observed live 2026-08-10: w2v_orchestrator ``site_capture`` scraping
# https://www.anthropic.com/news logged exactly that sentence, silently lost the
# page text, and produced an ungrounded promo script. Most of the public web
# serves gzip/br, so this is the common path, not an edge case.
# --------------------------------------------------------------------------- #

_GZIP_BODY = b"<html><body>ground me</body></html>"


def _compressing_handler(encoding: str, raw: bytes) -> Callable[[httpx.Request], httpx.Response]:
    """Serve ``raw`` compressed, advertising ``encoding`` — like any real CDN."""

    def handler(_request: httpx.Request) -> httpx.Response:
        if encoding == "gzip":
            import gzip

            body = gzip.compress(raw)
        elif encoding == "deflate":
            import zlib

            body = zlib.compress(raw)
        else:  # pragma: no cover - guarded by the parametrize list
            raise AssertionError(encoding)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8", "content-encoding": encoding},
            content=body,
        )

    return handler


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
@pytest.mark.parametrize("max_fetch_bytes", [None, DEFAULT_MAX_FETCH_BYTES])
@pytest.mark.asyncio
async def test_safe_get_decodes_compressed_body_exactly_once(
    encoding: str, max_fetch_bytes: int | None
) -> None:
    """Both the capped and uncapped paths must return the DECODED body, once."""
    transport = httpx.MockTransport(_compressing_handler(encoding, _GZIP_BODY))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(
            client, "https://public.example.com/page", max_fetch_bytes=max_fetch_bytes
        )
    assert resp.content == _GZIP_BODY
    assert resp.text == _GZIP_BODY.decode()


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_safe_get_sync_decodes_compressed_body_exactly_once(encoding: str) -> None:
    """Sync path shares ``_materialize`` and therefore shared the same defect."""
    transport = httpx.MockTransport(_compressing_handler(encoding, _GZIP_BODY))
    with httpx.Client(transport=transport) as client:
        resp = safe_get_sync(client, "https://public.example.com/page")
    assert resp.content == _GZIP_BODY
    assert resp.text == _GZIP_BODY.decode()


@pytest.mark.asyncio
async def test_materialized_response_does_not_claim_a_stale_content_encoding() -> None:
    """The header must not outlive the encoding it describes.

    ``resp.content`` is decoded plaintext; leaving ``Content-Encoding: gzip`` on it
    is a lie that any downstream re-parse (or a caller re-materializing the
    Response) would trip over exactly as httpx did.
    """
    transport = httpx.MockTransport(_compressing_handler("gzip", _GZIP_BODY))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/page")
    assert "content-encoding" not in resp.headers
    assert int(resp.headers["content-length"]) == len(_GZIP_BODY)


@pytest.mark.asyncio
async def test_content_encoding_identity_does_not_survive_materialization() -> None:
    """``identity`` is still a claim about the wire body, and still false of a decoded one.

    The rule is a property — "httpx has already decoded this" — not a list of
    codec names, so ``identity`` is dropped alongside ``gzip``. Pinning it here
    keeps anyone from "fixing" :func:`_materialize` into a ``== "gzip"`` special
    case that leaves the other values behind.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": "identity"}, content=_GZIP_BODY)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/page")
    assert resp.content == _GZIP_BODY
    assert "content-encoding" not in resp.headers


@pytest.mark.asyncio
async def test_stacked_content_encodings_decode_once_and_leave_no_header() -> None:
    """RFC 9110 allows a coding LIST; httpx unwinds every layer before we see it.

    ``Content-Encoding: gzip, deflate`` means gzip was applied first and deflate
    second, so the body is ``deflate(gzip(raw))`` and httpx's MultiDecoder peels
    both. One stale header would re-run BOTH decoders, so the multi-coding case
    is strictly worse than the single one — assert the whole header is gone, not
    just one coding.
    """
    import gzip as _gzip
    import zlib as _zlib

    body = _zlib.compress(_gzip.compress(_GZIP_BODY))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": "gzip, deflate"}, content=body)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/page")
    assert resp.content == _GZIP_BODY
    assert "content-encoding" not in resp.headers


@pytest.mark.asyncio
async def test_redirect_hop_to_a_compressed_final_response_decodes_once() -> None:
    """The body is read on the LAST hop, so a plain 302 into a gzip origin is the real shape.

    Every production caller of this helper reaches its payload through at least
    one redirect (``http`` -> ``https``, apex -> ``www``, CDN shuffles). A test
    that only ever compresses a direct 200 leaves the common path uncovered.
    """
    final = _compressing_handler("gzip", _GZIP_BODY)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return _redirect_to("https://cdn.example.com/final-gz")
        return final(request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/start")
    assert resp.status_code == 200
    assert resp.content == _GZIP_BODY
    assert "content-encoding" not in resp.headers


@pytest.mark.asyncio
async def test_uncompressed_response_keeps_every_header_that_is_not_wire_only() -> None:
    """The drop must be surgical: three wire headers, and nothing else.

    A fix phrased as "rebuild with fewer headers" is one careless edit away from
    losing ``Set-Cookie`` (repeated, so it also pins that ``multi_items()`` is
    used rather than a dict comprehension that would keep only the last one) or
    ``Content-Type`` (which ``.text``/``.json()`` need for charset). The response
    here is NOT compressed, so this is the no-regression half as well.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[
                ("content-type", "application/json; charset=utf-8"),
                ("set-cookie", "a=1"),
                ("set-cookie", "b=2"),
                ("x-ratelimit-remaining", "4999"),
                ("transfer-encoding", "chunked"),
            ],
            content=b'{"ok": true}',
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/page")

    assert resp.json() == {"ok": True}
    assert resp.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert resp.headers["content-type"] == "application/json; charset=utf-8"
    assert resp.headers["x-ratelimit-remaining"] == "4999"
    # Wire-only, all three gone; Content-Length recomputed from the bytes held.
    assert "transfer-encoding" not in resp.headers
    assert "content-encoding" not in resp.headers
    assert int(resp.headers["content-length"]) == len(b'{"ok": true}')


# --------------------------------------------------------------------------- #
# max_fetch_bytes vs. compression. The cap is a SECURITY control (bounded
# memory), so which side of the decoder it counts on is load-bearing and must
# be pinned by name rather than left to be re-derived.
#
# MEASURED 2026-08-19: the cap counts DECODED bytes. `aiter_bytes()` yields
# post-decoder chunks, so the running total is the expanded size. That is the
# safe direction — a 2 kB gzip that expands to 2 MB is refused under a 100 kB
# cap. Counting wire bytes instead would let any compression bomb through.
#
# The `Content-Length` pre-check (`_content_length_exceeds`) is the OTHER unit:
# it reads the wire header, i.e. the COMPRESSED length. It is only a fast path
# that can refuse early; it can never permit more than the streaming counter
# allows, so the mismatch is conservative. These tests pin both halves.
# --------------------------------------------------------------------------- #

_BOMB_RAW = b"A" * 2_000_000  # 2 MB decoded


def _bomb_handler(request: httpx.Request) -> httpx.Response:
    import gzip as _gzip

    return httpx.Response(
        200, headers={"content-encoding": "gzip"}, content=_gzip.compress(_BOMB_RAW)
    )


@pytest.mark.asyncio
async def test_max_fetch_bytes_counts_decoded_not_wire_bytes_so_a_compression_bomb_is_refused() -> (
    None
):
    """~2 kB on the wire, 2 MB decoded, 100 kB cap -> refused.

    The compressed body is far UNDER the cap, so the `Content-Length` pre-check
    waves it through; only the streaming counter can catch it, and it can only
    catch it because it counts decoded bytes. If this ever starts returning a
    body, the cap has been moved to `aiter_raw()` and the OOM guard is defeated
    by any `Content-Encoding: gzip`.
    """
    transport = httpx.MockTransport(_bomb_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get(client, "https://public.example.com/bomb", max_fetch_bytes=100_000)


def test_max_fetch_bytes_sync_counts_decoded_not_wire_bytes_so_a_compression_bomb_is_refused() -> (
    None
):
    """Sync twin shares the cap loop as well as ``_materialize``."""
    transport = httpx.MockTransport(_bomb_handler)
    with httpx.Client(transport=transport) as client, pytest.raises(ResponseTooLargeError):
        safe_get_sync(client, "https://public.example.com/bomb", max_fetch_bytes=100_000)


@pytest.mark.asyncio
async def test_compressed_body_under_the_decoded_cap_is_returned_whole() -> None:
    """The other side of the same boundary: a cap above the DECODED size must not fire."""
    transport = httpx.MockTransport(_bomb_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/bomb", max_fetch_bytes=5_000_000)
    assert len(resp.content) == len(_BOMB_RAW)


# --------------------------------------------------------------------------- #
# All FOUR public entry points funnel through ``_materialize``, so all four
# carried the defect. ``safe_get``/``safe_get_sync`` are covered above; these
# two pin the method-agnostic pair directly rather than through the GET wrapper.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_safe_request_decodes_a_compressed_body_exactly_once() -> None:
    transport = httpx.MockTransport(_compressing_handler("gzip", _GZIP_BODY))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_request(client, "POST", "https://public.example.com/api", json={"a": 1})
    assert resp.content == _GZIP_BODY
    assert "content-encoding" not in resp.headers


def test_safe_request_sync_decodes_a_compressed_body_exactly_once() -> None:
    transport = httpx.MockTransport(_compressing_handler("gzip", _GZIP_BODY))
    with httpx.Client(transport=transport) as client:
        resp = safe_request_sync(client, "POST", "https://public.example.com/api", json={"a": 1})
    assert resp.content == _GZIP_BODY
    assert "content-encoding" not in resp.headers


def test_safe_post_sync_decodes_a_compressed_body_exactly_once() -> None:
    transport = httpx.MockTransport(_compressing_handler("gzip", _GZIP_BODY))
    with httpx.Client(transport=transport) as client:
        resp = safe_post_sync(client, "https://public.example.com/api", json={"a": 1})
    assert resp.content == _GZIP_BODY
    assert "content-encoding" not in resp.headers


@pytest.mark.asyncio
async def test_an_uncompressed_image_fetch_passes_on_the_broken_code_too() -> None:
    """This is the shape that let the defect live in production for months.

    Eight of the nine fleet call sites fetch images/video. Servers do not gzip
    JPEG/PNG/MP4 — they are already compressed — so those responses carry no
    ``Content-Encoding`` and the double-decode never fires. The first consumer to
    fetch JSON (skill_builder's ``api_call`` step) hit it immediately.

    Kept deliberately as a NEGATIVE control: it passes with or without the fix.
    Its job is to stop someone from adding an image fixture, seeing it green, and
    concluding the compressed path is covered — it is not. The tests that
    actually bite are the ``_compressing_handler`` ones above.
    """
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=png)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://cdn.example.com/a.png")
    assert resp.content == png


# --------------------------------------------------------------------------- #
# safe_get_to_path — stream straight to disk, constant memory.
#
# WHY IT EXISTS: ``safe_get`` accumulates ``list[bytes]`` then ``b"".join``s it,
# so peak RSS is ~2x the body. ``video_compiler_service`` ingests tenant drone
# footage up to 4 GiB ⇒ ~8 GiB per concurrent fetch ⇒ OOMKilled pod.
#
# The security posture must be IDENTICAL to ``safe_get`` — it shares the one
# redirect/validation loop (``_safe_request_to_sink``) and only swaps the
# terminal. The tests below therefore pin BOTH halves: that the guards still
# fire on the streaming path, and that the file handling is atomic/self-cleaning.
# --------------------------------------------------------------------------- #


def _partials_in(directory: Path) -> list[Path]:
    """Every temp file ``_path_sink`` could have left behind in ``directory``."""
    return [p for p in directory.iterdir() if ".part-" in p.name]


@pytest.mark.asyncio
async def test_safe_get_to_path_writes_the_full_body_and_returns_dest(tmp_path: Path) -> None:
    """CONTROL: the happy path must land the exact bytes at ``dest`` and leave no temp."""
    dest = tmp_path / "clip.mp4"
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        got = await safe_get_to_path(client, "https://public.example.com/ok", dest)
    assert got == dest
    assert dest.read_bytes() == b"image-bytes"
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_shares_the_redirect_guard(tmp_path: Path) -> None:
    """The streaming terminal must not have escaped ``_validate_redirect``.

    A public host 302-ing to RFC1918 is the whole reason this module exists. If a
    new entry point re-implemented the loop, this is the test that would catch it
    — and nothing may be written when the hop is refused.
    """
    dest = tmp_path / "leak.bin"
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await safe_get_to_path(client, "https://public.example.com/to-internal", dest)
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_follows_a_public_redirect_to_the_final_body(tmp_path: Path) -> None:
    """The positive twin: a legal hop still reaches the FINAL body, not the 302's."""
    dest = tmp_path / "clip.mp4"
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get_to_path(client, "https://public.example.com/to-public", dest)
    assert dest.read_bytes() == b"final-bytes"


@pytest.mark.asyncio
async def test_safe_get_to_path_rejects_an_oversized_declared_content_length(
    tmp_path: Path,
) -> None:
    """The ``_content_length_exceeds`` pre-check must still fire before any body read."""
    dest = tmp_path / "huge.bin"
    transport = httpx.MockTransport(_oversized_declared_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get_to_path(client, "https://public.example.com/huge", dest)
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_aborts_an_undeclared_oversized_stream(tmp_path: Path) -> None:
    """A body that omits its length is stopped mid-flight by the running counter.

    The generator is UNBOUNDED: without the cap on the streaming path this does
    not fail, it fills the disk and never terminates. The chunk count proves the
    read was abandoned just past the cap rather than drained.
    """
    dest = tmp_path / "endless.bin"
    chunks_served = 0
    chunk = b"a" * 1024
    cap = 8 * 1024

    async def endless() -> AsyncIterator[bytes]:
        nonlocal chunks_served
        while True:
            chunks_served += 1
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=endless())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get_to_path(
                client, "https://public.example.com/endless", dest, max_fetch_bytes=cap
            )
    assert chunks_served <= (cap // len(chunk)) + 2
    assert not dest.exists()


@pytest.mark.asyncio
async def test_safe_get_to_path_removes_its_partial_file_on_a_cap_breach(tmp_path: Path) -> None:
    """THE REQUIREMENT: a refused download must leave NOTHING on disk.

    Separate from the abort test above because the failure modes differ: the cap
    could fire correctly and still orphan a multi-GiB ``.part-`` file, filling the
    node's ephemeral storage and getting the pod evicted.

    The body is streamed from an async generator so it carries NO
    ``Content-Length``. That is essential, not incidental: a declared length is
    refused by the ``_content_length_exceeds`` pre-check up in the loop, before the
    sink ever opens a file — so a fixed ``content=`` body would assert that no
    partial exists in a run that never created one, and the test would be vacuous.
    Here the sink genuinely opens the temp file and writes into it first.
    """
    dest = tmp_path / "partial.bin"

    async def slow_body() -> AsyncIterator[bytes]:
        for _ in range(40):
            yield b"b" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=slow_body())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get_to_path(
                client, "https://public.example.com/x", dest, max_fetch_bytes=1024
            )
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_counts_decoded_bytes_so_a_compression_bomb_is_refused(
    tmp_path: Path,
) -> None:
    """~2 kB on the wire, 2 MB decoded, 100 kB cap -> refused, nothing written.

    The compressed body is far UNDER the cap so the ``Content-Length`` pre-check
    waves it through; only the streaming counter can catch it, and only because it
    counts DECODED bytes. If this starts passing, the cap has moved to
    ``aiter_raw()`` and any ``Content-Encoding: gzip`` defeats the guard — which on
    this path means an arbitrarily large file on the node's disk, not just RAM.
    """
    dest = tmp_path / "bomb.bin"
    transport = httpx.MockTransport(_bomb_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get_to_path(
                client, "https://public.example.com/bomb", dest, max_fetch_bytes=100_000
            )
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_writes_a_compressed_body_decoded_and_whole(tmp_path: Path) -> None:
    """The other side of that boundary: under the DECODED cap, the file is the PLAINTEXT.

    Pins that the sink writes post-decoder bytes — writing the gzip stream to disk
    would hand ffmpeg an unreadable file while every size assertion still passed.
    """
    dest = tmp_path / "bomb.bin"
    transport = httpx.MockTransport(_bomb_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get_to_path(
            client, "https://public.example.com/bomb", dest, max_fetch_bytes=5_000_000
        )
    assert dest.read_bytes() == _BOMB_RAW


@pytest.mark.asyncio
async def test_safe_get_to_path_zero_cap_streams_unbounded(tmp_path: Path) -> None:
    """``max_fetch_bytes=0`` means UNLIMITED, matching every other wrapper.

    It must NOT mean "a cap of zero bytes" (which would refuse everything) and it
    must NOT silently fall back to the 64 MiB default.
    """
    dest = tmp_path / "big.bin"
    body = b"c" * (DEFAULT_MAX_FETCH_BYTES + 4096)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await safe_get_to_path(client, "https://public.example.com/big", dest, max_fetch_bytes=0)
    assert dest.stat().st_size == len(body)


@pytest.mark.asyncio
async def test_safe_get_to_path_publishes_atomically_via_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dest`` must NEVER exist in a truncated state — the rename is the publication.

    ``video_compiler_service``'s per-source dedupe makes one fetched file the shared
    ``-i`` input of N shots, so a torn file does not fail loudly: ffmpeg decodes a
    short clip and the delivered film is silently missing footage.

    Simulated by making the final ``Path.replace`` (``os.replace``) fail AFTER the
    body is fully written. A ``dest.open("wb")``-style implementation would already
    have created and truncated ``dest`` by this point; the atomic one leaves it
    absent.
    """
    dest = tmp_path / "clip.mp4"

    def boom(self: Path, target: object) -> None:
        raise OSError("rename refused")

    monkeypatch.setattr(safe_fetch_module.Path, "replace", boom)
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OSError, match="rename refused"):
            await safe_get_to_path(client, "https://public.example.com/ok", dest)
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_leaves_a_good_earlier_file_untouched_when_the_fetch_fails(
    tmp_path: Path,
) -> None:
    """A FAILED re-fetch must not destroy the file an earlier successful fetch published.

    ``dest.write_bytes`` would have truncated the good file the moment the second
    fetch started. Under the dedupe this is the difference between "one shot
    degrades" and "every shot sharing this source renders from a 0-byte input".

    Undeclared length again (see the test above): a declared one is refused before
    the sink opens anything, which would make this pass even against a sink that
    writes straight to ``dest``.
    """
    dest = tmp_path / "clip.mp4"
    dest.write_bytes(b"good-earlier-bytes")

    async def slow_body() -> AsyncIterator[bytes]:
        for _ in range(40):
            yield b"z" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=slow_body())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLargeError):
            await safe_get_to_path(
                client, "https://public.example.com/x", dest, max_fetch_bytes=1024
            )
    assert dest.read_bytes() == b"good-earlier-bytes"


@pytest.mark.asyncio
async def test_safe_get_to_path_cleans_up_and_re_raises_cancellation_unchanged(
    tmp_path: Path,
) -> None:
    """Cancellation is the LIVE path, not a curiosity.

    ``video_compiler_service`` cancels its download fan-out on job teardown. A bare
    ``except Exception`` would let ``CancelledError`` skip the unlink (orphaning the
    partial), and a handler that swallowed or re-wrapped it would break asyncio's
    cancellation contract — the task would look like it completed.
    """
    dest = tmp_path / "cancelled.bin"

    async def cancelling() -> AsyncIterator[bytes]:
        yield b"d" * 1024
        raise asyncio.CancelledError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=cancelling())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(asyncio.CancelledError):
            await safe_get_to_path(client, "https://public.example.com/cancel", dest)
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


# --------------------------------------------------------------------------- #
# safe_get_to_path — HTTP STATUS handling.
#
# THE DEFECT (found live by video_compiler_service): the streaming terminal had
# no status check at all, so a 404 streamed the error body to ``dest`` and
# returned it as a success — ``RETURNED OK: .../out.bin True
# b'<Error>NoSuchKey</Error>'``. The buffering caller it replaced did
# ``resp.raise_for_status()`` on the response it got back; streaming leaves no
# response to check, so an EXPIRED PRESIGNED MinIO URL stopped raising
# ``httpx.HTTPStatusError`` into ``_BAD_ASSET_ERRORS`` and started handing ffmpeg
# a few bytes of S3 error XML — a silently short or corrupt film instead of a
# cleanly degraded shot.
#
# The opt-in must stay opt-in: ``safe_request``/``safe_get``/``safe_post`` and the
# sync loop deliberately RETURN a non-2xx response, and fleet callers inspect it.
# --------------------------------------------------------------------------- #

#: What MinIO actually returns for an expired/absent key — the body the old code
#: wrote to disk and called a successful download.
_S3_ERROR_XML = b"<?xml version='1.0'?><Error><Code>NoSuchKey</Code></Error>"


def _status_handler(status: int) -> Callable[[httpx.Request], httpx.Response]:
    """A handler answering every request with ``status`` and an S3-shaped error body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=_S3_ERROR_XML)

    return handler


@pytest.mark.parametrize("status", [404, 500])
@pytest.mark.asyncio
async def test_safe_get_to_path_raises_on_an_error_status_and_writes_nothing(
    tmp_path: Path, status: int
) -> None:
    """A non-success status must raise and leave NO file — not even the error body.

    ``dest.exists()`` is the assertion that actually pins the reported defect: the
    old code returned ``dest`` with ``<Error>NoSuchKey</Error>`` inside it. The
    partial check pins the second half — the status is checked BEFORE the sink, so
    no temp file is ever opened either.
    """
    dest = tmp_path / "out.bin"
    transport = httpx.MockTransport(_status_handler(status))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await safe_get_to_path(client, "https://public.example.com/gone", dest)
    assert excinfo.value.response.status_code == status
    assert excinfo.value.request is not None
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_raises_when_a_redirect_chain_ends_in_an_error(
    tmp_path: Path,
) -> None:
    """The check must sit AFTER the redirect loop, not before it.

    A status check hoisted above the loop would treat the 302 itself as the final
    status and either raise on a legal redirect or — since 302 is not an error —
    wave the terminal 404 straight through to the sink. Only a check inside the
    non-redirect branch gets both right.
    """
    dest = tmp_path / "out.bin"

    def handler(request: httpx.Request) -> httpx.Response:
        if _host_of(request) == "public.example.com":
            return _redirect_to("https://cdn.example.com/gone")
        return httpx.Response(404, content=_S3_ERROR_XML)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await safe_get_to_path(client, "https://public.example.com/to-gone", dest)
    assert excinfo.value.response.status_code == 404
    assert not dest.exists()
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_leaves_a_good_earlier_file_untouched_when_the_refetch_404s(
    tmp_path: Path,
) -> None:
    """The cap-breach guarantee must hold for the STATUS path too.

    Under ``video_compiler_service``'s per-source dedupe one fetched file is the
    shared ``-i`` input of N shots. A re-fetch whose presigned URL has expired must
    degrade that one shot, not overwrite a good asset with S3 error XML and render
    every shot from it.
    """
    dest = tmp_path / "clip.mp4"
    dest.write_bytes(b"good-earlier-bytes")
    transport = httpx.MockTransport(_status_handler(404))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await safe_get_to_path(client, "https://public.example.com/gone", dest)
    assert dest.read_bytes() == b"good-earlier-bytes"
    assert _partials_in(tmp_path) == []


@pytest.mark.asyncio
async def test_safe_get_to_path_still_accepts_a_success_status(tmp_path: Path) -> None:
    """CONTROL for the raiser: 2xx is untouched — body lands, ``dest`` returned.

    Without this, making the check unconditional (``raise`` on every status) would
    still pass every test above.
    """
    dest = tmp_path / "clip.mp4"
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        got = await safe_get_to_path(client, "https://public.example.com/ok", dest)
    assert got == dest
    assert dest.read_bytes() == b"image-bytes"


@pytest.mark.asyncio
async def test_safe_get_does_not_raise_on_an_error_status() -> None:
    """THE OPT-IN IS OPT-IN: the buffering path keeps returning the non-2xx response.

    ``safe_request``/``safe_get``/``safe_post``/``safe_request_sync`` hand the caller
    the final response on purpose — several fleet callers read a non-2xx body
    (error envelopes, 304/404 probes). If ``raise_for_status`` ever defaults to True
    in ``_safe_request_to_sink``, this is the test that fails, before ~15 consumer
    repos discover it at runtime.
    """
    transport = httpx.MockTransport(_status_handler(404))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await safe_get(client, "https://public.example.com/gone")
    assert resp.status_code == 404
    assert resp.content == _S3_ERROR_XML


def test_safe_get_sync_does_not_raise_on_an_error_status() -> None:
    """Same guarantee for the sync loop, which has its own copy of the terminal."""
    transport = httpx.MockTransport(_status_handler(500))
    with httpx.Client(transport=transport) as client:
        resp = safe_get_sync(client, "https://public.example.com/gone")
    assert resp.status_code == 500
    assert resp.content == _S3_ERROR_XML
