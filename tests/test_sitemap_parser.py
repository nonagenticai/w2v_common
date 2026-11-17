"""Tests for w2v_common.clients.sitemap_parser.SitemapParser."""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from w2v_common.clients.sitemap_parser import SitemapParser

_SITEMAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/</loc></url>
  <url><loc>https://example.com/about</loc></url>
  <url><loc>https://example.com/contact</loc></url>
</urlset>
"""

_ROBOTS_TXT_WITH_SITEMAP = """\
User-agent: *
Disallow: /admin/

Sitemap: https://example.com/sitemap.xml
"""

_ROBOTS_TXT_WITHOUT_SITEMAP = """\
User-agent: *
Disallow: /admin/
"""


class TestDiscoverUrlsFromSitemap:
    async def test_discover_urls_from_sitemap(self) -> None:
        """Discover URLs when robots.txt points to a sitemap."""

        def handler(request: httpx.Request) -> httpx.Response:
            url_path = request.url.path
            if url_path == "/robots.txt":
                return httpx.Response(
                    200,
                    text=_ROBOTS_TXT_WITH_SITEMAP,
                    headers={"content-type": "text/plain"},
                )
            if url_path == "/sitemap.xml":
                return httpx.Response(
                    200,
                    text=_SITEMAP_XML,
                    headers={"content-type": "application/xml"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        parser = SitemapParser(base_url="https://example.com")

        real_async_client = httpx.AsyncClient

        def _async_client_with_transport(**kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_async_client(**kwargs)

        with patch.object(httpx, "AsyncClient", side_effect=_async_client_with_transport):
            urls = await parser.discover_urls()

        assert len(urls) == 3
        assert "https://example.com/" in urls
        assert "https://example.com/about" in urls
        assert "https://example.com/contact" in urls


class TestDiscoverUrlsNoRobotsFallback:
    async def test_discover_urls_no_robots_fallback(self) -> None:
        """When robots.txt has no Sitemap directive, fall back to /sitemap.xml."""

        def handler(request: httpx.Request) -> httpx.Response:
            url_path = request.url.path
            if url_path == "/robots.txt":
                return httpx.Response(
                    200,
                    text=_ROBOTS_TXT_WITHOUT_SITEMAP,
                    headers={"content-type": "text/plain"},
                )
            if url_path == "/sitemap.xml":
                return httpx.Response(
                    200,
                    text=_SITEMAP_XML,
                    headers={"content-type": "application/xml"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        parser = SitemapParser(base_url="https://example.com")

        real_async_client = httpx.AsyncClient

        def _async_client_with_transport(**kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_async_client(**kwargs)

        with patch.object(httpx, "AsyncClient", side_effect=_async_client_with_transport):
            urls = await parser.discover_urls()

        assert len(urls) == 3
        assert "https://example.com/" in urls
        assert "https://example.com/about" in urls
        assert "https://example.com/contact" in urls


# --------------------------------------------------------------------------- #
# Defect 3 (0.57.0): unscreened fetches + unbounded sitemap-index recursion
# --------------------------------------------------------------------------- #

# Hermetic DNS for the guard tests. No real network is touched.
_GUARD_DNS: dict[str, str] = {
    "example.com": "93.184.216.34",  # public -> allowed
    "internal.example.com": "10.0.0.5",  # RFC1918 -> must be refused
}


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Resolve test hostnames deterministically; literal IPs resolve to themselves."""

    def fake_getaddrinfo(host: str, *_args: object, **_kwargs: object) -> list:
        ip = _GUARD_DNS.get(host, host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    yield


def _install_transport(handler: Any) -> Any:
    """Patch ``httpx.AsyncClient`` so the parser's client uses ``handler``."""
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(**kwargs)

    return patch.object(httpx, "AsyncClient", side_effect=_factory)


_INTERNAL_INDEX = """\
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>http://internal.example.com/sitemap.xml</loc></sitemap>
</sitemapindex>
"""


class TestSitemapFetchesAreScreened:
    async def test_nested_loc_pointing_at_an_internal_host_is_not_fetched(
        self, _stub_dns: None
    ) -> None:
        """A sitemap index is attacker-controlled input; a nested ``<loc>`` must
        not be able to steer the fetch at an estate-internal address."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(f"{request.headers.get('host', '')}{request.url.path}")
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text=_ROBOTS_TXT_WITHOUT_SITEMAP)
            if request.url.path == "/sitemap.xml" and "internal" not in request.headers.get(
                "host", ""
            ):
                return httpx.Response(200, text=_INTERNAL_INDEX)
            return httpx.Response(200, text=_SITEMAP_XML)

        parser = SitemapParser(base_url="https://example.com")
        with _install_transport(handler):
            await parser.discover_urls()

        internal_hits = [s for s in seen if "internal.example.com" in s]
        assert internal_hits == [], (
            f"parser fetched an estate-internal URL steered by sitemap content: {internal_hits}"
        )


class TestSitemapRecursionIsBounded:
    async def test_deep_sitemap_index_chain_is_depth_capped(self, _stub_dns: None) -> None:
        """A chain of nested sitemap indexes must not recurse without bound."""
        depth = 30
        fetches: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            fetches.append(path)
            if path == "/robots.txt":
                return httpx.Response(
                    200, text="User-agent: *\nSitemap: https://example.com/s0.xml"
                )
            if path.startswith("/s"):
                n = int(path.removeprefix("/s").removesuffix(".xml"))
                if n >= depth - 1:
                    return httpx.Response(200, text=_SITEMAP_XML)
                return httpx.Response(
                    200,
                    text=(
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                        f"<sitemap><loc>https://example.com/s{n + 1}.xml</loc></sitemap>"
                        "</sitemapindex>"
                    ),
                )
            return httpx.Response(404)

        parser = SitemapParser(base_url="https://example.com")
        with _install_transport(handler):
            await parser.discover_urls()

        sitemap_fetches = [f for f in fetches if f != "/robots.txt"]
        assert len(sitemap_fetches) <= 10, (
            f"recursed {len(sitemap_fetches)} levels deep into a nested sitemap index "
            "with no depth cap"
        )

    async def test_wide_sitemap_index_is_fetch_capped(self, _stub_dns: None) -> None:
        """A sitemap index listing thousands of children must not be fully walked."""
        children = 500
        fetches: list[str] = []

        index_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(
                f"<sitemap><loc>https://example.com/c{i}.xml</loc></sitemap>"
                for i in range(children)
            )
            + "</sitemapindex>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            fetches.append(path)
            if path == "/robots.txt":
                return httpx.Response(
                    200, text="User-agent: *\nSitemap: https://example.com/index.xml"
                )
            if path == "/index.xml":
                return httpx.Response(200, text=index_xml)
            return httpx.Response(200, text=_SITEMAP_XML)

        parser = SitemapParser(base_url="https://example.com")
        with _install_transport(handler):
            await parser.discover_urls()

        sitemap_fetches = [f for f in fetches if f != "/robots.txt"]
        assert len(sitemap_fetches) <= 150, (
            f"fetched {len(sitemap_fetches)} sitemaps from one index with no total cap "
            "— unbounded fetch amplification"
        )


# --------------------------------------------------------------------------- #
# 0.62.0: failures must be visible under the fleet's LOG_LEVEL=WARNING
# --------------------------------------------------------------------------- #


def _records_at(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == level and r.name == "w2v_common.clients.sitemap_parser"
    ]


class TestFailuresLogAtWarning:
    """Prod runs ``LOG_LEVEL=WARNING``; a DEBUG failure line is invisible there."""

    async def test_robots_transport_failure_logs_at_warning(
        self, caplog: pytest.LogCaptureFixture, _stub_dns: None
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, text=_SITEMAP_XML)

        parser = SitemapParser(base_url="https://example.com")
        with caplog.at_level(logging.WARNING), _install_transport(handler):
            urls = await parser.discover_urls()

        assert urls  # fell back to /sitemap.xml
        warnings = _records_at(caplog, logging.WARNING)
        assert any("Could not fetch robots.txt" in r.getMessage() for r in warnings), (
            f"expected a WARNING for the robots.txt transport failure, got {caplog.records}"
        )

    async def test_sitemap_fetch_failure_logs_at_warning(
        self, caplog: pytest.LogCaptureFixture, _stub_dns: None
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text=_ROBOTS_TXT_WITHOUT_SITEMAP)
            raise httpx.ConnectError("connection refused", request=request)

        parser = SitemapParser(base_url="https://example.com")
        with caplog.at_level(logging.WARNING), _install_transport(handler):
            urls = await parser.discover_urls()

        assert urls == []
        warnings = _records_at(caplog, logging.WARNING)
        assert any("Could not fetch sitemap" in r.getMessage() for r in warnings), (
            f"expected a WARNING for the sitemap fetch failure, got {caplog.records}"
        )

    async def test_malformed_sitemap_xml_logs_at_warning_with_reason(
        self, caplog: pytest.LogCaptureFixture, _stub_dns: None
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text=_ROBOTS_TXT_WITHOUT_SITEMAP)
            return httpx.Response(200, text="<urlset><url>", headers={"content-type": "text/xml"})

        parser = SitemapParser(base_url="https://example.com")
        with caplog.at_level(logging.WARNING), _install_transport(handler):
            urls = await parser.discover_urls()

        assert urls == []
        warnings = _records_at(caplog, logging.WARNING)
        matching = [r for r in warnings if "Could not parse sitemap XML" in r.getMessage()]
        assert matching, f"expected a WARNING for the XML parse failure, got {caplog.records}"
        # The reason must be in the message, not just the URL.
        assert "https://example.com/sitemap.xml:" in matching[0].getMessage()
        assert len(matching[0].getMessage().split("sitemap.xml:")[1].strip()) > 0
