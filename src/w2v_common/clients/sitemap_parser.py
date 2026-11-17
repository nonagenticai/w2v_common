"""Fast path: parse robots.txt and sitemap.xml for URL discovery.

Every URL this module fetches is attacker-influenced: ``robots.txt`` names the
sitemaps, and a sitemap index names further sitemaps, recursively. All fetches
therefore go through :func:`w2v_common.net.safe_get`, this library's own
resolve-and-pin guard, rather than a raw ``httpx`` GET — an estate-internal or
metadata address named by remote content is refused before a connection is
opened.

Recursion into nested ``<sitemap><loc>`` entries is bounded on three axes
(:data:`MAX_SITEMAP_DEPTH`, :data:`MAX_SITEMAP_FETCHES`,
:data:`MAX_DISCOVERED_URLS`) so a hostile or merely broken sitemap index cannot
drive a ``RecursionError`` or turn one call into unbounded outbound fetch
amplification.
"""

from __future__ import annotations

import logging
import re
from xml.etree import (
    ElementTree,  # nosec B405 -- used only as type reference; parsing uses defusedxml
)

import defusedxml.ElementTree as DefusedET
import httpx

from w2v_common.clients._ssl import get_ssl_context
from w2v_common.net import ResponseTooLargeError, SSRFError, safe_get

logger = logging.getLogger(__name__)

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Nesting levels of ``<sitemapindex>`` followed before giving up. Real sitemap
# indexes are one level deep; the spec permits two. Five is generous and keeps
# the recursion far away from the interpreter's own limit.
MAX_SITEMAP_DEPTH = 5
# Hard ceiling on sitemap documents fetched per ``discover_urls()`` call. A
# single index may legitimately list hundreds of children; an index that lists
# tens of thousands is a fetch-amplification lever, not a site map.
MAX_SITEMAP_FETCHES = 100
# Hard ceiling on discovered URLs. Stops a 50 MB urlset from being accumulated
# in memory by a caller that only ever wanted a fast-path hint.
MAX_DISCOVERED_URLS = 50_000
# Sitemap documents are XML text. 32 MiB is far above any real sitemap (the
# spec caps an uncompressed sitemap at 50 MB / 50k URLs) and far below the
# 64 MiB default that ``safe_get`` applies to arbitrary artifacts.
MAX_SITEMAP_BYTES = 32 * 1024 * 1024

# Errors that mean "this URL is not fetchable, skip it" rather than "abort".
# SSRFError covers a blocked/unresolvable host; ValueError covers a malformed
# URL or unsupported scheme named inside the sitemap.
_SKIPPABLE_FETCH_ERRORS = (httpx.HTTPError, SSRFError, ResponseTooLargeError, ValueError)


class SitemapParser:
    """Parse robots.txt and sitemap.xml to discover URLs quickly."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def discover_urls(self) -> list[str]:
        """Try to discover URLs via sitemap. Returns empty list if no sitemap found."""
        sitemap_urls = await self._find_sitemap_urls()
        if not sitemap_urls:
            return []

        all_urls: list[str] = []
        # Budget shared across the whole call, not per sitemap: a hostile index
        # must not be able to multiply its allowance by listing more roots.
        budget = _FetchBudget()
        async with httpx.AsyncClient(timeout=15.0, verify=get_ssl_context()) as client:
            for sitemap_url in sitemap_urls:
                urls = await self._parse_sitemap(client, sitemap_url, depth=0, budget=budget)
                all_urls.extend(urls)
                if len(all_urls) >= MAX_DISCOVERED_URLS:
                    break

        deduped = list(dict.fromkeys(all_urls))  # deduplicate preserving order
        return deduped[:MAX_DISCOVERED_URLS]

    async def _find_sitemap_urls(self) -> list[str]:
        """Check robots.txt for Sitemap directives, fallback to /sitemap.xml."""
        async with httpx.AsyncClient(timeout=10.0, verify=get_ssl_context()) as client:
            try:
                resp = await safe_get(
                    client,
                    f"{self.base_url}/robots.txt",
                    max_fetch_bytes=MAX_SITEMAP_BYTES,
                )
                if resp.status_code == 200:
                    sitemaps = re.findall(
                        r"^Sitemap:\s*(.+)$", resp.text, re.MULTILINE | re.IGNORECASE
                    )
                    if sitemaps:
                        return [s.strip() for s in sitemaps]
            except _SKIPPABLE_FETCH_ERRORS as exc:
                logger.warning("Could not fetch robots.txt for %s: %s", self.base_url, exc)

        # Fallback: try /sitemap.xml directly
        return [f"{self.base_url}/sitemap.xml"]

    async def _parse_sitemap(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        depth: int,
        budget: _FetchBudget,
    ) -> list[str]:
        """Parse a sitemap XML file, handling sitemap indexes recursively.

        ``depth`` and ``budget`` bound the recursion: a sitemap index is remote
        content that names the next fetch, so without both a chain of indexes
        recurses until the interpreter's stack limit and a wide index turns one
        call into thousands of outbound requests.
        """
        if depth > MAX_SITEMAP_DEPTH:
            logger.debug("Sitemap depth cap (%d) reached at %s", MAX_SITEMAP_DEPTH, url)
            return []
        if not budget.take():
            logger.debug("Sitemap fetch cap (%d) reached at %s", MAX_SITEMAP_FETCHES, url)
            return []

        root = await self._fetch_sitemap_root(client, url)
        if root is None:
            return []

        # Sitemap index: <sitemapindex><sitemap><loc>...</loc></sitemap></sitemapindex>
        urls = await self._walk_index(client, root, depth=depth, budget=budget)
        # URL set: <urlset><url><loc>...</loc></url></urlset>
        urls.extend(_extract_locs(root, ".//sm:url/sm:loc"))
        return urls[:MAX_DISCOVERED_URLS]

    @staticmethod
    async def _fetch_sitemap_root(
        client: httpx.AsyncClient, url: str
    ) -> ElementTree.Element | None:
        """Fetch and parse one sitemap document, or None if it is unusable."""
        try:
            resp = await safe_get(client, url, max_fetch_bytes=MAX_SITEMAP_BYTES)
            if resp.status_code != 200:
                return None
        except _SKIPPABLE_FETCH_ERRORS as exc:
            logger.warning("Could not fetch sitemap %s: %s", url, exc)
            return None
        try:
            return DefusedET.fromstring(resp.text)
        except ElementTree.ParseError as exc:
            logger.warning("Could not parse sitemap XML: %s: %s", url, exc)
            return None

    async def _walk_index(
        self,
        client: httpx.AsyncClient,
        root: ElementTree.Element,
        *,
        depth: int,
        budget: _FetchBudget,
    ) -> list[str]:
        """Recurse into the nested ``<sitemap><loc>`` entries of a sitemap index."""
        urls: list[str] = []
        for loc in _extract_locs(root, ".//sm:sitemap/sm:loc"):
            if budget.exhausted or len(urls) >= MAX_DISCOVERED_URLS:
                break
            urls.extend(await self._parse_sitemap(client, loc, depth=depth + 1, budget=budget))
        return urls


def _extract_locs(root: ElementTree.Element, xpath: str) -> list[str]:
    """Return the stripped text of every ``<loc>`` matching ``xpath``."""
    return [el.text.strip() for el in root.findall(xpath, SITEMAP_NS) if el.text]


class _FetchBudget:
    """Shared, monotonically-consumed allowance of sitemap fetches per call."""

    def __init__(self, limit: int = MAX_SITEMAP_FETCHES) -> None:
        self._remaining = limit

    @property
    def exhausted(self) -> bool:
        return self._remaining <= 0

    def take(self) -> bool:
        """Consume one fetch; return False when the budget is spent."""
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True
