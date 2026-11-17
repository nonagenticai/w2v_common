"""URL normalization, deduplication, and link graph construction."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse, urlunparse


class URLGraph:
    """Build and query a directed link graph of discovered URLs."""

    def __init__(self, domain: str) -> None:
        self.domain = domain.lower()
        self._edges: dict[str, set[str]] = {}
        self._seen: set[str] = set()

    @staticmethod
    def _coerce_url(value: object) -> str | None:
        """Convert bytes/string-like URL values to a clean text URL."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore").strip()
            return value or None
        return None

    def normalize(self, url: object, base_url: object | None = None) -> str | None:
        """Normalize a URL. Returns None if it's external or should be excluded."""
        url_str = self._coerce_url(url)
        if not url_str:
            return None

        base_url_str = self._coerce_url(base_url) if base_url is not None else None
        if base_url_str:
            url_str = urljoin(base_url_str, url_str)

        parsed = urlparse(url_str)
        if parsed.scheme not in ("http", "https", ""):
            return None
        if parsed.hostname and parsed.hostname != self.domain:
            return None

        # Strip fragment, normalize trailing slash
        path = parsed.path.rstrip("/") or "/"
        return urlunparse(
            (
                parsed.scheme or "https",
                parsed.hostname or self.domain,
                path,
                parsed.params,
                parsed.query,
                "",  # no fragment
            )
        )

    def add_page(self, url: str, links: object) -> list[str]:
        """Add a page and its outgoing links. Returns new unseen internal URLs."""
        url = self.normalize(url) or url
        self._seen.add(url)

        new_urls: list[str] = []
        normalized_links: set[str] = set()
        link_values = links if isinstance(links, (list, tuple, set)) else []
        for link in link_values:
            norm = self.normalize(link, base_url=url)
            if norm:
                normalized_links.add(norm)
                if norm not in self._seen:
                    new_urls.append(norm)
                    self._seen.add(norm)

        self._edges[url] = normalized_links
        return new_urls

    def should_exclude(self, url: str, exclude_patterns: list[str]) -> bool:
        """Check if a URL matches any exclusion pattern."""
        return any(re.search(pattern, url) for pattern in exclude_patterns)

    def to_dict(self) -> dict[str, list[str]]:
        """Export the link graph as a dict."""
        return {url: sorted(links) for url, links in self._edges.items()}

    @property
    def seen_urls(self) -> set[str]:
        """Return a copy of all URLs discovered so far."""
        return self._seen.copy()
