"""Direct HTTP client for ``webscraper_service`` (replaces the KAG scrape hop).

KAG was only ever a gateway in front of webscraper: its ``/ingest/scrape/*``
routes forwarded to webscraper's ``POST /api/scrape/url`` and
``POST /api/scrape/bulk`` (or to the retired ingestion service). This client
calls webscraper directly. Wire contract: ``webscraper_service``
``src/schemas/scrape.py`` (``ScrapeRequest`` / ``BulkScrapeRequest`` /
``ScrapeResponse`` / ``BulkScrapeResponse``) on origin/main.

Auth: webscraper accepts any active licence, but the licence MUST travel as
``X-Tenant-Id: Bearer <licence>``; a request without one is billed to a default
tenant. So, unlike the deleted KAG scrape client, there is no plain-``tenant_id`` fallback:
every request carries a Bearer licence, in every ``WEBSCRAPER_AUTH_MODE``.

* ``shared`` (default): the shared licence (``license_token`` argument, else the
  lane's ``DOWNSTREAM_LICENSE_TOKEN``). Construction RAISES when neither is set.
* ``tenant_licence``: the bound tenant's minted ``w2v`` licence per request
  (``TenantLicenceLane.event_hooks``); no tenant bound RAISES before the send.

Errors: a non-2xx answer (SSRF guard 400, credential-resolution 400/403,
validation 422, 5xx) RAISES ``httpx.HTTPStatusError`` with the redacted body. A
page webscraper could not scrape comes back as HTTP 200 with
``status: "failed"``; that is mapped to ``ScrapeResponse.error``.
"""

from __future__ import annotations

import base64
import binascii
import logging
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, NotRequired, TypedDict, cast

import httpx

from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.auth.tenant_licence_lane import TenantLicenceLane
from w2v_common.clients.base import raise_for_status_with_body
from w2v_common.discovery import service_url

from ..tenant import MissingTenantError

logger = logging.getLogger(__name__)

#: webscraper ``BulkScrapeRequest.urls`` is ``max_length=100``. Enforced here so
#: an oversized batch fails at the call site, before any HTTP request.
MAX_BULK_URLS: Final = 100

#: Slack added on top of the scrape deadline for the HTTP read timeout, so the
#: service answers (with ``status: "failed"``) before the client gives up.
_HTTP_TIMEOUT_SLACK_SECONDS: Final = 30.0

_HOP: Final = "webscraper"

WaitUntil = Literal["load", "domcontentloaded", "networkidle", "commit"]


@dataclass
class ImageRef:
    """One image webscraper discovered on a page (``ImageInfo`` subset).

    ``content_base64`` is only populated when the request set
    ``download_images=True``.
    """

    src: str
    alt: str = ""
    mime_type: str | None = None
    content_base64: str | None = None

    @property
    def content(self) -> bytes | None:
        """Decoded image bytes, or ``None`` when not downloaded / not valid base64."""
        if not self.content_base64:
            return None
        try:
            return base64.b64decode(self.content_base64, validate=True)
        except binascii.Error, ValueError:
            return None


@dataclass
class ScrapeResponse:
    """One scraped page, normalised from webscraper's ``ScrapeResponse``.

    Field names are kept from the KAG-era model (``html``, ``text``, ``links`` as
    URL strings, ``screenshot_path``); ``images`` is now ``list[ImageRef]`` —
    callers that used the old ``list[str]`` read ``.src``.
    """

    url: str
    title: str = ""
    html: str = ""
    text: str = ""
    markdown: str = ""
    final_url: str | None = None
    links: list[str] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    structured_data: dict[str, object] = field(default_factory=dict)
    screenshot_path: Path | None = None
    status_code: int = 200
    error: str | None = None


class _ScrapeOptions(TypedDict, total=False):
    """Request fields shared by ``ScrapeRequest`` and ``BulkScrapeRequest``."""

    scraper_type: str
    javascript_enabled: bool
    screenshot: bool
    download_images: bool
    timeout: float
    auth_config: dict[str, object]
    delay: NotRequired[float]
    wait_until: NotRequired[str]
    wait_for_selector: NotRequired[str]


class _UrlRequest(_ScrapeOptions, total=False):
    url: str


class _BulkRequest(_ScrapeOptions, total=False):
    urls: list[str]
    follow_links: bool
    max_depth: int


class WebscraperClient:
    """Async client for webscraper ``POST /api/scrape/url`` and ``/api/scrape/bulk``.

    ``base_url`` defaults to ``service_url("webscraper")`` (override with
    ``W2V_WEBSCRAPER_URL``). ``licence_lane`` defaults to
    ``TenantLicenceLane("webscraper")`` (env ``WEBSCRAPER_AUTH_MODE``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        ssl_context: ssl.SSLContext | bool = True,
        *,
        licence_lane: TenantLicenceLane | None = None,
    ) -> None:
        self.base_url = (base_url or service_url(_HOP)).rstrip("/")
        self.ssl_context = ssl_context
        lane = licence_lane if licence_lane is not None else TenantLicenceLane(_HOP)
        #: Set only in ``tenant_licence`` mode; then nothing credential-shaped is
        #: baked into the pooled client and the lane's hooks mint per request.
        self.licence_lane: TenantLicenceLane | None = lane if lane.tenant_licence else None
        if self.licence_lane is not None:
            self.license_token = ""  # nosec B105 — no SA licence: the lane supplies one per call
        else:
            self.license_token = license_token or lane.shared_token or ""
            if not self.license_token:
                raise MissingTenantError(
                    "WebscraperClient: no licence — pass license_token, set "
                    "DOWNSTREAM_LICENSE_TOKEN, or WEBSCRAPER_AUTH_MODE=tenant_licence. "
                    "webscraper bills a request without X-Tenant-Id: Bearer to a "
                    "default tenant, so an unlicensed client is refused here"
                )
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the pooled httpx client, creating it lazily on first use."""
        if self._client is None or self._client.is_closed:
            if self.licence_lane is not None:
                self._client = httpx.AsyncClient(
                    base_url=f"{self.base_url}/",
                    timeout=60.0,
                    verify=self.ssl_context,
                    event_hooks=self.licence_lane.event_hooks(),
                )
            else:
                # Opaque licences go in X-Tenant-Id: Bearer; Authorization is
                # reserved for Keycloak JWTs. identity_event_hook preserves a
                # pre-set Bearer X-Tenant-Id and stamps licence/subject ids.
                self._client = httpx.AsyncClient(
                    base_url=f"{self.base_url}/",
                    headers={"X-Tenant-Id": f"Bearer {self.license_token}"},
                    timeout=60.0,
                    verify=self.ssl_context,
                    event_hooks={"request": [identity_event_hook]},
                )
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client, releasing connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # -- request building ------------------------------------------------------

    @staticmethod
    def _scraper_type(*, javascript_enabled: bool, screenshot: bool) -> str:
        """Playwright when JS or a screenshot is needed, else the light scrapy backend."""
        return "playwright" if javascript_enabled or screenshot else "scrapy"

    @classmethod
    def _options(
        cls,
        *,
        javascript_enabled: bool,
        screenshot: bool,
        download_images: bool,
        timeout: float,
        auth_config: Mapping[str, object] | None,
        delay: float | None,
        wait_until: WaitUntil | None,
        wait_for_selector: str | None,
    ) -> _ScrapeOptions:
        opts: _ScrapeOptions = {
            "scraper_type": cls._scraper_type(
                javascript_enabled=javascript_enabled, screenshot=screenshot
            ),
            "javascript_enabled": javascript_enabled,
            "screenshot": screenshot,
            "download_images": download_images,
            "timeout": timeout,
        }
        if auth_config:
            opts["auth_config"] = dict(auth_config)
        if delay is not None:
            opts["delay"] = delay
        if wait_until is not None:
            opts["wait_until"] = wait_until
        if wait_for_selector is not None:
            opts["wait_for_selector"] = wait_for_selector
        return opts

    # -- response mapping ------------------------------------------------------

    @staticmethod
    def _decode_screenshot(
        *,
        url: str,
        payload: dict[str, object],
        screenshot_enabled: bool,
        screenshot_dir: Path | None,
    ) -> Path | None:
        """Persist an inline base64 screenshot when the API returns one."""
        if not screenshot_enabled or screenshot_dir is None:
            return None
        screenshot_base64 = payload.get("screenshot_base64")
        if not isinstance(screenshot_base64, str) or not screenshot_base64:
            return None
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        fname = url.replace("https://", "").replace("http://", "").replace("/", "_")[:80] + ".png"
        screenshot_path = screenshot_dir / fname
        screenshot_path.write_bytes(base64.b64decode(screenshot_base64))
        return screenshot_path

    @staticmethod
    def _payload_error(payload: dict[str, object]) -> str | None:
        """Failure reason for a 200-body page whose ``status`` is not ``completed``.

        webscraper answers a page it could not scrape (timeout, 403, bad
        credential) with HTTP 200, ``status: "failed"`` and ``error``. Without
        this mapping that reads as an empty success and feeds blank content on.
        """
        status = str(payload.get("status") or "").lower()
        reason = payload.get("error")
        if status != "completed" or reason:
            return str(reason or f"scrape failed (status={status or 'unknown'!r})")
        return None

    @staticmethod
    def _dicts(raw: object) -> list[dict[str, object]]:
        """The dict entries of a JSON list (anything else is dropped)."""
        if not isinstance(raw, list):
            return []
        return [cast("dict[str, object]", item) for item in raw if isinstance(item, dict)]

    @classmethod
    def _links(cls, raw: object) -> list[str]:
        """``LinkInfo.url`` of every discovered link."""
        return [
            href for item in cls._dicts(raw) if isinstance(href := item.get("url"), str) and href
        ]

    @staticmethod
    def _opt_str(value: object) -> str | None:
        return value if isinstance(value, str) and value else None

    @classmethod
    def _images(cls, raw: object) -> list[ImageRef]:
        """``ImageInfo`` -> ``ImageRef``; entries without a ``src`` are dropped."""
        out: list[ImageRef] = []
        for item in cls._dicts(raw):
            src = cls._opt_str(item.get("src"))
            if src is None:
                continue
            out.append(
                ImageRef(
                    src=src,
                    alt=str(item.get("alt") or ""),
                    mime_type=cls._opt_str(item.get("mime_type")),
                    content_base64=cls._opt_str(item.get("content_base64")),
                )
            )
        return out

    def _scrape_response_from_payload(
        self,
        *,
        fallback_url: str,
        payload: dict[str, object],
        status_code: int,
        screenshot_enabled: bool,
        screenshot_dir: Path | None,
    ) -> ScrapeResponse:
        """Normalise one webscraper ``ScrapeResponse`` into the local model."""
        page_url = str(payload.get("url") or fallback_url)
        structured = payload.get("structured_data")
        return ScrapeResponse(
            url=page_url,
            title=str(payload.get("title") or ""),
            html=str(payload.get("html_content") or ""),
            text=str(payload.get("text_content") or ""),
            markdown=str(payload.get("markdown_content") or ""),
            final_url=self._opt_str(payload.get("final_url")),
            links=self._links(payload.get("links")),
            images=self._images(payload.get("images")),
            structured_data=cast("dict[str, object]", structured)
            if isinstance(structured, dict)
            else {},
            screenshot_path=self._decode_screenshot(
                url=page_url,
                payload=payload,
                screenshot_enabled=screenshot_enabled,
                screenshot_dir=screenshot_dir,
            ),
            status_code=status_code,
            error=self._payload_error(payload),
        )

    # -- public API ------------------------------------------------------------

    async def scrape_url(
        self,
        url: str,
        *,
        download_images: bool = False,
        auth_config: Mapping[str, object] | None = None,
        timeout: float = 30.0,  # noqa: ASYNC109 — why: webscraper per-page deadline (1-300 s), sent in the body
        javascript_enabled: bool = True,
        screenshot: bool = False,
        delay: float | None = None,
        wait_until: WaitUntil | None = None,
        wait_for_selector: str | None = None,
        screenshot_dir: Path | None = None,
    ) -> ScrapeResponse:
        """Scrape one URL via ``POST /api/scrape/url``.

        ``auth_config`` (at least ``credential_id``) makes webscraper resolve the
        credential through the secrets service under the caller's licence and
        log in before scraping. Raises ``httpx.HTTPStatusError`` on non-2xx.
        """
        client = await self._get_client()
        body: _UrlRequest = {
            "url": url,
            **self._options(
                javascript_enabled=javascript_enabled,
                screenshot=screenshot,
                download_images=download_images,
                timeout=timeout,
                auth_config=auth_config,
                delay=delay,
                wait_until=wait_until,
                wait_for_selector=wait_for_selector,
            ),
        }
        resp = await client.post(
            "api/scrape/url",
            json=body,
            timeout=timeout + (delay or 0.0) + _HTTP_TIMEOUT_SLACK_SECONDS,
        )
        raise_for_status_with_body(resp, _HOP)
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"webscraper /api/scrape/url returned {type(data).__name__}")
        return self._scrape_response_from_payload(
            fallback_url=url,
            payload=data,
            status_code=resp.status_code,
            screenshot_enabled=screenshot,
            screenshot_dir=screenshot_dir,
        )

    async def scrape_bulk(
        self,
        urls: list[str],
        *,
        download_images: bool = False,
        auth_config: Mapping[str, object] | None = None,
        timeout: float = 30.0,  # noqa: ASYNC109 — why: webscraper per-URL deadline (1-300 s), sent in the body
        javascript_enabled: bool = True,
        screenshot: bool = False,
        delay: float | None = None,
        wait_until: WaitUntil | None = None,
        wait_for_selector: str | None = None,
        screenshot_dir: Path | None = None,
    ) -> list[ScrapeResponse]:
        """Scrape up to 100 URLs synchronously via ``POST /api/scrape/bulk``.

        Returns one ``ScrapeResponse`` per result, ``html`` included. With
        ``auth_config`` webscraper logs in once and reuses the session for all
        URLs. Raises ``ValueError`` for more than 100 URLs (before any HTTP
        call) and ``httpx.HTTPStatusError`` on non-2xx.
        """
        if len(urls) > MAX_BULK_URLS:
            raise ValueError(
                f"webscraper bulk scrape accepts at most {MAX_BULK_URLS} URLs, got {len(urls)}"
            )
        if not urls:
            return []
        client = await self._get_client()
        body: _BulkRequest = {
            "urls": list(urls),
            "follow_links": False,
            "max_depth": 1,
            **self._options(
                javascript_enabled=javascript_enabled,
                screenshot=screenshot,
                download_images=download_images,
                timeout=timeout,
                auth_config=auth_config,
                delay=delay,
                wait_until=wait_until,
                wait_for_selector=wait_for_selector,
            ),
        }
        # Worst case the service scrapes sequentially; bound the read by that.
        http_timeout = (timeout + (delay or 0.0)) * len(urls) + _HTTP_TIMEOUT_SLACK_SECONDS
        resp = await client.post("api/scrape/bulk", json=body, timeout=http_timeout)
        raise_for_status_with_body(resp, _HOP)
        data = resp.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ValueError("webscraper /api/scrape/bulk response has no 'results' list")
        return [
            self._scrape_response_from_payload(
                fallback_url=str(item.get("url") or ""),
                payload=item,
                status_code=resp.status_code,
                screenshot_enabled=screenshot,
                screenshot_dir=screenshot_dir,
            )
            for item in results
            if isinstance(item, dict)
        ]


__all__ = ["MAX_BULK_URLS", "ImageRef", "ScrapeResponse", "WebscraperClient"]
