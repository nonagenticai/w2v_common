"""WebscraperClient against a mocked webscraper_service (respx).

Payload shapes mirror webscraper_service ``src/schemas/scrape.py`` on origin/main
(``ScrapeResponse`` flat for /api/scrape/url, ``BulkScrapeResponse.results`` for
/api/scrape/bulk). Header assertions are on the request the mock received, with
the client's REAL event hooks installed.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from w2v_common.auth.bundle_tokens import (
    BundleTokenMinter,
    reset_bundle_token_minter,
    set_bundle_token_minter,
)
from w2v_common.auth.identity_context import (
    NoTenantForLicenceError,
    clear_identity_context,
    set_identity_context,
)
from w2v_common.auth.tenant_licence_lane import TenantLicenceLane
from w2v_common.clients.webscraper import (
    MAX_BULK_URLS,
    ImageRef,
    ScrapeResponse,
    WebscraperClient,
)
from w2v_common.tenant import MissingTenantError, TenantSource

BASE = "http://webscraper.test:9119"
SHARED = "shared-downstream-licence"  # pragma: allowlist secret
TENANT_A = "0199a1b2-0000-7000-8000-00000000000a"
TOKEN_URL = "http://kc.test/realms/secure-apps/protocol/openid-connect/token"
PNG = b"\x89PNG\r\n\x1a\nfake"
PNG_B64 = base64.b64encode(PNG).decode()
_ENV = (
    "BUNDLE_MINTER_CLIENT_ID",
    "BUNDLE_MINTER_CLIENT_SECRET",
    "BUNDLE_MINTER_TOKEN_URL",
    "BUNDLE_MINTER_BUNDLE",
    "WEBSCRAPER_AUTH_MODE",
    "W2V_WEBSCRAPER_URL",
    "W2V_LOCAL",
)


def _page(url: str = "https://example.com", **over: object) -> dict[str, object]:
    page: dict[str, object] = {
        "url": url,
        "final_url": url + "/",
        "html_content": "<h1>Example</h1>",
        "text_content": "Example Domain",
        "markdown_content": "# Example",
        "title": "Example Domain",
        "metadata": {"status_code": 200},
        "structured_data": {"json_ld": [{"@type": "WebSite"}]},
        "links": [
            {"text": "More", "url": "https://iana.org/domains", "score": 0.0},
            {"text": "no url"},
        ],
        "images": [
            {"src": "https://example.com/logo.png", "alt": "logo", "mime_type": "image/png"},
            {"alt": "missing src is skipped"},
        ],
        "status": "completed",
        "error": None,
        "scraper_used": "PlaywrightScraper",
        "processing_time_seconds": 1.2,
    }
    page.update(over)
    return page


def _body(route: respx.Route) -> dict[str, object]:
    return json.loads(route.calls.last.request.content)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SHARED)
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    reset_bundle_token_minter()
    clear_identity_context()
    yield monkeypatch
    clear_identity_context()
    reset_bundle_token_minter()


@pytest.fixture
def licences(env: pytest.MonkeyPatch) -> list[str]:
    apps: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "minter-sa", "expires_in": 300})
        app = request.url.params["app"]
        apps.append(app)
        return httpx.Response(
            201,
            json={
                "token": f"lic-{app}-{len(apps)}",
                "expires_at": datetime.fromtimestamp(4_000_000_000, tz=UTC).isoformat(),
                "tenant_id": request.url.path.split("/")[3],
                "app": app,
                "license_id": f"lid-{app}-{len(apps)}",
            },
        )

    set_bundle_token_minter(
        BundleTokenMinter(
            licenses_url="http://lic.test",
            token_url=TOKEN_URL,
            client_id="w2v-bundle-minter",
            client_secret="s",  # pragma: allowlist secret
            transport=httpx.MockTransport(handler),
        )
    )
    return apps


def _bind() -> None:
    set_identity_context(
        TENANT_A, "inbound-licence-id", "subject-1", tenant_source=TenantSource.LICENSE_TOKEN
    )


# -- scrape_url: flat payload ----------------------------------------------------


async def test_scrape_url_maps_flat_payload(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/api/scrape/url").mock(return_value=httpx.Response(200, json=_page()))
        result = await client.scrape_url("https://example.com")
    assert isinstance(result, ScrapeResponse)
    assert result.error is None
    assert result.url == "https://example.com"
    assert result.final_url == "https://example.com/"
    assert result.title == "Example Domain"
    assert result.html == "<h1>Example</h1>"
    assert result.text == "Example Domain"
    assert result.markdown == "# Example"
    assert result.structured_data == {"json_ld": [{"@type": "WebSite"}]}
    assert result.links == ["https://iana.org/domains"]
    assert result.images == [
        ImageRef(src="https://example.com/logo.png", alt="logo", mime_type="image/png")
    ]
    assert result.status_code == 200
    body = _body(route)
    assert body == {
        "url": "https://example.com",
        "scraper_type": "playwright",
        "javascript_enabled": True,
        "screenshot": False,
        "download_images": False,
        "timeout": 30.0,
    }
    await client.close()


async def test_scrape_url_static_page_uses_light_backend(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/api/scrape/url").mock(return_value=httpx.Response(200, json=_page()))
        await client.scrape_url("https://example.com", javascript_enabled=False)
    assert _body(route)["scraper_type"] == "scrapy"
    await client.close()


async def test_failed_status_maps_to_error(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    failed = _page(status="failed", error="Timeout 30000ms exceeded", text_content=None)
    with respx.mock(base_url=BASE) as mock:
        mock.post("/api/scrape/url").mock(return_value=httpx.Response(200, json=failed))
        result = await client.scrape_url("https://example.com")
    assert result.error == "Timeout 30000ms exceeded"
    assert result.text == ""
    await client.close()


async def test_non_completed_status_without_error_still_fails(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE) as mock:
        mock.post("/api/scrape/url").mock(
            return_value=httpx.Response(200, json=_page(status="running"))
        )
        result = await client.scrape_url("https://example.com")
    assert result.error == "scrape failed (status='running')"
    await client.close()


@pytest.mark.parametrize(
    "detail",
    [
        "URL resolves to a private address: http://10.0.0.1/",
        "credential cred-1 could not be resolved",
    ],
)
async def test_400_raises_with_body(env: pytest.MonkeyPatch, detail: str) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE) as mock:
        mock.post("/api/scrape/url").mock(return_value=httpx.Response(400, json={"detail": detail}))
        mock.post("/api/scrape/bulk").mock(
            return_value=httpx.Response(400, json={"detail": detail})
        )
        with pytest.raises(httpx.HTTPStatusError, match="webscraper returned 400") as exc:
            await client.scrape_url("http://10.0.0.1/")
        assert detail in str(exc.value)
        with pytest.raises(httpx.HTTPStatusError) as exc_bulk:
            await client.scrape_bulk(["http://10.0.0.1/"])
        assert detail in str(exc_bulk.value)
    await client.close()


# -- base64 decode -----------------------------------------------------------------


async def test_downloaded_image_and_screenshot_are_decoded(
    env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = WebscraperClient(BASE)
    page = _page(
        images=[
            {
                "src": "https://example.com/a.png",
                "mime_type": "image/png",
                "content_base64": PNG_B64,
            }
        ],
        screenshot_base64=PNG_B64,
    )
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/api/scrape/url").mock(return_value=httpx.Response(200, json=page))
        result = await client.scrape_url(
            "https://example.com", download_images=True, screenshot=True, screenshot_dir=tmp_path
        )
    assert _body(route)["download_images"] is True
    assert result.images[0].content == PNG
    assert result.screenshot_path is not None
    assert result.screenshot_path.read_bytes() == PNG
    await client.close()


def test_image_content_is_none_when_absent_or_invalid() -> None:
    assert ImageRef(src="x").content is None
    assert ImageRef(src="x", content_base64="not base64!!").content is None


# -- scrape_bulk -------------------------------------------------------------------


async def test_scrape_bulk_maps_results_with_html(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    urls = ["https://example.com", "https://example.org"]
    payload = {
        "total_urls": 2,
        "completed": 1,
        "failed": 1,
        "results": [
            _page(urls[0]),
            _page(urls[1], status="failed", error="403 Forbidden", html_content=None),
        ],
        "processing_time_seconds": 2.0,
    }
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/api/scrape/bulk").mock(return_value=httpx.Response(200, json=payload))
        results = await client.scrape_bulk(urls, javascript_enabled=False)
    assert [r.url for r in results] == urls
    assert results[0].html == "<h1>Example</h1>"
    assert results[0].error is None
    assert results[1].error == "403 Forbidden"
    body = _body(route)
    assert body["urls"] == urls
    assert body["follow_links"] is False
    assert body["max_depth"] == 1
    assert body["scraper_type"] == "scrapy"
    await client.close()


async def test_bulk_over_limit_raises_before_any_http(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    urls = [f"https://example.com/{i}" for i in range(MAX_BULK_URLS + 1)]
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        route = mock.post("/api/scrape/bulk")
        with pytest.raises(ValueError, match="at most 100"):
            await client.scrape_bulk(urls)
    assert not route.called
    await client.close()


async def test_bulk_at_limit_is_sent(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    urls = [f"https://example.com/{i}" for i in range(MAX_BULK_URLS)]
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/api/scrape/bulk").mock(
            return_value=httpx.Response(200, json={"results": [_page(u) for u in urls]})
        )
        results = await client.scrape_bulk(urls)
    assert len(results) == MAX_BULK_URLS
    assert route.called
    await client.close()


async def test_bulk_empty_list_sends_nothing(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        route = mock.post("/api/scrape/bulk")
        assert await client.scrape_bulk([]) == []
    assert not route.called


async def test_bulk_without_results_list_raises(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE) as mock:
        mock.post("/api/scrape/bulk").mock(return_value=httpx.Response(200, json={"oops": 1}))
        with pytest.raises(ValueError, match="results"):
            await client.scrape_bulk(["https://example.com"])
    await client.close()


# -- passthrough -------------------------------------------------------------------

_AUTH = {"type": "form", "credential_id": "cred-1", "login_url": "https://example.com/login"}


async def test_auth_config_and_settling_controls_pass_through(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE)
    with respx.mock(base_url=BASE) as mock:
        url_route = mock.post("/api/scrape/url").mock(
            return_value=httpx.Response(200, json=_page())
        )
        bulk_route = mock.post("/api/scrape/bulk").mock(
            return_value=httpx.Response(200, json={"results": [_page()]})
        )
        await client.scrape_url(
            "https://example.com",
            auth_config=_AUTH,
            delay=1.5,
            wait_until="networkidle",
            wait_for_selector="#app",
            timeout=45.0,
        )
        await client.scrape_bulk(["https://example.com"], auth_config=_AUTH, delay=0.5)
    url_body = _body(url_route)
    assert url_body["auth_config"] == _AUTH
    assert url_body["delay"] == 1.5
    assert url_body["wait_until"] == "networkidle"
    assert url_body["wait_for_selector"] == "#app"
    assert url_body["timeout"] == 45.0
    bulk_body = _body(bulk_route)
    assert bulk_body["auth_config"] == _AUTH
    assert bulk_body["delay"] == 0.5
    assert "wait_until" not in bulk_body
    assert "wait_for_selector" not in bulk_body
    await client.close()


# -- licence lane: a Bearer licence on EVERY request, in every mode ----------------


async def test_shared_mode_sends_shared_licence_from_env(env: pytest.MonkeyPatch) -> None:
    _bind()
    client = WebscraperClient(BASE)
    assert client.licence_lane is None
    with respx.mock(base_url=BASE) as mock:
        url_route = mock.post("/api/scrape/url").mock(
            return_value=httpx.Response(200, json=_page())
        )
        bulk_route = mock.post("/api/scrape/bulk").mock(
            return_value=httpx.Response(200, json={"results": [_page()]})
        )
        await client.scrape_url("https://example.com")
        await client.scrape_bulk(["https://example.com"])
    for route in (url_route, bulk_route):
        headers = route.calls.last.request.headers
        assert headers["X-Tenant-Id"] == f"Bearer {SHARED}"
        assert "Authorization" not in headers
    await client.close()


async def test_shared_mode_explicit_token_wins(env: pytest.MonkeyPatch) -> None:
    client = WebscraperClient(BASE, license_token="explicit-licence")  # pragma: allowlist secret
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/api/scrape/url").mock(return_value=httpx.Response(200, json=_page()))
        await client.scrape_url("https://example.com")
    assert route.calls.last.request.headers["X-Tenant-Id"] == "Bearer explicit-licence"
    await client.close()


def test_shared_mode_without_any_licence_refuses_construction(
    env: pytest.MonkeyPatch,
) -> None:
    env.delenv("DOWNSTREAM_LICENSE_TOKEN")
    with pytest.raises(MissingTenantError, match="no licence"):
        WebscraperClient(BASE)


async def test_tenant_licence_mode_sends_minted_w2v_licence(licences: list[str]) -> None:
    _bind()
    client = WebscraperClient(
        BASE,
        license_token=SHARED,
        licence_lane=TenantLicenceLane("webscraper", mode="tenant_licence"),
    )
    with respx.mock(base_url=BASE) as mock:
        url_route = mock.post("/api/scrape/url").mock(
            return_value=httpx.Response(200, json=_page())
        )
        bulk_route = mock.post("/api/scrape/bulk").mock(
            return_value=httpx.Response(200, json={"results": [_page()]})
        )
        await client.scrape_url("https://example.com")
        await client.scrape_bulk(["https://example.com"])
    assert licences == ["w2v"]  # minted once for app w2v, then cached
    for route in (url_route, bulk_route):
        headers = route.calls.last.request.headers
        assert headers["X-Tenant-Id"] == "Bearer lic-w2v-1"
        assert headers["X-License-Id"] == "lid-w2v-1"
        assert "X-Effective-Tenant-Id" not in headers
        assert SHARED not in str(headers)
    await client.close()


async def test_tenant_licence_mode_from_env(licences: list[str], env: pytest.MonkeyPatch) -> None:
    env.setenv("WEBSCRAPER_AUTH_MODE", "tenant_licence")
    client = WebscraperClient(BASE)
    assert client.licence_lane is not None
    assert client.licence_lane.app == "w2v"
    await client.close()


async def test_tenant_licence_mode_without_tenant_raises_and_sends_nothing(
    licences: list[str],
) -> None:
    client = WebscraperClient(
        BASE, licence_lane=TenantLicenceLane("webscraper", mode="tenant_licence")
    )
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        route = mock.post("/api/scrape/url")
        with pytest.raises(NoTenantForLicenceError):
            await client.scrape_url("https://example.com")
    assert not route.called
    await client.close()


def test_default_base_url_comes_from_discovery(env: pytest.MonkeyPatch) -> None:
    assert WebscraperClient().base_url == "http://webscraper.webscraper:9119"
    env.setenv("W2V_WEBSCRAPER_URL", "http://localhost:9119/")
    assert WebscraperClient().base_url == "http://localhost:9119"
