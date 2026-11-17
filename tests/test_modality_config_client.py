"""Tests for ModalityConfigClient (resolve → route, fail-open, 409, validation)."""

from __future__ import annotations

import httpx
import pytest

from w2v_common.clients import (
    ModalityBYORequiredError,
    ModalityConfigClient,
    ModalityRoute,
)


def _client_with(handler) -> ModalityConfigClient:
    c = ModalityConfigClient(base_url="http://wm.test")
    # Pre-seed the pooled client with a MockTransport so _get_client reuses it.
    c._client = httpx.AsyncClient(
        base_url="http://wm.test",
        transport=httpx.MockTransport(handler),
    )
    return c


@pytest.mark.asyncio
async def test_resolve_byo_route() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tenant-config/modality/tts/resolve"
        return httpx.Response(
            200,
            json={
                "configured": True,
                "provider": "elevenlabs",
                "model": "eleven_turbo_v2_5",
                "base_url": None,
                "api_key": "xi-decrypted",  # pragma: allowlist secret
                "cost_provider": "elevenlabs",
                "cost_source": "worker-management",
                "byo": True,
                "role": "__default__",
            },
        )

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.configured is True
    assert route.byo is True
    # Legacy resolve response (no owner_provided key) → tenant-BYO default.
    assert route.owner_provided is False
    assert route.provider == "elevenlabs"
    assert route.api_key == "xi-decrypted"  # pragma: allowlist secret
    assert route.cost_provider == "elevenlabs"
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_owner_provided_route() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "configured": True,
                "provider": "openai",
                "model": "tts-1",
                "api_key": "sk-decrypted",  # pragma: allowlist secret
                "cost_provider": "openai",
                "cost_source": "worker-management",
                "byo": True,
                "owner_provided": True,
                "role": "__default__",
            },
        )

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.byo is True
    assert route.owner_provided is True
    assert route.api_key == "sk-decrypted"  # pragma: allowlist secret
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_tier_forwarded() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["tier"] = request.url.params.get("tier", "")
        return httpx.Response(200, json={"configured": False, "byo": False})

    c = _client_with(handler)
    await c.resolve("asr", tier="high")
    assert seen["tier"] == "high"
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_product_forwarded() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["product"] = request.url.params.get("product", "")
        seen["tier"] = request.url.params.get("tier", "")
        return httpx.Response(200, json={"configured": False, "byo": False})

    c = _client_with(handler)
    await c.resolve("image", product="code_graph_helper")
    assert seen["product"] == "code_graph_helper"
    # product alone must not add a spurious tier param.
    assert seen["tier"] == ""
    # tier + product coexist.
    await c.resolve("image", tier="high", product="code_graph_helper")
    assert seen["product"] == "code_graph_helper"
    assert seen["tier"] == "high"
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_no_product_sends_no_param() -> None:
    seen: dict[str, bool] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_product"] = "product" in request.url.params
        return httpx.Response(200, json={"configured": False, "byo": False})

    c = _client_with(handler)
    await c.resolve("tts")
    assert seen["has_product"] is False
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_not_configured_is_platform_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"configured": False, "byo": False})

    c = _client_with(handler)
    route = await c.resolve("image")
    assert route.configured is False
    assert route.byo is False
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_409_raises_byorequired() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "must configure"})

    c = _client_with(handler)
    with pytest.raises(ModalityBYORequiredError):
        await c.resolve("tts")
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_5xx_fails_open() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route == ModalityRoute.platform_default()
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_transport_error_fails_open() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.configured is False
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_malformed_body_fails_open() -> None:
    """A 200 with an unparseable body must still fail open, not raise.

    Reproduces the defect: WM (or a proxy in front of it) can return 200 with
    a truncated/HTML body. ``resp.json()`` then raises ``json.JSONDecodeError``,
    which is NOT an ``httpx.HTTPError`` — a guard scoped to that type lets it
    escape, breaking the documented "any transport/5xx failure fails open"
    contract for exactly the class of failure a caller most needs it for.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route == ModalityRoute.platform_default()
    await c.aclose()


@pytest.mark.asyncio
async def test_resolve_unknown_modality_raises() -> None:
    c = ModalityConfigClient(base_url="http://wm.test")
    with pytest.raises(ValueError, match="unknown modality"):
        await c.resolve("bogus")


def test_vlm_is_a_valid_modality() -> None:
    # The modality-gateway VLM route resolves "vlm"; the client-side guard must
    # accept it or resolve() 500s (regression: it once rejected vlm as unknown).
    from w2v_common.clients.modality_config_client import VALID_MODALITIES

    assert "vlm" in VALID_MODALITIES


# ---------------------------------------------------------------------------
# resolution_failed: "COULD NOT RESOLVE" must be distinguishable from
# "resolved: not configured".
#
# resolve() fails OPEN to platform_default() on transport error / 5xx /
# unparseable body. platform_default() means "use the platform provider", which
# is harmless for an UNCONFIGURED tenant and NOT harmless for a byo_required
# one — the 409 policy gate is bypassed and the caller spends PLATFORM
# credentials for a tenant contractually required to bring its own key. Before
# this flag the return shape could not express the difference: byo=False read
# as an affirmative posture rather than a measurement GAP.
#
# These pin BOTH directions. Asserting only that the flag is True on failure
# would pass against a field hardcoded to True.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolution_failed_is_set_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.resolution_failed is True, (
        "a transport error fails open to the platform default; the route must say so"
    )
    await c.aclose()


@pytest.mark.asyncio
async def test_resolution_failed_is_set_on_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.resolution_failed is True, (
        "a 5xx fails open to the platform default; the route must say so"
    )
    await c.aclose()


@pytest.mark.asyncio
async def test_resolution_failed_is_set_on_unparseable_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.resolution_failed is True
    await c.aclose()


@pytest.mark.asyncio
async def test_resolution_failed_is_not_set_on_a_normal_resolve() -> None:
    """The negative half. A route that resolved is not a measurement gap."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"configured": True, "byo": True, "provider": "elevenlabs"})

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.resolution_failed is False
    await c.aclose()


@pytest.mark.asyncio
async def test_resolution_failed_is_not_set_on_a_resolved_unconfigured_tenant() -> None:
    """The discrimination that is the whole point: unconfigured != unresolved.

    Both routes carry configured=False/byo=False, but only one is an answer.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"configured": False, "byo": False})

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.configured is False
    assert route.resolution_failed is False, (
        "a tenant WM says is unconfigured is a resolved answer, not a resolution failure"
    )
    await c.aclose()


@pytest.mark.asyncio
async def test_resolution_failed_is_not_set_on_a_clean_409() -> None:
    """409 raises the policy error; it never produces a fail-open route at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "must configure"})

    c = _client_with(handler)
    with pytest.raises(ModalityBYORequiredError):
        await c.resolve("tts")
    await c.aclose()


def test_platform_default_defaults_to_not_failed() -> None:
    assert ModalityRoute.platform_default().resolution_failed is False
    assert ModalityRoute.platform_default(resolution_failed=True).resolution_failed is True


@pytest.mark.asyncio
async def test_resolution_failed_does_not_change_route_equality() -> None:
    """Byte-compatibility guard for ~19 consumers.

    ``resolution_failed`` is ``compare=False``: a consumer (or an existing test)
    comparing a fail-open route against ``platform_default()`` must keep the
    answer it had before this field existed. If someone drops ``compare=False``,
    this reds.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    c = _client_with(handler)
    route = await c.resolve("tts")
    assert route.resolution_failed is True
    assert route == ModalityRoute.platform_default()
    await c.aclose()


def test_resolution_failed_is_appended_last_with_a_default() -> None:
    """Positional construction of the pre-existing fields must be unaffected."""
    import dataclasses

    names = [f.name for f in dataclasses.fields(ModalityRoute)]
    assert names[-1] == "resolution_failed", (
        "the new field must be LAST or it shifts every positional constructor arg"
    )
    # Every field except `configured` keeps a default, so ModalityRoute(True)
    # still constructs.
    assert ModalityRoute(True).resolution_failed is False


@pytest.mark.asyncio
async def test_failopen_is_logged_at_error_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A silent credential-spend window must be loud.

    Pinned because the loudness is the whole point: at WARNING this sat in the
    same bucket as routine retries and nobody alerted on it.
    """
    import logging

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    c = _client_with(handler)
    with caplog.at_level(logging.WARNING, logger="w2v_common.clients.modality_config_client"):
        await c.resolve("tts")
    levels = {r.levelno for r in caplog.records}
    assert logging.ERROR in levels, (
        f"fail-open must log at ERROR; saw {[logging.getLevelName(x) for x in levels]}"
    )
    await c.aclose()


@pytest.mark.asyncio
async def test_failopen_increments_the_counter() -> None:
    """The metric is the alertable signal; a log line alone is not."""
    from w2v_common.observability.metrics import modality_resolve_failopen_total

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    before = modality_resolve_failopen_total.labels(
        modality="asr", reason="server_error"
    )._value.get()
    c = _client_with(handler)
    await c.resolve("asr")
    after = modality_resolve_failopen_total.labels(
        modality="asr", reason="server_error"
    )._value.get()
    assert after == before + 1, "the fail-open path must increment its counter"
    await c.aclose()
