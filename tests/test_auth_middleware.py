"""Tests for w2v_common.auth.middleware."""

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from w2v_common.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    AuthMiddleware,
    _apply_effective_tenant,
    _get_license_cache,
    _introspect_license,
)
from w2v_common.tenant import TenantError


@pytest.fixture(autouse=True)
def _clear_license_cache():
    """Clear the license cache (positive, negative, single-flight locks)."""
    import w2v_common.auth.middleware as mod

    mod._license_cache = None
    mod._license_neg_cache = None
    mod._license_stale_cache = None
    mod._license_token_not_found_since = None
    mod._license_tnf_probe = None
    mod._introspect_locks.clear()
    yield
    mod._license_cache = None
    mod._license_neg_cache = None
    mod._license_stale_cache = None
    mod._license_token_not_found_since = None
    mod._license_tnf_probe = None
    mod._introspect_locks.clear()


def _mock_httpx_client(*, post_return=None, post_side_effect=None):
    """Create a mock httpx.AsyncClient that works as async context manager.

    httpx.AsyncClient is used as:
        async with httpx.AsyncClient(...) as client:
            resp = await client.post(...)

    So the constructor returns the object itself, and __aenter__ returns self.
    """
    mock_client = MagicMock()
    if post_side_effect:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    else:
        mock_client.post = AsyncMock(return_value=post_return)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with AuthMiddleware for testing."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/data")
    async def data(request: Request):
        user = request.state.user
        return {
            "user_sub": user.sub,
            "tenant_id": request.state.tenant_id,
            # Surfaced for the license-token-preservation regression test —
            # downstream code (notably the orchestrator's crawl stage) needs
            # the *raw* opaque token to forward to webscraper, since the
            # service-account fallback resolves to a different tenant inside
            # secrets-service and 404s the BOLA check.
            "license_token": getattr(request.state, "license_token", "<UNSET>"),
        }

    @app.get("/")
    async def root():
        return {"msg": "root"}

    @app.get("/docs")
    async def docs():
        return {"msg": "docs"}

    @app.get("/api/health")
    async def api_health():
        return {"status": "ok"}

    @app.get("/api/health/ready")
    async def api_health_ready():
        return {"status": "ready"}

    @app.get("/health/ready")
    async def health_ready():
        return {"status": "ready"}

    @app.get("/livez")
    async def livez():
        return {"status": "ok"}

    @app.get("/live")
    async def live():
        return {"status": "ok"}

    return app


def _reload_with_trusted_cidrs(monkeypatch: pytest.MonkeyPatch, value: str | None):
    """Reload the middleware module with TRUSTED_CIDRS set (or unset) at import."""
    import importlib

    import w2v_common.auth.middleware as mod

    if value is None:
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
    else:
        monkeypatch.setenv("TRUSTED_CIDRS", value)
    return importlib.reload(mod)


@pytest.fixture
def restore_middleware_module():
    """Reload the middleware once more after the test, from a clean env."""
    import importlib
    import os

    import w2v_common.auth.middleware as mod

    yield
    saved = os.environ.pop("TRUSTED_CIDRS", None)
    try:
        importlib.reload(mod)
    finally:
        if saved is not None:
            os.environ["TRUSTED_CIDRS"] = saved


@pytest.mark.usefixtures("restore_middleware_module")
class TestIsTrustedNetwork:
    """Tests for _is_trusted_network.

    TRUSTED_CIDRS defaults to EMPTY (cell plan B22): no address — not even
    loopback, not the k3s pod CIDR, not a compose bridge — is trusted unless
    the operator names the network.
    """

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "::1",
            "10.42.5.10",  # k3s pod CIDR
            "10.43.0.1",  # k3s service CIDR
            "172.18.0.5",  # docker-compose bridge
            "10.0.0.1",  # customer LAN
            "192.168.1.241",  # SSR-proxy bypass vector (2026-04-18)
            "8.8.8.8",
        ],
    )
    def test_nothing_trusted_by_default(self, monkeypatch: pytest.MonkeyPatch, host: str) -> None:
        mod = _reload_with_trusted_cidrs(monkeypatch, None)
        assert mod.TRUSTED_NETWORKS == []
        assert mod._is_trusted_network(host) is False

    @pytest.mark.parametrize("raw", ["", " ", ",", " , ,"])
    def test_empty_or_blank_env_trusts_nothing(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        mod = _reload_with_trusted_cidrs(monkeypatch, raw)
        assert mod.TRUSTED_NETWORKS == []
        assert mod._is_trusted_network("127.0.0.1") is False

    def test_explicit_env_trusts_exactly_the_named_networks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _reload_with_trusted_cidrs(monkeypatch, " 10.42.0.0/16 , ::1/128 ,")
        assert mod._is_trusted_network("10.42.5.10") is True
        assert mod._is_trusted_network("::1") is True
        # Neighbours of the named networks stay untrusted.
        assert mod._is_trusted_network("10.43.0.1") is False
        assert mod._is_trusted_network("127.0.0.1") is False
        assert mod._is_trusted_network("172.18.0.5") is False

    @pytest.mark.parametrize("host", [None, "", "not-an-ip"])
    def test_unparseable_host_never_trusted(
        self, monkeypatch: pytest.MonkeyPatch, host: str | None
    ) -> None:
        mod = _reload_with_trusted_cidrs(monkeypatch, "0.0.0.0/0,::/0")
        assert mod._is_trusted_network(host) is False


@pytest.mark.usefixtures("restore_middleware_module")
class TestTrustLaneRequiresExplicitCidrs:
    """AUTH_TRUST_INCLUSTER_IDENTITY alone must not trust a compose/LAN peer."""

    TENANT = "5b3f6a2e-8c1d-4e7a-9f20-3c4d5e6f7a81"

    def _app(self, mod):
        app = FastAPI()
        app.add_middleware(mod.AuthMiddleware)

        @app.get("/api/whoami")
        async def whoami():
            return {"ok": True}

        return app

    async def _get(self, app, peer: str):
        transport = ASGITransport(app=app, client=(peer, 40000))
        async with AsyncClient(transport=transport, base_url="http://svc") as client:
            return await client.get("/api/whoami", headers={"X-Tenant-Id": self.TENANT})

    async def test_flag_on_default_cidrs_refuses_bridge_peer_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AUTH_TRUST_INCLUSTER_IDENTITY", "true")
        mod = _reload_with_trusted_cidrs(monkeypatch, None)
        with caplog.at_level("WARNING", logger=mod.logger.name):
            resp = await self._get(self._app(mod), "172.18.0.5")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert "TRUSTED_CIDRS is empty" in caplog.text

    async def test_flag_on_explicit_cidrs_admits_named_peer_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AUTH_TRUST_INCLUSTER_IDENTITY", "true")
        mod = _reload_with_trusted_cidrs(monkeypatch, "10.42.0.0/16")
        app = self._app(mod)
        named = await self._get(app, "10.42.5.10")
        other = await self._get(app, "172.18.0.5")
        assert named.status_code == 200, named.text
        assert other.status_code == 401
        assert other.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"


class TestGetLicenseCache:
    """Tests for _get_license_cache."""

    def test_creates_ttl_cache_with_default_config(self):
        from cachetools import TTLCache

        cache = _get_license_cache()
        assert isinstance(cache, TTLCache)
        # Default maxsize from env or hardcoded default
        assert cache.maxsize == 1000

    def test_returns_same_cache_instance(self):
        cache1 = _get_license_cache()
        cache2 = _get_license_cache()
        assert cache1 is cache2


class TestIntrospectLicense:
    """Tests for _introspect_license."""

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_successful_introspection(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000001",
        }

        mock_instance = _mock_httpx_client(post_return=mock_resp)
        mock_client_cls.return_value = mock_instance

        result = await _introspect_license("valid-token")
        assert result == {"active": True, "tenant_id": "0195f48b-0000-7000-8000-000000000001"}
        # Verify the token was sent as JSON payload to the introspection endpoint
        from w2v_common.auth.middleware import LICENSE_INTROSPECT_URL

        mock_instance.post.assert_called_once_with(
            LICENSE_INTROSPECT_URL, json={"token": "valid-token"}
        )

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_inactive_token_returned_but_not_cached(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False}

        mock_client_cls.return_value = _mock_httpx_client(post_return=mock_resp)

        result = await _introspect_license("inactive-token")
        assert result == {"active": False}
        # Inactive tokens should NOT be cached (only active ones are)
        cache = _get_license_cache()
        assert len(cache) == 0

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_non_200_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_client_cls.return_value = _mock_httpx_client(post_return=mock_resp)

        result = await _introspect_license("some-token")
        assert result is None

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_http_error(self, mock_client_cls):
        import httpx

        mock_client_cls.return_value = _mock_httpx_client(
            post_side_effect=httpx.HTTPError("connection failed")
        )

        result = await _introspect_license("some-token")
        assert result is None

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_timeout_error(self, mock_client_cls):
        mock_client_cls.return_value = _mock_httpx_client(
            post_side_effect=TimeoutError("timed out")
        )

        result = await _introspect_license("some-token")
        assert result is None

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_cache_hit_avoids_second_http_call(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "tenant_id": "cached"}

        mock_instance = _mock_httpx_client(post_return=mock_resp)
        mock_client_cls.return_value = mock_instance

        # First call populates cache
        result1 = await _introspect_license("cached-token")
        assert result1 is not None
        assert result1["active"] is True
        assert result1["tenant_id"] == "cached"

        # Second call should hit cache -- no additional HTTP call
        result2 = await _introspect_license("cached-token")
        assert result2 is not None
        assert result2["active"] is True
        assert result2["tenant_id"] == "cached"

        # The HTTP client should have only been constructed once (for the first call)
        assert mock_client_cls.call_count == 1

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_concurrent_cold_cache_burst_single_flight(self, mock_client_cls):
        """A burst of concurrent requests for the same cold token collapses
        into exactly ONE upstream introspection (single-flight).

        Regression guard for the 2026-06-27 TTS-narration 401: without the
        per-token lock, N concurrent cache-misses each POST the licenses
        endpoint, stampeding it on a CPU-saturated host until the timeout fires.
        """
        import asyncio

        post_calls = 0

        async def _slow_post(*_args, **_kwargs):
            nonlocal post_calls
            post_calls += 1
            # Yield so sibling tasks reach the lock while this one is "in flight".
            await asyncio.sleep(0.02)
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"active": True, "tenant_id": "burst"}
            return resp

        mock_client_cls.return_value = _mock_httpx_client(post_side_effect=_slow_post)

        results = await asyncio.gather(*(_introspect_license("burst-token") for _ in range(12)))

        assert all(r is not None and r["tenant_id"] == "burst" for r in results)
        # Exactly one upstream call despite 12 concurrent cold-cache requests.
        assert post_calls == 1
        assert mock_client_cls.call_count == 1

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_transient_failure_serves_stale_last_known_good(self, mock_client_cls):
        """A timeout AFTER a prior successful introspection serves the stale
        last-known-good result instead of denying (the e2e 401 flake).

        Reproduces the real failure: the positive cache expires (simulated by
        clearing it), the refresh introspect times out, and a token that was
        valid seconds ago must NOT be 401'd.
        """
        import w2v_common.auth.middleware as mod

        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {"active": True, "tenant_id": "t-stale"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=good)

        first = await _introspect_license("stale-token")
        assert first == {"active": True, "tenant_id": "t-stale"}

        # Simulate the positive cache (CACHE_LICENSE_TTL) expiring while the
        # longer-lived stale cache still holds the good result.
        mod._license_cache = None
        mod._introspect_locks.clear()

        mock_client_cls.return_value = _mock_httpx_client(
            post_side_effect=TimeoutError("timed out")
        )
        served = await _introspect_license("stale-token")
        assert served == {"active": True, "tenant_id": "t-stale"}  # stale-served, not None

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_definitive_inactive_not_served_stale(self, mock_client_cls):
        """A definitive 200 active=false denies even when a stale good exists —
        a genuine revocation is honored, never masked by stale-while-error.
        """
        import w2v_common.auth.middleware as mod

        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {"active": True, "tenant_id": "t-rev"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=good)
        seeded = await _introspect_license("rev-token")
        assert seeded is not None
        assert seeded["active"] is True

        mod._license_cache = None
        mod._introspect_locks.clear()

        revoked = MagicMock()
        revoked.status_code = 200
        revoked.json.return_value = {"active": False, "reason": "revoked"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=revoked)
        result = await _introspect_license("rev-token")
        assert result == {"active": False, "reason": "revoked"}  # denies, not stale

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_cold_cache_transient_failure_still_fails_closed(self, mock_client_cls):
        """A transient failure with NO prior good entry still returns None
        (caller denies) — stale-while-error never opens the cold-cache path.
        """
        mock_client_cls.return_value = _mock_httpx_client(
            post_side_effect=TimeoutError("timed out")
        )
        assert await _introspect_license("never-seen-token") is None

    # ----- token_not_found rotation-race grace (mirrors worker_management #148) -----

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_token_not_found_warm_stale_serves_last_known_good(self, mock_client_cls):
        """token_not_found on a token seen active moments ago serves warm stale
        rather than 401 — the additive-rotation propagation race.
        """
        import w2v_common.auth.middleware as mod

        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {"active": True, "tenant_id": "t-rot"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=good)
        seeded = await _introspect_license("rot-token")
        assert seeded is not None
        assert seeded["active"] is True

        # Positive cache expires; stale still warm. The refresh introspect now
        # gets a token_not_found (new token row not yet visible after rotation).
        mod._license_cache = None
        mod._introspect_locks.clear()

        not_found = MagicMock()
        not_found.status_code = 200
        not_found.json.return_value = {"active": False, "reason": "token_not_found"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=not_found)
        served = await _introspect_license("rot-token")
        assert served == {"active": True, "tenant_id": "t-rot"}  # stale-served, not denied

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_token_not_found_cold_returns_none_not_negcached(self, mock_client_cls):
        """token_not_found with NO warm stale returns None (retryable) and does
        NOT neg-cache — so recovery is immediate once the new token propagates.
        """
        import w2v_common.auth.middleware as mod

        not_found = MagicMock()
        not_found.status_code = 200
        not_found.json.return_value = {"active": False, "reason": "token_not_found"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=not_found)

        assert await _introspect_license("cold-rot-token") is None
        # NOT neg-cached: recovery is not gated by the 60s neg TTL.
        assert len(mod._get_license_neg_cache()) == 0
        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {"active": True, "tenant_id": "t-recovered"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=good)
        mod._introspect_locks.clear()
        # The ~5s tnf_probe throttle holds the None outcome; once it expires the
        # next request re-introspects and recovers. Simulate the TTL lapsing.
        mod._license_tnf_probe = None
        recovered = await _introspect_license("cold-rot-token")
        assert recovered is not None
        assert recovered["active"] is True

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_token_invalid_is_not_graced(self, mock_client_cls):
        """A revoked/expired token (reason token_invalid) 401s immediately even
        with a warm stale entry — grace is scoped ONLY to token_not_found.
        """
        import w2v_common.auth.middleware as mod

        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {"active": True, "tenant_id": "t-inv"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=good)
        seeded = await _introspect_license("inv-token")
        assert seeded is not None
        assert seeded["active"] is True

        mod._license_cache = None
        mod._introspect_locks.clear()

        invalid = MagicMock()
        invalid.status_code = 200
        invalid.json.return_value = {"active": False, "reason": "token_invalid"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=invalid)
        result = await _introspect_license("inv-token")
        assert result == {"active": False, "reason": "token_invalid"}  # denies, not stale

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_token_not_found_grace_expires_to_authoritative_deny(
        self, mock_client_cls, monkeypatch
    ):
        """After the grace window elapses a still-missing token settles to an
        authoritative inactive (neg-cached), not a perpetual stale/None flap.
        """
        import w2v_common.auth.middleware as mod

        # Fixed 120s window driven by a fake monotonic clock we advance by hand.
        monkeypatch.setenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", "120")
        clock = {"t": 1000.0}
        monkeypatch.setattr(mod._time, "monotonic", lambda: clock["t"])

        not_found = MagicMock()
        not_found.status_code = 200
        not_found.json.return_value = {"active": False, "reason": "token_not_found"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=not_found)

        # First sighting: within grace, cold -> None (retryable).
        assert await _introspect_license("exp-token") is None
        mod._introspect_locks.clear()
        mod._license_tnf_probe = None  # let the short throttle lapse
        # Advance past the window: authoritative deny + neg-cached.
        clock["t"] += 121.0
        result = await _introspect_license("exp-token")
        assert result == {"active": False, "reason": "token_not_found"}
        assert len(mod._get_license_neg_cache()) == 1  # neg-cached authoritative

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_token_not_found_grace_disabled_denies_immediately(
        self, mock_client_cls, monkeypatch
    ):
        """LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS=0 restores the old instant deny."""
        import w2v_common.auth.middleware as mod

        monkeypatch.setenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", "0")
        not_found = MagicMock()
        not_found.status_code = 200
        not_found.json.return_value = {"active": False, "reason": "token_not_found"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=not_found)

        result = await _introspect_license("disabled-token")
        assert result == {"active": False, "reason": "token_not_found"}
        assert len(mod._get_license_neg_cache()) == 1  # neg-cached authoritative

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_token_not_found_throttles_upstream_reprobes(self, mock_client_cls):
        """Repeated requests for a missing token do NOT hammer licenses.

        The grace path deliberately avoids neg-caching, so without a throttle
        every request would POST licenses. The tnf_probe caches the transient
        outcome for a few seconds → exactly ONE upstream call for a burst.
        """
        import w2v_common.auth.middleware as mod

        post_calls = 0

        def _count(*_a, **_k):
            nonlocal post_calls
            post_calls += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"active": False, "reason": "token_not_found"}
            return resp

        client = _mock_httpx_client()
        client.post = AsyncMock(side_effect=_count)
        mock_client_cls.return_value = client

        results = await asyncio.gather(
            *(_introspect_license("burst-missing-token") for _ in range(15))
        )
        assert all(r is None for r in results)
        assert post_calls == 1, f"expected 1 upstream POST, got {post_calls}"
        assert len(mod._get_license_neg_cache()) == 0  # still not neg-cached

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_malformed_grace_env_does_not_500(self, mock_client_cls, monkeypatch):
        """A bad LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS must not crash auth.

        The value is read on the request path with no enclosing handler; an
        unguarded float() would turn every request into a 500. It falls back to
        the 120s default, so a token_not_found is still graced (not raised).
        """
        import w2v_common.auth.middleware as mod

        monkeypatch.setenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", "not-a-number")
        assert mod._token_not_found_grace() == 120.0  # no ValueError escapes

        not_found = MagicMock()
        not_found.status_code = 200
        not_found.json.return_value = {"active": False, "reason": "token_not_found"}
        mock_client_cls.return_value = _mock_httpx_client(post_return=not_found)
        # Graced (cold → None), NOT a 500/raise.
        assert await _introspect_license("bad-env-token") is None

    async def test_grace_env_rejects_nan_and_inf(self, monkeypatch):
        """NaN (compares False, silently disabling) and inf (never elapses) are
        rejected in favor of the safe default."""
        import w2v_common.auth.middleware as mod

        for bad in ("nan", "inf", "-1"):
            monkeypatch.setenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", bad)
            assert mod._token_not_found_grace() == 120.0


class TestAuthMiddlewareIntegration:
    """Integration tests for AuthMiddleware with FastAPI."""

    async def test_public_endpoint_no_auth_required(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200

    async def test_root_endpoint_public(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200

    async def test_docs_endpoint_public(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/docs")
            assert resp.status_code == 200

    async def test_api_health_endpoint_public(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200

    async def test_api_health_ready_public(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health/ready")
            assert resp.status_code == 200

    async def test_health_ready_public(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/ready")
            assert resp.status_code == 200

    async def test_livez_public(self):
        # Regression: `/livez` (k8s liveness) must be auth-exempt like `/readyz`.
        # It was originally missing, so probing pods 401'd and crashlooped.
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/livez")
            assert resp.status_code == 200

    async def test_live_public(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/live")
            assert resp.status_code == 200

    async def test_protected_endpoint_no_token_untrusted_network(self):
        """Test that untrusted network without token gets 401."""
        app = _make_app()
        transport = ASGITransport(app=app)
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data")
                assert resp.status_code == 401
                body = resp.json()
                assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
                assert "Missing authentication token" in body["error"]["message"]
                # 401 envelope is aligned to the fleet shape #2: status +
                # request_id + timestamp are present alongside code/message.
                assert body["error"]["status"] == 401
                assert body["error"]["request_id"]
                assert body["error"]["timestamp"].endswith("Z")

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_protected_endpoint_with_valid_x_tenant_id_bearer(self, mock_introspect):
        """License tokens travel in `X-Tenant-Id: Bearer <token>`, not Authorization."""
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000abc",
            "subject_id": "user-1",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "Bearer valid-token-123"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["tenant_id"] == "0195f48b-0000-7000-8000-000000000abc"
            assert data["user_sub"] == "0195f48b-0000-7000-8000-000000000abc"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_effective_tenant_honored_for_delegating_subject(
        self, mock_introspect, monkeypatch
    ):
        """A trusted service account (its subject in AUTH_DELEGATING_SUBJECTS) may
        re-scope the request to X-Effective-Tenant-Id — this is how a background
        run's real tenant reaches downstream for attribution + BYO."""
        sa_subject = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-0000000005ac",
            "subject_id": sa_subject,
        }
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", sa_subject)
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000099",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["tenant_id"] == "0195f48b-0000-7000-8000-000000000099"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_effective_tenant_ignored_for_non_delegating_subject(
        self, mock_introspect, monkeypatch
    ):
        """A NON-delegating subject (e.g. a customer) cannot spoof another tenant
        via X-Effective-Tenant-Id — the introspected tenant always wins."""
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000c05",
            "subject_id": "some-customer-subject",
        }
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "8382f0b3-c082-4f97-8d47-120ce5a18e05")
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data",
                headers={
                    "X-Tenant-Id": "Bearer customer-token",
                    "X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-00000000071c",
                },
            )
            assert resp.status_code == 200
            # Spoof rejected: stays on the customer's own introspected tenant.
            assert resp.json()["tenant_id"] == "0195f48b-0000-7000-8000-000000000c05"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_effective_tenant_inert_without_config(self, mock_introspect, monkeypatch):
        """With AUTH_DELEGATING_SUBJECTS unset (default), the side-channel is inert
        even for what would be a delegating subject — fail-closed by default."""
        monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-0000000005ac",
            "subject_id": "8382f0b3-c082-4f97-8d47-120ce5a18e05",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data",
                headers={
                    "X-Tenant-Id": "Bearer sa-token",
                    "X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000099",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["tenant_id"] == "0195f48b-0000-7000-8000-0000000005ac"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_protected_endpoint_with_inactive_x_tenant_id_token(self, mock_introspect):
        mock_introspect.return_value = {
            "active": False,
            "reason": "expired",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "Bearer expired-token"})
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"]["code"] == "INVALID_TOKEN"
            assert "expired" in body["error"]["message"]

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_protected_endpoint_introspection_returns_none(self, mock_introspect):
        mock_introspect.return_value = None
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "Bearer bad-token"})
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"]["code"] == "INVALID_TOKEN"
            assert "invalid_token" in body["error"]["message"]

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_active_license_with_empty_tenant_is_rejected(self, mock_introspect):
        """Three-piece invariant: an active license with no tenant_id must 401.

        Accepting it would authenticate an "empty tenant" and bypass tenant
        isolation downstream — an empty tenant is always a bug, never valid.
        """
        for bad in ({"active": True, "subject_id": "user-1"}, {"active": True, "tenant_id": "   "}):
            mock_introspect.return_value = bad
            app = _make_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"X-Tenant-Id": "Bearer tok"})
                assert resp.status_code == 401, bad
                assert resp.json()["error"]["code"] == "INVALID_TOKEN"

    @patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=False)
    async def test_opaque_token_in_authorization_is_introspected_as_a_license_token(
        self, _mock_trusted, mock_introspect
    ):
        """CONTRACT CHANGED (platform licensing): an opaque (non-JWT) bearer in
        ``Authorization`` IS a licence token and is introspected.

        This test previously asserted the opposite — that only ``X-Tenant-Id``
        could carry a licence token. That rule cannot survive the platform's
        ``license-authz`` ext-authz gate, which reads the token from
        ``Authorization`` ONLY; keeping it would 401 every caller the moment w2v
        is gated.

        The 2026-04-26 regression this test was written for is NOT weakened: the
        bug was a *Keycloak JWT* being submitted to licence introspection, and
        that is still impossible — ``dispatch`` consumes and hard-fails every
        JWT-SHAPED bearer before this lane is reached (see the sibling test
        ``test_keycloak_jwt_in_authorization_does_not_hit_license_introspect``
        and ``tests/test_auth_platform_license.py``).
        """
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000abc",
            "subject_id": "user-1",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data",
                headers={"Authorization": "Bearer opaque-license-looking-token"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["tenant_id"] == "0195f48b-0000-7000-8000-000000000abc"
        mock_introspect.assert_awaited_once_with("opaque-license-looking-token")

    @patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=False)
    async def test_keycloak_jwt_in_authorization_does_not_hit_license_introspect(
        self, _mock_trusted, mock_introspect
    ):
        """Regression for 2026-04-26: a Keycloak USER JWT forwarded by an
        upstream caller (e.g. FR) lands in Authorization. It must be routed
        to JWT validation (or rejected if validator unconfigured), NOT
        submitted to license introspection — that fallback is what produced
        the FR 401s on /api/assets/procedures and /api/assets/secrets.
        """
        # JWT-shaped (three dot-separated segments) but unsigned; without a
        # configured KEYCLOAK_URL the middleware now hard-rejects it.
        fake_jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.sig"  # pragma: allowlist secret
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"Authorization": f"Bearer {fake_jwt}"})
            assert resp.status_code == 401
        mock_introspect.assert_not_awaited()

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_x_tenant_id_bearer_token(self, mock_introspect):
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000f04",
            "subject_id": "user-2",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "Bearer my-license-token"})
            assert resp.status_code == 200

    @patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_access(self, mock_trusted, _mock_flag):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data")
            # 0.67.0: the trusted network vouches for the CALLER, never for the
            # TENANT. There is no DEFAULT_TENANT_ID any more — this used to 200
            # with sub=00000000-... , a valid UUID that survived every type check
            # and dirty-row audit while mis-attributing cost.
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    @patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_with_x_tenant_id(self, mock_trusted, _mock_flag):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data", headers={"X-Tenant-Id": "0195f48b-0000-7000-8000-000000000c05"}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["tenant_id"] == "0195f48b-0000-7000-8000-000000000c05"

    @patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_rejects_slug_x_tenant_id(self, mock_trusted, _mock_flag):
        """A legacy slug tenant is rejected, not coerced into a uuid column."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "custom-tenant"})
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    @patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_with_whitespace_x_tenant_id(self, _t, _f):
        """Whitespace-only X-Tenant-Id is rejected, not defaulted.

        History: it first minted tenant_id="" (isolation bypass), then the
        nil-UUID sentinel (silent mis-attribution). Both were substitutions.
        """
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "   "})
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    @patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_with_empty_x_tenant_id(self, _t, _f):
        """Explicit empty X-Tenant-Id is rejected, not defaulted."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": ""})
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    @patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=False)
    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_identity_can_be_disabled(self, _mock_trusted, _mock_flag):
        """Opt-out off: in-cluster source IP without a token FAILS CLOSED.

        Regression for project_orchestrator_trusted_networks_bypass_2026-04-18.
        Previously the middleware minted a synthetic n8n-dev identity for any
        request from an in-cluster pod IP, which let the SvelteKit SSR proxy
        elicit 200 with sub=n8n-dev for unauth external browsers. Setting
        AUTH_TRUST_INCLUSTER_IDENTITY=false (the orchestrator does this in its
        entry point) closes the bypass.
        """
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data")
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
            assert "Missing authentication token" in body["error"]["message"]

    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_identity_disabled_by_default(self, _mock_trusted, monkeypatch):
        """Default OFF (0.6.9 fail-closed flip): in-cluster source IP without
        a license token is rejected even when the source is in TRUSTED_CIDRS.

        Phases 1 (X-Tenant-Id routing) and 2 (bare-client license-token
        plumbing) are both shipped, so the trusted-network fallback is no
        longer load-bearing for any caller. Operators can opt back in via
        AUTH_TRUST_INCLUSTER_IDENTITY=true if needed mid-migration.

        As of 0.6.10 the value is read from AuthSettings.trust_incluster_identity
        at call time (the env var still feeds the field via Pydantic settings).
        """
        from w2v_common.config import AuthSettings

        monkeypatch.delenv("AUTH_TRUST_INCLUSTER_IDENTITY", raising=False)
        assert AuthSettings().trust_incluster_identity is False, (
            "AuthSettings.trust_incluster_identity default must be False as of 0.6.9/0.6.10"
        )

        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data")
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
            assert "Missing authentication token" in body["error"]["message"]

    @patch("w2v_common.auth.middleware._is_trusted_network", return_value=True)
    async def test_trusted_network_env_var_flips_behavior(self, _mock_trusted, monkeypatch):
        """0.6.10 behavior preservation: AUTH_TRUST_INCLUSTER_IDENTITY=true env var
        still flips the trust-incluster behavior, now via AuthSettings.
        """
        from w2v_common.auth.middleware import _trust_incluster_identity

        monkeypatch.setenv("AUTH_TRUST_INCLUSTER_IDENTITY", "true")
        assert _trust_incluster_identity() is True

        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/data")
            # Trust flag flips ON, but with no X-Tenant-Id there is no fallback
            # tenant to mint any more — the lane rejects.
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    @patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock)
    async def test_real_license_token_still_works(self, mock_introspect):
        """Regression: legitimate license-token auth must not regress."""
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000c05",
            "subject_id": "real-user",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data", headers={"X-Tenant-Id": "Bearer real-license-token"}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["tenant_id"] == "0195f48b-0000-7000-8000-000000000c05"
            assert data["user_sub"] == "0195f48b-0000-7000-8000-000000000c05"
        mock_introspect.assert_awaited_once()

    @patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock)
    async def test_license_token_preserved_on_request_state(self, mock_introspect):
        """Regression: AuthMiddleware must stash the *raw* opaque token on
        ``request.state.license_token`` so downstream code can forward it.

        The post-promote E2E happy-path 404 root cause was that the orchestrator's
        crawl stage fell back to ``settings.downstream.license_token`` (its own
        service-account token) when calling webscraper — because the caller's
        opaque token had been thrown away after introspection. webscraper then
        called secrets-service with the wrong tenant identity and 404'd the
        credential lookup.
        """
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": "0195f48b-0000-7000-8000-000000000c05",
            "subject_id": "real-user",
            "license_id": "lic-123",
        }
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/data",
                headers={"X-Tenant-Id": "Bearer the-actual-opaque-license-token"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["license_token"] == "the-actual-opaque-license-token"

    async def test_trailing_slash_normalization(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/")
            assert resp.status_code in (200, 307)


class TestPublicEndpoints:
    """Test that public endpoints are correctly defined."""

    def test_public_endpoints_contains_health(self):
        assert "/health" in PUBLIC_ENDPOINTS
        assert "/healthz" in PUBLIC_ENDPOINTS
        assert "/ready" in PUBLIC_ENDPOINTS

    def test_public_endpoints_liveness_readiness_symmetric(self):
        # Every readiness path must have its liveness twin (and vice versa),
        # so a service can use either probe convention without a 401 crashloop.
        assert "/readyz" in PUBLIC_ENDPOINTS
        assert "/livez" in PUBLIC_ENDPOINTS
        assert "/ready" in PUBLIC_ENDPOINTS
        assert "/live" in PUBLIC_ENDPOINTS

    def test_public_endpoints_contains_docs(self):
        assert "/docs" in PUBLIC_ENDPOINTS
        assert "/redoc" in PUBLIC_ENDPOINTS
        assert "/openapi.json" in PUBLIC_ENDPOINTS

    def test_public_prefixes(self):
        # PUBLIC_PREFIXES is a tuple (used with str.startswith())
        assert isinstance(PUBLIC_PREFIXES, tuple)
        assert "/docs" in PUBLIC_PREFIXES
        assert "/health/" in PUBLIC_PREFIXES


def _req(headers: dict[str, str], path: str = "/") -> Request:
    """Build a minimal Starlette Request carrying the given headers."""
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": path, "headers": raw})


class TestApplyEffectiveTenant:
    """Direct unit tests for the _apply_effective_tenant honour-gate."""

    _SA = "8382f0b3-c082-4f97-8d47-120ce5a18e05"

    def test_forwards_subject_and_license_on_delegation(self, monkeypatch):
        """A delegating subject's forwarded X-Subject-Id / X-License-Id carry the
        run's own identity so per-user metering attributes to the customer, not
        the service account."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req(
            {
                "X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000099",
                "X-Subject-Id": "run-user-sub",
                "X-License-Id": "run-license-id",
            }
        )
        tenant, subject, license_id = _apply_effective_tenant(
            req, self._SA, uuid.UUID("0195f48b-0000-7000-8000-0000000005aa"), "sa-sub", "sa-license"
        )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000099")
        assert subject == "run-user-sub"
        assert license_id == "run-license-id"

    def test_delegated_absent_license_is_not_the_sa_license(self, monkeypatch):
        """Tenant re-scoped + NO forwarded licence ⇒ licence is dropped to None.

        ⛔ This is issue #170. ``license_id`` here is the DELEGATING service
        account's own licence, owned by the SA's tenant. The tenant has just
        moved to somebody else, so keeping it billed 724 live cost events to one
        tenant under another tenant's active licence.

        ⚠️ The honest answer is an ABSENCE. This test fails the moment anyone
        reinstates ``fwd_license or license_id`` to make the unlicensed counter
        look smaller — that counter going up is the instrument telling the truth.

        Subject is deliberately NOT dropped: subject is *who acted* (billing
        carries it as ``actor_id``), and the SA genuinely is the actor. Only the
        licence is a tenant-bound entitlement.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000099"})
        tenant, subject, license_id = _apply_effective_tenant(
            req, self._SA, uuid.UUID("0195f48b-0000-7000-8000-0000000005aa"), "sa-sub", "sa-license"
        )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000099")
        assert subject == "sa-sub"
        assert license_id is None, "the delegator's own licence must not follow the tenant re-scope"

    def test_delegated_empty_license_header_is_not_the_sa_license(self, monkeypatch):
        """An explicitly EMPTY X-License-Id is the normal delegated shape.

        ``get_identity_headers()`` stamps ``X-License-Id: ""`` whenever the
        delegating run's own identity has no licence (both the Keycloak-JWT and
        trusted-network lanes bind ``license_id=None``), so this — not an absent
        header — is what production actually sent on the mis-billed events.
        Whitespace must be treated the same as empty.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        for blank in ("", "   "):
            req = _req(
                {
                    "X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000099",
                    "X-License-Id": blank,
                }
            )
            _, _, license_id = _apply_effective_tenant(
                req,
                self._SA,
                uuid.UUID("0195f48b-0000-7000-8000-0000000005aa"),
                "sa-sub",
                "sa-license",
            )
            assert license_id is None, f"blank={blank!r} must not fall back to the SA licence"

    def test_delegating_subject_not_re_scoping_keeps_its_own_license(self, monkeypatch):
        """⭐ NEGATIVE CONTROL — the 122,106 correctly-billed events.

        A delegating service account making an ORDINARY call for its own tenant
        (no effective-tenant header, or one naming its own tenant) is not a
        delegation at all. Its tenant never moves, so its licence is still its
        own tenant's and MUST survive untouched. The same licence that rides the
        724 mis-billed events is billed correctly on 122,106 others by exactly
        this path — dropping licences here would be the real regression.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        own = uuid.UUID("0195f48b-0000-7000-8000-0000000005aa")
        for headers in ({}, {"X-Effective-Tenant-Id": str(own)}):
            tenant, subject, license_id = _apply_effective_tenant(
                _req(headers), self._SA, own, "sa-sub", "sa-license"
            )
            assert tenant == own
            assert subject == "sa-sub"
            assert license_id == "sa-license", f"headers={headers} must not drop the licence"

    def test_non_delegating_caller_keeps_its_own_license(self, monkeypatch):
        """NEGATIVE CONTROL — an ordinary customer request is wholly unchanged."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        own = uuid.UUID("0195f48b-0000-7000-8000-000000000c05")
        for headers in ({}, {"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-00000000071c"}):
            tenant, subject, license_id = _apply_effective_tenant(
                _req(headers), "some-customer-subject", own, "cust-sub", "cust-lic"
            )
            assert (tenant, subject, license_id) == (own, "cust-sub", "cust-lic")

    def test_no_return_pairs_a_re_scoped_tenant_with_the_callers_license(self):
        """Structural guard: no ``return`` may hand back ``... or license_id``.

        The behavioural tests above already fail if ``fwd_license or license_id``
        comes back, but they cannot say WHY. This one names the defect in the
        source so a reviewer reading the failure sees the exact expression that
        caused issue #170.

        Asserts on the AST, not on text, so a comment mentioning the expression
        cannot trip it and a reformat cannot disarm it. It fails loudly if it
        cannot find the returns it is supposed to inspect — a guard that matches
        nothing must not read as a pass.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(_apply_effective_tenant)))
        returns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Tuple)
            and len(node.value.elts) == 3
        ]
        assert len(returns) >= 3, (
            f"guard is blind: expected the 3 (tenant, subject, license) returns, found "
            f"{len(returns)}. _apply_effective_tenant was restructured — re-target this test."
        )
        for node in returns:
            # Re-assert what the comprehension above already established. The
            # narrowing does not survive into this loop for a static checker,
            # and `ty check` (whole repo, tests included) is a CI gate.
            tup = node.value
            assert isinstance(tup, ast.Tuple)
            lic = tup.elts[2]
            if isinstance(lic, ast.BoolOp) and isinstance(lic.op, ast.Or):
                names = {v.id for v in lic.values if isinstance(v, ast.Name)}
                assert "license_id" not in names, (
                    f"line {node.lineno}: `... or license_id` re-attaches the DELEGATOR's "
                    "licence to a re-scoped tenant. That is issue #170 (724 mis-billed "
                    "events). An absent forwarded licence must yield None."
                )

    def test_accepted_delegation_stamps_tenant_delegated_true(self, monkeypatch):
        """S95: an ACCEPTED re-scope (allowlisted subject, header present, tenant
        differs) must stamp ``request.state.tenant_delegated = True`` — this is
        the marker w2v_orchestrator's ``require_known_tenant`` reads to treat a
        delegated-but-unregistered tenant as known instead of 403ing it."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000099"})
        tenant, _, _ = _apply_effective_tenant(
            req, self._SA, uuid.UUID("0195f48b-0000-7000-8000-0000000005aa"), "sa-sub", "sa-license"
        )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000099")
        assert req.state.tenant_delegated is True

    def test_non_delegating_subject_does_not_set_marker(self, monkeypatch):
        """A customer token (subject not in AUTH_DELEGATING_SUBJECTS) asserting a
        different tenant must NOT get the marker, even though it is rejected
        rather than silently dropped — the marker means "trusted delegator
        vouched for this", not "a header was present"."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-00000000071c"})
        tenant, _, _ = _apply_effective_tenant(
            req,
            "some-customer-subject",
            uuid.UUID("0195f48b-0000-7000-8000-000000000c05"),
            "cust-sub",
            "cust-lic",
        )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000c05")
        assert req.state.tenant_delegated is False

    def test_no_header_does_not_set_marker(self, monkeypatch):
        """A delegating subject making an ordinary call (no header at all) is not
        a delegation — the marker must stay False."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({})
        _apply_effective_tenant(
            req, self._SA, uuid.UUID("0195f48b-0000-7000-8000-0000000005aa"), "sa-sub", "sa-license"
        )
        assert req.state.tenant_delegated is False

    def test_equal_tenant_header_does_not_set_marker(self, monkeypatch):
        """A delegating subject whose header names its OWN tenant is a no-op,
        not a re-scope — the marker must stay False."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        own = uuid.UUID("0195f48b-0000-7000-8000-0000000005aa")
        req = _req({"X-Effective-Tenant-Id": str(own)})
        _apply_effective_tenant(req, self._SA, own, "sa-sub", "sa-license")
        assert req.state.tenant_delegated is False

    def test_invalid_effective_tenant_raises_and_does_not_set_marker(self, monkeypatch):
        """A delegating subject asserting a non-UUID effective tenant is rejected
        (TenantError) by the caller — the marker must not have been flipped True
        before the raise."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({"X-Effective-Tenant-Id": "not-a-uuid"})
        with pytest.raises(TenantError):
            _apply_effective_tenant(
                req,
                self._SA,
                uuid.UUID("0195f48b-0000-7000-8000-0000000005aa"),
                "sa-sub",
                "sa-license",
            )
        assert req.state.tenant_delegated is False

    def test_rejected_header_logs_warning(self, monkeypatch, caplog):
        """A non-delegating subject that sends an effective-tenant header is logged
        (spoof attempt or a receiver missing AUTH_DELEGATING_SUBJECTS config)."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-00000000071c"})
        with caplog.at_level("WARNING", logger="w2v_common.auth.middleware"):
            tenant, _, _ = _apply_effective_tenant(
                req,
                "some-customer-subject",
                uuid.UUID("0195f48b-0000-7000-8000-000000000c05"),
                "cust-sub",
                "cust-lic",
            )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000c05")
        assert "effective-tenant header rejected" in caplog.text

    def test_no_header_no_log(self, monkeypatch, caplog):
        """No effective-tenant header ⇒ silent no-op, no warning noise."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({})
        with caplog.at_level("WARNING", logger="w2v_common.auth.middleware"):
            tenant, _, _ = _apply_effective_tenant(
                req,
                "some-customer-subject",
                uuid.UUID("0195f48b-0000-7000-8000-000000000c05"),
                "cust-sub",
                "cust-lic",
            )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000c05")
        assert "effective-tenant header rejected" not in caplog.text
        assert "/api/health/" in PUBLIC_PREFIXES

    def test_equal_tenant_header_no_log(self, monkeypatch, caplog):
        """A non-delegating subject whose effective-tenant header equals its OWN
        introspected tenant is a benign no-op (the S2S hook stamps the header
        unconditionally) ⇒ no warning. Only a header asserting a DIFFERENT tenant
        is a real re-scope attempt worth surfacing — otherwise the warning is
        fleet-wide log spam that drowns the genuine-spoof signal."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        req = _req({"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-000000000c05"})
        with caplog.at_level("WARNING", logger="w2v_common.auth.middleware"):
            tenant, _, _ = _apply_effective_tenant(
                req,
                "some-customer-subject",
                uuid.UUID("0195f48b-0000-7000-8000-000000000c05"),
                "cust-sub",
                "cust-lic",
            )
        assert tenant == uuid.UUID("0195f48b-0000-7000-8000-000000000c05")
        assert "effective-tenant header rejected" not in caplog.text

    def test_rejected_header_counts_and_names_both_tenants(self, monkeypatch, caplog):
        """The reject path must be ATTRIBUTABLE.

        Keeping the caller's own tenant is correct and stays observe-only — a
        403 would break live narration for any currently-unlisted sender, and an
        unlisted sender IS the misconfiguration this branch anticipates. But the
        warning named only the subject, so an operator could not tell which
        customer was under-billed. Counter + both tenant ids close that.
        """
        from prometheus_client import REGISTRY

        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        subject = "some-customer-subject"
        asserted = "0195f48b-0000-7000-8000-00000000071c"
        kept = uuid.UUID("0195f48b-0000-7000-8000-000000000c05")

        def _count() -> float:
            value = REGISTRY.get_sample_value("w2v_auth_effective_tenant_rejected_total")
            return 0.0 if value is None else value

        before = _count()
        req = _req({"X-Effective-Tenant-Id": asserted})
        with caplog.at_level("WARNING", logger="w2v_common.auth.middleware"):
            tenant, _, _ = _apply_effective_tenant(req, subject, kept, "cust-sub", "cust-lic")
        assert tenant == kept
        assert _count() - before == 1.0
        assert asserted in caplog.text, "the under-billed customer must be named"
        assert str(kept) in caplog.text, "the tenant that ate the usage must be named"
        assert subject in caplog.text

    def test_rejected_counter_carries_no_subject_label(self, monkeypatch):
        """CARDINALITY GUARD — this series must stay unlabelled.

        This branch fires for ANY non-delegating caller that stamps the header,
        and that population is every end user on the fleet: a proxy
        misconfigured to stamp it would mint one series per subject and blow up
        ``/metrics``. The subject is named in the warning (asserted above), so
        nothing diagnostic is lost. Asserting the labelled lookup is ``None``
        fails loudly if anyone re-adds the label.
        """
        from prometheus_client import REGISTRY

        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        subject = "cardinality-probe-subject"
        kept = uuid.UUID("0195f48b-0000-7000-8000-000000000c05")
        _apply_effective_tenant(
            _req({"X-Effective-Tenant-Id": "0195f48b-0000-7000-8000-00000000071c"}),
            subject,
            kept,
            "cust-sub",
            "cust-lic",
        )

        assert (
            REGISTRY.get_sample_value(
                "w2v_auth_effective_tenant_rejected_total", {"subject": subject}
            )
            is None
        ), "the rejected counter must not be labelled by subject (unbounded cardinality)"
        # ...and the unlabelled series does exist, so the assertion above is not
        # vacuously true because the metric stopped being recorded entirely.
        assert REGISTRY.get_sample_value("w2v_auth_effective_tenant_rejected_total") is not None
        # `_labelnames` is private and untyped, so reach it via getattr to keep
        # `ty check` (which covers tests in CI) resolving.
        collector = REGISTRY._names_to_collectors["w2v_auth_effective_tenant_rejected_total"]
        assert getattr(collector, "_labelnames", None) == ()

    def test_no_reject_does_not_move_the_counter(self, monkeypatch):
        """NEGATIVE CONTROL — a benign equal-tenant header is not a rejection."""
        from prometheus_client import REGISTRY

        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        subject = "quiet-customer-subject"
        own = uuid.UUID("0195f48b-0000-7000-8000-000000000c05")

        def _count() -> float:
            value = REGISTRY.get_sample_value("w2v_auth_effective_tenant_rejected_total")
            return 0.0 if value is None else value

        before = _count()
        for headers in ({}, {"X-Effective-Tenant-Id": str(own)}):
            _apply_effective_tenant(_req(headers), subject, own, "cust-sub", "cust-lic")
        assert _count() == before

    def _missing_count(self, subject: str) -> float:
        from prometheus_client import REGISTRY

        value = REGISTRY.get_sample_value(
            "w2v_auth_effective_tenant_missing_total", {"subject": subject}
        )
        return 0.0 if value is None else value

    @pytest.mark.parametrize("headers", [{}, {"X-Effective-Tenant-Id": "   "}])
    def test_delegating_caller_without_header_counts_and_warns(self, monkeypatch, caplog, headers):
        """The last invisible path on the delegation route.

        An ALLOWLISTED delegator that does not stamp X-Effective-Tenant-Id keeps
        the SERVICE ACCOUNT's tenant, so the real customer is never billed. That
        used to produce no exception, no log and no metric. Observe-only by
        design — it must keep working — but it must be VISIBLE, naming which
        caller failed to stamp and which tenant wrongly absorbed the cost.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        sa_tenant = uuid.UUID("0195f48b-0000-7000-8000-0000000005aa")
        before = self._missing_count(self._SA)

        req = _req(headers)
        with caplog.at_level("WARNING", logger="w2v_common.auth.middleware"):
            tenant, subject, license_id = _apply_effective_tenant(
                req, self._SA, sa_tenant, "sa-sub", "sa-license"
            )

        # Behaviour is UNCHANGED — warn, never raise.
        assert (tenant, subject, license_id) == (sa_tenant, "sa-sub", "sa-license")
        assert req.state.tenant_delegated is False
        assert self._missing_count(self._SA) - before == 1.0
        assert "effective-tenant header missing" in caplog.text
        assert self._SA in caplog.text, "the caller that failed to stamp must be named"
        assert str(sa_tenant) in caplog.text, "the tenant that ate the cost must be named"

    def test_delegating_caller_asserting_own_tenant_stays_silent(self, monkeypatch, caplog):
        """NEGATIVE CONTROL — the OTHER half of the split condition.

        An allowlisted delegator whose header names its OWN tenant is legitimate
        and expected (the S2S hook stamps it unconditionally). It must not touch
        the missing-header counter and must not log: counting it would bury the
        real signal under fleet-wide volume.

        ⛔ This asserts the ABSENCE of any WARNING-or-above record from this
        module's logger, NOT the absence of one substring. A substring match is
        a test that goes blind the moment somebody rewords the message — it then
        passes while the fleet-wide log spam it exists to prevent is back. The
        silence itself is the contract, so the silence is what is asserted.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        sa_tenant = uuid.UUID("0195f48b-0000-7000-8000-0000000005aa")
        before = self._missing_count(self._SA)

        req = _req({"X-Effective-Tenant-Id": str(sa_tenant)})
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="w2v_common.auth.middleware"):
            tenant, subject, license_id = _apply_effective_tenant(
                req, self._SA, sa_tenant, "sa-sub", "sa-license"
            )

        assert (tenant, subject, license_id) == (sa_tenant, "sa-sub", "sa-license")
        assert req.state.tenant_delegated is False
        assert self._missing_count(self._SA) == before
        noisy = [
            (r.name, r.levelname, r.getMessage())
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.name.startswith("w2v_common.auth.middleware")
        ]
        assert noisy == [], f"a legitimate self-asserting delegator must log nothing: {noisy}"

    def test_non_delegating_caller_without_header_is_not_counted(self, monkeypatch):
        """NEGATIVE CONTROL — the metric is about DELEGATORS failing to stamp.

        An ordinary customer request carries no effective-tenant header and never
        should; counting it would make the series meaningless.
        """
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        subject = "some-customer-subject"
        before = self._missing_count(subject)
        _apply_effective_tenant(
            _req({}),
            subject,
            uuid.UUID("0195f48b-0000-7000-8000-000000000c05"),
            "cust-sub",
            "cust-lic",
        )
        assert self._missing_count(subject) == before

    # --- AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES ---------------------------------

    _SA_TENANT = uuid.UUID("0195f48b-0000-7000-8000-0000000005aa")

    def _missing_warnings(self, caplog) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name.startswith("w2v_common.auth.middleware")
            and "effective-tenant header missing" in r.getMessage()
        ]

    def _call_missing(self, caplog, path: str):
        req = _req({}, path=path)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="w2v_common.auth.middleware"):
            result = _apply_effective_tenant(req, self._SA, self._SA_TENANT, "sa-sub", "sa-license")
        # Identity is never changed by the unbilled-prefix setting.
        assert result == (self._SA_TENANT, "sa-sub", "sa-license")
        assert req.state.tenant_delegated is False
        return req

    def test_unbilled_prefix_suppresses_missing_warning_and_counter(self, monkeypatch, caplog):
        """A worker poll (POST /api/tts/claim → 204, never bills) cannot
        mis-attribute cost, so it must not warn nor move the missing counter."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        monkeypatch.setenv("AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES", "/api/tts/, /other/")
        before = self._missing_count(self._SA)
        self._call_missing(caplog, "/api/tts/claim")
        assert self._missing_count(self._SA) == before
        assert self._missing_warnings(caplog) == []
        debug = [r for r in caplog.records if "unbilled path" in r.getMessage()]
        assert len(debug) == 1 and debug[0].levelno == logging.DEBUG

    def test_unbilled_prefix_does_not_cover_other_paths(self, monkeypatch, caplog):
        """REGRESSION GUARD — a billing path outside the prefix still warns + counts."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        monkeypatch.setenv("AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES", "/api/tts/")
        before = self._missing_count(self._SA)
        self._call_missing(caplog, "/api/tts-jobs/synthesize")
        assert self._missing_count(self._SA) - before == 1.0
        assert len(self._missing_warnings(caplog)) == 1

    @pytest.mark.parametrize("path", ["/api/ttsx/claim", "/api/ttsx", "/api/tts/../render"])
    def test_unbilled_prefix_matches_only_at_segment_boundary(self, monkeypatch, caplog, path):
        """'/api/tts/' must never cover '/api/ttsx/...' (raw startswith bug) nor a
        traversal that resolves elsewhere."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        monkeypatch.setenv("AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES", "/api/tts/")
        before = self._missing_count(self._SA)
        self._call_missing(caplog, path)
        assert self._missing_count(self._SA) - before == 1.0
        assert len(self._missing_warnings(caplog)) == 1

    def test_unbilled_prefix_default_empty_keeps_old_behaviour(self, monkeypatch, caplog):
        """Setting unset (default) ⇒ every path warns + counts, as before 0.81.0."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        monkeypatch.delenv("AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES", raising=False)
        before = self._missing_count(self._SA)
        self._call_missing(caplog, "/api/tts/claim")
        assert self._missing_count(self._SA) - before == 1.0
        assert len(self._missing_warnings(caplog)) == 1

    def test_unbilled_prefix_still_honours_a_sent_header(self, monkeypatch):
        """The setting is observability-only: a delegator that DOES stamp the
        header on an unbilled path is still re-scoped exactly as today."""
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", self._SA)
        monkeypatch.setenv("AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES", "/api/tts/")
        eff = "0195f48b-0000-7000-8000-000000000099"
        req = _req({"X-Effective-Tenant-Id": eff}, path="/api/tts/claim")
        tenant, _, _ = _apply_effective_tenant(
            req, self._SA, self._SA_TENANT, "sa-sub", "sa-license"
        )
        assert tenant == uuid.UUID(eff)
        assert req.state.tenant_delegated is True


class TestExtraPublicEndpoints:
    """Per-instance ``extra_public_endpoints`` (the OAuth-callback hatch).

    Services with a public callback/webhook endpoint (e.g. youtube_upload's
    ``/api/auth/callback``) pass the path here instead of subclassing
    AuthMiddleware or mutating the shared module-level ``PUBLIC_ENDPOINTS``
    (which would leak the path to every service sharing the process).
    """

    def _app(self) -> FastAPI:
        app = FastAPI()
        app.add_middleware(AuthMiddleware, extra_public_endpoints={"/api/auth/callback"})

        @app.get("/api/auth/callback")
        async def callback():
            return {"ok": True}

        @app.get("/api/data")
        async def data():
            return {"ok": True}

        return app

    async def test_extra_public_endpoint_bypasses_auth(self):
        app = self._app()
        transport = ASGITransport(app=app)
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/auth/callback")
                assert resp.status_code == 200
                assert resp.json() == {"ok": True}

    async def test_extra_public_endpoint_trailing_slash_normalized(self):
        # A configured path WITH a trailing slash must still match the request
        # path WITHOUT one: __init__ normalizes the stored set to the same
        # rstrip("/") form dispatch() compares against.
        app = FastAPI()
        app.add_middleware(AuthMiddleware, extra_public_endpoints={"/api/auth/callback/"})

        @app.get("/api/auth/callback")
        async def callback():
            return {"ok": True}

        transport = ASGITransport(app=app)
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/auth/callback")
                assert resp.status_code == 200

    async def test_normal_endpoint_still_protected(self):
        app = self._app()
        transport = ASGITransport(app=app)
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data")
                assert resp.status_code == 401

    def test_default_is_empty_and_does_not_mutate_shared_set(self):
        app = FastAPI()
        before = frozenset(PUBLIC_ENDPOINTS)
        mw = AuthMiddleware(app, extra_public_endpoints={"/api/auth/callback"})
        assert mw._extra_public_endpoints == frozenset({"/api/auth/callback"})
        # The instance override must NOT bleed into the module-level policy.
        assert frozenset(PUBLIC_ENDPOINTS) == before
        assert "/api/auth/callback" not in PUBLIC_ENDPOINTS
        # Default construction carries no extra endpoints.
        assert AuthMiddleware(app)._extra_public_endpoints == frozenset()
