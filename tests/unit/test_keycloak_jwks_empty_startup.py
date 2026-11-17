"""JWKS startup behavior: empty cache + fetch failure must raise loudly.

A silent ``_refresh_jwks`` failure with an empty cache leaves the validator
permanently broken — every subsequent JWT validation hits ``unknown_kid``
and surfaces as a 401 with no diagnostic that the JWKS path was unreachable.
0.16.3 raises in that case so startup crashes loudly.
"""

from __future__ import annotations

import time

import httpx
import pytest

from w2v_common.auth.keycloak import KeycloakJWTValidator


def _validator() -> KeycloakJWTValidator:
    return KeycloakJWTValidator(
        keycloak_url="http://kc.local",
        realm="testrealm",
        audiences=[],
    )


class TestPrefetchJwks:
    async def test_raises_when_cache_empty_and_fetch_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        v = _validator()

        async def boom(*args, **kwargs):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr("httpx.AsyncClient.get", boom)

        with pytest.raises(httpx.HTTPError):
            await v.prefetch_jwks()

    async def test_soft_fail_when_cache_has_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        v = _validator()
        # Prime the cache so soft-fail path triggers.
        v._jwks_by_kid = {"existing-kid": {"kid": "existing-kid", "kty": "RSA"}}
        v._jwks_fetched_at = time.monotonic()

        async def boom(*args, **kwargs):
            raise httpx.ConnectError("transient blip")

        monkeypatch.setattr("httpx.AsyncClient.get", boom)

        # No exception — cached set still serves traffic until rotation.
        await v.prefetch_jwks()
        assert "existing-kid" in v._jwks_by_kid

    async def test_successful_fetch_populates_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        v = _validator()
        payload = {"keys": [{"kid": "k1", "kty": "RSA"}, {"kid": "k2", "kty": "RSA"}]}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, json=payload)

        transport = httpx.MockTransport(handler)
        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        await v.prefetch_jwks()
        assert set(v._jwks_by_kid.keys()) == {"k1", "k2"}


class TestIsReady:
    def test_empty_cache_is_not_ready(self) -> None:
        assert _validator().is_ready() is False

    def test_fresh_cache_is_ready(self) -> None:
        v = _validator()
        v._jwks_by_kid = {"k1": {"kid": "k1"}}
        v._jwks_fetched_at = time.monotonic()
        assert v.is_ready() is True

    def test_severely_stale_cache_is_not_ready(self) -> None:
        v = _validator()
        v._jwks_by_kid = {"k1": {"kid": "k1"}}
        v._jwks_ttl = 60
        # 3x TTL ago — beyond the 2x cutoff
        v._jwks_fetched_at = time.monotonic() - 200
        assert v.is_ready() is False
