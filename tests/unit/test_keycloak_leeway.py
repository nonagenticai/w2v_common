"""JWT clock-skew leeway in ``KeycloakJWTValidator``.

Sub-second NTP drift between pods and Keycloak triggers ``ExpiredSignatureError``
on otherwise-valid tokens. The 10-second leeway absorbs that drift so a valid
token signed less than 10s past its ``exp`` still validates.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from w2v_common.auth.keycloak import KeycloakJWTValidator


def _generate_rsa_keypair() -> tuple[str, dict[str, str]]:
    """Return (private_pem, jwk_dict) for an RS256 signing pair."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_key = private_key.public_key()
    numbers = public_key.public_numbers()
    import base64

    def _b64url(n: int) -> str:
        b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
        return base64.urlsafe_b64encode(b).decode().rstrip("=")

    jwk = {
        "kty": "RSA",
        "kid": "test-kid",
        "alg": "RS256",
        "use": "sig",
        "n": _b64url(numbers.n),
        "e": _b64url(numbers.e),
    }
    return private_pem, jwk


@pytest.fixture
def signing_material() -> tuple[str, dict[str, str]]:
    return _generate_rsa_keypair()


def _make_validator(jwk: dict[str, str], leeway: float = 10.0) -> KeycloakJWTValidator:
    v = KeycloakJWTValidator(
        keycloak_url="http://kc.local",
        realm="testrealm",
        audiences=[],
        leeway=leeway,
    )
    v._jwks_by_kid = {"test-kid": jwk}
    v._jwks_fetched_at = time.monotonic()
    return v


def _make_token(private_pem: str, *, exp_offset: int) -> str:
    now = int(time.time())
    payload = {
        "iss": "http://kc.local/realms/testrealm",
        "sub": "user-1",
        "exp": now + exp_offset,
        "iat": now - 60,
    }
    return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": "test-kid"})


class TestLeeway:
    async def test_token_expired_within_leeway_window_passes(
        self, signing_material: tuple[str, dict[str, str]]
    ) -> None:
        private_pem, jwk = signing_material
        validator = _make_validator(jwk, leeway=10.0)
        token = _make_token(private_pem, exp_offset=-5)

        claims, reason = await validator.validate(token)

        assert reason is None
        assert claims is not None
        assert claims["sub"] == "user-1"

    async def test_token_expired_beyond_leeway_window_fails(
        self, signing_material: tuple[str, dict[str, str]]
    ) -> None:
        private_pem, jwk = signing_material
        validator = _make_validator(jwk, leeway=10.0)
        token = _make_token(private_pem, exp_offset=-30)

        claims, reason = await validator.validate(token)

        assert claims is None
        assert reason is not None
        assert "jwt_decode_failed" in reason

    async def test_zero_leeway_rejects_just_expired(
        self, signing_material: tuple[str, dict[str, str]]
    ) -> None:
        """Sanity check: with leeway=0 the same near-expiry token would 401."""

        private_pem, jwk = signing_material
        validator = _make_validator(jwk, leeway=0.0)
        token = _make_token(private_pem, exp_offset=-5)

        claims, reason = await validator.validate(token)

        assert claims is None
        assert reason is not None


class TestEnvDrivenLeeway:
    def test_env_var_feeds_validator_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.local")
        monkeypatch.setenv("KEYCLOAK_REALM", "testrealm")
        monkeypatch.setenv("KEYCLOAK_JWT_LEEWAY", "30")

        from w2v_common.auth import keycloak as kc_module

        with (
            patch.object(kc_module, "_validator", None),
            patch.object(kc_module, "_validator_initialized", False),
        ):
            v = kc_module.get_jwt_validator()
            assert v is not None
            assert v._leeway == 30.0
            kc_module.reset_jwt_validator()
