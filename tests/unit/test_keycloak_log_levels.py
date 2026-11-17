"""JWT rejection reasons must be visible under the fleet's ``LOG_LEVEL=WARNING``.

Every w2v service runs with ``LOG_LEVEL=WARNING`` in production, so a failure
logged at DEBUG is not merely quiet — it does not exist. w2v services are
internal-mesh only (edge auth was removed 2026-04-18), so a JWT that fails to
parse, names an unknown ``kid``, or fails validation is genuinely anomalous
rather than attacker-driven background noise: each one is an operator signal.

These tests pin the LEVEL, not merely the fact that something was logged.
"""

from __future__ import annotations

import base64
import logging
import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from w2v_common.auth.keycloak import KeycloakJWTValidator


def _generate_rsa_keypair() -> tuple[str, dict[str, str]]:
    """Return ``(private_pem, jwk)`` for an RS256 signing pair with kid ``test-kid``."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    numbers = private_key.public_key().public_numbers()

    def _b64url(n: int) -> str:
        raw = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    jwk = {
        "kty": "RSA",
        "kid": "test-kid",
        "alg": "RS256",
        "use": "sig",
        "n": _b64url(numbers.n),
        "e": _b64url(numbers.e),
    }
    return private_pem, jwk


_LOGGER_NAME = "w2v_common.auth.keycloak"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == _LOGGER_NAME]


def _validator(jwk: dict[str, str]) -> KeycloakJWTValidator:
    v = KeycloakJWTValidator(
        keycloak_url="http://kc.local",
        realm="testrealm",
        audiences=[],
    )
    v._jwks_by_kid = {"test-kid": jwk}
    v._jwks_fetched_at = time.monotonic()
    return v


class TestJwtFailuresLogAtWarning:
    async def test_header_parse_failure_logs_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Three dot-separated segments, but the header is not decodable JSON."""
        _, jwk = _generate_rsa_keypair()
        validator = _validator(jwk)

        with caplog.at_level(logging.WARNING):
            claims, reason = await validator.validate("aaaa.bbbb.cccc")

        assert claims is None
        assert reason is not None and reason.startswith("header_parse_failed")
        assert any("JWT header parse failed" in r.getMessage() for r in _warnings(caplog)), (
            f"expected WARNING, got {[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )

    async def test_unknown_kid_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """An unknown ``kid`` survives a forced JWKS refresh — key rotation drift."""
        private_pem, jwk = _generate_rsa_keypair()
        validator = _validator(jwk)
        token = jwt.encode(
            {
                "iss": "http://kc.local/realms/testrealm",
                "sub": "user-1",
                "exp": int(time.time()) + 300,
            },
            private_pem,
            algorithm="RS256",
            headers={"kid": "rotated-away-kid"},
        )

        with (
            caplog.at_level(logging.WARNING),
            patch.object(validator, "_refresh_jwks", new=AsyncMock(return_value=None)),
        ):
            claims, reason = await validator.validate(token)

        assert claims is None
        assert reason == "unknown_kid: rotated-away-kid"
        assert any("No signing key for kid=" in r.getMessage() for r in _warnings(caplog)), (
            f"expected WARNING, got {[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )

    async def test_decode_failure_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """An expired token (beyond leeway) is a validation failure operators see."""
        private_pem, jwk = _generate_rsa_keypair()
        validator = _validator(jwk)
        token = jwt.encode(
            {
                "iss": "http://kc.local/realms/testrealm",
                "sub": "user-1",
                "exp": int(time.time()) - 3600,
            },
            private_pem,
            algorithm="RS256",
            headers={"kid": "test-kid"},
        )

        with caplog.at_level(logging.WARNING):
            claims, reason = await validator.validate(token)

        assert claims is None
        assert reason is not None and reason.startswith("jwt_decode_failed")
        matching = [r for r in _warnings(caplog) if "JWT validation failed" in r.getMessage()]
        assert matching, (
            f"expected WARNING, got {[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        # The reason must travel with the line, not just the fact of failure.
        assert "Signature has expired" in matching[0].getMessage()
