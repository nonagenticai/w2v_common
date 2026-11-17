"""P7 intra-cell JWT: mint, verify, keyring hot reload, outbound hook, middleware lane.

Every test here fails on 0.90.0: ``w2v_common.auth.cell_jwt`` did not exist, no
outbound request carried ``X-W2V-Cell-Token`` and ``AuthMiddleware`` had no
cell lane / ``CELL_AUTH_MODE``.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import OKPAlgorithm
from prometheus_client import REGISTRY
from starlette.requests import Request
from starlette.routing import Match

from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    CELL_TOKEN_TYPE,
    CellAuthMode,
    CellClaims,
    CellConfigError,
    CellKeyring,
    CellScope,
    CellTokenError,
    cell_audience_for_url,
    cell_auth_mode,
    mint_cell_token,
    mint_service_cell_token,
    parse_jwks,
    reset_cell_state,
    verify_cell_token,
)
from w2v_common.auth.cell_service import require_cell_service_caller
from w2v_common.auth.entitlement import require_license
from w2v_common.auth.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    clear_identity_context,
    get_actor_id_from_context,
    get_attribution_subject_id_from_context,
    get_license_id_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
    identity_event_hook,
    service_cell_token_hook,
    service_cell_token_hook_sync,
    set_identity_context,
    set_w2v_job_context,
    set_w2v_project_context,
)
from w2v_common.auth.middleware import PUBLIC_ENDPOINTS, PUBLIC_PREFIX_BASES, AuthMiddleware
from w2v_common.discovery import cell_services, service_url
from w2v_common.tenant import VOUCHED_TENANT_SOURCES, TenantSource

DEPLOYMENT = "strato-w2v"
CALLEE = "analysis-service"
ORCH = "w2v-orchestrator"
BDD = "bdd-playwright-service"
WORKER = "tts-gateway-worker"
TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000c3111")
OTHER_TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000c3222")
ACTOR = "0198f1d2-0000-7000-8000-0000000ac701"
PROJECT = "0198f1d2-0000-7000-8000-0000000b0001"
RUN = "0198f1d2-0000-7000-8000-00000000a0a1"
SA_SUBJECT = "01a0dd02-0000-7000-8000-000000000001"
CUSTOMER_SUBJECT = "0198f1d2-0000-7000-8000-0000000c5501"


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


def _jwk(private: Ed25519PrivateKey, kid: str, iss: str) -> dict[str, Any]:
    entry = json.loads(OKPAlgorithm.to_jwk(private.public_key()))
    entry.update({"kid": kid, "iss": iss})
    return entry


@pytest.fixture(scope="module")
def orch_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(scope="module")
def bdd_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(scope="module")
def worker_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def jwks_file(
    tmp_path: Path,
    orch_key: Ed25519PrivateKey,
    bdd_key: Ed25519PrivateKey,
    worker_key: Ed25519PrivateKey,
) -> Path:
    path = tmp_path / "jwks.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    _jwk(orch_key, "orch-1", ORCH),
                    _jwk(bdd_key, "bdd-1", BDD),
                    _jwk(worker_key, "worker-1", WORKER),
                ]
            }
        )
    )
    return path


@pytest.fixture(autouse=True)
def cell_env(monkeypatch: pytest.MonkeyPatch, jwks_file: Path) -> Iterator[None]:
    """This process is the CALLEE (analysis-service) unless a test says otherwise."""
    for name in (
        "CELL_JWT_SIGNING_KEY",
        "CELL_JWT_KID",
        "CELL_ALLOWED_CALLERS",
        "CELL_AUTH_MODE",
        "AUTH_DELEGATING_SUBJECTS",
        "SERVICE_ACTOR_ID",
        "W2V_LOCAL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CELL_SERVICE_NAME", CALLEE)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks_file))
    # Since 0.91.1 unset/empty CELL_ALLOWED_CALLERS denies every cell caller;
    # tests that are not about the allow-list opt into "any keyring issuer".
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    reset_cell_state()
    clear_identity_context()
    yield
    reset_cell_state()
    clear_identity_context()


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ORCH,
        "aud": CALLEE,
        "cell": DEPLOYMENT,
        "tenant": str(TENANT),
        "run": RUN,
        "actor": ACTOR,
        "project": PROJECT,
        "iat": now,
        "nbf": now,
        "exp": now + 60,
        "jti": uuid.uuid7().hex,
    }
    for key, value in overrides.items():
        if value is _DROP:
            claims.pop(key, None)
        else:
            claims[key] = value
    return claims


_DROP = object()


def _token(
    key: Ed25519PrivateKey,
    *,
    kid: str = "orch-1",
    headers: dict[str, Any] | None = None,
    **overrides: Any,
) -> str:
    hdr = {"kid": kid, "typ": CELL_TOKEN_TYPE}
    hdr.update(headers or {})
    return jwt.encode(_claims(**overrides), key, algorithm="EdDSA", headers=hdr)


def _b64(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def _reason(token: str, **kwargs: Any) -> str:
    with pytest.raises(CellTokenError) as exc:
        verify_cell_token(token, **kwargs)
    return exc.value.reason


def _pem(key: Ed25519PrivateKey) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


# --------------------------------------------------------------------------- #
# verify_cell_token — accept + every rejection path
# --------------------------------------------------------------------------- #


def test_valid_token_round_trips_every_claim(orch_key: Ed25519PrivateKey) -> None:
    claims = verify_cell_token(_token(orch_key))
    assert claims.issuer == ORCH
    assert claims.audience == CALLEE
    assert claims.deployment == DEPLOYMENT
    assert claims.tenant_id == TENANT
    assert (claims.run, claims.actor, claims.project) == (RUN, ACTOR, PROJECT)


def test_null_run_actor_project_are_accepted(orch_key: Ed25519PrivateKey) -> None:
    claims = verify_cell_token(_token(orch_key, run=None, actor=None, project=None))
    assert (claims.run, claims.actor, claims.project) == (None, None, None)


def test_wrong_audience_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, aud="slide-service")) == "wrong_audience"


def test_audience_list_is_rejected_even_when_it_contains_us(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, aud=[CALLEE, "slide-service"])) == "wrong_audience"


def test_expired_token_is_rejected_after_skew(orch_key: Ed25519PrivateKey) -> None:
    token = _token(orch_key, iat=1_000_000, nbf=1_000_000, exp=1_000_060)
    assert _reason(token, now=1_000_066) == "expired"


def test_expiry_tolerates_five_seconds_of_skew(orch_key: Ed25519PrivateKey) -> None:
    token = _token(orch_key, iat=1_000_000, nbf=1_000_000, exp=1_000_060)
    assert verify_cell_token(token, now=1_000_065).issuer == ORCH


def test_future_token_is_rejected_beyond_skew(orch_key: Ed25519PrivateKey) -> None:
    token = _token(orch_key, iat=1_000_000, nbf=1_000_000, exp=1_000_060)
    assert _reason(token, now=1_000_000 - 6) == "not_yet_valid"


def test_lifetime_over_sixty_seconds_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    now = int(time.time())
    assert _reason(_token(orch_key, iat=now, nbf=now, exp=now + 3600)) == "bad_lifetime"


def test_unknown_kid_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, kid="nobody-1")) == "unknown_kid"


def test_kid_signing_as_a_foreign_issuer_is_rejected(bdd_key: Ed25519PrivateKey) -> None:
    """bdd's own key, bdd's own kid, but claiming to BE the orchestrator."""
    assert _reason(_token(bdd_key, kid="bdd-1", iss=ORCH)) == "issuer_mismatch"


def test_key_presented_under_another_services_kid_fails_signature(
    bdd_key: Ed25519PrivateKey,
) -> None:
    assert _reason(_token(bdd_key, kid="orch-1")) == "bad_signature"


def test_wrong_deployment_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, cell="other-w2v")) == "wrong_deployment"


def test_replay_of_the_same_token_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    token = _token(orch_key)
    verify_cell_token(token)
    assert _reason(token) == "replay"


def test_a_forged_token_does_not_poison_the_replay_cache(
    orch_key: Ed25519PrivateKey, bdd_key: Ed25519PrivateKey
) -> None:
    jti = uuid.uuid7().hex
    assert _reason(_token(bdd_key, kid="orch-1", jti=jti)) == "bad_signature"
    assert verify_cell_token(_token(orch_key, jti=jti)).jti == jti


def test_alg_none_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    unsigned = (
        _b64({"alg": "none", "typ": CELL_TOKEN_TYPE, "kid": "orch-1"}) + "." + _b64(_claims()) + "."
    )
    assert _reason(unsigned) == "bad_alg"


def test_hs256_with_the_public_key_as_secret_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    """Classic key confusion: HMAC keyed with the (public) verification key."""
    public_raw = orch_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    forged = jwt.encode(
        _claims(),
        public_raw,
        algorithm="HS256",
        headers={"kid": "orch-1", "typ": CELL_TOKEN_TYPE},
    )
    assert _reason(forged) == "bad_alg"


def test_rs256_is_rejected() -> None:
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(
        _claims(), rsa_key, algorithm="RS256", headers={"kid": "orch-1", "typ": CELL_TOKEN_TYPE}
    )
    assert _reason(forged) == "bad_alg"


def test_tampered_claims_fail_the_signature(orch_key: Ed25519PrivateKey) -> None:
    header, _payload, signature = _token(orch_key).split(".")
    tampered = f"{header}.{_b64(_claims(tenant=str(OTHER_TENANT)))}.{signature}"
    assert _reason(tampered) == "bad_signature"


def test_wrong_typ_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, headers={"typ": "JWT"})) == "bad_type"


def test_embedded_jwk_header_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, headers={"jku": "https://evil.test/jwks"})) == "bad_header"


@pytest.mark.parametrize("claim", ["tenant", "jti", "cell", "exp", "aud"])
def test_missing_required_claim_is_rejected(orch_key: Ed25519PrivateKey, claim: str) -> None:
    drop: dict[str, Any] = {claim: _DROP}
    assert _reason(_token(orch_key, **drop)) == "missing_claim"


def test_nil_tenant_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, tenant=str(uuid.UUID(int=0)))) == "invalid_tenant"


def test_non_uuid_actor_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    assert _reason(_token(orch_key, actor="svc-w2v")) == "invalid_claim"


def test_malformed_token_is_rejected() -> None:
    assert _reason("not-a-jwt") == "malformed"
    assert _reason("a.b.c") == "malformed"
    assert _reason("x" * 5000) == "malformed"


def test_allowed_callers_restricts_issuers(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    bdd_key: Ed25519PrivateKey,
) -> None:
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", f"{ORCH}, tts-gateway-worker")
    assert verify_cell_token(_token(orch_key)).issuer == ORCH
    assert _reason(_token(bdd_key, kid="bdd-1", iss=BDD)) == "caller_not_allowed"


def test_unset_allowed_callers_denies_every_caller(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """0.91.0 read unset as "any keyring issuer" — fail-open. Now: deny all."""
    monkeypatch.delenv("CELL_ALLOWED_CALLERS")
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    with caplog.at_level(logging.WARNING, logger="w2v_common.auth.cell_jwt"):
        assert _reason(_token(orch_key)) == "caller_not_allowed"
        assert _reason(_token(orch_key)) == "caller_not_allowed"
    warnings = [r for r in caplog.records if "CELL_ALLOWED_CALLERS" in r.getMessage()]
    assert len(warnings) == 1, "warned once per process, not per request"
    assert warnings[0].levelno == logging.WARNING
    assert CALLEE in warnings[0].getMessage(), "the warning names the service"


@pytest.mark.parametrize("raw", ["", "  ", " , ,"])
def test_empty_allowed_callers_denies_every_caller(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, raw: str
) -> None:
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", raw)
    assert _reason(_token(orch_key)) == "caller_not_allowed"


def test_star_allowed_callers_accepts_any_keyring_issuer(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    bdd_key: Ed25519PrivateKey,
) -> None:
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    assert verify_cell_token(_token(orch_key)).issuer == ORCH
    assert verify_cell_token(_token(bdd_key, kid="bdd-1", iss=BDD)).issuer == BDD


def test_no_in_cell_callers_sentinel_denies_every_caller(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    bdd_key: Ed25519PrivateKey,
) -> None:
    """gitops sets this on the orchestrator (no in-cell caller may reach it)."""
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "_no-in-cell-callers_")
    assert _reason(_token(orch_key)) == "caller_not_allowed"
    assert _reason(_token(bdd_key, kid="bdd-1", iss=BDD)) == "caller_not_allowed"


def test_middleware_init_warns_when_no_cell_callers_configured(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from w2v_common.auth.middleware import AuthMiddleware

    async def _app(scope: object, receive: object, send: object) -> None:  # pragma: no cover
        return None

    monkeypatch.delenv("CELL_ALLOWED_CALLERS")
    monkeypatch.setenv("CELL_AUTH_MODE", "off")
    with caplog.at_level(logging.WARNING, logger="w2v_common.auth.cell_jwt"):
        AuthMiddleware(_app)
    assert not [r for r in caplog.records if "CELL_ALLOWED_CALLERS" in r.getMessage()]
    for mode in ("dual", "enforce"):
        reset_cell_state()
        caplog.clear()
        monkeypatch.setenv("CELL_AUTH_MODE", mode)
        with caplog.at_level(logging.WARNING, logger="w2v_common.auth.cell_jwt"):
            AuthMiddleware(_app)
            AuthMiddleware(_app)
        hits = [r for r in caplog.records if "CELL_ALLOWED_CALLERS" in r.getMessage()]
        assert len(hits) == 1 and CALLEE in hits[0].getMessage(), mode


def test_unconfigured_receiver_rejects(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.delenv("CELL_DEPLOYMENT_ID")
    assert _reason(_token(orch_key)) == "not_configured"


def test_missing_keyring_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_JWKS_PATH", str(tmp_path / "absent.json"))
    reset_cell_state()
    assert _reason(_token(orch_key)) == "keyring_unavailable"


# --------------------------------------------------------------------------- #
# Keyring hot reload
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_keyring_reload_adds_a_kid_without_restart(
    tmp_path: Path, orch_key: Ed25519PrivateKey
) -> None:
    new_key = Ed25519PrivateKey.generate()
    path = tmp_path / "ring.json"
    path.write_text(json.dumps({"keys": [_jwk(orch_key, "orch-1", ORCH)]}))
    clock = _Clock()
    ring = CellKeyring(str(path), reload_seconds=60, clock=clock)
    token = _token(new_key, kid="orch-2")
    with pytest.raises(CellTokenError, match="unknown_kid"):
        verify_cell_token(token, keyring=ring)

    path.write_text(
        json.dumps({"keys": [_jwk(orch_key, "orch-1", ORCH), _jwk(new_key, "orch-2", ORCH)]})
    )
    clock.t += 59
    with pytest.raises(CellTokenError, match="unknown_kid"):
        verify_cell_token(token, keyring=ring)  # not re-read yet
    clock.t += 1
    assert verify_cell_token(token, keyring=ring).issuer == ORCH


def test_corrupt_keyring_keeps_the_last_good_one(
    tmp_path: Path, orch_key: Ed25519PrivateKey
) -> None:
    path = tmp_path / "ring.json"
    path.write_text(json.dumps({"keys": [_jwk(orch_key, "orch-1", ORCH)]}))
    clock = _Clock()
    ring = CellKeyring(str(path), reload_seconds=60, clock=clock)
    assert ring.kids() == {"orch-1"}
    before = _sample("w2v_cell_jwks_reload_total", {"result": "invalid"})
    path.write_text("{not json")
    clock.t += 60
    assert verify_cell_token(_token(orch_key), keyring=ring).issuer == ORCH
    assert _sample("w2v_cell_jwks_reload_total", {"result": "invalid"}) == before + 1


def test_deleted_keyring_file_empties_the_ring(tmp_path: Path, orch_key: Ed25519PrivateKey) -> None:
    path = tmp_path / "ring.json"
    path.write_text(json.dumps({"keys": [_jwk(orch_key, "orch-1", ORCH)]}))
    clock = _Clock()
    ring = CellKeyring(str(path), reload_seconds=60, clock=clock)
    assert ring.available()
    path.unlink()
    clock.t += 60
    with pytest.raises(CellTokenError, match="keyring_unavailable"):
        verify_cell_token(_token(orch_key), keyring=ring)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update({"d": "c2VjcmV0"}),
        lambda e: e.update({"crv": "Ed448"}),
        lambda e: e.update({"alg": "HS256"}),
        lambda e: e.pop("iss"),
        lambda e: e.update({"x": "AAAA"}),
    ],
    ids=["private-material", "ed448", "foreign-alg", "no-iss", "short-key"],
)
def test_parse_jwks_rejects_off_spec_entries(orch_key: Ed25519PrivateKey, mutate: Any) -> None:
    entry = _jwk(orch_key, "orch-1", ORCH)
    mutate(entry)
    with pytest.raises(ValueError):
        parse_jwks({"keys": [entry]})


def test_parse_jwks_rejects_duplicate_kids(orch_key: Ed25519PrivateKey) -> None:
    entry = _jwk(orch_key, "orch-1", ORCH)
    with pytest.raises(ValueError, match="duplicate"):
        parse_jwks({"keys": [entry, dict(entry)]})


# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "mode"),
    [
        (None, CellAuthMode.OFF),
        ("", CellAuthMode.OFF),
        ("dual", CellAuthMode.DUAL),
        ("ENFORCE", CellAuthMode.ENFORCE),
        ("enforced", CellAuthMode.ENFORCE),  # typo fails CLOSED
    ],
)
def test_cell_auth_mode(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, mode: CellAuthMode
) -> None:
    if raw is None:
        monkeypatch.delenv("CELL_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("CELL_AUTH_MODE", raw)
    assert cell_auth_mode() is mode


# --------------------------------------------------------------------------- #
# Signer + outbound hook
# --------------------------------------------------------------------------- #


@pytest.fixture
def as_orchestrator(monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey) -> None:
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(orch_key))
    monkeypatch.setenv("CELL_JWT_KID", "orch-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", ORCH)


def _verify_as(monkeypatch: pytest.MonkeyPatch, token: str, callee: str = CALLEE) -> Any:
    monkeypatch.setenv("CELL_SERVICE_NAME", callee)
    return verify_cell_token(token)


def test_mint_then_verify(monkeypatch: pytest.MonkeyPatch, as_orchestrator: None) -> None:
    token = mint_cell_token(CALLEE, tenant=TENANT, run=RUN, actor=ACTOR, project=PROJECT)
    assert jwt.get_unverified_header(token) == {
        "alg": "EdDSA",
        "kid": "orch-1",
        "typ": CELL_TOKEN_TYPE,
    }
    claims = _verify_as(monkeypatch, token)
    assert claims.issuer == ORCH
    assert claims.expires_at - claims.issued_at == 60


def test_mint_refuses_a_nil_tenant(as_orchestrator: None) -> None:
    with pytest.raises(ValueError):
        mint_cell_token(CALLEE, tenant=uuid.UUID(int=0))


def test_half_configured_signer_is_loud(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    monkeypatch.delenv("CELL_JWT_KID")
    with pytest.raises(CellConfigError, match="CELL_JWT_KID"):
        mint_cell_token(CALLEE, tenant=TENANT)


def test_non_ed25519_signing_key_is_loud(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    rsa_pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", rsa_pem)
    with pytest.raises(CellConfigError, match="Ed25519"):
        mint_cell_token(CALLEE, tenant=TENANT)


def _request(url: str, headers: dict[str, str] | None = None) -> httpx.Request:
    return httpx.Request("POST", url, headers=headers or {})


async def test_hook_stamps_a_token_carrying_the_context_claims(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM, actor_id=ACTOR)
    set_w2v_project_context(PROJECT)
    set_w2v_job_context(RUN)
    request = _request(f"{service_url('analysis')}/api/analyze")
    await identity_event_hook(request)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER])
    assert claims.tenant_id == TENANT
    assert (claims.actor, claims.project, claims.run) == (ACTOR, PROJECT, RUN)
    assert claims.audience == CALLEE
    # dual-send: the legacy tenant header still goes out
    assert request.headers["X-Tenant-Id"] == str(TENANT)


async def test_every_send_mints_a_fresh_token(as_orchestrator: None) -> None:
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    first = _request(f"{service_url('slide_service')}/api/x")
    second = _request(f"{service_url('slide_service')}/api/x")
    await identity_event_hook(first)
    await identity_event_hook(second)
    jti_1 = jwt.decode(first.headers[CELL_TOKEN_HEADER], options={"verify_signature": False})
    jti_2 = jwt.decode(second.headers[CELL_TOKEN_HEADER], options={"verify_signature": False})
    assert jti_1["jti"] != jti_2["jti"]
    assert jti_1["aud"] == "slide-service"


async def test_resending_the_same_request_replaces_the_token(as_orchestrator: None) -> None:
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    request = _request(f"{service_url('analysis')}/api/x")
    await identity_event_hook(request)
    first = request.headers[CELL_TOKEN_HEADER]
    await identity_event_hook(request)
    assert request.headers[CELL_TOKEN_HEADER] != first
    assert len(request.headers.get_list(CELL_TOKEN_HEADER)) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://worker-management.workers:8887/workers/x/message",
        "http://webscraper.webscraper:9119/api/scrape/url",
        "http://tts.example:9202/api/tts/synthesize",
        "https://example.com/hook",
        # right host, wrong port: not the peer
        "http://analysis-service.w2v:9999/api/x",
    ],
)
async def test_out_of_cell_destinations_never_get_a_token(as_orchestrator: None, url: str) -> None:
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    request = _request(url, headers={CELL_TOKEN_HEADER: "copied-from-inbound"})
    await identity_event_hook(request)
    assert CELL_TOKEN_HEADER not in request.headers


@pytest.mark.parametrize(
    "source", [None, TenantSource.UNVOUCHED_HEADER, TenantSource.AUTH_DISABLED]
)
async def test_an_unvouched_tenant_is_never_signed_into_a_cell_token(
    as_orchestrator: None, source: TenantSource | None
) -> None:
    """A cell token makes the tenant VOUCHED (``CELL_JWT``) one hop later.

    Signing a tenant this service could not itself delegate (a reaper that
    dropped provenance, a bare-header lane) would launder it: the receiver
    could then spend it at a metered peer. No token is minted; the legacy
    headers still go out, so ``dual`` behaves as before and ``enforce`` 401s.
    """
    set_identity_context(TENANT, None, None, tenant_source=source)
    request = _request(f"{service_url('content')}/api/x")
    await identity_event_hook(request)
    assert CELL_TOKEN_HEADER not in request.headers
    assert request.headers["X-Tenant-Id"] == str(TENANT)
    assert _sample("w2v_cell_token_unminted_total", {"reason": "unvouched_tenant"}) >= 1


async def test_no_signing_key_means_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode-off contract: consumers without keys (modality_gateway & co) are untouched."""
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    request = _request(f"{service_url('analysis')}/api/x")
    await identity_event_hook(request)
    assert CELL_TOKEN_HEADER not in request.headers


async def test_service_actor_never_enters_a_cell_token(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", "e7330ae1-0000-7000-8000-000000000001")
    set_identity_context(
        TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM
    )  # service-originated, no human actor
    request = _request(f"{service_url('analysis')}/api/x")
    await identity_event_hook(request)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER])
    assert claims.actor is None


def test_audience_is_resolved_through_discovery_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("W2V_LOCAL", "1")
    assert cell_audience_for_url("http://localhost:6303/api/x") == "analysis-service"
    assert cell_audience_for_url("http://localhost:6320/api/x") == "tts-gateway"
    assert cell_audience_for_url("http://localhost:8887/api/x") is None
    monkeypatch.setenv("W2V_SLIDE_SERVICE_URL", "http://slides.test:1234")
    assert cell_audience_for_url("http://slides.test:1234/api/x") == "slide-service"


def test_cell_services_are_exactly_the_w2v_namespace() -> None:
    peers = cell_services()
    assert peers["orchestrator"] == "w2v-orchestrator"
    assert "worker_management" not in peers
    assert "tts" not in peers  # tailnet host service, out of cell


#: Every w2v-namespace peer and the cell identity (``aud``) it must keep. Pinned
#: literally so admitting video-compiler cannot silently re-key an existing peer.
_W2V_NAMESPACE_PEERS: dict[str, str] = {
    "orchestrator": "w2v-orchestrator",
    "analysis": "analysis-service",
    "html_render": "html-render-service",
    "content": "content-service",
    "video_production": "video-production-service",
    "tts_gateway": "tts-gateway",
    "ai_content_generation": "ai-content-generation-service",
    "app_inspector": "app-inspector-service",
    "slide_service": "slide-service",
    "bdd_playwright": "bdd-playwright-service",
    "course": "course-service",
    "youtube_upload": "youtube-upload-service",
    "frontend_test_runner": "frontend-test-runner-service",
    "video_analyzer": "video-analyzer-service",
}


def test_video_compiler_is_an_explicit_cell_member_outside_the_w2v_namespace() -> None:
    # video-compiler lives in namespace ``video-compiler``, not ``w2v``. Without
    # an explicit membership the orchestrator never mints it a cell token and it
    # can never run CELL_AUTH_MODE=enforce.
    assert cell_services()["video_compile"] == "video-compiler"
    url = service_url("video_compile")
    assert url == "http://video-compiler.video-compiler:8200"
    assert cell_audience_for_url(f"{url}/api/compile") == "video-compiler"


def test_video_compiler_audience_honours_discovery_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("W2V_LOCAL", "1")
    assert cell_audience_for_url("http://localhost:8200/api/compile") == "video-compiler"
    monkeypatch.setenv("W2V_VIDEO_COMPILE_URL", "http://vc.test:9999")
    assert cell_audience_for_url("http://vc.test:9999/api/compile") == "video-compiler"


@pytest.mark.parametrize(
    "name",
    [
        "inference_router",
        "licenses",
        "webscraper",
        "media_discovery",
        "worker_management",
        "artifacts",
        "modality_gateway",
    ],
)
def test_other_non_w2v_namespace_services_stay_out_of_cell(name: str) -> None:
    # Admitting video-compiler must not widen the namespace rule: every other
    # cross-namespace platform service still gets NO cell token.
    assert name not in cell_services()
    assert cell_audience_for_url(f"{service_url(name)}/api/x") is None


def test_cell_membership_is_the_w2v_namespace_plus_exactly_video_compiler() -> None:
    assert cell_services() == {**_W2V_NAMESPACE_PEERS, "video_compile": "video-compiler"}


@pytest.mark.parametrize(("name", "aud"), sorted(_W2V_NAMESPACE_PEERS.items()))
def test_existing_w2v_peer_audiences_are_unchanged(name: str, aud: str) -> None:
    assert cell_audience_for_url(f"{service_url(name)}/api/x") == aud


async def test_hook_mints_a_video_compiler_audience_token(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM, actor_id=ACTOR)
    request = _request(f"{service_url('video_compile')}/api/compile")
    await identity_event_hook(request)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER], callee="video-compiler")
    assert claims.audience == "video-compiler"
    assert claims.issuer == ORCH
    assert claims.tenant_id == TENANT


def test_cell_jwt_is_a_vouched_source() -> None:
    assert TenantSource.CELL_JWT in VOUCHED_TENANT_SOURCES


# --------------------------------------------------------------------------- #
# AuthMiddleware lane
# --------------------------------------------------------------------------- #


async def _whoami(request: Request) -> dict[str, Any]:
    tenant = get_tenant_id_from_context()
    source = get_tenant_source_from_context()
    return {
        "tenant": str(tenant) if tenant else None,
        "source": source.value if source else None,
        "cell_caller": getattr(request.state, "cell_caller", None),
        "actor": get_actor_id_from_context(),
        "subject": get_subject_id_from_context(),
        "w2v_project": get_w2v_project_id_from_context(),
        "w2v_job": get_w2v_job_id_from_context(),
        "delegated": getattr(request.state, "tenant_delegated", None),
    }


async def _whoami_attribution(request: Request) -> dict[str, Any]:
    body = await _whoami(request)
    body["attribution_subject"] = get_attribution_subject_id_from_context()
    body["license_id"] = get_license_id_from_context()
    return body


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/whoami", _whoami, methods=["GET"])
    app.add_api_route("/api/whoami-attr", _whoami_attribution, methods=["GET"])
    return app


def _introspection(tenant: uuid.UUID, subject: str) -> dict[str, Any]:
    return {"active": True, "tenant_id": str(tenant), "subject_id": subject}


async def _call(
    headers: dict[str, str],
    introspection: dict[str, Any] | None = None,
    path: str = "/api/whoami",
) -> tuple[int, dict[str, Any], AsyncMock]:
    with patch(
        "w2v_common.auth.middleware._introspect_license",
        new_callable=AsyncMock,
        return_value=introspection,
    ) as introspect:
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://t") as c:
            resp = await c.get(path, headers=headers)
    return resp.status_code, resp.json(), introspect


@pytest.mark.parametrize("mode", ["dual", "enforce"])
async def test_valid_cell_token_lands_on_the_cell_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, mode: str
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    before = _sample("w2v_auth_license_lane_total", {"lane": "cell-jwt"})
    status, body, introspect = await _call(
        {
            CELL_TOKEN_HEADER: _token(orch_key),
            "X-W2V-Project-Id": "0198f1d2-0000-7000-8000-0000000bad01",  # signed claim wins
            "X-Actor-Id": "0198f1d2-0000-7000-8000-0000000bad02",  # never from a header
            "X-Subject-Id": SA_SUBJECT,
        }
    )
    assert status == 200, body
    assert body == {
        "tenant": str(TENANT),
        "source": "cell-jwt",
        "cell_caller": ORCH,
        "actor": ACTOR,
        # 0.93.0: this token (no signed subject claim = a 0.92.x minter) gets
        # its subject from X-Subject-Id as ATTRIBUTION ONLY — never the
        # authz-visible subject. See the signed-attribution tests below.
        "subject": "",
        "w2v_project": PROJECT,
        "w2v_job": RUN,
        "delegated": False,
    }
    introspect.assert_not_awaited()
    assert _sample("w2v_auth_license_lane_total", {"lane": "cell-jwt"}) == before + 1


async def test_dual_invalid_token_falls_through_to_the_licence_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    before = _sample("w2v_cell_jwt_rejected_total", {"reason": "wrong_audience", "scope": "tenant"})
    status, body, introspect = await _call(
        {
            CELL_TOKEN_HEADER: _token(orch_key, aud="slide-service"),
            "X-Tenant-Id": "Bearer customer-licence",
        },
        _introspection(OTHER_TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 200
    assert body["source"] == "license-token"
    assert body["tenant"] == str(OTHER_TENANT)
    assert body["cell_caller"] is None
    introspect.assert_awaited_once()
    assert (
        _sample("w2v_cell_jwt_rejected_total", {"reason": "wrong_audience", "scope": "tenant"})
        == before + 1
    )


async def test_dual_unset_allowed_callers_falls_through_not_401(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    """0.91.1 fail-closed allow-list must not 401 live traffic during ``dual``."""
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    monkeypatch.delenv("CELL_ALLOWED_CALLERS")
    before = _sample(
        "w2v_cell_jwt_rejected_total", {"reason": "caller_not_allowed", "scope": "tenant"}
    )
    status, body, introspect = await _call(
        {CELL_TOKEN_HEADER: _token(orch_key), "X-Tenant-Id": "Bearer customer-licence"},
        _introspection(OTHER_TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 200, body
    assert body["source"] == "license-token"
    assert body["cell_caller"] is None
    introspect.assert_awaited_once()
    assert (
        _sample("w2v_cell_jwt_rejected_total", {"reason": "caller_not_allowed", "scope": "tenant"})
        == before + 1
    )


async def test_enforce_invalid_token_is_401_without_introspection(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    expired = _token(orch_key, iat=1_000_000, nbf=1_000_000, exp=1_000_060)
    status, body, introspect = await _call(
        {CELL_TOKEN_HEADER: expired, "X-Tenant-Id": "Bearer customer-licence"},
        _introspection(OTHER_TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 401
    assert body["error"]["code"] == "CELL_TOKEN_INVALID"
    assert expired not in json.dumps(body)
    introspect.assert_not_awaited()


async def test_enforce_refuses_licence_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    status, body, introspect = await _call(
        {"X-Tenant-Id": "Bearer sa-licence", EFFECTIVE_TENANT_HEADER: str(TENANT)},
        _introspection(OTHER_TENANT, SA_SUBJECT),
    )
    assert status == 401
    assert body["error"]["code"] == "CELL_TOKEN_REQUIRED"
    introspect.assert_not_awaited()


async def test_enforce_refuses_a_delegating_subject_without_the_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    status, body, _ = await _call(
        {"X-Tenant-Id": "Bearer sa-licence"}, _introspection(OTHER_TENANT, SA_SUBJECT)
    )
    assert status == 401
    assert body["error"]["code"] == "CELL_TOKEN_REQUIRED"


async def test_enforce_accepts_a_plain_customer_licence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    status, body, _ = await _call(
        {"X-Tenant-Id": "Bearer customer-licence"},
        _introspection(OTHER_TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 200
    assert body["source"] == "license-token"


async def test_dual_counts_legacy_delegation_per_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    before = _sample("w2v_cell_legacy_delegation_total", {"caller": SA_SUBJECT})
    status, body, _ = await _call(
        {"X-Tenant-Id": "Bearer sa-licence", EFFECTIVE_TENANT_HEADER: str(TENANT)},
        _introspection(OTHER_TENANT, SA_SUBJECT),
    )
    assert status == 200
    assert body["tenant"] == str(TENANT)  # dual: legacy delegation still works
    assert body["delegated"] is True
    assert _sample("w2v_cell_legacy_delegation_total", {"caller": SA_SUBJECT}) == before + 1


async def test_customer_licence_is_not_counted_as_legacy_cell_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    before = _sample("w2v_cell_legacy_delegation_total", {"caller": "unlisted"})
    await _call({"X-Tenant-Id": "Bearer c"}, _introspection(OTHER_TENANT, CUSTOMER_SUBJECT))
    assert _sample("w2v_cell_legacy_delegation_total", {"caller": "unlisted"}) == before


async def test_mode_off_ignores_the_cell_header_entirely(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.delenv("CELL_AUTH_MODE", raising=False)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    with patch("w2v_common.auth.middleware.verify_cell_token") as verifier:
        status, _, _ = await _call({CELL_TOKEN_HEADER: _token(orch_key)})
        assert status == 401  # no other credential → the old answer
        status, body, _ = await _call(
            {"X-Tenant-Id": "Bearer sa-licence", EFFECTIVE_TENANT_HEADER: str(TENANT)},
            _introspection(OTHER_TENANT, SA_SUBJECT),
        )
    verifier.assert_not_called()
    assert status == 200
    assert body["tenant"] == str(TENANT)
    assert body["source"] == "license-token"


async def test_enforce_with_missing_keyring_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("CELL_JWKS_PATH", str(tmp_path / "gone.json"))
    reset_cell_state()
    status, body, _ = await _call({CELL_TOKEN_HEADER: _token(orch_key)})
    assert status == 401
    assert "keyring_unavailable" in body["error"]["message"]


async def test_dual_with_missing_keyring_counts_and_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    monkeypatch.setenv("CELL_JWKS_PATH", str(tmp_path / "gone.json"))
    reset_cell_state()
    before = _sample(
        "w2v_cell_jwt_rejected_total", {"reason": "keyring_unavailable", "scope": "unknown"}
    )
    status, body, _ = await _call(
        {CELL_TOKEN_HEADER: _token(orch_key), "X-Tenant-Id": "Bearer c"},
        _introspection(OTHER_TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 200
    assert body["source"] == "license-token"
    assert (
        _sample(
            "w2v_cell_jwt_rejected_total", {"reason": "keyring_unavailable", "scope": "unknown"}
        )
        == before + 1
    )


async def test_rejected_token_is_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    token = _token(orch_key, aud="slide-service")
    with caplog.at_level("DEBUG"):
        await _call({CELL_TOKEN_HEADER: token})
    assert "wrong_audience" in caplog.text
    assert token not in caplog.text
    assert token.split(".")[2] not in caplog.text


async def test_cell_lane_tenant_may_be_delegated_onward(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    """The callee's own metered hop (SA bearer) must not raise UnvouchedTenantError."""
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    seen: dict[str, str] = {}

    async def _onward(request: Request) -> dict[str, str]:
        hop = httpx.Request(
            "POST",
            f"{service_url('worker_management')}/workers/x/message",
            headers={"X-Tenant-Id": "Bearer callee-sa-licence"},
        )
        await identity_event_hook(hop)
        seen.update(hop.headers)
        return {}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/onward", _onward, methods=["GET"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/onward", headers={CELL_TOKEN_HEADER: _token(orch_key)})
    assert resp.status_code == 200
    assert seen[EFFECTIVE_TENANT_HEADER.lower()] == str(TENANT)
    assert CELL_TOKEN_HEADER.lower() not in seen  # WM is out of cell


async def test_licence_gate_is_skipped_on_the_cell_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    scope = {"type": "http", "method": "GET", "path": "/api/x", "headers": [], "state": {}}
    request = Request(scope)
    request.state.cell_caller = ORCH
    request.state.tenant_id = str(TENANT)
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.CELL_JWT)
    try:
        with patch(
            "w2v_common.auth.entitlement.get_license_status", new_callable=AsyncMock
        ) as status:
            await require_license(request)
    finally:
        clear_identity_context()
    status.assert_not_awaited()


async def test_inbound_marker_is_set_on_the_cell_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    """Service-originated fallback must stay off for cell-lane work too."""
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    from w2v_common.auth.identity_context import is_inbound_request

    seen: list[bool] = []

    async def _probe(request: Request) -> dict[str, bool]:
        seen.append(is_inbound_request())
        return {}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/probe", _probe, methods=["GET"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/probe", headers={CELL_TOKEN_HEADER: _token(orch_key)})
    assert resp.status_code == 200
    assert seen == [True]


async def test_half_configured_signer_fails_only_in_cell_calls(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    monkeypatch.delenv("CELL_JWT_KID")
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    out_of_cell = _request("http://webscraper.webscraper:9119/api/scrape/url")
    await identity_event_hook(out_of_cell)  # unaffected
    assert CELL_TOKEN_HEADER not in out_of_cell.headers
    with pytest.raises(CellConfigError):
        await identity_event_hook(_request(f"{service_url('analysis')}/api/x"))


# --------------------------------------------------------------------------- #
# 0.91.2 — ``dual`` keeps the licence lane when legacy credentials are present
#
# On 0.91.1 a valid cell token won BEFORE the licence lane, so in ``dual``
# request.state.license / license_token and the outbound caller_license_token
# were left unset for every in-cell call that ALSO carried a licence — and
# consumers on main still read them (youtube_upload caller_is_orchestrator_sa,
# tts_gateway worker gate, KAG/artifact clients forwarding caller_license_token).
# ``dual`` must be behaviour-preserving: the licence lane runs exactly as in
# ``off``, and the verified cell caller is recorded alongside it.
# --------------------------------------------------------------------------- #


_CELL_ONLY_STATE = {"cell_caller", "cell_claims", "cell_scope", "cell_auth_lane"}


def _state_app(seen: list[dict[str, Any]]) -> FastAPI:
    from w2v_common.auth.identity_context import (
        get_caller_license_token,
        get_license_id_from_context,
    )

    async def _dump(request: Request) -> dict[str, Any]:
        state = dict(request.state._state)
        tenant = get_tenant_id_from_context()
        source = get_tenant_source_from_context()
        state["ctx"] = {
            "tenant": str(tenant) if tenant else None,
            "source": source.value if source else None,
            "license_id": get_license_id_from_context(),
            "subject": get_subject_id_from_context(),
            "actor": get_actor_id_from_context(),
            "caller_license_token": get_caller_license_token(),
            "w2v_project": get_w2v_project_id_from_context(),
            "w2v_job": get_w2v_job_id_from_context(),
        }
        seen.append(state)
        return {}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/state", _dump, methods=["GET"])
    return app


async def _state_call(
    headers: dict[str, str], introspection: dict[str, Any] | None
) -> tuple[int, dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    with patch(
        "w2v_common.auth.middleware._introspect_license",
        new_callable=AsyncMock,
        return_value=introspection,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=_state_app(seen)), base_url="http://t"
        ) as c:
            resp = await c.get("/api/state", headers=headers)
    assert resp.status_code == 200, resp.text
    assert len(seen) == 1
    return resp.status_code, seen[0]


@pytest.mark.parametrize(
    ("delegating", "legacy_headers"),
    [
        # the orchestrator SA delegating a run's tenant (the live in-cell shape)
        (True, {"X-Tenant-Id": "Bearer sa-licence", EFFECTIVE_TENANT_HEADER: str(TENANT)}),
        # platform-shape licence, no delegation
        (False, {"Authorization": "Bearer opaque-sa-licence"}),
    ],
)
async def test_dual_with_both_credentials_keeps_every_licence_lane_field(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    delegating: bool,
    legacy_headers: dict[str, str],
) -> None:
    if delegating:
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    sa_tenant = OTHER_TENANT if delegating else TENANT
    introspection = {
        "active": True,
        "tenant_id": str(sa_tenant),
        "subject_id": SA_SUBJECT,
        "license_id": "lic-sa",
    }
    extra = {"X-Subject-Id": SA_SUBJECT, "X-W2V-Project-Id": PROJECT}

    monkeypatch.delenv("CELL_AUTH_MODE", raising=False)
    _, off_state = await _state_call({**legacy_headers, **extra}, introspection)

    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    both_before = _sample("w2v_cell_dual_both_total", {"caller": ORCH, "scope": "tenant"})
    legacy_before = _sample("w2v_cell_legacy_delegation_total", {"caller": SA_SUBJECT})
    _, dual_state = await _state_call(
        {CELL_TOKEN_HEADER: _token(orch_key), **legacy_headers, **extra}, introspection
    )

    # Every request.state field (and the identity ContextVars) identical to mode off.
    assert {k: v for k, v in dual_state.items() if k not in _CELL_ONLY_STATE} == off_state
    assert dual_state["license"] == introspection
    assert dual_state["license_token"] in {"sa-licence", "opaque-sa-licence"}
    assert dual_state["ctx"]["caller_license_token"] == dual_state["license_token"]
    assert dual_state["ctx"]["source"] == "license-token"
    # ...plus the verified cell caller, recorded alongside.
    assert dual_state["cell_caller"] == ORCH
    assert str(dual_state["cell_claims"].tenant_id) == str(TENANT)
    assert (
        _sample("w2v_cell_dual_both_total", {"caller": ORCH, "scope": "tenant"}) == both_before + 1
    )
    # A request WITH a valid cell token is not legacy-only traffic: no flip signal.
    assert _sample("w2v_cell_legacy_delegation_total", {"caller": SA_SUBJECT}) == legacy_before


async def test_dual_with_both_credentials_keeps_the_jwt_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    """A forwarded user JWT keeps winning over the cell lane in ``dual`` too."""
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    validator = AsyncMock()
    validator.validate.return_value = (
        {"sub": CUSTOMER_SUBJECT, "tenant_id": str(TENANT), "preferred_username": "u"},
        None,
    )
    with patch("w2v_common.auth.middleware.get_jwt_validator", return_value=validator):
        _, state = await _state_call(
            {CELL_TOKEN_HEADER: _token(orch_key), "Authorization": "Bearer a.b.c"}, None
        )
    assert state["ctx"]["source"] == "jwt-claim"
    assert state["ctx"]["actor"] == CUSTOMER_SUBJECT
    assert state["cell_caller"] == ORCH


async def test_dual_legacy_only_counts_the_flip_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)
    before = _sample("w2v_cell_legacy_delegation_total", {"caller": SA_SUBJECT})
    both_before = _sample("w2v_cell_dual_both_total", {"caller": ORCH, "scope": "tenant"})
    _, state = await _state_call(
        {"X-Tenant-Id": "Bearer sa-licence", EFFECTIVE_TENANT_HEADER: str(TENANT)},
        _introspection(OTHER_TENANT, SA_SUBJECT),
    )
    assert state["license_token"] == "sa-licence"
    assert "cell_caller" not in state
    assert _sample("w2v_cell_legacy_delegation_total", {"caller": SA_SUBJECT}) == before + 1
    assert _sample("w2v_cell_dual_both_total", {"caller": ORCH, "scope": "tenant"}) == both_before


async def test_dual_cell_only_is_accepted_on_the_cell_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    both_before = _sample("w2v_cell_dual_both_total", {"caller": ORCH, "scope": "tenant"})
    _, state = await _state_call({CELL_TOKEN_HEADER: _token(orch_key)}, None)
    assert state["ctx"]["source"] == "cell-jwt"
    assert state["cell_caller"] == ORCH
    assert state["license"] is None
    assert state["license_token"] is None
    assert _sample("w2v_cell_dual_both_total", {"caller": ORCH, "scope": "tenant"}) == both_before


async def test_dual_tenant_mismatch_keeps_legacy_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    before = _sample(
        "w2v_cell_jwt_rejected_total", {"reason": "tenant_mismatch", "scope": "tenant"}
    )
    with caplog.at_level(logging.WARNING):
        _, state = await _state_call(
            {CELL_TOKEN_HEADER: _token(orch_key), "X-Tenant-Id": "Bearer customer-licence"},
            _introspection(OTHER_TENANT, CUSTOMER_SUBJECT),
        )
    assert state["tenant_id"] == str(OTHER_TENANT)  # legacy wins
    assert state["ctx"]["tenant"] == str(OTHER_TENANT)
    assert state["license_token"] == "customer-licence"
    assert (
        _sample("w2v_cell_jwt_rejected_total", {"reason": "tenant_mismatch", "scope": "tenant"})
        == before + 1
    )
    assert "tenant mismatch" in caplog.text


async def test_dual_agreeing_tenant_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    before = _sample(
        "w2v_cell_jwt_rejected_total", {"reason": "tenant_mismatch", "scope": "tenant"}
    )
    await _state_call(
        {CELL_TOKEN_HEADER: _token(orch_key), "X-Tenant-Id": "Bearer sa"},
        _introspection(TENANT, SA_SUBJECT),
    )
    assert (
        _sample("w2v_cell_jwt_rejected_total", {"reason": "tenant_mismatch", "scope": "tenant"})
        == before
    )


async def test_enforce_with_both_credentials_still_takes_the_cell_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    _, state = await _state_call(
        {CELL_TOKEN_HEADER: _token(orch_key), "X-Tenant-Id": "Bearer sa"},
        _introspection(OTHER_TENANT, SA_SUBJECT),
    )
    assert state["ctx"]["source"] == "cell-jwt"
    assert state["license_token"] is None


# --------------------------------------------------------------------------- #
# 0.91.2 review — the licence GATE must also be behaviour-preserving in ``dual``
#
# ``require_license`` skipped the per-tenant licence check whenever
# ``request.state.cell_caller`` was set. 0.91.2 records ``cell_caller`` on the
# dual+legacy path too, so with ``W2V_LICENSE_GATE_ENABLED=true`` (live on every
# w2v service) an in-cell call carrying both credentials skipped the gate that
# ``off`` runs — and on a tenant MISMATCH the legacy tenant (which no signature
# vouches for) was served ungated. The skip belongs to the cell LANE only.
# --------------------------------------------------------------------------- #


async def _gated_status(
    headers: dict[str, str], introspection: dict[str, Any] | None
) -> tuple[int, int]:
    """(HTTP status, licence-status lookups) for a LICENSE_GATE-mounted route."""
    from w2v_common.auth.entitlement import LICENSE_GATE, _result

    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    async def _ok() -> dict[str, str]:
        return {"ok": "1"}

    app.add_api_route("/api/gated", _ok, methods=["GET"], dependencies=list(LICENSE_GATE))
    lookup = AsyncMock(return_value=_result("none"))
    with (
        patch(
            "w2v_common.auth.middleware._introspect_license",
            new_callable=AsyncMock,
            return_value=introspection,
        ),
        patch("w2v_common.auth.entitlement.get_license_status", lookup),
        patch("w2v_common.auth.entitlement.get_licenses_read_client"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.get("/api/gated", headers=headers)
    return resp.status_code, lookup.await_count


@pytest.mark.parametrize("tenant", [TENANT, OTHER_TENANT], ids=["agreeing", "mismatch"])
async def test_dual_with_both_credentials_still_runs_the_licence_gate(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, tenant: uuid.UUID
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    legacy = {"X-Tenant-Id": "Bearer sa-licence"}
    introspection = _introspection(tenant, SA_SUBJECT)

    monkeypatch.delenv("CELL_AUTH_MODE", raising=False)
    off = await _gated_status(legacy, introspection)
    assert off[1] == 1 and off[0] != 200  # gate ran and refused an unlicensed tenant

    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    dual = await _gated_status({CELL_TOKEN_HEADER: _token(orch_key), **legacy}, introspection)
    assert dual == off


async def test_dual_jwt_user_with_valid_cell_header_still_runs_the_licence_gate(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    validator = AsyncMock()
    validator.validate.return_value = (
        {"sub": CUSTOMER_SUBJECT, "tenant_id": str(TENANT), "preferred_username": "u"},
        None,
    )
    with patch("w2v_common.auth.middleware.get_jwt_validator", return_value=validator):
        status_code, lookups = await _gated_status(
            {CELL_TOKEN_HEADER: _token(orch_key), "Authorization": "Bearer a.b.c"}, None
        )
    assert lookups == 1
    assert status_code != 200


async def test_dual_cell_only_still_skips_the_licence_gate(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    assert await _gated_status({CELL_TOKEN_HEADER: _token(orch_key)}, None) == (200, 0)


# --------------------------------------------------------------------------- #
# 0.92.0 — service-scope (tenant-less) cell tokens
#
# tts-gateway workers poll ``POST /api/tts/claim`` with no tenant. On 0.91.2
# ``mint_cell_token`` required a tenant, ``verify_cell_token`` refused a
# tenant-less token as ``invalid_tenant`` and ``CellClaims.tenant_id`` was
# required, so a worker could never present a cell token at all. A service
# token is accepted ONLY on a route that opts in with
# ``require_cell_service_caller`` and yields an identity with NO tenant.
# --------------------------------------------------------------------------- #


def _service_token(
    key: Ed25519PrivateKey, *, kid: str = "worker-1", iss: str = WORKER, **overrides: Any
) -> str:
    claims: dict[str, Any] = {
        "iss": iss,
        "scope": "service",
        "tenant": None,
        "run": None,
        "actor": None,
        "project": None,
    }
    claims.update(overrides)
    return _token(key, kid=kid, **claims)


@pytest.fixture
def as_worker(monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey) -> None:
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(worker_key))
    monkeypatch.setenv("CELL_JWT_KID", "worker-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", WORKER)


# -- mint ------------------------------------------------------------------- #


def test_mint_service_scope_with_a_tenant_is_an_error(as_worker: None) -> None:
    with pytest.raises(ValueError, match="service"):
        mint_cell_token(CALLEE, tenant=TENANT, scope="service")


@pytest.mark.parametrize("tenant", [None, uuid.UUID(int=0), str(uuid.UUID(int=0)), ""])
def test_mint_tenant_scope_without_a_real_tenant_is_an_error(as_worker: None, tenant: Any) -> None:
    """A tenant-scope mint never silently becomes a service token."""
    with pytest.raises(ValueError):
        mint_cell_token(CALLEE, tenant=tenant)
    with pytest.raises(ValueError):
        mint_cell_token(CALLEE, tenant=tenant, scope="tenant")


@pytest.mark.parametrize(
    "extra", [{"run": RUN}, {"actor": ACTOR}, {"project": PROJECT}], ids=["run", "actor", "project"]
)
def test_mint_service_scope_refuses_tenant_scoped_claims(
    as_worker: None, extra: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        mint_cell_token(CALLEE, scope="service", **extra)


def test_mint_unknown_scope_is_an_error(as_worker: None) -> None:
    with pytest.raises(ValueError):
        mint_cell_token(CALLEE, scope="global")


def test_mint_service_then_verify_on_an_opted_route(
    monkeypatch: pytest.MonkeyPatch, as_worker: None
) -> None:
    token = mint_service_cell_token(CALLEE)
    payload = jwt.decode(token, options={"verify_signature": False})
    assert payload["scope"] == "service"
    assert payload["tenant"] is None
    monkeypatch.setenv("CELL_SERVICE_NAME", CALLEE)
    claims = verify_cell_token(token, service_callers=frozenset({WORKER}))
    assert claims.scope is CellScope.SERVICE
    assert claims.tenant_id is None
    assert (claims.run, claims.actor, claims.project) == (None, None, None)


def test_minted_tenant_tokens_now_carry_an_explicit_scope(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    token = mint_cell_token(CALLEE, tenant=TENANT)
    assert jwt.decode(token, options={"verify_signature": False})["scope"] == "tenant"
    claims = _verify_as(monkeypatch, token)
    assert claims.scope is CellScope.TENANT
    assert claims.tenant_id == TENANT


# -- verify ----------------------------------------------------------------- #


def test_service_token_is_rejected_where_no_route_opted_in(worker_key: Ed25519PrivateKey) -> None:
    assert _reason(_service_token(worker_key)) == "service_scope_not_allowed"


def test_service_token_from_a_caller_the_route_did_not_name_is_rejected(
    orch_key: Ed25519PrivateKey,
) -> None:
    token = _service_token(orch_key, kid="orch-1", iss=ORCH)
    assert _reason(token, service_callers=frozenset({WORKER})) == "service_caller_not_allowed"


def test_service_token_still_needs_cell_allowed_callers(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", ORCH)
    token = _service_token(worker_key)
    assert _reason(token, service_callers=frozenset({WORKER})) == "caller_not_allowed"


@pytest.mark.parametrize(
    "tenant",
    [str(TENANT), str(uuid.UUID(int=0)), "", "00000000-0000-0000-0000-000000000000"],
    ids=["real", "nil", "empty", "nil-literal"],
)
def test_service_token_carrying_any_tenant_value_is_rejected(
    worker_key: Ed25519PrivateKey, tenant: str
) -> None:
    token = _service_token(worker_key, tenant=tenant)
    assert _reason(token, service_callers=frozenset({WORKER})) == "invalid_tenant"


def test_service_token_may_omit_the_tenant_claim(worker_key: Ed25519PrivateKey) -> None:
    token = _service_token(worker_key, tenant=_DROP)
    claims = verify_cell_token(token, service_callers=frozenset({WORKER}))
    assert claims.tenant_id is None


@pytest.mark.parametrize(
    "extra", [{"run": RUN}, {"actor": ACTOR}, {"project": PROJECT}], ids=["run", "actor", "project"]
)
def test_service_token_carrying_tenant_scoped_claims_is_rejected(
    worker_key: Ed25519PrivateKey, extra: dict[str, str]
) -> None:
    token = _service_token(worker_key, **extra)
    assert _reason(token, service_callers=frozenset({WORKER})) == "invalid_claim"


@pytest.mark.parametrize("scope", ["global", "", 1, None, "SERVICE"])
def test_unknown_scope_is_rejected(orch_key: Ed25519PrivateKey, scope: Any) -> None:
    assert _reason(_token(orch_key, scope=scope)) == "invalid_claim"


def test_tenant_scope_token_with_null_tenant_is_rejected(orch_key: Ed25519PrivateKey) -> None:
    """``scope=tenant`` + no tenant is NOT a service token — never a downgrade."""
    assert _reason(_token(orch_key, scope="tenant", tenant=None)) == "invalid_tenant"
    assert _reason(_token(orch_key, tenant=None), service_callers=frozenset({ORCH})) == (
        "invalid_tenant"
    )


def test_old_token_without_scope_is_a_tenant_token(orch_key: Ed25519PrivateKey) -> None:
    token = _token(orch_key)
    assert "scope" not in jwt.decode(token, options={"verify_signature": False})
    claims = verify_cell_token(token)
    assert claims.scope is CellScope.TENANT
    assert claims.tenant_id == TENANT


def test_tenant_token_is_accepted_on_an_opted_route_too(orch_key: Ed25519PrivateKey) -> None:
    """Opting a route in to service callers does not change tenant-token checks."""
    claims = verify_cell_token(_token(orch_key), service_callers=frozenset({WORKER}))
    assert claims.scope is CellScope.TENANT


def test_a_refused_service_token_does_not_burn_its_jti(worker_key: Ed25519PrivateKey) -> None:
    token = _service_token(worker_key)
    assert _reason(token) == "service_scope_not_allowed"
    assert verify_cell_token(token, service_callers=frozenset({WORKER})).issuer == WORKER


# -- receiver dependency + middleware --------------------------------------- #


def test_require_cell_service_caller_needs_an_explicit_caller_list() -> None:
    with pytest.raises(ValueError):
        require_cell_service_caller(set())
    with pytest.raises(ValueError):
        require_cell_service_caller({"*"})
    with pytest.raises(ValueError):
        require_cell_service_caller({" "})


async def _service_whoami(request: Request) -> dict[str, Any]:
    body = await _whoami(request)
    body["state_tenant"] = request.state.tenant_id
    body["scope"] = getattr(request.state, "cell_scope", None)
    body["license"] = getattr(request.state, "license", "unset")
    return body


def _service_app(*, allow_other_lanes: bool = False, gated: bool = False) -> FastAPI:
    from w2v_common.auth.entitlement import LICENSE_GATE

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    deps = [Depends(require_cell_service_caller({WORKER}, allow_other_lanes=allow_other_lanes))]
    if gated:
        deps += list(LICENSE_GATE)
    router = APIRouter(prefix="/api/tts", dependencies=deps)
    router.add_api_route("/claim", _service_whoami, methods=["POST"])
    app.include_router(router)
    app.add_api_route("/api/whoami", _service_whoami, methods=["POST"])
    return app


async def _service_call(
    path: str,
    headers: dict[str, str],
    introspection: dict[str, Any] | None = None,
    **app_kwargs: Any,
) -> tuple[int, dict[str, Any], AsyncMock]:
    with patch(
        "w2v_common.auth.middleware._introspect_license",
        new_callable=AsyncMock,
        return_value=introspection,
    ) as introspect:
        transport = ASGITransport(app=_service_app(**app_kwargs))
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post(path, headers=headers)
    return resp.status_code, resp.json(), introspect


@pytest.mark.parametrize("mode", ["dual", "enforce"])
async def test_service_token_on_the_opted_route_yields_a_tenantless_identity(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey, mode: str
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    before = _sample("w2v_auth_license_lane_total", {"lane": "cell-jwt-service"})
    status, body, introspect = await _service_call(
        "/api/tts/claim",
        {
            CELL_TOKEN_HEADER: _service_token(worker_key),
            # never honoured on a service token: there is no tenant to scope to
            "X-Subject-Id": SA_SUBJECT,
            "X-W2V-Project-Id": PROJECT,
            EFFECTIVE_TENANT_HEADER: str(TENANT),
        },
    )
    assert status == 200, body
    assert body["tenant"] is None
    assert body["state_tenant"] is None  # never a sentinel
    assert body["source"] is None  # nothing vouched: tenant-scoped code fails loud
    assert body["cell_caller"] == WORKER
    assert body["scope"] == "service"
    assert body["license"] is None
    # no human, no forwarded subject, no project/run: nothing tenant-scoped rides along
    assert not body["actor"]
    assert not body["subject"]
    assert not body["w2v_project"]
    assert not body["w2v_job"]
    introspect.assert_not_awaited()
    assert _sample("w2v_auth_license_lane_total", {"lane": "cell-jwt-service"}) == before + 1


async def test_enforce_service_token_on_a_route_that_did_not_opt_in_is_401(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    labels = {"reason": "service_scope_not_allowed", "scope": "service"}
    before = _sample("w2v_cell_jwt_rejected_total", labels)
    status, body, _ = await _service_call(
        "/api/whoami", {CELL_TOKEN_HEADER: _service_token(worker_key)}
    )
    assert status == 401, body
    assert _sample("w2v_cell_jwt_rejected_total", labels) == before + 1


async def test_dual_service_token_on_a_route_that_did_not_opt_in_is_not_authenticated(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    status, body, _ = await _service_call(
        "/api/whoami", {CELL_TOKEN_HEADER: _service_token(worker_key)}
    )
    assert status == 401, body


@pytest.mark.parametrize("mode", ["dual", "enforce"])
async def test_service_token_from_a_disallowed_caller_is_refused_on_the_opted_route(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, mode: str
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    labels = {"reason": "service_caller_not_allowed", "scope": "service"}
    before = _sample("w2v_cell_jwt_rejected_total", labels)
    status, body, _ = await _service_call(
        "/api/tts/claim", {CELL_TOKEN_HEADER: _service_token(orch_key, kid="orch-1", iss=ORCH)}
    )
    assert status == 401, body
    assert _sample("w2v_cell_jwt_rejected_total", labels) == before + 1


async def test_strict_service_route_refuses_a_tenant_cell_token(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    status, body, _ = await _service_call("/api/tts/claim", {CELL_TOKEN_HEADER: _token(orch_key)})
    assert status == 403, body


async def test_strict_service_route_refuses_a_licence_only_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    status, body, _ = await _service_call(
        "/api/tts/claim",
        {"X-Tenant-Id": "Bearer worker-licence"},
        _introspection(TENANT, SA_SUBJECT),
    )
    assert status == 403, body


async def test_allow_other_lanes_keeps_the_licence_worker_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migration shape: the claim route accepts service tokens AND the legacy worker."""
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    status, body, _ = await _service_call(
        "/api/tts/claim",
        {"X-Tenant-Id": "Bearer worker-licence"},
        _introspection(TENANT, SA_SUBJECT),
        allow_other_lanes=True,
    )
    assert status == 200, body
    assert body["tenant"] == str(TENANT)
    assert body["scope"] is None


async def test_dual_service_token_with_legacy_credential_keeps_legacy_and_skips_mismatch(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    """0.91.2 rule holds: legacy lane runs as off; the service token is only recorded.

    A service token has no tenant, so the dual tenant-mismatch check must not
    compare ``None`` against the legacy tenant and cry mismatch.
    """
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    mismatch = {"reason": "tenant_mismatch", "scope": "service"}
    mismatch_tenant = {"reason": "tenant_mismatch", "scope": "tenant"}
    both = {"caller": WORKER, "scope": "service"}
    before = (
        _sample("w2v_cell_jwt_rejected_total", mismatch),
        _sample("w2v_cell_jwt_rejected_total", mismatch_tenant),
        _sample("w2v_cell_dual_both_total", both),
    )
    status, body, introspect = await _service_call(
        "/api/tts/claim",
        {CELL_TOKEN_HEADER: _service_token(worker_key), "X-Tenant-Id": "Bearer worker-licence"},
        _introspection(TENANT, SA_SUBJECT),
        allow_other_lanes=True,
    )
    assert status == 200, body
    introspect.assert_awaited_once()
    assert body["tenant"] == str(TENANT)  # legacy lane, exactly as off
    assert body["source"] == "license-token"
    assert body["license"] == _introspection(TENANT, SA_SUBJECT)
    assert body["cell_caller"] == WORKER  # recorded alongside
    assert body["scope"] == "service"
    assert _sample("w2v_cell_jwt_rejected_total", mismatch) == before[0]
    assert _sample("w2v_cell_jwt_rejected_total", mismatch_tenant) == before[1]
    assert _sample("w2v_cell_dual_both_total", both) == before[2] + 1


async def test_dual_recorded_service_token_does_not_admit_a_strict_service_route(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    """dual + legacy credential = exactly ``off``: ``off`` 403s this route, so must ``dual``.

    The service token is only RECORDED there (``cell_claims``) while the licence
    lane authenticated a tenant — the dependency must not admit on a recording
    (the 0.91.2 ``require_license`` rule, applied to service admission).
    """
    headers = {CELL_TOKEN_HEADER: _service_token(worker_key), "X-Tenant-Id": "Bearer w"}
    statuses = {}
    for mode in ("off", "dual"):
        monkeypatch.setenv("CELL_AUTH_MODE", mode)
        statuses[mode], _, _ = await _service_call(
            "/api/tts/claim", headers, _introspection(TENANT, SA_SUBJECT)
        )
    assert statuses == {"off": 403, "dual": 403}


async def test_dual_recorded_service_token_returns_none_on_a_lenient_route(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    """``allow_other_lanes=True`` + dual + legacy: the handler sees the legacy caller (None)."""
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    seen: list[CellClaims | None] = []
    dep = require_cell_service_caller({WORKER}, allow_other_lanes=True)

    async def _claim(svc: CellClaims | None = Depends(dep)) -> dict[str, Any]:
        seen.append(svc)
        return {}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/tts/claim", _claim, methods=["POST"])
    with patch(
        "w2v_common.auth.middleware._introspect_license",
        new_callable=AsyncMock,
        return_value=_introspection(TENANT, SA_SUBJECT),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                "/api/tts/claim",
                headers={CELL_TOKEN_HEADER: _service_token(worker_key), "X-Tenant-Id": "Bearer w"},
            )
    assert resp.status_code == 200, resp.text
    assert seen == [None]


async def test_licence_gate_is_not_skipped_for_a_service_caller(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    """No tenant → no entitlement: a gated route refuses a service caller, loudly."""
    from w2v_common.auth.entitlement import _result

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    lookup = AsyncMock(return_value=_result("none"))
    with (
        patch("w2v_common.auth.entitlement.get_license_status", lookup),
        patch("w2v_common.auth.entitlement.get_licenses_read_client"),
    ):
        status, body, _ = await _service_call(
            "/api/tts/claim", {CELL_TOKEN_HEADER: _service_token(worker_key)}, gated=True
        )
    assert status != 200, body
    assert lookup.await_count == 1
    assert lookup.await_args is not None
    assert lookup.await_args.kwargs["tenant_id"] is None


# -- outbound (non-web workers) -------------------------------------------- #


async def test_service_hook_signs_a_tenantless_in_cell_call(
    monkeypatch: pytest.MonkeyPatch, as_worker: None
) -> None:
    hook = service_cell_token_hook()
    request = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    await hook(request)
    monkeypatch.setenv("CELL_SERVICE_NAME", "tts-gateway")
    verified = verify_cell_token(
        request.headers[CELL_TOKEN_HEADER], service_callers=frozenset({WORKER})
    )
    assert verified.scope is CellScope.SERVICE
    assert verified.audience == "tts-gateway"
    assert verified.tenant_id is None


def _host_mismatch_count() -> float:
    return _sample("w2v_cell_token_unminted_total", {"reason": "service_hook_host_mismatch"})


async def test_service_hook_with_explicit_audience_signs_its_in_cell_host(
    as_worker: None,
) -> None:
    hook = service_cell_token_hook(audience="tts-gateway")
    request = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    await hook(request)
    payload = jwt.decode(request.headers[CELL_TOKEN_HEADER], options={"verify_signature": False})
    assert (payload["aud"], payload["scope"], payload["tenant"]) == ("tts-gateway", "service", None)


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/api/tts/claim",
        # Same name, not the in-cell endpoint: a non-discovery URL is foreign too.
        "https://tts-gateway.example/api/tts/claim",
    ],
)
async def test_service_hook_explicit_audience_never_signs_a_foreign_host(
    as_worker: None, url: str
) -> None:
    """An explicit audience is not a licence to sign whatever host the client hits.

    A token minted for ``tts-gateway`` and sent to another host can be replayed
    against the claim route within its TTL, so the host must BE that audience.
    """
    before = _host_mismatch_count()
    request = _request(url, headers={CELL_TOKEN_HEADER: "copied"})
    await service_cell_token_hook(audience="tts-gateway")(request)
    assert CELL_TOKEN_HEADER not in request.headers
    assert _host_mismatch_count() == before + 1


async def test_service_hook_explicit_audience_never_signs_another_in_cell_peer(
    as_worker: None,
) -> None:
    """In cell but a different callee: the token would be for the wrong ``aud``."""
    before = _host_mismatch_count()
    request = _request(f"{service_url('analysis')}/api/analyze")
    await service_cell_token_hook(audience="tts-gateway")(request)
    assert CELL_TOKEN_HEADER not in request.headers
    assert _host_mismatch_count() == before + 1


@pytest.mark.parametrize("audience", [None, "tts-gateway"])
async def test_service_hook_redirect_to_a_foreign_host_carries_no_token(
    as_worker: None, audience: str | None
) -> None:
    """follow_redirects: the hook re-runs per hop, and the foreign hop gets nothing."""
    in_cell = f"{service_url('tts_gateway')}/api/tts/claim"
    seen: list[tuple[str, str | None]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get(CELL_TOKEN_HEADER)))
        if request.url.host == "attacker.example":
            return httpx.Response(200)
        return httpx.Response(307, headers={"Location": "https://attacker.example/steal"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        follow_redirects=True,
        event_hooks={"request": [service_cell_token_hook(audience=audience)]},
    ) as client:
        resp = await client.post(in_cell)
    assert resp.status_code == 200
    assert len(seen) == 2
    assert seen[0][1] is not None, "the in-cell hop must be signed"
    assert seen[1] == ("attacker.example", None)


async def test_service_hook_strips_out_of_cell(as_worker: None) -> None:
    before = _host_mismatch_count()
    request = _request("https://example.com/x", headers={CELL_TOKEN_HEADER: "copied"})
    await service_cell_token_hook()(request)
    assert CELL_TOKEN_HEADER not in request.headers
    assert _host_mismatch_count() == before + 1


async def test_service_hook_never_overrides_a_bound_tenant(as_worker: None) -> None:
    """Explicit service scope must not paper over a tenant the context carries."""
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    before = _sample("w2v_cell_token_unminted_total", {"reason": "service_hook_tenant_bound"})
    request = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    await identity_event_hook(request)
    tenant_token = request.headers[CELL_TOKEN_HEADER]
    await service_cell_token_hook()(request)
    assert request.headers[CELL_TOKEN_HEADER] == tenant_token
    assert (
        _sample("w2v_cell_token_unminted_total", {"reason": "service_hook_tenant_bound"})
        == before + 1
    )


async def test_service_hook_without_signing_key_is_a_noop() -> None:
    request = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    await service_cell_token_hook()(request)
    assert CELL_TOKEN_HEADER not in request.headers


async def test_identity_hook_never_downgrades_a_lost_tenant_to_service_scope(
    as_worker: None,
) -> None:
    """A tenant-scope call that lost its tenant is never re-signed as the service.

    Without credentials it fails loud (``MissingTenantError``, unchanged); on a
    service-account licence it goes out with NO cell token at all.
    """
    from w2v_common.tenant import MissingTenantError

    bare = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    with pytest.raises(MissingTenantError):
        await identity_event_hook(bare)
    assert CELL_TOKEN_HEADER not in bare.headers

    sa = _request(
        f"{service_url('tts_gateway')}/api/tts/claim",
        headers={"X-Tenant-Id": "Bearer sa-licence", CELL_TOKEN_HEADER: "copied"},
    )
    await identity_event_hook(sa)
    assert CELL_TOKEN_HEADER not in sa.headers


# -- end to end: the worker claim under enforce ----------------------------- #


async def test_enforce_simulation_worker_claim(
    monkeypatch: pytest.MonkeyPatch, as_worker: None
) -> None:
    """Worker (no request context) signs; tts-gateway in enforce serves the claim."""
    outbound = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    await service_cell_token_hook()(outbound)
    token = outbound.headers[CELL_TOKEN_HEADER]

    # ...now this process is the receiver.
    monkeypatch.delenv("CELL_JWT_SIGNING_KEY")
    monkeypatch.delenv("CELL_JWT_KID")
    monkeypatch.setenv("CELL_SERVICE_NAME", "tts-gateway")
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", f"{ORCH},{WORKER}")
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    status, body, introspect = await _service_call("/api/tts/claim", {CELL_TOKEN_HEADER: token})
    assert status == 200, body
    assert body["tenant"] is None
    assert body["cell_caller"] == WORKER
    introspect.assert_not_awaited()
    # the same token on a tenant route of the same receiver is refused
    status, _, _ = await _service_call("/api/whoami", {CELL_TOKEN_HEADER: token})
    assert status == 401


def _routing_app() -> FastAPI:
    async def _h() -> dict[str, Any]:
        return {}

    inner = FastAPI()
    router = APIRouter(
        prefix="/api/tts", dependencies=[Depends(require_cell_service_caller({WORKER}))]
    )
    router.add_api_route("/claim", _h, methods=["POST"])
    router.add_api_route("/status", _h, methods=["GET"])
    inner.include_router(router)
    inner.add_api_route("/api/tts/{job}", _h, methods=["POST"])  # registered AFTER: never shadows
    inner.add_api_route("/api/other", _h, methods=["POST"])
    outer = FastAPI()
    outer.include_router(router)
    outer.mount("/sub", inner)
    return outer


@pytest.mark.parametrize(
    ("method", "path", "opted"),
    [
        ("POST", "/api/tts/claim", True),
        ("POST", "/sub/api/tts/claim", True),  # mounted sub-app followed
        ("POST", "/api/tts/claim/", False),  # trailing slash: Starlette redirects, no opt-in
        ("GET", "/api/tts/claim", False),  # method mismatch (405)
        ("OPTIONS", "/api/tts/claim", False),
        ("HEAD", "/api/tts/status", False),  # APIRoute does not add HEAD: 405 too
        ("POST", "/api/other", False),
        ("POST", "/sub/api/other", False),
        ("POST", "/sub/api/tts/x", False),  # path-param sibling did not opt in
        ("POST", "/nope", False),  # 404
        ("POST", "/API/tts/claim", False),
        ("POST", "//api/tts/claim", False),
    ],
)
def test_route_opt_in_lookup_mirrors_starlette_and_fails_closed(
    method: str, path: str, opted: bool
) -> None:
    from w2v_common.auth.cell_service import route_cell_service_callers

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "app": _routing_app(),
        "headers": [],
        "query_string": b"",
    }
    assert route_cell_service_callers(scope) == (frozenset({WORKER}) if opted else None)


def test_route_opt_in_lookup_without_an_app_is_none() -> None:
    from w2v_common.auth.cell_service import route_cell_service_callers

    assert route_cell_service_callers({"type": "http", "method": "POST", "path": "/x"}) is None


# -- route resolution across FastAPI's include_router shapes ---------------- #
#
# FastAPI >= 0.141 no longer COPIES an included router's routes into the parent:
# it wraps the router in ``_IncludedRouter``, whose ``matches()`` reports only
# that the inclusion matched, and whose matched APIRoute is the ORIGINAL route
# (its dependant lacks the dependencies added at inclusion time). 0.92.0 skipped
# the wrapper, so every service token at an included route was refused
# ``service_scope_not_allowed``. The fleet locks 0.135 .. 0.142, so these cases
# run on the repo lock AND (CI job ``fastapi-compat``) on the oldest locked line.

OTHER = "other-worker"


def _opt(*callers: str) -> Any:
    return Depends(require_cell_service_caller(set(callers)))


def _include_shapes_app() -> FastAPI:
    async def _h() -> dict[str, Any]:
        return {}

    app = FastAPI()

    # 1. route-level dependency inside an included router
    per_route = APIRouter(prefix="/r1")
    per_route.add_api_route("/claim", _h, methods=["POST"], dependencies=[_opt(WORKER)])
    per_route.add_api_route("/plain", _h, methods=["POST"])
    app.include_router(per_route, prefix="/api")

    # 2. opt-in given at INCLUDE time on a router that declares nothing itself
    plain = APIRouter()
    plain.add_api_route("/claim", _h, methods=["POST"])
    app.include_router(plain, prefix="/api/r2", dependencies=[_opt(WORKER)])

    # 3. nested include_router, prefixes at every level, a typed path param,
    #    opt-ins at two levels (union) and a non-opted sibling
    leaf = APIRouter(prefix="/jobs", dependencies=[_opt(OTHER)])
    leaf.add_api_route("/{shard:int}/claim", _h, methods=["POST"])
    mid = APIRouter(prefix="/mid", dependencies=[_opt(WORKER)])
    mid.include_router(leaf)
    mid.add_api_route("/free", _h, methods=["POST"])
    top = APIRouter(prefix="/r3")
    top.include_router(mid)
    unopted_mid = APIRouter(prefix="/bare")
    unopted_mid.add_api_route("/x", _h, methods=["POST"])
    top.include_router(unopted_mid)
    app.include_router(top, prefix="/api")

    # 4. dependency declared in the endpoint signature (dependant sub-tree)
    async def _sig(_c: Any = _opt(WORKER)) -> dict[str, Any]:
        return {}

    sig = APIRouter()
    sig.add_api_route("/claim", _sig, methods=["POST"])
    app.include_router(sig, prefix="/api/r4")

    # 5. mounted sub-app whose opted route is itself included
    sub = FastAPI()
    sub_router = APIRouter(prefix="/api/tts", dependencies=[_opt(WORKER)])
    sub_router.add_api_route("/claim", _h, methods=["POST"])
    sub.include_router(sub_router)
    sub.add_api_route("/api/other", _h, methods=["POST"])
    app.mount("/sub", sub)

    # 6. an included Mount (FastAPI rebuilds it with the include prefix)
    mounting = APIRouter()
    inner = FastAPI()
    inner.include_router(per_route)
    mounting.mount("/inner", inner)
    app.include_router(mounting, prefix="/api/r6")

    # 7. a NON-opted route included FIRST shadows an opted one on the same path
    shadow = APIRouter()
    shadow.add_api_route("/api/r7/claim", _h, methods=["POST"])
    app.include_router(shadow)
    shadowed = APIRouter(dependencies=[_opt(WORKER)])
    shadowed.add_api_route("/api/r7/claim", _h, methods=["POST"])
    app.include_router(shadowed)
    return app


def _app_level_deps_app() -> FastAPI:
    async def _h() -> dict[str, Any]:
        return {}

    app = FastAPI(dependencies=[_opt(WORKER)])
    router = APIRouter(prefix="/api/q")
    router.add_api_route("/claim", _h, methods=["POST"])
    app.include_router(router)
    return app


def _scope(app: FastAPI, method: str, path: str) -> dict[str, Any]:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "app": app,
        "headers": [],
        "query_string": b"",
    }


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/api/r1/claim", {WORKER}),
        ("POST", "/api/r1/plain", None),
        ("GET", "/api/r1/claim", None),  # 405
        ("POST", "/api/r2/claim", {WORKER}),  # include-time dependency
        ("POST", "/api/r3/mid/jobs/7/claim", {WORKER, OTHER}),  # nested: union
        ("POST", "/api/r3/mid/jobs/x/claim", None),  # int convertor: 404
        ("POST", "/api/r3/mid/free", {WORKER}),
        ("POST", "/api/r3/bare/x", None),
        ("POST", "/api/r4/claim", {WORKER}),  # signature dependency
        ("POST", "/sub/api/tts/claim", {WORKER}),  # mount -> included router
        ("POST", "/sub/api/other", None),
        ("POST", "/api/r6/inner/r1/claim", {WORKER}),  # included mount -> included
        ("POST", "/api/r6/inner/r1/plain", None),
        ("POST", "/api/r7/claim", None),  # first FULL match wins, as in Starlette
        ("POST", "/nope", None),
    ],
)
async def test_route_opt_in_lookup_sees_every_include_router_shape(
    method: str, path: str, expected: set[str] | None
) -> None:
    from w2v_common.auth.cell_service import route_cell_service_callers

    app = _include_shapes_app()
    # Ground truth is what the app actually DISPATCHES on this FastAPI: before
    # 0.141 ``include_router`` silently drops a router's Mounts (r6 is a 404
    # there), so the lookup must agree with the router, not with the intent.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # 403 = routed, and the opt-in dependency refused this lane-less call
        served = (await c.request(method, path)).status_code in (200, 403)
    if expected is not None and not served:
        assert path.startswith("/api/r6/"), f"opted route {path} is not served"
        expected = None
    got = route_cell_service_callers(_scope(app, method, path))
    assert got == (frozenset(expected) if expected is not None else None)


def test_route_opt_in_lookup_counts_app_level_dependencies() -> None:
    from w2v_common.auth.cell_service import route_cell_service_callers

    got = route_cell_service_callers(_scope(_app_level_deps_app(), "POST", "/api/q/claim"))
    assert got == frozenset({WORKER})


def test_route_opt_in_lookup_leaves_the_request_scope_untouched() -> None:
    """Probing must not leave FastAPI routing bookkeeping on the live scope."""
    from w2v_common.auth.cell_service import route_cell_service_callers

    scope = _scope(_include_shapes_app(), "POST", "/api/r3/mid/jobs/7/claim")
    before = dict(scope)
    route_cell_service_callers(scope)
    assert scope == before


async def test_service_token_reaches_a_nested_included_route_end_to_end(
    monkeypatch: pytest.MonkeyPatch, worker_key: Ed25519PrivateKey
) -> None:
    """The middleware lane admits the token AND the handler-time dependency agrees."""
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")

    leaf = APIRouter(prefix="/claim")
    leaf.add_api_route("/{shard}", _service_whoami, methods=["POST"])
    mid = APIRouter(prefix="/tts", dependencies=[_opt(WORKER)])
    mid.include_router(leaf)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(mid, prefix="/api")

    with patch(
        "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
    ) as introspect:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                "/api/tts/claim/3", headers={CELL_TOKEN_HEADER: _service_token(worker_key)}
            )
    assert resp.status_code == 200, resp.text
    assert resp.json()["cell_caller"] == WORKER
    introspect.assert_not_awaited()


# -- sync service hook (httpx.Client) ---------------------------------------- #


def test_sync_service_hook_signs_through_a_sync_client(
    monkeypatch: pytest.MonkeyPatch, as_worker: None
) -> None:
    seen: list[str | None] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(CELL_TOKEN_HEADER))
        return httpx.Response(200)

    with httpx.Client(
        transport=httpx.MockTransport(_handler),
        event_hooks={"request": [service_cell_token_hook_sync(audience="tts-gateway")]},
    ) as client:
        client.post(f"{service_url('tts_gateway')}/api/tts/claim")
    assert seen[0] is not None
    monkeypatch.setenv("CELL_SERVICE_NAME", "tts-gateway")
    verified = verify_cell_token(seen[0], service_callers=frozenset({WORKER}))
    assert (verified.scope, verified.audience, verified.tenant_id) == (
        CellScope.SERVICE,
        "tts-gateway",
        None,
    )


@pytest.mark.parametrize(
    ("audience", "url"),
    [
        ("tts-gateway", "https://attacker.example/api/tts/claim"),
        ("tts-gateway", "https://tts-gateway.example/api/tts/claim"),
        ("tts-gateway", "ANALYSIS"),  # another in-cell peer
        (None, "https://example.com/x"),
    ],
)
def test_sync_service_hook_applies_the_same_host_rules(
    as_worker: None, audience: str | None, url: str
) -> None:
    if url == "ANALYSIS":
        url = f"{service_url('analysis')}/api/analyze"
    before = _host_mismatch_count()
    request = _request(url, headers={CELL_TOKEN_HEADER: "copied"})
    service_cell_token_hook_sync(audience=audience)(request)
    assert CELL_TOKEN_HEADER not in request.headers
    assert _host_mismatch_count() == before + 1


def test_sync_service_hook_redirect_to_a_foreign_host_carries_no_token(
    as_worker: None,
) -> None:
    seen: list[tuple[str, str | None]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get(CELL_TOKEN_HEADER)))
        if request.url.host == "attacker.example":
            return httpx.Response(200)
        return httpx.Response(307, headers={"Location": "https://attacker.example/steal"})

    with httpx.Client(
        transport=httpx.MockTransport(_handler),
        follow_redirects=True,
        event_hooks={"request": [service_cell_token_hook_sync()]},
    ) as client:
        client.post(f"{service_url('tts_gateway')}/api/tts/claim")
    assert seen[0][1] is not None
    assert seen[1] == ("attacker.example", None)


def test_sync_service_hook_never_overrides_a_bound_tenant(as_worker: None) -> None:
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.JWT_CLAIM)
    before = _sample("w2v_cell_token_unminted_total", {"reason": "service_hook_tenant_bound"})
    request = _request(f"{service_url('tts_gateway')}/api/tts/claim", {CELL_TOKEN_HEADER: "t"})
    service_cell_token_hook_sync()(request)
    assert request.headers[CELL_TOKEN_HEADER] == "t"
    assert (
        _sample("w2v_cell_token_unminted_total", {"reason": "service_hook_tenant_bound"})
        == before + 1
    )


def test_sync_service_hook_without_signing_key_is_a_noop() -> None:
    request = _request(f"{service_url('tts_gateway')}/api/tts/claim")
    service_cell_token_hook_sync()(request)
    assert CELL_TOKEN_HEADER not in request.headers


# -- private-API drift fails closed for the service lane only ---------------- #


@pytest.mark.parametrize(
    "drifted",
    [
        pytest.param(lambda self, scope, extra: None, id="signature-changed"),
        pytest.param(lambda self, scope: (Match.FULL, {}, None), id="three-tuple"),
        pytest.param(lambda self, scope: (Match.FULL, {}, None, None), id="no-route"),
        pytest.param(lambda self, scope: "matched", id="not-a-tuple"),
    ],
)
def test_included_router_match_drift_refuses_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, drifted: Any
) -> None:
    """A FastAPI that changes the private ``_IncludedRouter._match`` gets refusal, not 500."""
    import fastapi.routing

    from w2v_common.auth.cell_service import route_cell_service_callers

    included = getattr(fastapi.routing, "_IncludedRouter", None)
    if included is None:
        pytest.skip("this FastAPI copies included routes (< 0.141)")
    app = _include_shapes_app()
    monkeypatch.setattr(included, "_match", drifted)
    assert route_cell_service_callers(_scope(app, "POST", "/api/r2/claim")) is None


async def test_a_crashing_route_lookup_refuses_service_tokens_and_spares_tenant_tokens(
    monkeypatch: pytest.MonkeyPatch,
    orch_key: Ed25519PrivateKey,
    worker_key: Ed25519PrivateKey,
) -> None:
    """The lookup runs for EVERY cell token; its failure must not 500 tenant traffic."""
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")

    def _boom(*_a: Any, **_k: Any) -> None:
        raise TypeError("_match() takes 3 positional arguments but 2 were given")

    monkeypatch.setattr("w2v_common.auth.cell_service._match_routes", _boom)
    status, body, _ = await _call({CELL_TOKEN_HEADER: _token(orch_key)})
    assert status == 200, body
    assert body["tenant"] == str(TENANT)
    before = _sample(
        "w2v_cell_jwt_rejected_total", {"reason": "service_scope_not_allowed", "scope": "service"}
    )
    status, body, _ = await _service_call(
        "/api/tts/claim", {CELL_TOKEN_HEADER: _service_token(worker_key)}
    )
    assert status == 401, body
    assert (
        _sample(
            "w2v_cell_jwt_rejected_total",
            {"reason": "service_scope_not_allowed", "scope": "service"},
        )
        == before + 1
    )


# --------------------------------------------------------------------------- #
# 0.93.0 — signed attribution: subject / license are claims, not headers
# --------------------------------------------------------------------------- #

LICENSE = "0198f1d2-0000-7000-8000-00000011c001"
SPOOF_SUBJECT = "0198f1d2-0000-7000-8000-0000005f0001"
SPOOF_LICENSE = "0198f1d2-0000-7000-8000-0000005f0002"


def test_mint_verify_round_trips_subject_and_license(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    token = mint_cell_token(CALLEE, tenant=TENANT, subject_id=CUSTOMER_SUBJECT, license_id=LICENSE)
    raw = jwt.decode(token, options={"verify_signature": False})
    assert (raw["subject"], raw["license"]) == (CUSTOMER_SUBJECT, LICENSE)
    claims = _verify_as(monkeypatch, token)
    assert (claims.subject, claims.license_id) == (CUSTOMER_SUBJECT, LICENSE)
    assert claims.attribution_signed is True


def test_minted_tenant_token_always_carries_both_keys(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    """Presence is the version signal: a 0.93.0 token with nothing bound still
    signs ``subject: null`` / ``license: null`` so the receiver ignores headers."""
    token = mint_cell_token(CALLEE, tenant=TENANT)
    raw = jwt.decode(token, options={"verify_signature": False})
    assert "subject" in raw and raw["subject"] is None
    assert "license" in raw and raw["license"] is None
    claims = _verify_as(monkeypatch, token)
    assert (claims.subject, claims.license_id, claims.attribution_signed) == (None, None, True)


def test_service_scope_token_carries_neither(
    monkeypatch: pytest.MonkeyPatch, as_worker: None
) -> None:
    token = mint_service_cell_token("tts-gateway")
    raw = jwt.decode(token, options={"verify_signature": False})
    assert raw.get("subject") is None
    assert raw.get("license") is None
    monkeypatch.setenv("CELL_SERVICE_NAME", "tts-gateway")
    claims = verify_cell_token(token, service_callers=frozenset({WORKER}))
    assert (claims.subject, claims.license_id, claims.attribution_signed) == (None, None, False)


def test_mint_service_scope_refuses_subject_or_license(as_worker: None) -> None:
    with pytest.raises(ValueError, match="subject/license"):
        mint_cell_token("tts-gateway", scope="service", subject_id=SA_SUBJECT)
    with pytest.raises(ValueError, match="subject/license"):
        mint_cell_token("tts-gateway", scope="service", license_id=LICENSE)


@pytest.mark.parametrize("bad", ["", "has space", "x" * 257, "tab\there"])
def test_mint_refuses_a_malformed_subject_or_license(as_orchestrator: None, bad: str) -> None:
    with pytest.raises(ValueError, match="subject"):
        mint_cell_token(CALLEE, tenant=TENANT, subject_id=bad)
    with pytest.raises(ValueError, match="license"):
        mint_cell_token(CALLEE, tenant=TENANT, license_id=bad)


def test_service_token_carrying_a_subject_or_license_is_rejected(
    worker_key: Ed25519PrivateKey,
) -> None:
    callers = frozenset({WORKER})
    for claim in ("subject", "license"):
        pair: dict[str, Any] = {"subject": None, "license": None, claim: SA_SUBJECT}
        token = _service_token(worker_key, **pair)
        assert _reason(token, service_callers=callers) == "invalid_claim"


@pytest.mark.parametrize("present", ["subject", "license"])
def test_half_an_attribution_pair_is_rejected(orch_key: Ed25519PrivateKey, present: str) -> None:
    half: dict[str, Any] = {present: SA_SUBJECT}
    assert _reason(_token(orch_key, **half)) == "invalid_claim"


@pytest.mark.parametrize("bad", [7, "", "two words", ["x"]])
def test_malformed_signed_attribution_is_rejected(orch_key: Ed25519PrivateKey, bad: Any) -> None:
    assert _reason(_token(orch_key, subject=bad, license=LICENSE)) == "invalid_claim"
    assert _reason(_token(orch_key, subject=SA_SUBJECT, license=bad)) == "invalid_claim"


@pytest.mark.parametrize("mode", ["dual", "enforce"])
async def test_signed_claims_win_over_conflicting_headers(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, mode: str
) -> None:
    """The finding: a cell caller could bill any licence / assert any subject
    via headers. With signed claims the headers are ignored entirely."""
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    before = _sample("w2v_cell_unsigned_attribution_total", {"caller": ORCH})
    status, body, _ = await _call(
        {
            CELL_TOKEN_HEADER: _token(orch_key, subject=CUSTOMER_SUBJECT, license=LICENSE),
            "X-Subject-Id": SPOOF_SUBJECT,
            "X-License-Id": SPOOF_LICENSE,
        },
        path="/api/whoami-attr",
    )
    assert status == 200, body
    assert body["subject"] == CUSTOMER_SUBJECT
    assert body["attribution_subject"] == CUSTOMER_SUBJECT
    assert body["license_id"] == LICENSE
    assert _sample("w2v_cell_unsigned_attribution_total", {"caller": ORCH}) == before


async def test_signed_null_attribution_ignores_the_headers(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    """A 0.93.0 caller with nothing bound signs null: a header cannot fill it."""
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    status, body, _ = await _call(
        {
            CELL_TOKEN_HEADER: _token(orch_key, subject=None, license=None),
            "X-Subject-Id": SPOOF_SUBJECT,
            "X-License-Id": SPOOF_LICENSE,
        },
        path="/api/whoami-attr",
    )
    assert status == 200, body
    assert (body["subject"], body["attribution_subject"], body["license_id"]) == ("", "", "")


async def test_unsigned_token_falls_back_to_headers_as_attribution_only(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, caplog: pytest.LogCaptureFixture
) -> None:
    """A 0.92.x peer (no subject/license claims): headers bill as before, the
    fallback is counted per caller, and the header subject never becomes the
    authz-visible subject (tts_gateway's worker gate, orchestrator SubjectIdDep)."""
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    before = _sample("w2v_cell_unsigned_attribution_total", {"caller": ORCH})
    with caplog.at_level(logging.DEBUG, logger="w2v_common.auth.middleware"):
        status, body, _ = await _call(
            {
                CELL_TOKEN_HEADER: _token(orch_key),  # no subject/license keys
                "X-Subject-Id": SA_SUBJECT,
                "X-License-Id": LICENSE,
            },
            path="/api/whoami-attr",
        )
    assert status == 200, body
    assert body["subject"] == ""  # authz sees no subject
    assert body["attribution_subject"] == SA_SUBJECT  # billing still does
    assert body["license_id"] == LICENSE
    assert _sample("w2v_cell_unsigned_attribution_total", {"caller": ORCH}) == before + 1
    assert any("no signed subject/license" in r.getMessage() for r in caplog.records)


async def test_unsigned_header_subject_is_forwarded_but_never_signed_onward(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, bdd_key: Ed25519PrivateKey
) -> None:
    """The callee re-signs only what it AUTHENTICATED: a header subject from a
    0.92.x caller rides the legacy header onward but is not laundered into a
    signed claim the next hop would trust. The licence (billing-only) is."""
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(bdd_key))
    monkeypatch.setenv("CELL_JWT_KID", "bdd-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", BDD)
    seen: dict[str, str] = {}

    async def _onward(request: Request) -> dict[str, str]:
        hop = httpx.Request("POST", f"{service_url('analysis')}/api/x")
        await identity_event_hook(hop)
        seen.update(hop.headers)
        return {}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/onward", _onward, methods=["GET"])
    inbound = _token(orch_key, aud=BDD)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get(
            "/api/onward",
            headers={
                CELL_TOKEN_HEADER: inbound,
                "X-Subject-Id": SA_SUBJECT,
                "X-License-Id": LICENSE,
            },
        )
    assert resp.status_code == 200
    assert seen["x-subject-id"] == SA_SUBJECT
    onward = jwt.decode(seen[CELL_TOKEN_HEADER.lower()], options={"verify_signature": False})
    assert onward["subject"] is None
    assert onward["license"] == LICENSE


async def test_signed_attribution_is_signed_onward(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, bdd_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(bdd_key))
    monkeypatch.setenv("CELL_JWT_KID", "bdd-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", BDD)
    seen: dict[str, str] = {}

    async def _onward(request: Request) -> dict[str, str]:
        hop = httpx.Request("POST", f"{service_url('analysis')}/api/x")
        await identity_event_hook(hop)
        seen.update(hop.headers)
        return {}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/onward", _onward, methods=["GET"])
    inbound = _token(orch_key, aud=BDD, subject=CUSTOMER_SUBJECT, license=LICENSE)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get(
            "/api/onward", headers={CELL_TOKEN_HEADER: inbound, "X-Subject-Id": SPOOF_SUBJECT}
        )
    assert resp.status_code == 200
    assert seen["x-subject-id"] == CUSTOMER_SUBJECT
    onward = jwt.decode(seen[CELL_TOKEN_HEADER.lower()], options={"verify_signature": False})
    assert (onward["subject"], onward["license"]) == (CUSTOMER_SUBJECT, LICENSE)


async def test_hook_signs_the_bound_subject_and_license(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    set_identity_context(TENANT, LICENSE, CUSTOMER_SUBJECT, tenant_source=TenantSource.JWT_CLAIM)
    request = _request(f"{service_url('analysis')}/api/analyze")
    await identity_event_hook(request)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER])
    assert (claims.subject, claims.license_id) == (CUSTOMER_SUBJECT, LICENSE)


async def test_hook_omits_a_malformed_bound_subject_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch, as_orchestrator: None
) -> None:
    set_identity_context(TENANT, LICENSE, "not a valid id", tenant_source=TenantSource.JWT_CLAIM)
    request = _request(f"{service_url('analysis')}/api/analyze")
    await identity_event_hook(request)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER])
    assert (claims.subject, claims.license_id) == (None, LICENSE)


# --------------------------------------------------------------------------- #
# Evidence counters: zero series from startup + the w2v_cell_auth_total denominator
# --------------------------------------------------------------------------- #

_AUTH_SERIES = (
    ("cell", "tenant"),
    ("cell", "service"),
    ("dual", "tenant"),
    ("dual", "service"),
    ("legacy", "none"),
)


def _auth_snapshot() -> dict[tuple[str, str], float]:
    return {
        (lane, scope): _sample("w2v_cell_auth_total", {"lane": lane, "scope": scope})
        for lane, scope in _AUTH_SERIES
    }


def _delta(before: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    after = _auth_snapshot()
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


def test_fresh_middleware_exposes_zero_cell_auth_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh process (fresh registry) publishes every static series at 0 on setup.

    Without pre-creation a labelled Counter emits NOTHING until its first
    ``.labels().inc()``, so "no legacy traffic" and "no traffic" both read as an
    absent series. Fresh counters on a private registry stand in for a fresh
    process (the module counters on the default REGISTRY already have samples).
    """
    from prometheus_client import CollectorRegistry, Counter

    from w2v_common.auth import cell_jwt as cj

    reg = CollectorRegistry()
    monkeypatch.setattr(
        cj,
        "cell_auth_total",
        Counter("w2v_cell_auth_total", "d", ["lane", "scope"], registry=reg),
    )
    monkeypatch.setattr(
        cj,
        "cell_jwt_rejected_total",
        Counter("w2v_cell_jwt_rejected_total", "d", ["reason", "scope"], registry=reg),
    )
    monkeypatch.setattr(
        cj,
        "cell_legacy_delegation_total",
        Counter("w2v_cell_legacy_delegation_total", "d", ["caller"], registry=reg),
    )
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SA_SUBJECT)

    def _value(name: str, labels: dict[str, str]) -> float | None:
        return reg.get_sample_value(name, labels)

    # Before setup: absent, which is exactly the ambiguity being removed.
    assert _value("w2v_cell_auth_total", {"lane": "legacy", "scope": "none"}) is None

    AuthMiddleware(FastAPI())

    for lane, scope in _AUTH_SERIES:
        assert _value("w2v_cell_auth_total", {"lane": lane, "scope": scope}) == 0.0
    for reason, scope in cj.CELL_REJECT_SERIES:
        assert _value("w2v_cell_jwt_rejected_total", {"reason": reason, "scope": scope}) == 0.0
    for caller in ("unlisted", "unverified", SA_SUBJECT):
        assert _value("w2v_cell_legacy_delegation_total", {"caller": caller}) == 0.0
    # Only the pre-created series exist: nothing was counted.
    auth_samples = [
        s
        for m in reg.collect()
        if m.name == "w2v_cell_auth"
        for s in m.samples
        if s.name == "w2v_cell_auth_total"
    ]
    assert len(auth_samples) == len(_AUTH_SERIES)
    assert all(s.value == 0.0 for s in auth_samples)


def test_every_rejection_reason_in_the_source_is_pre_created() -> None:
    """A new ``CellTokenError("x")`` / ``record_cell_rejection("x")`` needs a zero series."""
    import re

    from w2v_common.auth import cell_jwt as cj
    from w2v_common.auth import middleware as mw

    pattern = re.compile(r'(?:CellTokenError|record_cell_rejection)\(\s*"([a-z_]+)"')
    used = {
        reason
        for module in (cj, mw)
        for reason in pattern.findall(Path(str(module.__file__)).read_text())
    }
    assert used, "pattern matched nothing — the guard would be vacuous"
    assert "tenant_mismatch" in used
    assert used <= {reason for reason, _ in cj.CELL_REJECT_SERIES}


async def test_cell_lane_request_counts_exactly_the_cell_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    before = _auth_snapshot()
    status, body, _ = await _call({CELL_TOKEN_HEADER: _token(orch_key)})
    assert status == 200, body
    assert body["source"] == "cell-jwt"
    assert _delta(before) == {("cell", "tenant"): 1.0}


async def test_dual_both_request_counts_exactly_the_dual_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    before = _auth_snapshot()
    status, body, _ = await _call(
        {CELL_TOKEN_HEADER: _token(orch_key), "X-Tenant-Id": "Bearer sa-licence"},
        _introspection(TENANT, SA_SUBJECT),
    )
    assert status == 200, body
    assert body["source"] == "license-token"
    assert _delta(before) == {("dual", "tenant"): 1.0}


@pytest.mark.parametrize("mode", ["off", "dual"])
async def test_legacy_request_counts_exactly_the_legacy_lane(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    before = _auth_snapshot()
    status, body, _ = await _call(
        {"X-Tenant-Id": "Bearer customer-licence"},
        _introspection(TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 200, body
    assert _delta(before) == {("legacy", "none"): 1.0}


async def test_dual_invalid_cell_token_on_the_legacy_lane_counts_legacy(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    before = _auth_snapshot()
    status, _, _ = await _call(
        {
            CELL_TOKEN_HEADER: _token(orch_key, aud="slide-service"),
            "X-Tenant-Id": "Bearer customer-licence",
        },
        _introspection(TENANT, CUSTOMER_SUBJECT),
    )
    assert status == 200
    assert _delta(before) == {("legacy", "none"): 1.0}


@pytest.mark.parametrize("mode", ["dual", "enforce"])
async def test_unauthenticated_request_counts_no_lane(
    monkeypatch: pytest.MonkeyPatch, orch_key: Ed25519PrivateKey, mode: str
) -> None:
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    before = _auth_snapshot()
    status, _, _ = await _call({CELL_TOKEN_HEADER: _token(orch_key, aud="slide-service")})
    assert status == 401
    assert _delta(before) == {}


# Every exempt path the middleware short-circuits BEFORE any lane runs: the
# fleet-wide PUBLIC_ENDPOINTS, one path under each PUBLIC_PREFIXES base, and an
# instance-level ``extra_public_endpoints`` entry. Enumerated from the source so
# a newly exempted path is covered automatically.
_EXTRA_PUBLIC = "/api/auth/callback"
_EXEMPT_PATHS = sorted(
    {*PUBLIC_ENDPOINTS, *(f"{base}/probe" for base in PUBLIC_PREFIX_BASES), _EXTRA_PUBLIC}
)


def test_exempt_path_enumeration_is_not_vacuous() -> None:
    assert {"/health", "/readyz", "/livez", "/metrics", "/health/probe"} <= set(_EXEMPT_PATHS)
    assert len(PUBLIC_PREFIX_BASES) >= 1


@pytest.mark.parametrize("mode", ["off", "dual"])
@pytest.mark.parametrize("path", _EXEMPT_PATHS)
async def test_exempt_probe_counts_no_lane_but_authenticated_request_does(
    monkeypatch: pytest.MonkeyPatch, mode: str, path: str
) -> None:
    """Probes and /metrics must never inflate the w2v_cell_auth_total denominator.

    The legacy share of this counter decides production auth flips: a kubelet
    probe or Prometheus scrape counted as ``lane=legacy`` would make every
    service look legacy-dependent forever. The wrap sits AFTER the exemption; a
    wrap moved above it counts every probe. The control on the SAME app proves
    the counter is live in this test (an always-empty delta cannot pass it).
    """
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    app = FastAPI()
    app.add_middleware(AuthMiddleware, extra_public_endpoints=[_EXTRA_PUBLIC])
    app.add_api_route("/api/whoami", _whoami, methods=["GET"])
    with patch(
        "w2v_common.auth.middleware._introspect_license",
        new_callable=AsyncMock,
        return_value=_introspection(TENANT, CUSTOMER_SUBJECT),
    ) as introspect:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            before = _auth_snapshot()
            resp = await c.get(path)
            assert resp.status_code != 401, (path, resp.status_code)
            assert _delta(before) == {}, path
            introspect.assert_not_awaited()

            control_before = _auth_snapshot()
            ok = await c.get("/api/whoami", headers={"X-Tenant-Id": "Bearer customer-licence"})
            assert ok.status_code == 200, ok.text
            assert _delta(control_before) == {("legacy", "none"): 1.0}
