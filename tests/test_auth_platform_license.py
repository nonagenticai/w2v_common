"""w2v_common must speak the PLATFORM's licensing conventions.

Context (measured, not assumed):

* The platform's ``license-authz`` Istio ext-authz gate reads the licence token
  from ``Authorization: Bearer <opaque>`` ONLY. It never reads ``x-tenant-id``
  inbound.
* On ALLOW it INJECTS ``x-tenant-id`` (a BARE tenant UUID) together with
  ``x-license-id``, ``x-subject-id``, ``x-actor-id`` and friends.
* w2v's own convention is the opposite: the token travels in
  ``X-Tenant-Id: Bearer <opaque>``.

So ``X-Tenant-Id`` means TWO different things depending on the hop, and w2v has
to handle both without letting either weaken the other. These tests pin that
seam. Every one of them fails on the pre-change middleware:

* the ``Authorization``-bearer tests 401 (non-JWT bearers were never routed to
  introspection),
* the injected-identity tests 401 (the lane did not exist),
* the refusal tests would pass vacuously there, and are here to keep the new
  lane from becoming a spoofing primitive.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import ipaddress
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

import w2v_common.auth.middleware as mw
from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_license_id_from_context,
    get_subject_id_from_context,
    get_tenant_source_from_context,
    is_tenant_vouched,
)
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.tenant import TenantSource

pytestmark = pytest.mark.asyncio

TENANT = "0195f48b-0000-7000-8000-0000000000aa"
LICENSE_ID = "lic-7788"
SUBJECT_ID = "0195f48b-0000-7000-8000-0000000000bb"
ACTOR_ID = "0195f48b-0000-7000-8000-0000000000cc"


@pytest.fixture(autouse=True)
def _isolate_auth_state(monkeypatch: pytest.MonkeyPatch):
    """Clear the licence caches and force every new knob to its DEFAULT.

    ``delenv`` matters as much as the caches: the new mesh lane defaults OFF and
    a leaked env var from another test would make the "refused when the setting
    is off" assertions pass for the wrong reason.
    """
    for name in (
        "_license_cache",
        "_license_neg_cache",
        "_license_stale_cache",
        "_license_token_not_found_since",
        "_license_tnf_probe",
    ):
        setattr(mw, name, None)
    mw._introspect_locks.clear()
    monkeypatch.delenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", raising=False)
    monkeypatch.delenv("AUTH_MESH_VOUCH_SECRET", raising=False)
    monkeypatch.delenv("AUTH_TRUST_INCLUSTER_IDENTITY", raising=False)
    # TRUSTED_CIDRS defaults EMPTY since 0.89.0; name the ASGITransport peer
    # explicitly, as an operator must, so the lanes are reachable at all.
    monkeypatch.setattr(mw, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")])
    yield
    for name in (
        "_license_cache",
        "_license_neg_cache",
        "_license_stale_cache",
        "_license_token_not_found_since",
        "_license_tnf_probe",
    ):
        setattr(mw, name, None)
    mw._introspect_locks.clear()


def _make_app() -> FastAPI:
    """A minimal app whose protected route reports the RESOLVED identity.

    It reads the ContextVars, not ``request.state``, because provenance
    (``tenant_source``) only exists there — and provenance is the thing that
    decides whether a downstream metered hop is allowed.
    """
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/api/data")
    async def data(request: Request):  # pragma: no cover - exercised via HTTP
        source = get_tenant_source_from_context()
        return {
            "tenant_id": request.state.tenant_id,
            "user_sub": request.state.user.sub,
            "tenant_source": None if source is None else str(source),
            "vouched": is_tenant_vouched(),
            "license_id": get_license_id_from_context(),
            "subject_id": get_subject_id_from_context(),
            "actor_id": get_actor_id_from_context(),
            "has_raw_token": bool(getattr(request.state, "license_token", None)),
        }

    return app


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=_make_app()), base_url="http://test")


def _active(**overrides):
    payload = {
        "active": True,
        "tenant_id": TENANT,
        "subject_id": SUBJECT_ID,
        "license_id": LICENSE_ID,
    }
    payload.update(overrides)
    return payload


def _lane_value(lane: str) -> float:
    return mw.auth_license_lane_total.labels(lane=lane)._value.get()


# ---------------------------------------------------------------------------
# 1. Authorization: Bearer <opaque> — the platform convention.
# ---------------------------------------------------------------------------


class TestOpaqueAuthorizationBearer:
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_opaque_bearer_on_authorization_authenticates(self, introspect):
        """PROVES: the shape license-authz speaks now authenticates.

        Pre-change this 401'd — ``_extract_license_token`` looked at
        ``X-Tenant-Id`` only, so a non-JWT ``Authorization`` bearer fell all the
        way through to AUTHENTICATION_REQUIRED. That is the exact 401 that would
        have hit every caller the moment w2v was put behind the gate.
        """
        introspect.return_value = _active()
        async with _client() as client:
            resp = await client.get(
                "/api/data", headers={"Authorization": "Bearer opaque-licence-token"}
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tenant_id"] == TENANT
        assert body["tenant_source"] == TenantSource.LICENSE_TOKEN
        assert body["vouched"] is True
        # The token was presented TO US, so it stays forwardable.
        assert body["has_raw_token"] is True
        introspect.assert_awaited_once_with("opaque-licence-token")

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_opaque_bearer_rejected_when_introspection_says_inactive(self, introspect):
        """PROVES: accepting the new shape did not make it a free pass."""
        introspect.return_value = {"active": False, "reason": "revoked"}
        async with _client() as client:
            resp = await client.get("/api/data", headers={"Authorization": "Bearer revoked-tok"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "INVALID_TOKEN"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_jwt_shaped_bearer_is_never_introspected(self, introspect):
        """PROVES: JWT validation is UNCHANGED and unweakened.

        A three-segment bearer still takes the Keycloak branch and still hard
        fails when no validator is configured. It must NOT be laundered into the
        new licence lane — submitting a JWT to introspection is the 2026-04-26
        prod-401 bug, and it would also let an unverifiable JWT be treated as an
        opaque credential.
        """
        async with _client() as client:
            resp = await client.get(
                "/api/data", headers={"Authorization": "Bearer header.payload.signature"}
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "JWT_VALIDATOR_UNAVAILABLE"
        introspect.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Legacy X-Tenant-Id: Bearer — REQUIRED to keep working.
# ---------------------------------------------------------------------------


class TestLegacyXTenantIdBearer:
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_legacy_x_tenant_id_bearer_still_authenticates(self, introspect):
        """PROVES: every existing caller keeps working (they all send this)."""
        introspect.return_value = _active()
        before = _lane_value("legacy-x-tenant-id")
        async with _client() as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": "Bearer legacy-tok"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["tenant_id"] == TENANT
        introspect.assert_awaited_once_with("legacy-tok")
        # The deprecation burn-down signal actually moves.
        assert _lane_value("legacy-x-tenant-id") == before + 1

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_legacy_header_wins_over_a_service_account_authorization(self, introspect):
        """PROVES: precedence cannot silently re-attribute existing traffic.

        ``BaseServiceClient`` puts the CALLER's token on ``X-Tenant-Id`` while
        its constructor-baked SERVICE-ACCOUNT token sits on ``Authorization``.
        If the new ``Authorization`` lane were preferred, every delegated call in
        the fleet would flip from the customer tenant to the SA tenant — a silent
        mis-attribution, not an error. The caller's token must still win.
        """
        introspect.return_value = _active()
        async with _client() as client:
            resp = await client.get(
                "/api/data",
                headers={
                    "X-Tenant-Id": "Bearer caller-tok",
                    "Authorization": "Bearer service-account-tok",
                },
            )
        assert resp.status_code == 200, resp.text
        introspect.assert_awaited_once_with("caller-tok")


# ---------------------------------------------------------------------------
# 3. The license-authz INJECTED-identity lane.
# ---------------------------------------------------------------------------


def _injected(**extra) -> dict[str, str]:
    headers = {
        "X-Tenant-Id": TENANT,  # BARE uuid, not "Bearer ..."
        "X-License-Id": LICENSE_ID,
        "X-Subject-Id": SUBJECT_ID,
    }
    headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# A DELIBERATELY INDEPENDENT reimplementation of license-authz's vouch producer.
#
# This is written from the SPEC in license-authz's ``src/vouch.py`` docstring —
# it imports NOTHING from ``w2v_common.auth.middleware``. That is the whole
# point: if either side's canonicalisation drifts (field order, separator,
# absent-vs-empty, encoding, hash, hex case), these tests go red. Importing the
# middleware's own constants here would make the tests agree with the consumer
# by construction and catch nothing — which is exactly how the bare-secret
# mismatch survived code review on both sides in the first place.
#
# Spec, verbatim from the producer:
#   wire      : v1:<unix_ts>:<nonce_hex>:<hmac_sha256_hex>
#   message   : "v1\n<ts>\n<nonce>\n" + "\n".join(f"{name}={value or ''}")
#               over the ten VOUCHED_HEADERS, in this fixed order
#   mac       : HMAC-SHA256(secret.utf8, message.utf8).hexdigest()
#   absent    : bound as the empty string
# ---------------------------------------------------------------------------

VOUCH_SECRET = "shared-with-license-authz-LICENSE_VOUCH_SECRET"  # pragma: allowlist secret

_SPEC_VOUCHED_HEADERS = (
    "x-tenant-id",
    "x-license-id",
    "x-subject-id",
    "x-app-id",
    "x-plan-id",
    "x-plan-code",
    "x-app-key",
    "x-actor-id",
    "x-caller-client-id",
    "x-caller-roles",
)


def _spec_canonical_message(headers: dict[str, str], *, timestamp: int, nonce: str) -> str:
    lower = {k.lower(): v for k, v in headers.items()}
    lines = ["v1", str(timestamp), nonce]
    lines.extend(f"{name}={lower.get(name) or ''}" for name in _SPEC_VOUCHED_HEADERS)
    return "\n".join(lines)


def _spec_compute_vouch(
    headers: dict[str, str],
    *,
    secret: str = VOUCH_SECRET,
    timestamp: int | None = None,
    nonce: str = "a1b2c3d4e5f60718",  # pragma: allowlist secret
) -> str:
    ts = int(time.time()) if timestamp is None else timestamp
    mac = hmac.new(
        secret.encode("utf-8"),
        _spec_canonical_message(headers, timestamp=ts, nonce=nonce).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"v1:{ts}:{nonce}:{mac}"


def _vouched(
    *,
    secret: str = VOUCH_SECRET,
    timestamp: int | None = None,
    mac_over: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
    **extra: str,
) -> dict[str, str]:
    """Injected header set plus a genuine producer-format vouch over it.

    ``mac_over`` lets a test sign a DIFFERENT tuple than the one it sends — the
    transplant attack the MAC exists to stop.

    ``overrides`` takes a prebuilt header dict. Header names are not valid
    Python identifiers, so a full set has to arrive as a dict rather than as
    ``**kwargs``; routing it through a typed parameter keeps the unpack from
    being checked against ``timestamp``/``mac_over``.
    """
    headers = _injected(**{**(overrides or {}), **extra})
    signed = headers if mac_over is None else mac_over
    headers["X-License-Vouch"] = _spec_compute_vouch(signed, secret=secret, timestamp=timestamp)
    return headers


class TestMeshVouchedLane:
    async def test_injected_identity_is_vouched_and_not_unvouched_header(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: a gated request authenticates with a NON-UNVOUCHED provenance.

        This is the point of the whole change. ``MESH_VOUCHED`` — not
        ``UNVOUCHED_HEADER`` — is what lets a gated request survive its first
        metered downstream hop instead of failing closed there.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        async with _client() as client:
            resp = await client.get("/api/data", headers=_injected(**{"X-Actor-Id": ACTOR_ID}))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tenant_id"] == TENANT
        assert body["tenant_source"] == TenantSource.MESH_VOUCHED
        assert body["tenant_source"] != TenantSource.UNVOUCHED_HEADER
        assert body["vouched"] is True
        assert body["license_id"] == LICENSE_ID
        assert body["subject_id"] == SUBJECT_ID
        assert body["actor_id"] == ACTOR_ID
        # The gate consumed the licence; we never saw a token to forward.
        assert body["has_raw_token"] is False

    async def test_injected_identity_refused_when_setting_is_off(self):
        """PROVES: the lane is INERT by default — headers alone buy nothing.

        Default-off is the answer to "we cannot distinguish injection from a
        forgery": until an operator asserts only the gate can reach this port,
        the same headers are worth exactly zero.
        """
        async with _client() as client:
            resp = await client.get("/api/data", headers=_injected())
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    async def test_bare_tenant_alone_from_untrusted_peer_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: a lone bare ``x-tenant-id`` never authenticates.

        Even with the lane ON, a partial set (no licence id, no subject id) is
        not an injected identity. It falls through, and with the trusted-network
        lane off it is refused.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        async with _client() as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": TENANT})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    async def test_partial_injected_set_does_not_reach_the_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: tenant+licence without a subject is still not this lane."""
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        headers = _injected()
        del headers["X-Subject-Id"]
        async with _client() as client:
            resp = await client.get("/api/data", headers=headers)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    async def test_injected_identity_ignored_from_off_cluster_peer(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: the peer-IP floor is real, not decorative."""
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with _client() as client:
                resp = await client.get("/api/data", headers=_injected())
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    async def test_vouch_secret_mismatch_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: with a secret configured, headers alone are refused.

        This is the only condition in the lane an in-mesh peer genuinely cannot
        forge. When it is configured it is MANDATORY — a missing or wrong
        ``X-License-Vouch`` is a hard refusal, never a downgrade to the weaker
        IP-only check.

        Note the BARE SECRET case: echoing the secret back used to be accepted.
        It is now a refusal, because the wire value is an HMAC and the secret
        never travels.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        async with _client() as client:
            missing = await client.get("/api/data", headers=_injected())
            wrong = await client.get(
                "/api/data", headers=_injected(**{"X-License-Vouch": "not-the-secret"})
            )
            bare_secret = await client.get(
                "/api/data", headers=_injected(**{"X-License-Vouch": VOUCH_SECRET})
            )
            right = await client.get("/api/data", headers=_vouched())
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"
        assert wrong.status_code == 401
        assert wrong.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"
        assert bare_secret.status_code == 401
        assert bare_secret.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"
        assert right.status_code == 200, right.text
        assert right.json()["tenant_source"] == TenantSource.MESH_VOUCHED

    async def test_injected_non_uuid_tenant_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: the lane still funnels through the one tenant parser.

        No lane may substitute a default or coerce a slug — a bad injected
        tenant is a 401, not an invented UUID.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        async with _client() as client:
            resp = await client.get("/api/data", headers=_injected(**{"X-Tenant-Id": "n8n-dev"}))
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    async def test_injected_nil_tenant_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: the banned nil-UUID sentinel cannot enter through the gate lane."""
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        async with _client() as client:
            resp = await client.get(
                "/api/data", headers=_injected(**{"X-Tenant-Id": str(uuid.UUID(int=0))})
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "TENANT_REQUIRED"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_bearer_prefixed_tenant_header_is_never_laundered_into_the_lane(
        self, introspect, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: a FAILED credential cannot be upgraded by sibling headers.

        ``X-Tenant-Id: Bearer <tok>`` is a credential. If introspection rejects
        it, the presence of ``x-license-id``/``x-subject-id`` must not then let
        the same request in through the injected-identity lane with a VOUCHED
        provenance.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        introspect.return_value = {"active": False, "reason": "revoked"}
        async with _client() as client:
            resp = await client.get(
                "/api/data", headers=_injected(**{"X-Tenant-Id": "Bearer revoked-tok"})
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "INVALID_TOKEN"


# ---------------------------------------------------------------------------
# 3b. CROSS-REPO CONTRACT: the vouch w2v verifies must be the vouch
#     license-authz actually emits.
#
# Every test below fails on the pre-change middleware, which compared the
# presented header against the BARE SECRET. A genuine producer-format vouch is
# not the secret, so it never matched: the gate would have emitted a valid
# vouch, w2v would have refused it, and the lane would have been dead in the
# silent direction with both repos looking implemented.
# ---------------------------------------------------------------------------


class TestVouchIsAVerifiedHmac:
    async def test_genuine_producer_vouch_verifies(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: the exact value license-authz computes is ACCEPTED.

        The MAC here is built by ``_spec_compute_vouch`` — an independent
        reimplementation of the producer's algorithm from its spec — so this
        goes red if EITHER side's canonicalisation drifts.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        async with _client() as client:
            resp = await client.get("/api/data", headers=_vouched())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tenant_id"] == TENANT
        assert body["tenant_source"] == TenantSource.MESH_VOUCHED
        assert body["vouched"] is True

    async def test_vouch_verifies_over_the_full_ten_field_set(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: the whole injected set is bound, not just tenant/licence/subject.

        Also pins the ABSENT==EMPTY rule from both directions in one request:
        ``x-actor-id`` etc. are signed as "" when the gate omits them, and the
        fields below are signed with real values when it does not.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        full = {
            "X-Actor-Id": ACTOR_ID,
            "X-App-Id": "app-42",
            "X-Plan-Id": "plan-9",
            "X-Plan-Code": "enterprise",
            "X-App-Key": "key-abc",
            "X-Caller-Client-Id": "orchestrator",
            "X-Caller-Roles": "svc,metered",
        }
        async with _client() as client:
            resp = await client.get("/api/data", headers=_vouched(overrides=full))
        assert resp.status_code == 200, resp.text
        assert resp.json()["actor_id"] == ACTOR_ID

    @pytest.mark.parametrize(
        "field",
        [
            "X-Tenant-Id",
            "X-License-Id",
            "X-Subject-Id",
            "X-App-Id",
            "X-Plan-Id",
            "X-Plan-Code",
            "X-App-Key",
            "X-Actor-Id",
            "X-Caller-Client-Id",
            "X-Caller-Roles",
        ],
    )
    async def test_vouch_over_different_values_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ):
        """PROVES: EACH of the ten bound fields is really bound.

        A peer captures a valid vouch and re-aims it at another tenant, or just
        adds ``x-caller-roles: admin`` under an otherwise-valid MAC. Every one
        of those is a refusal — this is the transplant/escalation attack, and
        parametrising it stops a field being quietly dropped from the tuple
        while nine others keep the test green.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        signed = _injected(
            **{
                "X-App-Id": "app-42",
                "X-Plan-Id": "plan-9",
                "X-Plan-Code": "enterprise",
                "X-App-Key": "key-abc",
                "X-Actor-Id": ACTOR_ID,
                "X-Caller-Client-Id": "orchestrator",
                "X-Caller-Roles": "svc",
            }
        )
        sent = dict(signed)
        sent[field] = (
            "0195f48b-0000-7000-8000-0000000000ff" if field == "X-Tenant-Id" else "tampered"
        )
        sent["X-License-Vouch"] = _spec_compute_vouch(signed)
        async with _client() as client:
            resp = await client.get("/api/data", headers=sent)
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"

    async def test_stale_vouch_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: an old-but-genuine vouch cannot be replayed forever.

        Without a freshness bound, one captured header set is a permanent
        credential for that tenant — the gate is the only party that re-checks
        the licence, and a replay never reaches it.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        old = int(time.time()) - (mw.MESH_VOUCH_MAX_AGE_DEFAULT + 30)
        async with _client() as client:
            resp = await client.get("/api/data", headers=_vouched(timestamp=old))
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"

    async def test_future_dated_vouch_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: the window is SYMMETRIC.

        A forward-dated timestamp must not buy a vouch a longer life than the
        window allows.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        ahead = int(time.time()) + (mw.MESH_VOUCH_MAX_AGE_DEFAULT + 30)
        async with _client() as client:
            resp = await client.get("/api/data", headers=_vouched(timestamp=ahead))
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"

    async def test_freshness_window_is_configurable(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: the same vouch is refused at 60s and accepted at a widened window."""
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        headers = _vouched(timestamp=int(time.time()) - 300)
        async with _client() as client:
            tight = await client.get("/api/data", headers=headers)
            monkeypatch.setenv("AUTH_MESH_VOUCH_MAX_AGE_SECONDS", "900")
            wide = await client.get("/api/data", headers=headers)
        assert tight.status_code == 401
        assert wide.status_code == 200, wide.text

    async def test_bad_max_age_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: a typo cannot silently disable the replay bound."""
        monkeypatch.setenv("AUTH_MESH_VOUCH_MAX_AGE_SECONDS", "not-a-number")
        assert mw._mesh_vouch_max_age() == mw.MESH_VOUCH_MAX_AGE_DEFAULT
        monkeypatch.setenv("AUTH_MESH_VOUCH_MAX_AGE_SECONDS", "0")
        assert mw._mesh_vouch_max_age() == mw.MESH_VOUCH_MAX_AGE_DEFAULT
        monkeypatch.setenv("AUTH_MESH_VOUCH_MAX_AGE_SECONDS", "-1")
        assert mw._mesh_vouch_max_age() == mw.MESH_VOUCH_MAX_AGE_DEFAULT

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda v: v[:-1] + ("0" if v[-1] != "0" else "1"), id="last_mac_byte"),
            pytest.param(
                lambda v: (
                    v.split(":")[0]
                    + ":"
                    + v.split(":")[1]
                    + ":"
                    + v.split(":")[2]
                    + ":"
                    + ("0" if v.split(":")[3][0] != "0" else "1")
                    + v.split(":")[3][1:]
                ),
                id="first_mac_byte",
            ),
            pytest.param(lambda v: "v2" + v[2:], id="wrong_version"),
            pytest.param(lambda v: ":".join(v.split(":")[:3]), id="truncated"),
            pytest.param(lambda v: v.replace(":", "|", 1), id="wrong_separator"),
            pytest.param(lambda v: v.split(":")[3], id="mac_only"),
            pytest.param(
                lambda v: "v1:" + v.split(":")[1] + ":ffffffffffffffff:" + v.split(":")[3],
                id="swapped_nonce",
            ),
        ],
    )
    async def test_tampered_vouch_is_refused(self, monkeypatch: pytest.MonkeyPatch, mutate):
        """PROVES: every part of the wire value is load-bearing.

        Flipping ONE hex nibble of the MAC — at either end, so a prefix- or
        suffix-only comparison cannot pass — the version tag, the nonce, or the
        framing is a refusal.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        headers = _vouched()
        headers["X-License-Vouch"] = mutate(headers["X-License-Vouch"])
        async with _client() as client:
            resp = await client.get("/api/data", headers=headers)
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"

    async def test_wrong_secret_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """PROVES: AUTH_MESH_VOUCH_SECRET must equal LICENSE_VOUCH_SECRET.

        A well-formed, fresh vouch computed under a DIFFERENT key is refused —
        the two env vars are one key, and a mismatch is a hard failure rather
        than a silent acceptance.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        async with _client() as client:
            resp = await client.get("/api/data", headers=_vouched(secret="a-different-key"))
        assert resp.status_code == 401, resp.text
        assert resp.json()["error"]["code"] == "MESH_VOUCH_REQUIRED"

    async def test_mac_comparison_is_constant_time(self):
        """PROVES: the MAC compare uses hmac.compare_digest, not ``==``.

        A short-circuiting comparison leaks the length of the matching prefix
        and makes the MAC forgeable byte-by-byte. This is asserted structurally
        because a timing measurement on a loaded CI box is a flake, not a test:
        the source must call ``hmac.compare_digest`` and must not fall back to
        an ``==`` on the MAC.
        """
        src = inspect.getsource(mw._verify_mesh_vouch)
        assert "hmac.compare_digest(expected, mac)" in src
        assert "expected == mac" not in src
        assert "mac == expected" not in src

    async def test_valid_vouch_does_not_bypass_the_other_two_conditions(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: the vouch is an ADDITIONAL requirement, not a replacement.

        A perfectly valid vouch still gets nowhere from an off-cluster peer, and
        still gets nowhere while the lane flag is OFF.
        """
        monkeypatch.setenv("AUTH_MESH_VOUCH_SECRET", VOUCH_SECRET)
        # (1) flag OFF — the lane does not exist, vouch or no vouch.
        async with _client() as client:
            flag_off = await client.get("/api/data", headers=_vouched())
        assert flag_off.status_code == 401
        assert flag_off.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

        # (2) off-cluster peer — the IP floor still applies.
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with _client() as client:
                off_cluster = await client.get("/api/data", headers=_vouched())
        assert off_cluster.status_code == 401
        assert off_cluster.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

        # ...and the complete injected set is still required.
        headers = _vouched()
        del headers["X-Subject-Id"]
        async with _client() as client:
            incomplete = await client.get("/api/data", headers=headers)
        assert incomplete.status_code == 401
        assert incomplete.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    async def test_no_secret_configured_keeps_warn_once_behaviour(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: the unkeyed rollout is unchanged — warn once, still allow.

        license-authz also emits nothing when its own key is unset, so this is
        the honest "neither side can vouch yet" state, not a downgrade.
        """
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setattr(mw, "_warned_mesh_vouch_unsecured", False, raising=False)
        async with _client() as client:
            resp = await client.get("/api/data", headers=_injected())
        assert resp.status_code == 200, resp.text
        assert resp.json()["tenant_source"] == TenantSource.MESH_VOUCHED
        assert mw._warned_mesh_vouch_unsecured is True

    async def test_canonical_message_is_byte_identical_to_the_producer_spec(self):
        """PROVES the contract directly, with no HTTP in the way.

        The consumer's canonicalisation and the spec-derived one must produce
        the SAME bytes — including the absent-binds-as-empty rule, which is the
        subtlety most likely to be "cleaned up" on one side only.
        """
        from starlette.datastructures import Headers

        raw = {
            "x-tenant-id": TENANT,
            "x-license-id": LICENSE_ID,
            "x-subject-id": SUBJECT_ID,
            "x-caller-roles": "svc,metered",
            # x-app-id / x-plan-id / x-plan-code / x-app-key / x-actor-id /
            # x-caller-client-id deliberately ABSENT -> must bind as "".
        }
        expected = _spec_canonical_message(raw, timestamp=1757700000, nonce="deadbeefcafe0001")
        actual = mw._mesh_vouch_canonical_message(
            Headers(raw), timestamp=1757700000, nonce="deadbeefcafe0001"
        )
        assert actual == expected
        assert actual.startswith("v1\n1757700000\ndeadbeefcafe0001\nx-tenant-id=")
        assert "\nx-app-id=\n" in actual
        assert actual.endswith("\nx-caller-roles=svc,metered")
        assert len(actual.split("\n")) == 3 + 10


# ---------------------------------------------------------------------------
# 4. The pre-existing trusted-network lane must keep its WEAK provenance.
# ---------------------------------------------------------------------------


class TestTrustedNetworkStillUnvouched:
    async def test_bare_tenant_on_trusted_network_stays_unvouched(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """PROVES: the new lane did not accidentally upgrade the old one.

        A bare ``X-Tenant-Id`` accepted because the CALLER is in-cluster is still
        ``UNVOUCHED_HEADER`` — accepted as a data scope, never delegatable.
        """
        monkeypatch.setenv("AUTH_TRUST_INCLUSTER_IDENTITY", "true")
        async with _client() as client:
            resp = await client.get("/api/data", headers={"X-Tenant-Id": TENANT})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tenant_source"] == TenantSource.UNVOUCHED_HEADER
        assert body["vouched"] is False
