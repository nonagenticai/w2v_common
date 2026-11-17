"""Entitlement gate: every w2v service must be able to refuse an unlicensed tenant.

These tests drive the gate through a real FastAPI app + TestClient, not by
calling the derivation directly, because the thing being promoted is an
END-TO-END refusal: state → HTTP status → error code → Retry-After. A test that
asserted only on ``LicenseStatusResult`` would stay green if the dependency were
never wired to a route — this repo's signature failure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from w2v_common.auth import entitlement
from w2v_common.auth.entitlement import (
    LICENSE_GATE,
    LICENSE_GATE_WS,
    LicenseStatusResult,
    LicenseStatusState,
    clear_license_status_cache,
    enforce_license_state,
    is_license_exempt_path,
)
from w2v_common.auth.licenses_read_client import (
    ActiveLicenseView,
    AppInfoView,
    LicensesClientError,
    LicenseView,
    set_licenses_read_client,
)

TENANT = "11111111-1111-1111-1111-111111111111"
SUBJECT = "22222222-2222-2222-2222-222222222222"


class FakeLicenses:
    """Stand-in for the read client. ``mode`` selects the upstream behaviour."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.active_calls = 0

    async def get_active_license(self, *, tenant_id: str, app_key: str):
        self.active_calls += 1
        if self.mode == "active":
            return ActiveLicenseView(
                license_id="lic-1",
                tenant_id=tenant_id,
                expires_at="2099-01-01T00:00:00+00:00",
                app=AppInfoView(key=app_key),
            )
        if self.mode == "boom":
            raise LicensesClientError("licenses unreachable", status_code=502)
        if self.mode == "weird":
            raise RuntimeError("socket exploded")
        # none / expired / revoked all collapse to a 404 upstream
        return None

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str):
        if self.mode == "expired":
            return [
                LicenseView(
                    id="lic-x",
                    tenant_id=tenant_id,
                    status="expired",
                    expires_at="2020-01-01T00:00:00+00:00",
                    app=AppInfoView(key="w2v"),
                )
            ]
        if self.mode == "revoked":
            return [
                LicenseView(
                    id="lic-y",
                    tenant_id=tenant_id,
                    status="revoked",
                    revoked_at="2024-01-01T00:00:00+00:00",
                    app=AppInfoView(key="w2v"),
                )
            ]
        return []


def build_app(fake: FakeLicenses | None) -> FastAPI:
    app = FastAPI()
    gated = APIRouter(dependencies=list(LICENSE_GATE))

    @gated.get("/api/things")
    async def things() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(gated)

    @app.middleware("http")
    async def _stamp_identity(request, call_next):
        request.state.tenant_id = TENANT
        request.state.subject_id = SUBJECT
        return await call_next(request)

    return app


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_license_status_cache()
    set_licenses_read_client(None)
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    yield
    clear_license_status_cache()
    set_licenses_read_client(None)


def client_for(mode: str) -> tuple[TestClient, FakeLicenses]:
    fake = FakeLicenses(mode)
    set_licenses_read_client(fake)
    return TestClient(build_app(fake), raise_server_exceptions=False), fake


# ---------------------------------------------------------------------------
# Entitled
# ---------------------------------------------------------------------------


def test_entitled_tenant_passes():
    client, _ = client_for("active")
    resp = client.get("/api/things")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": "yes"}


# ---------------------------------------------------------------------------
# Authoritative refusals → 402 license_<state>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,expected_code",
    [("none", "license_none"), ("expired", "license_expired"), ("revoked", "license_revoked")],
)
def test_authoritative_refusal_is_402_with_matching_code(mode, expected_code):
    client, _ = client_for(mode)
    resp = client.get("/api/things")
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == expected_code


# ---------------------------------------------------------------------------
# Indeterminate → 503 license_unavailable + Retry-After
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["boom", "weird"])
def test_upstream_failure_is_503_with_retry_after(mode):
    client, _ = client_for(mode)
    resp = client.get("/api/things")
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["error"] == "license_unavailable"
    assert resp.headers["Retry-After"] == "30"


def test_unrecognised_future_state_is_503_not_allowed():
    """The allowlist, directly: a state this build has never heard of must 503,
    never pass. A blocklist here is what turned an outage into a fleet ALLOW."""
    # Deliberately a state the Literal does not contain — that IS the scenario:
    # a future licenses release inventing one. cast states the intent.
    future = LicenseStatusResult(
        state=cast("LicenseStatusState", "entitled_v2"),
        has_license=True,
        expires_at=None,
        days_remaining=None,
        plan_code=None,
    )
    with pytest.raises(HTTPException) as exc:
        enforce_license_state(future)
    assert exc.value.status_code == 503
    # Bound through locals with an explicit shape assertion: HTTPException types
    # `detail` as str and `headers` as optional, so the checker cannot see the
    # dict/header this gate always sends. These assertions are STRONGER than a
    # bare subscript (a str detail or a missing headers dict now fails loudly),
    # never weaker — the 503 must still carry license_unavailable + Retry-After.
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail["error"] == "license_unavailable"
    headers = exc.value.headers
    assert headers is not None
    assert headers["Retry-After"] == "30"


# ---------------------------------------------------------------------------
# Stale-while-error — the layer that buys availability
# ---------------------------------------------------------------------------


def test_stale_while_error_serves_warm_verdict_when_licenses_unreachable():
    client, fake = client_for("active")
    assert client.get("/api/things").status_code == 200
    assert fake.active_calls == 1

    # Licenses falls over, and the 60s positive cache is dropped so the request
    # MUST go upstream: only the 900s stale window can save it.
    fake.mode = "boom"
    entitlement._status_cache.clear()

    resp = client.get("/api/things")
    assert resp.status_code == 200, resp.text
    assert fake.active_calls == 2  # it really did try upstream and fail


def test_licenses_unreachable_with_no_warm_verdict_is_503():
    client, _ = client_for("boom")
    resp = client.get("/api/things")
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["error"] == "license_unavailable"


# ---------------------------------------------------------------------------
# Dormant by default
# ---------------------------------------------------------------------------


def test_gate_off_is_a_no_op(monkeypatch):
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)
    client, fake = client_for("revoked")
    resp = client.get("/api/things")
    assert resp.status_code == 200, resp.text
    assert fake.active_calls == 0  # not even a lookup


def test_flag_defaults_to_false(monkeypatch):
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)
    assert entitlement.license_gate_enabled() is False
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    assert entitlement.license_gate_enabled() is True


# ---------------------------------------------------------------------------
# Health exemption — platform rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/health", "/healthz", "/ready"])
def test_health_endpoints_are_exempt_even_for_a_revoked_tenant(path):
    client, _ = client_for("revoked")
    resp = client.get(path)
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "path", ["/health", "/healthz", "/ready", "/health/live", "/api/health/ready"]
)
def test_exempt_path_predicate(path):
    assert is_license_exempt_path(path) is True


@pytest.mark.parametrize("path", ["/api/things", "/api/projects/1", "/healthy-tenant"])
def test_non_exempt_paths_are_gated(path):
    assert is_license_exempt_path(path) is False


# ---------------------------------------------------------------------------
# Exempt PREFIXES match at a SEGMENT BOUNDARY, never as a raw string prefix
#
# `is_license_exempt_path` matched exempt prefixes with a bare `str.startswith`.
# `"/docsomething".startswith("/docs")` is True, so a route named `/docs-export`
# or `/redocument` would have been silently UNAUTHENTICATED (AuthMiddleware
# shares the same prefix tuple) AND UNBILLED, with nothing to flag it.
#
# Severity, stated honestly: a fleet-wide sweep found NO route occupying any of
# these shapes, so this was a LANDMINE, not a live bypass. It is fixed before a
# route grows into it, not after.
#
# The exemption SET is unchanged — only how a prefix is matched.
# ---------------------------------------------------------------------------


#: Every liveness/readiness path the fleet actually probes. A 402 on any of
#: these crashloops the pod, so these assertions are the real blast radius of
#: this change — they must stay exempt. `/health` is tts-gateway's probe and
#: `/healthz` is video-compiler's.
_REAL_PROBE_PATHS = [
    "/health",
    "/health/live",
    "/health/ready",
    "/healthz",
    "/readyz",
    "/livez",
    "/live",
    "/ready",
    "/api/health",
    "/api/health/live",
    "/api/health/ready",
]


@pytest.mark.parametrize("path", _REAL_PROBE_PATHS)
def test_every_real_probe_path_stays_exempt(path):
    """The one thing this change must not break."""
    assert is_license_exempt_path(path) is True, (
        f"{path} is a live k8s probe path; a non-exempt probe is a 402 on liveness "
        "and a CrashLoopBackOff across the fleet"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/docs",
        "/docs/",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
        "/metrics",
        "/favicon.ico",
        "/health/",
        "/health/live/",
    ],
    ids=lambda p: p,
)
def test_legitimate_public_paths_stay_exempt(path):
    """Swagger, the OpenAPI schema and sub-paths of an exempt prefix keep working."""
    assert is_license_exempt_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/docsomething/anything",
        "/docs-internal/render",
        "/redocument",
    ],
    ids=lambda p: p,
)
def test_raw_string_prefix_no_longer_grants_exemption(path):
    """The measured landmine. Each of these passed `startswith` before this fix."""
    assert is_license_exempt_path(path) is False, (
        f"{path} shares a string prefix with an exempt path but is a DIFFERENT "
        "route; matching it would make it unauthenticated and unbilled"
    )


@pytest.mark.parametrize(
    "path",
    ["/healthy", "/health-admin", "/readyz-admin", "/livezone", "/api/healthcheck"],
    ids=lambda p: p,
)
def test_sibling_names_that_merely_share_a_prefix_are_gated(path):
    assert is_license_exempt_path(path) is False


@pytest.mark.parametrize(
    "path",
    ["/health/../x", "/api/health/../designs", "/docs/../admin", "/health/./x"],
    ids=lambda p: p,
)
def test_traversal_segments_never_inherit_an_exemption(path):
    """A `.`/`..` segment resolves somewhere else, so it cannot ride the prefix.

    Not independently exploitable today — FastAPI routes on the raw path so
    these 404, and Istio normalises before the pod — but the gate must not be
    the layer that relies on either.
    """
    assert is_license_exempt_path(path) is False


@pytest.mark.parametrize("path", ["/health/diagnostics", "/api/health/deep"], ids=lambda p: p)
def test_subpaths_of_an_exempt_prefix_remain_exempt_by_design(path):
    """Documents a deliberate NON-change, so nobody 'fixes' it later.

    These were flagged alongside the raw-prefix strings above, but they are not
    boundary bugs: `/health/` and `/api/health/` ARE exempt prefixes, and these
    sit under them at a proper segment boundary. Excluding them would shrink the
    exemption SET — a different decision, and the one that risks 402ing a probe.
    """
    assert is_license_exempt_path(path) is True


def test_gate_and_auth_middleware_share_one_matcher():
    """The two exemption lists must not drift; they never had separate matchers."""
    from w2v_common.auth.middleware import PUBLIC_PREFIXES, matches_public_prefix

    assert entitlement.LICENSE_EXEMPT_PREFIXES == PUBLIC_PREFIXES
    for path in (*_REAL_PROBE_PATHS, "/docsomething", "/redocument", "/healthy"):
        assert matches_public_prefix(path) == is_license_exempt_path(path) or (
            path in entitlement.LICENSE_EXEMPT_PATHS
        )


@pytest.mark.parametrize("path", ["/docsomething/anything", "/redocument", "/healthy"])
def test_auth_middleware_also_refuses_the_lookalike_paths(path):
    """AuthMiddleware shares PUBLIC_PREFIXES; the auth half of the landmine.

    An unbilled route is bad; an UNAUTHENTICATED one is worse, and the bare
    `startswith` at the middleware's public-path check made both true at once.
    """
    from w2v_common.auth.middleware import matches_public_prefix

    assert matches_public_prefix(path) is False


# ---------------------------------------------------------------------------
# Tenant scoping of the 404 refinement
#
# ``GET /licenses/user/{sub}`` is SUBJECT-scoped, not tenant-scoped: it returns
# every licence the subject holds, in every tenant. ``derive_inactive_state``
# filtered on app key ALONE, so a revoked w2v row the subject holds in tenant A
# decided the verdict in tenant B — where they simply never had a licence. The
# 402 was right either way (this branch has no active exit, so it never wrongly
# ALLOWED); the REASON CODE was wrong, and that code is what the client renders
# and what support triages on. The 401/403 path already scoped both dimensions
# via ``_in_gate_scope``; this is the same predicate, applied here too.
# ---------------------------------------------------------------------------

OTHER_TENANT = "33333333-3333-3333-3333-333333333333"


def _revoked_record(tenant_id: str) -> dict:
    return {
        "id": "lic-other",
        "tenant_id": tenant_id,
        "status": "revoked",
        "revoked_at": "2024-01-01T00:00:00+00:00",
        "app": {"key": "w2v"},
    }


def test_derive_inactive_state_ignores_another_tenants_revoked_row():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = entitlement.derive_inactive_state([_revoked_record(OTHER_TENANT)], "w2v", TENANT, now)
    assert result.state == "none", "tenant A's revoked row must not label tenant B revoked"


def test_derive_inactive_state_still_sees_its_own_revoked_row():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = entitlement.derive_inactive_state([_revoked_record(TENANT)], "w2v", TENANT, now)
    assert result.state == "revoked"


def test_derive_inactive_state_ignores_a_row_with_no_tenant():
    """A record we cannot prove in-scope is out of scope — same rule as the gate."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    record = _revoked_record(TENANT)
    del record["tenant_id"]
    assert entitlement.derive_inactive_state([record], "w2v", TENANT, now).state == "none"


class CrossTenantLicenses(FakeLicenses):
    """404 on /licenses/active, and an inventory holding ANOTHER tenant's revocation."""

    def __init__(self) -> None:
        super().__init__("none")

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str):
        return [
            LicenseView(
                id="lic-other",
                tenant_id=OTHER_TENANT,
                status="revoked",
                revoked_at="2024-01-01T00:00:00+00:00",
                app=AppInfoView(key="w2v"),
            )
        ]


def test_another_tenants_revocation_does_not_mislabel_this_tenants_402():
    fake = CrossTenantLicenses()
    set_licenses_read_client(fake)
    client = TestClient(build_app(fake), raise_server_exceptions=False)
    resp = client.get("/api/things")
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == "license_none", (
        "tenant B has no licence at all — a revoked row belonging to tenant A "
        "must not turn this into license_revoked"
    )


# ---------------------------------------------------------------------------
# WebSocket gate
#
# WS routes were COMPLETELY ungated: require_license is typed for a Request and
# reads request.url.path / request.state, neither of which a WebSocket has, and
# nothing branched on scope["type"]. An expired tenant was refused on the HTTP
# API and served on the socket (live consumer: tts_service /api/tts/ws). These
# tests drive a REAL websocket route through TestClient for the same reason the
# HTTP ones drive a real route: a test on the derivation alone would stay green
# with the dependency unmounted.
# ---------------------------------------------------------------------------


def build_ws_app(tenant: str | None = TENANT) -> FastAPI:
    """A WS route gated by LICENSE_GATE_WS.

    AuthMiddleware is a BaseHTTPMiddleware and never runs on a websocket scope,
    so identity is stamped by a dependency ordered BEFORE the gate — which is
    exactly what a real service has to do.
    """
    app = FastAPI()

    async def _stamp(websocket: WebSocket) -> None:
        if tenant is not None:
            websocket.state.tenant_id = tenant
            websocket.state.subject_id = SUBJECT

    @app.websocket("/api/stream", dependencies=[Depends(_stamp), *LICENSE_GATE_WS])
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return app


def ws_client_for(mode: str, tenant: str | None = TENANT) -> TestClient:
    set_licenses_read_client(FakeLicenses(mode))
    return TestClient(build_ws_app(tenant), raise_server_exceptions=False)


def test_ws_gate_off_lets_the_handshake_through(monkeypatch):
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)
    with ws_client_for("revoked").websocket_connect("/api/stream") as ws:
        assert ws.receive_text() == "hello"


def test_ws_licensed_tenant_completes_the_handshake():
    with ws_client_for("active").websocket_connect("/api/stream") as ws:
        assert ws.receive_text() == "hello"


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [("revoked", "license_revoked"), ("expired", "license_expired"), ("none", "license_none")],
)
def test_ws_unlicensed_tenant_is_closed_with_the_reason(mode: str, expected_code: str):
    client = ws_client_for(mode)
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/api/stream"):
        pass  # pragma: no cover — the handshake must never get here
    assert excinfo.value.code == entitlement.LICENSE_WS_CLOSE_POLICY
    assert excinfo.value.reason.startswith(expected_code)


def test_ws_with_no_tenant_is_closed_rather_than_admitted():
    """No identity stamped on the socket ⇒ 'none' ⇒ closed. Fail-CLOSED."""
    client = ws_client_for("active", tenant=None)
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/api/stream"):
        pass  # pragma: no cover
    assert excinfo.value.code == entitlement.LICENSE_WS_CLOSE_POLICY
    assert excinfo.value.reason.startswith("license_none")


def test_ws_licences_outage_closes_retryable_not_policy():
    """A licences outage must not look like a billing refusal — 1013, not 1008."""
    client = ws_client_for("boom")
    with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/api/stream"):
        pass  # pragma: no cover
    assert excinfo.value.code == entitlement.LICENSE_WS_CLOSE_UNAVAILABLE
    assert excinfo.value.reason.startswith(entitlement.LICENSE_UNAVAILABLE_ERROR)


def test_ws_close_reason_fits_the_rfc6455_byte_cap():
    reason = entitlement._ws_close_reason("license_revoked", "x" * 500)
    assert len(reason.encode("utf-8")) <= 123
    assert reason.startswith("license_revoked")


def test_require_license_http_behaviour_is_unchanged_by_the_ws_twin():
    """The HTTP dependency still returns 402 — it was not rerouted onto closes."""
    client, _ = client_for("revoked")
    resp = client.get("/api/things")
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == "license_revoked"
