"""A licenses-authored "no active licence" 404 must 402, not 503.

Observed from mini1's live venv (w2v_common 0.80.0): a delegated request for a
tenant with no w2v licence got 404 from ``/licenses/active``, then the reason
refinement read ``/licenses/user/{subject}``, which refuses a delegating SA with
403. That exception escaped to the fail-soft and the gate answered 503
``license_unavailable`` — "the platform is broken, retry" — for a tenant the
licences service had just authoritatively said holds no licence.

The trap these tests hold down in the other direction: a 404 is ALSO what a
wrong base URL, a missing route or a proxy with no matching path returns. Only
the 404 carrying licenses' own discriminator (``detail.error ==
"no_active_license"``, licenses ``_resolve_active_entitlement``) is
authoritative; any other 404 keeps today's behaviour.

These drive the REAL ``LicensesReadClient`` over an ``httpx.MockTransport`` so the
discriminator is exercised on the wire shape, not on a fake's return value.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from w2v_common.auth import licenses_read_client as lrc
from w2v_common.auth.entitlement import LICENSE_GATE, clear_license_status_cache

TENANT = "11111111-1111-1111-1111-111111111111"
SUBJECT = "22222222-2222-2222-2222-222222222222"


class Upstream:
    """Scripted licenses service; records every path it is asked for."""

    def __init__(
        self, active: Callable[[], httpx.Response], user: Callable[[], httpx.Response]
    ) -> None:
        self._active = active
        self._user = user
        self.active_calls = 0
        self.user_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/licenses/active":
            self.active_calls += 1
            return self._active()
        if request.url.path.startswith("/licenses/user/"):
            self.user_calls += 1
            return self._user()
        raise AssertionError(f"unexpected path {request.url.path}")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_license_status_cache()
    lrc.set_licenses_read_client(None)
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    yield
    clear_license_status_cache()
    lrc.set_licenses_read_client(None)


def gate_client(upstream: Upstream, *, subject: str | None = SUBJECT) -> TestClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    reader = lrc.LicensesReadClient(
        base_url="http://licenses", client=http, token_provider=_no_token
    )
    lrc.set_licenses_read_client(reader)

    app = FastAPI()
    gated = APIRouter(dependencies=list(LICENSE_GATE))

    @gated.get("/api/things")
    async def things() -> dict[str, str]:
        return {"ok": "yes"}

    app.include_router(gated)

    @app.middleware("http")
    async def _stamp_identity(request, call_next):
        request.state.tenant_id = TENANT
        request.state.subject_id = subject
        return await call_next(request)

    return TestClient(app, raise_server_exceptions=False)


async def _no_token() -> str | None:
    return None


def _json(status: int, body: object) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(status, json=body)


def _authoritative_404() -> httpx.Response:
    return httpx.Response(
        404,
        json={"detail": {"error": "no_active_license", "tenant_id": TENANT, "app_key": "w2v"}},
    )


def _inventory_403() -> httpx.Response:
    return httpx.Response(403, json={"detail": "Not authorized to read another subject's licenses"})


# ---------------------------------------------------------------------------
# (1) authoritative no-licence 404 → 402 license_none
# ---------------------------------------------------------------------------


def test_authoritative_404_without_subject_is_license_none_and_skips_inventory():
    upstream = Upstream(active=_authoritative_404, user=_inventory_403)
    resp = gate_client(upstream, subject=None).get("/api/things")
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == "license_none"
    assert upstream.user_calls == 0


def test_authoritative_404_with_inventory_refused_is_402_not_503():
    """THE live defect: the refinement's 403 must not turn a settled refusal into 503."""
    upstream = Upstream(active=_authoritative_404, user=_inventory_403)
    resp = gate_client(upstream).get("/api/things")
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == "license_none"
    assert upstream.user_calls == 1


def test_authoritative_404_still_refines_the_reason_when_inventory_is_readable():
    """The refinement is deliberate (#178): keep revoked/expired when we CAN read it."""
    upstream = Upstream(
        active=_authoritative_404,
        user=_json(
            200,
            [
                {
                    "id": "lic-y",
                    "tenant_id": TENANT,
                    "status": "revoked",
                    "revoked_at": "2024-01-01T00:00:00+00:00",
                    "app": {"key": "w2v"},
                }
            ],
        ),
    )
    resp = gate_client(upstream).get("/api/things")
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == "license_revoked"


# ---------------------------------------------------------------------------
# (2) a 404 WITHOUT the discriminator is NOT authoritative — unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route_miss",
    [
        _json(404, {"detail": "Not Found"}),  # FastAPI route miss
        lambda: httpx.Response(404, text="404 page not found"),  # proxy / wrong base URL
        _json(404, {"detail": {"error": "something_else"}}),
        lambda: httpx.Response(404),
    ],
)
def test_route_miss_404_with_inventory_refused_stays_503(route_miss):
    upstream = Upstream(active=route_miss, user=_inventory_403)
    resp = gate_client(upstream).get("/api/things")
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["error"] == "license_unavailable"


async def test_client_returns_none_for_a_route_miss_404_and_raises_only_on_the_discriminator():
    for response, authoritative in [
        (_authoritative_404, True),
        (_json(404, {"detail": "Not Found"}), False),
        (lambda: httpx.Response(404, text="nope"), False),
        (_json(404, ["no_active_license"]), False),
    ]:
        upstream = Upstream(active=response, user=_inventory_403)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler)) as http:
            reader = lrc.LicensesReadClient(
                base_url="http://licenses", client=http, token_provider=_no_token
            )
            if authoritative:
                with pytest.raises(lrc.LicenseNotActiveError) as info:
                    await reader.get_active_license(tenant_id=TENANT, app_key="w2v")
                assert info.value.status_code == 404
            else:
                assert await reader.get_active_license(tenant_id=TENANT, app_key="w2v") is None


# ---------------------------------------------------------------------------
# (3) 403 on /licenses/active → the deliberate inventory fallback, unchanged
# ---------------------------------------------------------------------------


def test_403_on_active_route_still_falls_back_to_inventory():
    upstream = Upstream(
        active=_json(403, {"detail": "forbidden"}),
        user=_json(
            200,
            [
                {
                    "id": "lic-1",
                    "tenant_id": TENANT,
                    "status": "active",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                    "app": {"key": "w2v"},
                }
            ],
        ),
    )
    resp = gate_client(upstream).get("/api/things")
    assert resp.status_code == 200, resp.text
    assert upstream.user_calls == 1


def test_403_on_active_route_with_inventory_refused_stays_503():
    upstream = Upstream(active=_json(403, {"detail": "forbidden"}), user=_inventory_403)
    resp = gate_client(upstream).get("/api/things")
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["error"] == "license_unavailable"


# ---------------------------------------------------------------------------
# (4) 200 → entitled, unchanged
# ---------------------------------------------------------------------------


def test_200_active_license_passes_without_inventory():
    upstream = Upstream(
        active=_json(
            200,
            {
                "license_id": "lic-1",
                "tenant_id": TENANT,
                "expires_at": "2099-01-01T00:00:00+00:00",
                "app": {"key": "w2v"},
            },
        ),
        user=_inventory_403,
    )
    resp = gate_client(upstream).get("/api/things")
    assert resp.status_code == 200, resp.text
    assert upstream.user_calls == 0
