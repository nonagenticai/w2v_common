"""Direct w2v cost events are attributed to the W2V project and job (0.85.0).

Owner ruling 2026-10-07: a cost row's ``project_id`` is the CALLING APP's own
project. For w2v that is the W2V project (``X-W2V-Project-Id`` /
``set_w2v_project_context``) — never the fr project (``X-Project-Id``). Before
0.85.0 the cost bus defaulted to the fr project, so every direct w2v emit (tts,
video-analyzer, video-compiler, html-render, image-gen) shipped ``project_id``
NULL. ``metadata.job_id`` likewise defaults to the bound W2V job.
"""

from __future__ import annotations

import ipaddress
import uuid
from typing import Any

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY
from starlette.requests import Request

import w2v_common.auth.middleware as mw
from w2v_common.auth.identity_context import (
    W2V_JOB_ID_HEADER,
    W2V_PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    clear_identity_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
    identity_event_hook,
    set_identity_context,
    set_w2v_job_context,
    set_w2v_project_context,
    tenant_level_project_id,
)
from w2v_common.metering import cost_bus


@pytest.fixture(autouse=True)
def _trust_the_loopback_test_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the test client's peer (127.0.0.1) in TRUSTED_CIDRS explicitly.

    TRUSTED_CIDRS defaults to EMPTY since 0.89.0, so a trust lane is inert
    unless the operator names the networks — this is that opt-in, scoped to
    the ASGITransport peer, not a re-widening of the default.
    """
    monkeypatch.setattr(mw, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")])


_TENANT = uuid.UUID("0195f48b-0000-7000-8000-000000000abc")
_OTHER_TENANT = uuid.UUID("0195f48b-0000-7000-8000-000000000def")
_FR_PROJECT = "0195f48b-1111-7000-8000-00000000beef"
_W2V_PROJECT = "0195f48b-4444-7000-8000-0000000000aa"
_RUN_ID = "0195f48b-5555-7000-8000-0000000000bb"
_TENANT_LEVEL = tenant_level_project_id(_TENANT)


@pytest.fixture(autouse=True)
def _reset() -> Any:
    clear_identity_context()
    yield
    clear_identity_context()


def _fields(**overrides: Any) -> dict[str, str]:
    kwargs: dict[str, Any] = dict(
        service_key="w2v-test",  # pragma: allowlist secret -- routing slug
        cost_source="w2v-test",
        tenant_id=_TENANT,
        license_id="",
        subject_id="",
        app_id="",
        provider="kokoro",
        model="kokoro-v1",
        served_model=None,
        input_tokens=1,
        output_tokens=0,
        latency_ms=3,
        cost_usd=0.0,
        provider_request_id="",
        usage_extra=None,
    )
    kwargs.update(overrides)
    return cost_bus._build_fields(**kwargs)


def _fallback(reason: str) -> float:
    value = REGISTRY.get_sample_value("w2v_cost_project_fallback_total", {"reason": reason})
    assert value is not None, f"series reason={reason} is not exported"
    return value


# ---------------------------------------------------------------------------
# project_id
# ---------------------------------------------------------------------------


def test_cost_event_takes_the_w2v_project_from_context() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    before = _fallback("tenant_level")
    assert _fields()["project_id"] == _W2V_PROJECT
    assert _fallback("tenant_level") == before


def test_fr_project_is_never_the_cost_project() -> None:
    """An fr project bound from X-Project-Id must not leak onto a w2v cost row."""
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Project-Id": _FR_PROJECT}, trusted=False)
    fields = _fields()
    assert fields["project_id"] != _FR_PROJECT
    assert fields["project_id"] == _TENANT_LEVEL
    # ...and with a W2V project also bound, the W2V one wins.
    set_w2v_project_context(_W2V_PROJECT)
    assert _fields()["project_id"] == _W2V_PROJECT


def test_tenant_level_fallback_is_counted() -> None:
    set_identity_context(_TENANT, None, "pays")
    before = _fallback("tenant_level")
    assert _fields()["project_id"] == _TENANT_LEVEL
    assert _fallback("tenant_level") == before + 1


def test_tenant_level_project_is_the_events_tenant_not_a_stale_one() -> None:
    """The fallback derives from the event's tenant argument."""
    set_identity_context(_TENANT, None, "pays")
    other_level = tenant_level_project_id(_OTHER_TENANT)
    assert _fields(tenant_id=_OTHER_TENANT)["project_id"] == other_level
    assert other_level != _TENANT_LEVEL


def test_no_project_reason_counted_when_nothing_derivable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cost_bus, "tenant_level_project_id", lambda _t: None)
    set_identity_context(_TENANT, None, "pays")
    before = _fallback("no_project")
    assert "project_id" not in _fields()
    assert _fallback("no_project") == before + 1


def test_fallback_series_are_preinitialised() -> None:
    for reason in cost_bus.COST_PROJECT_FALLBACK_REASONS:
        assert _fallback(reason) >= 0


# ---------------------------------------------------------------------------
# job_id
# ---------------------------------------------------------------------------


def test_job_id_defaults_from_the_w2v_job_context() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_job_context(_RUN_ID)
    assert _fields()["job_id"] == _RUN_ID


def test_job_id_absent_when_unbound() -> None:
    set_identity_context(_TENANT, None, "pays")
    assert "job_id" not in _fields()


def test_explicit_job_id_wins_over_context() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_job_context(_RUN_ID)
    assert _fields(metadata_extra={"job_id": "tts-job-7"})["job_id"] == "tts-job-7"


def test_explicit_none_job_id_suppresses_the_ambient_job() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_job_context(_RUN_ID)
    assert "job_id" not in _fields(metadata_extra={"job_id": None})


def test_job_id_defaults_alongside_other_metadata() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_job_context(_RUN_ID)
    fields = _fields(metadata_extra={"stage": "narrate"})
    assert fields["job_id"] == _RUN_ID
    assert fields["stage"] == "narrate"


@pytest.mark.parametrize("bad", ["", "has space", "a\r\nX-Evil: 1", "x" * 129, "-leading"])
def test_set_w2v_job_context_is_strict(bad: str) -> None:
    with pytest.raises(ValueError, match="w2v job id"):
        set_w2v_job_context(bad)


def test_tenant_change_clears_the_w2v_job_and_project() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    set_w2v_job_context(_RUN_ID)
    set_identity_context(_TENANT, None, "pays")  # same tenant: kept
    assert get_w2v_job_id_from_context() == _RUN_ID
    set_identity_context(_OTHER_TENANT, None, "pays")  # different tenant: cleared
    assert get_w2v_job_id_from_context() == ""
    assert get_w2v_project_id_from_context() == ""


async def test_emit_puts_w2v_project_and_job_on_the_stream() -> None:
    redis_conn = fakeredis.aioredis.FakeRedis(decode_responses=True)
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    set_w2v_job_context(_RUN_ID)
    await cost_bus.emit_cost_event(
        redis_conn,
        service_key="tts-service",  # pragma: allowlist secret -- routing slug
        cost_source="tts",
        tenant_id=_TENANT,
        provider="kokoro",
        model="kokoro-v1",
        input_tokens=10,
        output_tokens=0,
        latency_ms=5,
        license_id="lic",
        subject_id="sub",
        app_id="w2v",
        stream="test:costs",
    )
    entries = await redis_conn.xrange("test:costs")
    assert len(entries) == 1
    _, fields = entries[0]
    assert fields["project_id"] == _W2V_PROJECT
    assert fields["job_id"] == _RUN_ID
    await redis_conn.aclose()


# ---------------------------------------------------------------------------
# header round trip: orchestrator -> downstream hop -> cost event
# ---------------------------------------------------------------------------


async def test_outbound_hook_stamps_w2v_project_and_job() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    set_w2v_job_context(_RUN_ID)
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert request.headers[W2V_PROJECT_ID_HEADER] == _W2V_PROJECT
    assert request.headers[W2V_JOB_ID_HEADER] == _RUN_ID


async def test_outbound_hook_omits_unbound_w2v_job() -> None:
    set_identity_context(_TENANT, None, "pays")
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert W2V_JOB_ID_HEADER not in request.headers
    assert W2V_PROJECT_ID_HEADER not in request.headers


def test_malformed_inbound_job_header_is_ignored() -> None:
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({W2V_JOB_ID_HEADER: "bad value"}, trusted=False)
    assert get_w2v_job_id_from_context() == ""


async def test_header_round_trip_through_a_downstream_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orchestrator stamps both headers; the downstream middleware binds them;
    the downstream's own cost event carries them; its next hop forwards them."""
    # Upstream (orchestrator) side: stamp the outbound request.
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    set_w2v_job_context(_RUN_ID)
    upstream = httpx.Request("POST", "http://tts.invalid/api/tts/generate")
    await identity_event_hook(upstream)
    clear_identity_context()

    # Downstream (e.g. tts) side, behind the real AuthMiddleware.
    monkeypatch.setenv("AUTH_TRUST_INCLUSTER_IDENTITY", "true")
    app = FastAPI()
    app.add_middleware(mw.AuthMiddleware)

    @app.get("/api/emit")
    async def emit(request: Request) -> dict[str, Any]:
        onward = httpx.Request("GET", "http://next.invalid/x")
        await identity_event_hook(onward)
        fields = _fields()
        return {
            "cost_project": fields.get("project_id"),
            "cost_job": fields.get("job_id"),
            "onward_project": onward.headers.get(W2V_PROJECT_ID_HEADER),
            "onward_job": onward.headers.get(W2V_JOB_ID_HEADER),
        }

    headers = {
        "X-Tenant-Id": str(_TENANT),
        "X-Project-Id": _FR_PROJECT,  # an fr project on the wire must not win
        W2V_PROJECT_ID_HEADER: upstream.headers[W2V_PROJECT_ID_HEADER],
        W2V_JOB_ID_HEADER: upstream.headers[W2V_JOB_ID_HEADER],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.get("/api/emit", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "cost_project": _W2V_PROJECT,
        "cost_job": _RUN_ID,
        "onward_project": _W2V_PROJECT,
        "onward_job": _RUN_ID,
    }
    # Nothing leaked out of the request scope.
    assert get_w2v_job_id_from_context() == ""
    assert get_w2v_project_id_from_context() == ""
