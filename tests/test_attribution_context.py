"""X-Project-Id / X-Actor-Id attribution (ATTRIBUTION-PLAN "THE CONTRACT").

Headers are UUID-only and OMITTED when absent; billing reads the top-level
``project_id`` / ``actor_id`` of a cost event; ``X-Actor-Id`` is honoured only
from a trusted (delegating) caller.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

import w2v_common.auth.identity_context as idc
import w2v_common.auth.middleware as mw
from w2v_common.auth.identity_context import (
    bind_attribution_from_headers,
    clear_identity_context,
    get_actor_id_from_context,
    get_identity_headers,
    get_project_id_from_context,
    identity_event_hook,
    parse_attribution_uuid,
    set_identity_context,
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
_PROJECT = "0195f48b-1111-7000-8000-00000000beef"
_ACTOR = "0195f48b-2222-7000-8000-00000000cafe"
_SA_SUBJECT = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
_W2V_PROJECT = "0195f48b-4444-7000-8000-0000000000aa"
_TENANT_LEVEL = tenant_level_project_id(_TENANT)


@pytest.fixture(autouse=True)
def _reset() -> Any:
    clear_identity_context()
    for name in (
        "_license_cache",
        "_license_neg_cache",
        "_license_stale_cache",
        "_license_token_not_found_since",
        "_license_tnf_probe",
    ):
        setattr(mw, name, None)
    mw._introspect_locks.clear()
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
        provider="deepseek",
        model="deepseek-chat",
        served_model=None,
        input_tokens=1,
        output_tokens=2,
        latency_ms=3,
        cost_usd=0.0,
        provider_request_id="",
        usage_extra=None,
    )
    kwargs.update(overrides)
    return cost_bus._build_fields(**kwargs)


# ---------------------------------------------------------------------------
# strict UUID parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not-a-uuid",
        "00000000-0000-0000-0000-000000000000",  # the nil placeholder
        "0195f48b111170008000" + "00000000beef",  # bare hex: not canonical
        "{" + _PROJECT + "}",
        "urn:uuid:" + _PROJECT,
    ],
)
def test_parse_attribution_uuid_refuses_non_canonical_and_placeholder(raw: str) -> None:
    assert parse_attribution_uuid(raw) is None


def test_parse_attribution_uuid_canonicalises_case() -> None:
    assert parse_attribution_uuid(_PROJECT.upper()) == _PROJECT


# ---------------------------------------------------------------------------
# bind_attribution_from_headers
# ---------------------------------------------------------------------------


def test_project_bound_from_any_caller() -> None:
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Project-Id": _PROJECT}, trusted=False)
    assert get_project_id_from_context() == _PROJECT


def test_invalid_project_uuid_is_ignored_with_a_debug_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_identity_context(_TENANT, None, "pays")
    with caplog.at_level(logging.DEBUG, logger="w2v_common.auth.identity_context"):
        bind_attribution_from_headers(
            {"X-Project-Id": "proj-1", "X-Actor-Id": "someone"}, trusted=True
        )
    assert get_project_id_from_context() == ""
    assert get_actor_id_from_context() == ""
    assert any("non-UUID X-Project-Id" in r.getMessage() for r in caplog.records)
    assert any("non-UUID X-Actor-Id" in r.getMessage() for r in caplog.records)


def test_actor_not_bound_from_an_untrusted_caller() -> None:
    """The lane's own actor (here the JWT sub) survives an untrusted assertion."""
    set_identity_context(_TENANT, None, "pays", actor_id="jwt-sub")
    bind_attribution_from_headers({"X-Actor-Id": _ACTOR}, trusted=False)
    assert get_actor_id_from_context() == "jwt-sub"


def test_actor_bound_from_a_trusted_caller() -> None:
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"x-actor-id": _ACTOR}, trusted=True)
    assert get_actor_id_from_context() == _ACTOR


def test_reseed_clears_the_project() -> None:
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Project-Id": _PROJECT}, trusted=False)
    set_identity_context(_TENANT, None, "pays")
    assert get_project_id_from_context() == ""


# ---------------------------------------------------------------------------
# cost event: top-level project_id / actor_id
# ---------------------------------------------------------------------------


def test_w2v_project_emitted_top_level_never_the_fr_project() -> None:
    """Owner ruling 2026-10-07: the cost row's project is the w2v project."""
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers(
        {"X-Project-Id": _PROJECT, "X-W2V-Project-Id": _W2V_PROJECT, "X-Actor-Id": _ACTOR},
        trusted=True,
    )
    fields = _fields()
    assert fields["project_id"] == _W2V_PROJECT
    assert fields["actor_id"] == _ACTOR


def test_unbound_project_falls_back_to_the_tenant_level_project() -> None:
    set_identity_context(_TENANT, None, "pays")
    fields = _fields()
    assert fields["project_id"] == _TENANT_LEVEL
    assert "actor_id" not in fields


# ---------------------------------------------------------------------------
# cost event actor: bound human > SERVICE_ACTOR_ID (service-originated work
# only) > absent-and-counted. These run with NO inbound request marked, i.e.
# as a background task / scheduler would.
# ---------------------------------------------------------------------------


def _actor_missing(service: str, reason: str = idc.ACTOR_MISSING_NO_SERVICE_ACTOR) -> float:
    return idc.actor_missing_total.labels(service=service, reason=reason)._value.get()


def test_service_actor_id_is_the_fallback_when_no_actor_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    before = _actor_missing("w2v-test")
    assert _fields()["actor_id"] == _SA_SUBJECT
    assert _actor_missing("w2v-test") == before


def test_a_bound_human_actor_wins_over_service_actor_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Actor-Id": _ACTOR}, trusted=True)
    assert _fields()["actor_id"] == _ACTOR


def test_no_actor_and_no_service_actor_is_omitted_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERVICE_ACTOR_ID", raising=False)
    set_identity_context(_TENANT, None, "pays")
    before = _actor_missing("w2v-test")
    assert "actor_id" not in _fields()
    assert _actor_missing("w2v-test") == before + 1


@pytest.mark.parametrize(
    "bad",
    ["not-a-uuid", "00000000-0000-0000-0000-000000000000", "{" + _SA_SUBJECT + "}", "acted"],
)
def test_a_malformed_service_actor_id_is_refused_and_counted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", bad)
    monkeypatch.setattr(idc, "_warned_bad_service_actor", set())
    set_identity_context(_TENANT, None, "pays")
    before = _actor_missing("w2v-test")
    with caplog.at_level(logging.WARNING, logger=idc.__name__):
        fields = _fields()
        _fields()
    assert "actor_id" not in fields
    assert _actor_missing("w2v-test") == before + 2
    assert sum("SERVICE_ACTOR_ID" in r.getMessage() for r in caplog.records) == 1


def test_a_malformed_bound_actor_falls_back_to_service_actor_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-UUID lane value (mesh header verbatim) is never shipped as the actor."""
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays", actor_id="jwt-sub")
    assert _fields()["actor_id"] == _SA_SUBJECT


def test_an_explicit_none_actor_still_suppresses_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    assert "actor_id" not in _fields(metadata_extra={"actor_id": None})


# ---------------------------------------------------------------------------
# 0.88.0: serving an INBOUND request, the service never names itself.
# ---------------------------------------------------------------------------


def test_background_work_with_no_request_uses_the_service_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    assert not idc.is_inbound_request()
    assert _fields()["actor_id"] == _SA_SUBJECT
    assert get_identity_headers()["X-Actor-Id"] == _SA_SUBJECT


async def test_inbound_request_without_actor_is_absent_and_counted_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    idc.mark_inbound_request()
    reason = idc.ACTOR_MISSING_INBOUND
    cost_before = _actor_missing("w2v-test", reason)
    out_before = _actor_missing("outbound", reason)

    assert "actor_id" not in _fields()
    assert _actor_missing("w2v-test", reason) == cost_before + 1

    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert "X-Actor-Id" not in request.headers
    assert _SA_SUBJECT not in request.headers.values()
    assert _actor_missing("outbound", reason) == out_before + 1


async def test_inbound_request_with_an_actor_forwards_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    idc.mark_inbound_request()
    bind_attribution_from_headers({"X-Actor-Id": _ACTOR}, trusted=True)
    assert _fields()["actor_id"] == _ACTOR
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert request.headers.get_list("X-Actor-Id") == [_ACTOR]


def test_clear_identity_context_ends_the_inbound_mark() -> None:
    idc.mark_inbound_request()
    clear_identity_context()
    assert not idc.is_inbound_request()


def test_explicit_project_arg_overrides_context() -> None:
    other = "0195f48b-3333-7000-8000-000000000001"
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    assert _fields(project_id=other)["project_id"] == other


def test_explicit_non_uuid_project_arg_is_dropped_to_the_default_chain() -> None:
    set_identity_context(_TENANT, None, "pays")
    set_w2v_project_context(_W2V_PROJECT)
    assert _fields(project_id="not-a-uuid")["project_id"] == _W2V_PROJECT


def test_project_is_not_smuggled_through_metadata_extra() -> None:
    """billing reads project_id top-level, not from metadata; keep one path."""
    set_identity_context(_TENANT, None, "pays")
    assert _fields(metadata_extra={"project_id": _PROJECT})["project_id"] == _TENANT_LEVEL


# ---------------------------------------------------------------------------
# outbound forwarding
# ---------------------------------------------------------------------------


@pytest.fixture
def _forward_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ATTRIBUTION_FORWARD_HEADERS", "true")


@pytest.fixture
def _forward_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTH_ATTRIBUTION_FORWARD_HEADERS", raising=False)


# 0.81.0's outbound identity header set. With forwarding not opted in, every
# lane emits exactly this plus X-Actor-Id when a UUID actor is bound (0.86.0).
_V081_IDENTITY_KEYS = {"X-Tenant-Id", "X-License-Id", "X-Subject-Id", "X-Correlation-Id", "X-Stage"}


@pytest.mark.usefixtures("_forward_off")
async def test_flag_off_forwards_the_actor_but_not_the_project() -> None:
    """0.86.0: X-Actor-Id is unconditional; only X-Project-Id stays opt-in."""
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR}, trusted=True)
    # Binding still happens (cost events keep it) ...
    assert get_project_id_from_context() == _PROJECT
    assert get_actor_id_from_context() == _ACTOR
    # ... the actor goes on the wire, the fr project does not.
    assert set(get_identity_headers()) == _V081_IDENTITY_KEYS | {"X-Actor-Id"}
    assert get_identity_headers()["X-Actor-Id"] == _ACTOR
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert "X-Project-Id" not in request.headers
    assert request.headers["X-Actor-Id"] == _ACTOR


@pytest.mark.usefixtures("_forward_off")
async def test_unbound_actor_forwards_the_calling_services_own_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0.87.0: downstream records the CALLING service as actor, not itself."""
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    assert get_identity_headers()["X-Actor-Id"] == _SA_SUBJECT
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert request.headers["X-Actor-Id"] == _SA_SUBJECT


@pytest.mark.usefixtures("_forward_off")
async def test_a_bound_human_actor_wins_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Actor-Id": _ACTOR}, trusted=True)
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert request.headers.get_list("X-Actor-Id") == [_ACTOR]


@pytest.mark.usefixtures("_forward_off")
@pytest.mark.parametrize("sa", [None, "not-a-uuid", "00000000-0000-0000-0000-000000000000"])
async def test_no_valid_actor_anywhere_sends_no_header(
    monkeypatch: pytest.MonkeyPatch, sa: str | None
) -> None:
    if sa is None:
        monkeypatch.delenv("SERVICE_ACTOR_ID", raising=False)
    else:
        monkeypatch.setenv("SERVICE_ACTOR_ID", sa)
    set_identity_context(_TENANT, None, "pays")
    assert "X-Actor-Id" not in get_identity_headers()
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert "X-Actor-Id" not in request.headers


@pytest.mark.usefixtures("_forward_on")
def test_get_identity_headers_forwards_bound_attribution() -> None:
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR}, trusted=True)
    headers = get_identity_headers()
    assert headers["X-Project-Id"] == _PROJECT
    assert headers["X-Actor-Id"] == _ACTOR


@pytest.mark.usefixtures("_forward_on")
def test_get_identity_headers_omits_unbound_attribution() -> None:
    set_identity_context(_TENANT, None, "pays")
    headers = get_identity_headers()
    assert "X-Project-Id" not in headers
    assert "X-Actor-Id" not in headers


@pytest.mark.usefixtures("_forward_on")
async def test_identity_event_hook_forwards_bound_attribution() -> None:
    set_identity_context(_TENANT, None, "pays")
    bind_attribution_from_headers({"X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR}, trusted=True)
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert request.headers["X-Project-Id"] == _PROJECT
    assert request.headers["X-Actor-Id"] == _ACTOR


@pytest.mark.usefixtures("_forward_on")
async def test_identity_event_hook_sends_no_attribution_when_unbound() -> None:
    set_identity_context(_TENANT, None, "pays")
    request = httpx.Request("GET", "http://example.invalid/x")
    await identity_event_hook(request)
    assert "X-Project-Id" not in request.headers
    assert "X-Actor-Id" not in request.headers


# ---------------------------------------------------------------------------
# AuthMiddleware wiring + per-request lifecycle
# ---------------------------------------------------------------------------


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(mw.AuthMiddleware)

    @app.get("/api/attr")
    async def attr(request: Request) -> dict[str, str]:
        return {
            "project_id": get_project_id_from_context(),
            "actor_id": get_actor_id_from_context(),
            "cost_project": _fields().get("project_id", "<absent>"),
        }

    return app


async def _get(headers: dict[str, str]) -> dict[str, str]:
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://t") as client:
        resp = await client.get("/api/attr", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


@patch("w2v_common.auth.middleware._introspect_license")
async def test_license_lane_non_delegating_caller_cannot_assert_actor(
    mock_introspect: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
    monkeypatch.delenv("AUTH_ACTOR_ASSERTING_SUBJECTS", raising=False)
    mock_introspect.return_value = {
        "active": True,
        "tenant_id": str(_TENANT),
        "subject_id": "customer-subject",
    }
    body = await _get({"X-Tenant-Id": "Bearer tok", "X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR})
    assert body["project_id"] == _PROJECT
    # The fr project is bound but never becomes the cost row's project.
    assert body["cost_project"] == _TENANT_LEVEL
    assert body["actor_id"] == ""


@patch("w2v_common.auth.middleware._introspect_license")
async def test_license_lane_delegating_subject_alone_cannot_assert_actor(
    mock_introspect: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant re-scoping trust (AUTH_DELEGATING_SUBJECTS) confers NO actor trust."""
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", _SA_SUBJECT)
    monkeypatch.delenv("AUTH_ACTOR_ASSERTING_SUBJECTS", raising=False)
    mock_introspect.return_value = {
        "active": True,
        "tenant_id": str(_TENANT),
        "subject_id": _SA_SUBJECT,
    }
    body = await _get(
        {"X-Tenant-Id": "Bearer sa-tok", "X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR}
    )
    assert body["project_id"] == _PROJECT
    assert body["actor_id"] == ""


@patch("w2v_common.auth.middleware._introspect_license")
async def test_license_lane_actor_asserting_subject_may_assert_actor(
    mock_introspect: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actor trust comes ONLY from AUTH_ACTOR_ASSERTING_SUBJECTS (not delegating)."""
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
    monkeypatch.setenv("AUTH_ACTOR_ASSERTING_SUBJECTS", f"other-sub, {_SA_SUBJECT}")
    mock_introspect.return_value = {
        "active": True,
        "tenant_id": str(_TENANT),
        "subject_id": _SA_SUBJECT,
    }
    body = await _get(
        {"X-Tenant-Id": "Bearer sa-tok", "X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR}
    )
    assert body["project_id"] == _PROJECT
    assert body["actor_id"] == _ACTOR


@patch("w2v_common.auth.middleware._introspect_license")
async def test_license_lane_actor_allowlist_empty_means_nobody(
    mock_introspect: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_ACTOR_ASSERTING_SUBJECTS", " , ")
    mock_introspect.return_value = {
        "active": True,
        "tenant_id": str(_TENANT),
        "subject_id": _SA_SUBJECT,
    }
    body = await _get({"X-Tenant-Id": "Bearer sa-tok", "X-Actor-Id": _ACTOR})
    assert body["actor_id"] == ""


async def test_trusted_network_lane_binds_project_but_never_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_TRUST_INCLUSTER_IDENTITY", "true")
    body = await _get({"X-Tenant-Id": str(_TENANT), "X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR})
    assert body["project_id"] == _PROJECT
    assert body["actor_id"] == ""


async def test_jwt_lane_keeps_sub_as_actor_and_binds_project() -> None:
    validator = MagicMock()
    validator.validate = AsyncMock(
        return_value=({"sub": "jwt-sub", "tenant_id": str(_TENANT)}, None)
    )
    with patch("w2v_common.auth.middleware.get_jwt_validator", return_value=validator):
        body = await _get(
            {"Authorization": "Bearer a.b.c", "X-Project-Id": _PROJECT, "X-Actor-Id": _ACTOR}
        )
    assert body["project_id"] == _PROJECT
    assert body["actor_id"] == "jwt-sub"


@patch("w2v_common.auth.middleware._introspect_license")
async def test_no_cross_request_leak(
    mock_introspect: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Request 1 binds project+actor; request 2 (same client, no headers) sees none."""
    monkeypatch.setenv("AUTH_ACTOR_ASSERTING_SUBJECTS", _SA_SUBJECT)
    mock_introspect.return_value = {
        "active": True,
        "tenant_id": str(_TENANT),
        "subject_id": _SA_SUBJECT,
    }
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://t") as client:
        first = (
            await client.get(
                "/api/attr",
                headers={
                    "X-Tenant-Id": "Bearer sa-tok",
                    "X-Project-Id": _PROJECT,
                    "X-Actor-Id": _ACTOR,
                },
            )
        ).json()
        second = (await client.get("/api/attr", headers={"X-Tenant-Id": "Bearer sa-tok"})).json()
    assert first["project_id"] == _PROJECT and first["actor_id"] == _ACTOR
    assert second == {"project_id": "", "actor_id": "", "cost_project": _TENANT_LEVEL}
    # ...and nothing leaked into the enclosing (test) context either.
    assert get_project_id_from_context() == ""


async def test_concurrent_tasks_do_not_share_a_project() -> None:
    other = "0195f48b-3333-7000-8000-000000000001"
    set_identity_context(_TENANT, None, "pays")

    async def _one(project: str) -> str:
        bind_attribution_from_headers({"X-Project-Id": project}, trusted=False)
        await asyncio.sleep(0)
        return get_project_id_from_context()

    got = await asyncio.gather(_one(_PROJECT), _one(other))
    assert got == [_PROJECT, other]


# ---------------------------------------------------------------------------
# Inertness: outbound headers on the lanes that bind an actor WITHOUT any
# inbound attribution header (JWT UUID sub, mesh gate x-actor-id).
# ---------------------------------------------------------------------------


def _outbound_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(mw.AuthMiddleware)

    @app.get("/api/out")
    async def out(request: Request) -> dict[str, Any]:
        req = httpx.Request("GET", "http://example.invalid/x")
        await identity_event_hook(req)
        return {
            "identity": get_identity_headers(),
            "hook": {k.lower(): v for k, v in req.headers.items() if k.lower().startswith("x-")},
            "actor_id": get_actor_id_from_context(),
        }

    return app


async def _out(headers: dict[str, str]) -> dict[str, Any]:
    async with AsyncClient(
        transport=ASGITransport(app=_outbound_app()), base_url="http://t"
    ) as client:
        resp = await client.get("/api/out", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _jwt_uuid_sub_out() -> dict[str, Any]:
    validator = MagicMock()
    validator.validate = AsyncMock(return_value=({"sub": _ACTOR, "tenant_id": str(_TENANT)}, None))
    with patch("w2v_common.auth.middleware.get_jwt_validator", return_value=validator):
        return await _out({"Authorization": "Bearer a.b.c"})


async def _mesh_gate_actor_out(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
    return await _out(
        {
            "x-tenant-id": str(_TENANT),
            "x-license-id": "lic-1",
            "x-subject-id": "sub-1",
            "x-actor-id": _ACTOR,
        }
    )


@pytest.mark.usefixtures("_forward_off")
async def test_jwt_uuid_sub_is_forwarded_by_default() -> None:
    body = await _jwt_uuid_sub_out()
    assert body["actor_id"] == _ACTOR  # bound for this hop's cost events
    assert set(body["identity"]) == _V081_IDENTITY_KEYS | {"X-Actor-Id"}
    assert body["hook"]["x-actor-id"] == _ACTOR
    assert "x-project-id" not in body["hook"]


@pytest.mark.usefixtures("_forward_on")
async def test_jwt_uuid_sub_is_forwarded_when_opted_in() -> None:
    body = await _jwt_uuid_sub_out()
    assert body["identity"]["X-Actor-Id"] == _ACTOR
    assert body["hook"]["x-actor-id"] == _ACTOR


@pytest.mark.usefixtures("_forward_off")
async def test_mesh_gate_actor_is_forwarded_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = await _mesh_gate_actor_out(monkeypatch)
    assert body["actor_id"] == _ACTOR
    assert body["identity"]["X-Actor-Id"] == _ACTOR
    assert body["hook"]["x-actor-id"] == _ACTOR


@pytest.mark.usefixtures("_forward_on")
async def test_mesh_gate_actor_is_forwarded_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = await _mesh_gate_actor_out(monkeypatch)
    assert body["identity"]["X-Actor-Id"] == _ACTOR
    assert body["hook"]["x-actor-id"] == _ACTOR


async def test_auth_middleware_marks_the_request_inbound_so_no_service_actor_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mesh request with NO x-actor-id: the service must not name itself."""
    monkeypatch.setenv("SERVICE_ACTOR_ID", _SA_SUBJECT)
    monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
    before = _actor_missing("outbound", idc.ACTOR_MISSING_INBOUND)
    body = await _out({"x-tenant-id": str(_TENANT), "x-license-id": "lic-1", "x-subject-id": "s"})
    assert "X-Actor-Id" not in body["identity"]
    assert "x-actor-id" not in body["hook"]
    assert _actor_missing("outbound", idc.ACTOR_MISSING_INBOUND) >= before + 1
