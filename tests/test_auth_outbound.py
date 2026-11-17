"""Outbound tenant-header contract: shape by destination, delegation by provenance.

Mirrors kag's ``src/services/proxy.py`` rules. Every test here has been
mutation-checked: the guarded behaviour was broken in the source, the test was
observed RED, and the source reverted.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from w2v_common.auth.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    clear_identity_context,
    get_identity_headers,
    identity_event_hook,
    is_tenant_vouched,
    set_identity_context,
)
from w2v_common.auth.outbound import (
    METERED_DESTINATION_SERVICES,
    build_downstream_tenant_headers,
    is_metered_destination,
)
from w2v_common.discovery import service_url
from w2v_common.tenant import (
    VOUCHED_TENANT_SOURCES,
    MissingTenantError,
    TenantSource,
    UnvouchedTenantError,
)

TENANT = uuid.UUID("11111111-2222-4333-8444-555555555555")
# ``content`` rather than the ``analysis`` this used to name: analysis BECAME
# a metered destination in the owner flip of 2026-09-13 (it was the largest
# mis-filer in the fleet, 15,663 SA-tenant rows). Pick a peer declared
# TENANT_AGNOSTIC in ``auth.destination_tenancy`` so this constant cannot
# rot into a vacuous pass; the assert below fails loudly if it does.
PLAIN_BACKEND = service_url("content")
assert not is_metered_destination(PLAIN_BACKEND), "PLAIN_BACKEND must not be metered"
METERED = service_url("worker_management")


@pytest.fixture(autouse=True)
def _clean_context():
    clear_identity_context()
    yield
    clear_identity_context()


# ---------------------------------------------------------------------------
# The ladder itself
# ---------------------------------------------------------------------------


def test_only_license_and_vouched_and_jwt_are_vouched() -> None:
    """The vouched set is exactly the rungs backed by a verified credential.

    ``MESH_VOUCHED`` joined the set when w2v learned the platform's licensing
    conventions: the ``license-authz`` ext-authz gate introspects the licence at
    the waypoint and injects its verdict, so the credential WAS verified — just
    one hop earlier. Without it, every gated request would fail closed at its
    first metered hop. The lane that produces it is OFF by default and
    additionally peer-IP and (optionally) HMAC gated — see
    ``AuthMiddleware._dispatch_mesh_vouched``.
    """
    assert (
        frozenset(
            {
                TenantSource.LICENSE_TOKEN,
                TenantSource.VOUCHED_ONBEHALF,
                TenantSource.JWT_CLAIM,
                TenantSource.MESH_VOUCHED,
                TenantSource.MEMBERSHIP,
                TenantSource.TENANT_API_KEY,
                # P7 (0.91.0): signed by an in-cell peer's own Ed25519 key and
                # verified here (kid->iss, aud, cell, exp, jti).
                TenantSource.CELL_JWT,
            }
        )
        == VOUCHED_TENANT_SOURCES
    )
    # The three that must NEVER delegate.
    for source in (
        TenantSource.UNVOUCHED_HEADER,
        TenantSource.TRUSTED_NETWORK_DEFAULT,
        TenantSource.AUTH_DISABLED,
    ):
        assert source not in VOUCHED_TENANT_SOURCES


def test_metered_destination_is_matched_by_url_not_call_site() -> None:
    assert is_metered_destination(METERED) is True
    assert is_metered_destination(METERED + "/") is True
    assert is_metered_destination("worker_management") is True
    assert is_metered_destination(PLAIN_BACKEND) is False
    assert is_metered_destination("") is False


def test_modality_gateway_is_metered() -> None:
    """The gateway rejects a bare UUID, so it MUST get the Bearer+effective shape.

    kag names it outright: modality-gateway "is the ONLY downstream that demands
    a license token in ``X-Tenant-Id`` and rejects a bare UUID outright"
    (``kag/src/services/proxy.py::_is_license_introspecting_gateway``). 0.67.0
    omitted it from ``METERED_DESTINATION_SERVICES``, so w2v callers sent it a
    bare uuid it refuses -- a broken hop, not merely a missing header. It never
    shipped: the deploy gates held every affected service on its old image.

    Without this test the omission is invisible -- removing the name from the
    set left all 23 sibling tests green.
    """
    assert is_metered_destination("modality_gateway") is True
    assert is_metered_destination(service_url("modality_gateway")) is True
    assert is_metered_destination(service_url("modality_gateway") + "/v1/chat") is True


def test_video_production_is_metered() -> None:
    """Render cost is cost: the orchestrator's renders must name the real tenant.

    The other four metered peers consume INFERENCE spend; video_production
    consumes RENDER spend, which is just as billable. While its discovery name
    was absent from ``METERED_DESTINATION_SERVICES`` the hook sent NO delegation
    header on that hop at all, so every customer's render was attributed to the
    orchestrator's service-account tenant e148e8af-3751-49bf-a0fa-56ab46d31ddc
    — unattributable rows, not merely mislabelled ones. The orchestrator's
    client was never the bug; this classification is.
    """
    assert is_metered_destination("video_production") is True
    assert is_metered_destination(service_url("video_production")) is True
    assert is_metered_destination(service_url("video_production") + "/api/render") is True


def test_html_render_is_metered() -> None:
    """Same defect as ``video_production``, on the other render peer.

    ``html_render_service`` files each design/redesign job under whatever
    tenant the inbound credential introspects to. The orchestrator's STAGE
    path (``web_design``, ``web_design_v2``, ``website_review`` redesign,
    ``promo`` html-segments → ``DesignClient`` → ``LicensedServiceClient``)
    authenticates with the shared service-account licence token and nothing
    else, so while this discovery name was absent from
    ``METERED_DESTINATION_SERVICES`` the hook sent NO delegation header on the
    hop at all and every customer's redesign landed on the orchestrator SA
    tenant e148e8af-3751-49bf-a0fa-56ab46d31ddc — 1044 of 1070 rows in the
    service's own database. The read path filters
    ``RedesignJob.tenant_id == <caller tenant>``, so those jobs are 404 to the
    customer who paid for them. The client is not the bug; this
    classification is.
    """
    assert is_metered_destination("html_render") is True
    assert is_metered_destination(service_url("html_render")) is True
    assert is_metered_destination(service_url("html_render") + "/api/designs") is True


def test_metered_predicate_agrees_for_a_name_and_its_resolved_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predicate must give ONE answer for a peer, whichever shape it is asked in.

    Live gitops (clusters/strato/services/w2v/configmaps.yaml) sets
    ``W2V_TTS_URL`` to the modality-gateway. So ``service_url("tts")`` is a
    modality-gateway URL — a metered peer. The predicate used to resolve only
    the METERED NAMES through ``service_url`` and never the INPUT, so it
    answered False for ``"tts"`` and True for ``service_url("tts")``. Those two
    answers select different outbound tenant headers: the False branch sends a
    bare-UUID ``X-Tenant-Id`` to a gateway that introspects a license and
    rejects it.
    """
    monkeypatch.setenv("W2V_TTS_URL", "http://modality-gateway.modality-gateway:8890")
    by_url = is_metered_destination(service_url("tts"))
    by_name = is_metered_destination("tts")
    assert by_url is True, "the override points tts at the modality-gateway, a metered peer"
    assert by_name == by_url, (
        f"asymmetric resolution: is_metered_destination('tts')={by_name} but "
        f"is_metered_destination(service_url('tts'))={by_url}"
    )


def test_a_bare_name_without_an_override_is_still_matched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-existing bare-name path must keep working (no override in play)."""
    monkeypatch.delenv("W2V_WORKER_MANAGEMENT_URL", raising=False)
    assert is_metered_destination("worker_management") is True
    assert is_metered_destination("modality_gateway") is True


def test_a_bare_name_pointed_at_a_plain_backend_is_not_metered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative half: resolving the input must not make everything metered."""
    monkeypatch.delenv("W2V_TTS_URL", raising=False)
    assert is_metered_destination("tts") is False
    assert is_metered_destination("not_a_discovery_name") is False
    assert is_metered_destination("http://analysis.analysis:8004") is False


def test_env_override_extends_rather_than_replaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHANGED ON PURPOSE 2026-09-13. This test used to assert the opposite.

    ``W2V_METERED_DESTINATIONS`` used to REPLACE ``METERED_DESTINATION_SERVICES``.
    Its predecessor test pinned that deliberately, on the argument that a
    deployment must be able to say "here, kag is NOT metered". That need is
    real; replacement was the wrong way to serve it. Under replacement, the
    low-effort mistake — naming the ONE peer you want to add — demoted every
    other destination fleet-wide with no error and no log line, and an empty or
    typo'd value demoted ALL of them. The cheap keystrokes had the maximum blast
    radius and were indistinguishable from intent.

    So the semantics inverted: a bare name ADDS, and a demotion must be typed
    per destination as ``-name``. The dangerous edit is now the verbose one, and
    every malformed value raises instead of demoting. The demotion path and the
    failure modes are covered in ``tests/test_metered_destination_flip.py``.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "course")
    assert is_metered_destination("course") is True
    assert is_metered_destination("worker_management") is True
    assert is_metered_destination(METERED) is True


# ---------------------------------------------------------------------------
# A. build_downstream_tenant_headers
# ---------------------------------------------------------------------------


def test_plain_backend_gets_the_verbatim_tenant_header() -> None:
    headers = build_downstream_tenant_headers(TENANT, PLAIN_BACKEND, TenantSource.LICENSE_TOKEN)
    assert headers == {"X-Tenant-Id": str(TENANT)}
    assert EFFECTIVE_TENANT_HEADER not in headers


def test_plain_backend_does_not_gate_on_provenance() -> None:
    """An unvouched tenant is still a valid DATA SCOPE on a non-metered peer."""
    headers = build_downstream_tenant_headers(TENANT, PLAIN_BACKEND, TenantSource.UNVOUCHED_HEADER)
    assert headers == {"X-Tenant-Id": str(TENANT)}


@pytest.mark.parametrize(
    "source",
    [
        TenantSource.LICENSE_TOKEN,
        TenantSource.VOUCHED_ONBEHALF,
        TenantSource.JWT_CLAIM,
        TenantSource.MEMBERSHIP,
        TenantSource.TENANT_API_KEY,
    ],
)
def test_vouched_source_delegates_to_a_metered_destination(source: TenantSource) -> None:
    headers = build_downstream_tenant_headers(
        TENANT, METERED, source, downstream_license_token="svc-token"
    )
    assert headers == {
        "X-Tenant-Id": "Bearer svc-token",
        EFFECTIVE_TENANT_HEADER: str(TENANT),
    }


@pytest.mark.parametrize(
    "source",
    [
        TenantSource.UNVOUCHED_HEADER,
        TenantSource.TRUSTED_NETWORK_DEFAULT,
        TenantSource.AUTH_DISABLED,
    ],
)
def test_unvouched_source_raises_instead_of_delegating(source: TenantSource) -> None:
    with pytest.raises(UnvouchedTenantError) as exc:
        build_downstream_tenant_headers(
            TENANT, METERED, source, downstream_license_token="svc-token"
        )
    assert source.value in str(exc.value)


def test_absent_tenant_fails_closed_on_both_destination_shapes() -> None:
    for destination in (PLAIN_BACKEND, METERED):
        with pytest.raises(MissingTenantError):
            build_downstream_tenant_headers(
                None, destination, TenantSource.LICENSE_TOKEN, downstream_license_token="t"
            )


def test_metered_hop_without_a_service_token_is_a_named_config_failure() -> None:
    with pytest.raises(RuntimeError, match="DOWNSTREAM_LICENSE_TOKEN"):
        build_downstream_tenant_headers(
            TENANT, METERED, TenantSource.LICENSE_TOKEN, downstream_license_token=""
        )


def test_provenance_has_no_default() -> None:
    """Omitting the provenance must be a TypeError, never a silent escalation."""
    with pytest.raises(TypeError):
        build_downstream_tenant_headers(TENANT, METERED)  # ty: ignore[missing-argument]


# ---------------------------------------------------------------------------
# B. Provenance in the identity context
# ---------------------------------------------------------------------------


def test_provenance_defaults_to_unvouched_when_no_lane_recorded_one() -> None:
    set_identity_context(tenant_id=TENANT, license_id=None, subject_id=None)
    assert is_tenant_vouched() is False


def test_license_provenance_is_vouched() -> None:
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    assert is_tenant_vouched() is True


# ---------------------------------------------------------------------------
# C. The fail-open outbound sites
# ---------------------------------------------------------------------------


def test_get_identity_headers_fails_closed_when_unbound() -> None:
    with pytest.raises(MissingTenantError):
        get_identity_headers()


def test_get_identity_headers_never_emits_a_blank_tenant() -> None:
    set_identity_context(tenant_id=TENANT, license_id="lic", subject_id="sub")
    assert get_identity_headers()["X-Tenant-Id"] == str(TENANT)


async def test_event_hook_fails_closed_on_a_tenantless_call() -> None:
    request = httpx.Request("POST", "https://analysis-service.w2v/api/x")
    with pytest.raises(MissingTenantError):
        await identity_event_hook(request)


async def test_event_hook_allows_a_service_account_call_with_no_bound_tenant() -> None:
    """A CronJob on its own SA token is attributable and must keep working."""
    request = httpx.Request(
        "POST",
        "https://worker-management.workers/api/x",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    await identity_event_hook(request)
    assert request.headers["X-Tenant-Id"] == "Bearer sa-token"
    assert EFFECTIVE_TENANT_HEADER not in request.headers


async def test_event_hook_stamps_the_verbatim_tenant_on_a_plain_hop() -> None:
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.UNVOUCHED_HEADER,
    )
    request = httpx.Request("POST", "https://analysis-service.w2v/api/x")
    await identity_event_hook(request)
    assert request.headers["X-Tenant-Id"] == str(TENANT)
    assert EFFECTIVE_TENANT_HEADER not in request.headers


async def test_event_hook_delegates_only_for_a_vouched_tenant() -> None:
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    request = httpx.Request(
        "POST",
        "https://worker-management.workers/api/x",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    await identity_event_hook(request)
    assert request.headers["X-Tenant-Id"] == "Bearer sa-token"
    assert request.headers[EFFECTIVE_TENANT_HEADER] == str(TENANT)


async def test_event_hook_refuses_to_delegate_a_trusted_network_tenant() -> None:
    """The trusted-network lane vouches for the CALLER, never for the TENANT."""
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.UNVOUCHED_HEADER,
    )
    request = httpx.Request(
        "POST",
        "https://worker-management.workers/api/x",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    with pytest.raises(UnvouchedTenantError):
        await identity_event_hook(request)
    assert EFFECTIVE_TENANT_HEADER not in request.headers


async def test_event_hook_delegates_the_tenant_on_a_video_production_hop() -> None:
    """The end-to-end shape of the fix, at the hook the orchestrator actually uses."""
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    request = httpx.Request(
        "POST",
        service_url("video_production") + "/api/render",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    await identity_event_hook(request)
    assert request.headers["X-Tenant-Id"] == "Bearer sa-token"
    assert request.headers[EFFECTIVE_TENANT_HEADER] == str(TENANT)


async def test_event_hook_refuses_an_unvouched_video_production_hop() -> None:
    """Delegation is a privilege escalation on this peer too — it must raise, not skip."""
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.UNVOUCHED_HEADER,
    )
    request = httpx.Request(
        "POST",
        service_url("video_production") + "/api/render",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    with pytest.raises(UnvouchedTenantError):
        await identity_event_hook(request)
    assert EFFECTIVE_TENANT_HEADER not in request.headers


async def test_event_hook_delegates_the_tenant_on_an_html_render_hop() -> None:
    """The end-to-end shape of the fix, at the hook ``DesignClient`` actually uses.

    ``DesignClient`` is a ``LicensedServiceClient``: the orchestrator's own
    service-account licence is already baked into ``X-Tenant-Id: Bearer …``,
    so this is the ``service_credentialed`` branch of
    ``_apply_outbound_tenant``. The token must survive untouched (it is how
    html_render authenticates us) and ``X-Effective-Tenant-Id`` must name the
    run's REAL tenant beside it — html-render-service already carries
    ``AUTH_DELEGATING_SUBJECTS=8382f0b3-c082-4f97-8d47-120ce5a18e05`` with
    ``TENANT_RLS_MODE=enforce``, so the receiver honours it today.
    """
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    request = httpx.Request(
        "POST",
        service_url("html_render") + "/api/designs",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    await identity_event_hook(request)
    assert request.headers["X-Tenant-Id"] == "Bearer sa-token"
    assert request.headers[EFFECTIVE_TENANT_HEADER] == str(TENANT)


async def test_event_hook_refuses_an_unvouched_html_render_hop() -> None:
    """Delegation is a privilege escalation on this peer too — it must raise, not skip."""
    set_identity_context(
        tenant_id=TENANT,
        license_id=None,
        subject_id=None,
        tenant_source=TenantSource.UNVOUCHED_HEADER,
    )
    request = httpx.Request(
        "POST",
        service_url("html_render") + "/api/designs",
        headers={"X-Tenant-Id": "Bearer sa-token"},
    )
    with pytest.raises(UnvouchedTenantError):
        await identity_event_hook(request)
    assert EFFECTIVE_TENANT_HEADER not in request.headers


def test_kag_scrape_client_is_gone() -> None:
    """W0b: KAG is retired, so its scrape client and its metered hop are deleted.

    The package-level ``ScrapeResponse`` is the webscraper one now, and a
    request to the old gateway host gets the plain-backend shape (no
    delegation), because nothing meters there any more.
    """
    import importlib

    import w2v_common.clients as clients
    from w2v_common.clients import webscraper

    assert not hasattr(clients, "KagScrapeClient")
    assert "KagScrapeClient" not in clients.__all__
    assert clients.ScrapeResponse is webscraper.ScrapeResponse
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("w2v_common.clients.kag_scrape")

    assert "kag" not in METERED_DESTINATION_SERVICES
    headers = build_downstream_tenant_headers(
        TENANT,
        "https://kag.kag:8900",
        TenantSource.UNVOUCHED_HEADER,
        downstream_license_token="svc-token",
    )
    assert EFFECTIVE_TENANT_HEADER not in headers
