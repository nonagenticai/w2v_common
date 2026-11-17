"""WMClient's inherited WM client under ``WM_AUTH_MODE`` + ``tenant_licence_event_hook``.

0.109.0 deleted the WM ``message()`` backend and ``ModalityConfigClient``, and with
them every on-the-wire mint/re-mint test of those calls. What is left is the
pooled WM client the orchestrator's subclass still sends ``/tenant-config/...``
through (and re-mints on itself), so its credential shape stays pinned here.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from w2v_common.auth.identity_context import (
    NoTenantForLicenceError,
    clear_identity_context,
    set_identity_context,
    tenant_licence_event_hook,
)
from w2v_common.clients.wm_client import WMClient
from w2v_common.discovery import service_url
from w2v_common.tenant import TenantSource

TENANT_A = "0199a1b2-0000-7000-8000-00000000000a"
TENANT_B = "0199a1b2-0000-7000-8000-00000000000b"
SA_TOKEN = "sa-shared-licence"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _env(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SA_TOKEN)
    for name in ("WM_AUTH_MODE", "LLM_BACKEND", "LLM_IR_TOKEN", "LLM_IR_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    clear_identity_context()
    yield
    clear_identity_context()


def _wm_url() -> str:
    return service_url("worker_management")


def _bind(tenant: str) -> None:
    set_identity_context(tenant, "license-1", "subject-1", tenant_source=TenantSource.LICENSE_TOKEN)


def test_wm_client_rejects_unknown_auth_mode() -> None:
    with pytest.raises(ValueError, match="WM_AUTH_MODE"):
        WMClient(base_url="http://wm.test", auth_mode="tenant-licence")


def test_wm_client_reads_wm_auth_mode_env(monkeypatch) -> None:
    monkeypatch.setenv("WM_AUTH_MODE", "tenant_licence")
    assert WMClient(base_url="http://wm.test").auth_mode.value == "tenant_licence"


async def test_tenant_licence_hook_strips_a_delegation_header() -> None:
    _bind(TENANT_A)
    req = httpx.Request(
        "GET",
        _wm_url() + "/x",
        headers={"X-Tenant-Id": "Bearer lic-x", "X-Effective-Tenant-Id": TENANT_B},
    )
    await tenant_licence_event_hook(req)
    assert "X-Effective-Tenant-Id" not in req.headers
    assert req.headers["X-Tenant-Id"] == "Bearer lic-x"
    # The inbound licence id is never stamped on a tenant-licence hop.
    assert "X-License-Id" not in req.headers


@pytest.mark.parametrize("value", [None, TENANT_A, "Bearer ", "Bearer   "])
async def test_tenant_licence_hook_refuses_a_hop_without_a_licence(value) -> None:
    _bind(TENANT_A)
    headers = {} if value is None else {"X-Tenant-Id": value}
    req = httpx.Request("GET", _wm_url() + "/x", headers=headers)
    with pytest.raises(NoTenantForLicenceError):
        await tenant_licence_event_hook(req)


async def test_tenant_licence_pooled_client_holds_no_sa_credential() -> None:
    """Nothing credential-shaped is baked into the pooled WM client in tenant_licence mode."""
    transport = httpx.MockTransport(lambda r: httpx.Response(204))
    client = WMClient(base_url=_wm_url(), auth_mode="tenant_licence", transport=transport)
    assert "X-Tenant-Id" not in (await client._get_client()).headers
    delegating = WMClient(base_url=_wm_url(), auth_mode="delegation", transport=transport)
    assert (await delegating._get_client()).headers["X-Tenant-Id"] == f"Bearer {SA_TOKEN}"
