"""Tests for ArtifactClient — uses respx to mock the artifact_service HTTP API."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import respx

from w2v_common.clients.artifact_client import ArtifactClient

PROJECT_ID = uuid4()
ARTIFACT_ID = uuid4()
BASE_URL = "http://artifacts.test:8556"


@pytest.fixture
def client() -> ArtifactClient:
    return ArtifactClient(base_url=BASE_URL, license_token="tok-test")


async def test_upload_file_returns_artifact_id(client: ArtifactClient) -> None:
    payload = {
        "artifact_id": str(ARTIFACT_ID),
        "version": 1,
        "storage_key": f"projects/{PROJECT_ID}/step/p1/{ARTIFACT_ID}/v1/p1.demo.spec.ts",
        "size": 42,
        "sha256": "0" * 64,
    }
    with respx.mock(base_url=BASE_URL, assert_all_called=True) as mock:
        mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await client.upload_bytes(
            project_id=PROJECT_ID,
            filename="p1.demo.spec.ts",
            content=b"console.log('hi')",
            content_type="text/plain",
            kind="other",
            step="p1",
            tags=["w2v-script"],
        )
    assert result.artifact_id == ARTIFACT_ID
    assert result.size == 42
    assert result.sha256 == "0" * 64


async def test_upload_file_propagates_4xx(client: ArtifactClient) -> None:
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(415, json={"detail": "content-type not allowed"})
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await client.upload_bytes(
                project_id=PROJECT_ID,
                filename="bad.exe",
                content=b"x",
                content_type="application/x-msdownload",
                kind="other",
            )
    assert exc.value.response.status_code == 415


async def test_download_to_path_streams_bytes(client: ArtifactClient, tmp_path: Path) -> None:
    out = tmp_path / "p1.demo.spec.ts"
    body = b"// playwright script\nconsole.log('downloaded')"
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get(f"/projects/{PROJECT_ID}/artifacts/{ARTIFACT_ID}/download").mock(
            return_value=httpx.Response(
                200,
                content=body,
                headers={"Content-Type": "text/plain"},
            )
        )
        result_path = await client.download_to_path(
            project_id=PROJECT_ID,
            artifact_id=ARTIFACT_ID,
            dest=out,
        )
    assert result_path == out
    assert out.read_bytes() == body


async def test_download_404_raises(client: ArtifactClient, tmp_path: Path) -> None:
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get(f"/projects/{PROJECT_ID}/artifacts/{ARTIFACT_ID}/download").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await client.download_to_path(
                project_id=PROJECT_ID,
                artifact_id=ARTIFACT_ID,
                dest=tmp_path / "x.ts",
            )
    assert exc.value.response.status_code == 404


async def test_license_token_sent_as_bearer(client: ArtifactClient) -> None:
    """Confirm Authorization: Bearer <token> is set on every outbound call."""
    payload = {
        "artifact_id": str(ARTIFACT_ID),
        "version": 1,
        "storage_key": "projects/x",
        "size": 1,
        "sha256": "0" * 64,
    }
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=payload)
        )
        await client.upload_bytes(
            project_id=PROJECT_ID,
            filename="x.ts",
            content=b"x",
            content_type="text/plain",
            kind="other",
        )
    assert route.calls[0].request.headers.get("Authorization") == "Bearer tok-test"
    assert route.calls[0].request.headers.get("X-Tenant-Id") == "Bearer tok-test"


async def test_caller_token_overrides_bearer(client: ArtifactClient) -> None:
    """When caller_license_token is set in the ContextVar, ArtifactClient
    sends Authorization: Bearer <caller> + X-Tenant-Id: Bearer <caller>,
    overriding the constructor's service-account token. X-License-Id /
    X-Subject-Id come from identity_event_hook (additive UUID propagation).
    """
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        set_identity_context,
    )

    set_identity_context(
        tenant_id="0195f48b-0000-7000-8000-00000000007d",
        license_id="lic",
        subject_id="sub",
        caller_license_token="caller-tok",
    )
    try:
        payload = {
            "artifact_id": str(ARTIFACT_ID),
            "version": 1,
            "storage_key": "projects/x",
            "size": 1,
            "sha256": "0" * 64,
        }
        with respx.mock(base_url=BASE_URL) as mock:
            route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
                return_value=httpx.Response(200, json=payload)
            )
            await client.upload_bytes(
                project_id=PROJECT_ID,
                filename="x.ts",
                content=b"x",
                content_type="text/plain",
                kind="other",
            )
        assert route.calls[0].request.headers["Authorization"] == "Bearer caller-tok"
        assert route.calls[0].request.headers["X-Tenant-Id"] == "Bearer caller-tok"
        assert route.calls[0].request.headers["X-License-Id"] == "lic"
        assert route.calls[0].request.headers["X-Subject-Id"] == "sub"
    finally:
        clear_identity_context()
