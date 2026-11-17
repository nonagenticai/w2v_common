"""Unit tests for the shared w2v_common.storage.StorageService.

Local-backend tests use a real tmp dir; MinIO-backend tests inject a mocked
async client (no live MinIO). The presign/resign/object_exists logic is the
behavioral contract lifted from the orchestrator superset.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from w2v_common.config.base import StorageSettings
from w2v_common.storage import StorageService


def _local_settings(tmp_path) -> StorageSettings:
    return StorageSettings(backend="local", local_path=str(tmp_path))


def _minio_settings(**over) -> StorageSettings:
    # Heterogeneous field types (str / bool / int) live in this kwargs bag, so it
    # is a dict[str, Any] rather than the dict[str, str] inferred from the
    # literal — StorageSettings has bool + int fields that overrides may set.
    base: dict[str, Any] = {
        "backend": "minio",
        "minio_endpoint": "minio.example.com:9000",
        "minio_access_key": "k",
        "minio_secret_key": "s",
        "minio_bucket": "w2v-data",
    }
    base.update(over)
    return StorageSettings(**base)


# --- local backend ----------------------------------------------------------


@pytest.mark.asyncio
async def test_local_put_get_roundtrip(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.initialize()
    ref = await svc.put_bytes("a/b/c.bin", b"hello")
    assert (tmp_path / "a/b/c.bin").read_bytes() == b"hello"
    assert await svc.get_bytes(ref) == b"hello"
    # also retrievable by the relative key
    assert await svc.get_bytes("a/b/c.bin") == b"hello"


@pytest.mark.asyncio
async def test_local_get_missing_returns_none(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    assert await svc.get_bytes("nope.bin") is None


@pytest.mark.asyncio
async def test_local_list_and_delete_prefix(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("p/one.txt", b"1")
    await svc.put_bytes("p/sub/two.txt", b"2")
    listed = sorted(await svc.list_prefix("p"))
    assert listed == ["p/one.txt", "p/sub/two.txt"]
    assert await svc.delete_prefix("p") is True
    assert await svc.list_prefix("p") == []
    assert await svc.delete_prefix("p") is False  # already gone


@pytest.mark.asyncio
async def test_local_object_exists(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    ref = await svc.put_bytes("x.bin", b"z")
    assert await svc.object_exists(f"file://{ref}") is True
    assert await svc.object_exists(ref) is True  # bare local path
    assert await svc.object_exists(f"file://{tmp_path}/missing.bin") is False


@pytest.mark.asyncio
async def test_local_check_health(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.initialize()
    assert await svc.check_health() is True


@pytest.mark.asyncio
async def test_unsupported_backend_raises(tmp_path):
    svc = StorageService(StorageSettings(backend="local", local_path=str(tmp_path)))
    svc._backend = "weird"
    with pytest.raises(ValueError, match="Unsupported"):
        await svc.initialize()


# --- pure presign helpers ---------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "expected_netloc", "expected_scheme"),
    [
        ("minio.ext:9000", "minio.ext:9000", "http"),  # bare host:port keeps url scheme
        ("cdn.example.com", "cdn.example.com", "http"),  # bare host
        ("https://minio.ext:9000", "minio.ext:9000", "https"),  # full url overrides scheme
    ],
)
def test_rewrite_presigned_url(endpoint, expected_netloc, expected_scheme):
    original = "http://minio.example.com:9000/w2v-data/key.bin?X-Amz-Signature=abc"
    out = StorageService._rewrite_presigned_url(original, endpoint)
    assert f"{expected_scheme}://{expected_netloc}/w2v-data/key.bin" in out
    assert "X-Amz-Signature=abc" in out  # query preserved


def test_endpoint_netloc_forms():
    assert StorageService._endpoint_netloc("h:9000") == "h:9000"
    assert StorageService._endpoint_netloc("h") == "h"
    assert StorageService._endpoint_netloc("https://h:9000") == "h:9000"


def test_is_own_minio_host_and_bucket():
    svc = StorageService(_minio_settings(minio_external_endpoint="https://minio.ext:9000"))
    assert svc._is_own_minio_host("minio.example.com:9000") is True
    assert svc._is_own_minio_host("minio.ext:9000") is True
    assert svc._is_own_minio_host("evil.com") is False
    assert svc._is_own_minio_host("") is False
    assert svc._is_own_bucket("w2v-data") is True
    assert svc._is_own_bucket("w2v-compiled-videos") is True  # any w2v-* bucket
    assert svc._is_own_bucket("someone-else") is False
    assert svc._is_own_bucket("") is False


# --- MinIO backend (mocked client) ------------------------------------------


@pytest.mark.asyncio
async def test_presign_get_rewrites_external_host():
    svc = StorageService(_minio_settings(minio_external_endpoint="https://minio.ext:9000"))
    svc._minio_client = AsyncMock()
    svc._minio_client.presigned_get_object.return_value = (
        "http://minio.example.com:9000/w2v-data/k.bin?X-Amz-Signature=sig"
    )
    url = await svc.presign_get("k.bin")
    assert url is not None
    assert url.startswith("https://minio.ext:9000/w2v-data/k.bin")
    assert "X-Amz-Signature=sig" in url


@pytest.mark.asyncio
async def test_presign_get_local_backend_returns_none(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    assert await svc.presign_get("k.bin") is None


@pytest.mark.asyncio
async def test_object_exists_minio_true_false_indeterminate():
    from miniopy_async.error import S3Error

    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()

    # present
    svc._minio_client.stat_object.return_value = object()
    assert await svc.object_exists("http://minio.example.com:9000/w2v-data/k.bin") is True

    # definitively absent (NoSuchKey) — set the property's backing attr
    nosuchkey = S3Error.__new__(S3Error)
    nosuchkey._code = "NoSuchKey"
    svc._minio_client.stat_object.side_effect = nosuchkey
    assert await svc.object_exists("http://minio.example.com:9000/w2v-data/k.bin") is False

    # indeterminate (other S3 code)
    nobucket = S3Error.__new__(S3Error)
    nobucket._code = "NoSuchBucket"
    svc._minio_client.stat_object.side_effect = nobucket
    assert await svc.object_exists("http://minio.example.com:9000/w2v-data/k.bin") is None

    # indeterminate (transport error)
    svc._minio_client.stat_object.side_effect = OSError("boom")
    assert await svc.object_exists("http://minio.example.com:9000/w2v-data/k.bin") is None


@pytest.mark.asyncio
async def test_object_exists_minio_decodes_key():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.stat_object.return_value = object()
    # %20 in the URL must be decoded to a literal space for stat_object
    await svc.object_exists("http://minio.example.com:9000/w2v-data/a%20b,c.png")
    _bucket, key = svc._minio_client.stat_object.call_args.args
    assert key == "a b,c.png"


@pytest.mark.asyncio
async def test_resign_only_for_owned_url():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.presigned_get_object.return_value = (
        "http://minio.example.com:9000/w2v-data/k.bin?X-Amz-Signature=fresh"
    )

    # foreign host → None (never mint a pointer into our minio for a foreign URL)
    assert await svc.resign_presigned_url("http://evil.com/w2v-data/k.bin?x=1") is None
    # foreign bucket → None
    assert await svc.resign_presigned_url("http://minio.example.com:9000/other/k.bin?x=1") is None
    # non-http scheme → None
    assert await svc.resign_presigned_url("file:///w2v-data/k.bin") is None
    # owned host + owned bucket → fresh presign
    out = await svc.resign_presigned_url(
        "http://minio.example.com:9000/w2v-data/k.bin?X-Amz-Signature=stale"
    )
    assert out is not None
    assert "X-Amz-Signature=fresh" in out


# --- put_file (streaming, on-disk source) -----------------------------------


@pytest.mark.asyncio
async def test_local_put_file_roundtrip(tmp_path):
    svc = StorageService(_local_settings(tmp_path / "store"))
    await svc.initialize()
    src = tmp_path / "src.mp4"
    src.write_bytes(b"video-bytes")
    await svc.put_file("videos/out.mp4", src, content_type="video/mp4")
    assert (tmp_path / "store" / "videos/out.mp4").read_bytes() == b"video-bytes"
    assert await svc.get_bytes("videos/out.mp4") == b"video-bytes"
    assert src.read_bytes() == b"video-bytes"  # source is copied, not moved


@pytest.mark.asyncio
async def test_minio_put_file_streams_not_in_memory(tmp_path):
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    src = tmp_path / "big.mp4"
    src.write_bytes(b"x" * 1024)
    ref = await svc.put_file("out/big.mp4", src, content_type="video/mp4")
    assert ref == "out/big.mp4"
    # streamed via fput_object (constant memory), never read into RAM via put_object
    svc._minio_client.fput_object.assert_awaited_once_with(
        "w2v-data", "out/big.mp4", str(src), content_type="video/mp4"
    )
    svc._minio_client.put_object.assert_not_called()


@pytest.mark.asyncio
async def test_minio_put_file_bucket_override(tmp_path):
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    src = tmp_path / "a.wav"
    src.write_bytes(b"wav")
    await svc.put_file("clip.wav", src, content_type="audio/wav", bucket="w2v-audiobooks")
    bucket, key, _path = svc._minio_client.fput_object.call_args.args
    assert bucket == "w2v-audiobooks"
    assert key == "clip.wav"


# --- delete_object (single key) ---------------------------------------------


@pytest.mark.asyncio
async def test_local_delete_object(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("p/one.txt", b"1")
    await svc.put_bytes("p/two.txt", b"2")
    assert await svc.delete_object("p/one.txt") is True
    assert await svc.get_bytes("p/one.txt") is None
    assert await svc.get_bytes("p/two.txt") == b"2"  # sibling untouched
    assert await svc.delete_object("p/one.txt") is True  # idempotent (missing_ok)


@pytest.mark.asyncio
async def test_minio_delete_object():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    assert await svc.delete_object("k.bin") is True
    svc._minio_client.remove_object.assert_awaited_once_with("w2v-data", "k.bin")
    # bucket override
    await svc.delete_object("v.mp4", bucket="w2v-compiled-videos")
    bucket, _key = svc._minio_client.remove_object.call_args.args
    assert bucket == "w2v-compiled-videos"
    # storage error → False (never raises)
    svc._minio_client.remove_object.side_effect = OSError("boom")
    assert await svc.delete_object("k.bin") is False


# --- object_exists_key (bare key, not URL) ----------------------------------


@pytest.mark.asyncio
async def test_object_exists_key_local(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("slides/1.png", b"z")
    assert await svc.object_exists_key("slides/1.png") is True
    assert await svc.object_exists_key("slides/missing.png") is False


@pytest.mark.asyncio
async def test_object_exists_key_minio_true_false_indeterminate():
    from miniopy_async.error import S3Error

    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()

    svc._minio_client.stat_object.return_value = object()
    assert await svc.object_exists_key("slides/1.png") is True
    svc._minio_client.stat_object.assert_awaited_with("w2v-data", "slides/1.png")

    nosuchkey = S3Error.__new__(S3Error)
    nosuchkey._code = "NoSuchKey"
    svc._minio_client.stat_object.side_effect = nosuchkey
    assert await svc.object_exists_key("slides/1.png") is False

    svc._minio_client.stat_object.side_effect = OSError("boom")
    assert await svc.object_exists_key("slides/1.png") is None

    # bucket override reaches stat_object
    svc._minio_client.stat_object.side_effect = None
    svc._minio_client.stat_object.return_value = object()
    await svc.object_exists_key("k", bucket="w2v-other")
    assert svc._minio_client.stat_object.call_args.args[0] == "w2v-other"


# --- 0.39.0 keystone: SameFile guard, cross-bucket read, streaming range get ---


class _FakeContent:
    """Async-iterable body mirroring miniopy_async's response.content."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def iter_chunked(self, size: int):
        data = self._data

        async def _gen():
            for i in range(0, len(data), size):
                yield data[i : i + size]

        return _gen()


class _FakeResponse:
    """Stand-in for the miniopy_async streaming get_object response."""

    def __init__(self, data: bytes, *, status: int = 200, headers: dict | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(data)
        self._data = data
        self.closed = False
        self.released = False

    async def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True

    async def release(self) -> None:
        self.released = True


async def _drain(stream, chunk_size: int = 4) -> bytes:
    return b"".join([chunk async for chunk in stream.iter_chunks(chunk_size)])


# SameFile guard --------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_put_file_in_place_is_noop(tmp_path):
    """put_file where src already IS the destination key must not raise
    SameFileError (the video_production in-place compile pattern)."""
    svc = StorageService(_local_settings(tmp_path / "store"))
    await svc.initialize()
    dest = tmp_path / "store" / "vids" / "final.mp4"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"in-place-bytes")
    ref = await svc.put_file("vids/final.mp4", dest, content_type="video/mp4")
    assert ref == str(dest)
    assert dest.read_bytes() == b"in-place-bytes"  # untouched, not truncated


def test_same_path_helper(tmp_path):
    from w2v_common.storage.service import _same_path

    a = tmp_path / "f.bin"
    a.write_bytes(b"x")
    assert _same_path(a, a) is True
    assert _same_path(a, tmp_path / "g.bin") is False
    assert _same_path(tmp_path / "n1", tmp_path / "n2") is False
    assert _same_path(tmp_path / "same", tmp_path / "same") is True


# cross-bucket get_bytes ------------------------------------------------------


@pytest.mark.asyncio
async def test_minio_get_bytes_bucket_override():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.get_object.return_value = _FakeResponse(b"payload")
    data = await svc.get_bytes("k.bin", bucket="w2v-compiled-videos")
    assert data == b"payload"
    bucket, key = svc._minio_client.get_object.call_args.args
    assert bucket == "w2v-compiled-videos"
    assert key == "k.bin"


@pytest.mark.asyncio
async def test_minio_get_bytes_defaults_own_bucket():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.get_object.return_value = _FakeResponse(b"p")
    await svc.get_bytes("k.bin")
    assert svc._minio_client.get_object.call_args.args[0] == "w2v-data"


# _parse_byte_range -----------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 100, None),
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=10-", 100, (10, 99)),
        ("bytes=-20", 100, (80, 99)),
        ("bytes=0-999", 100, (0, 99)),  # clamped to size-1
        ("bytes=200-300", 100, None),  # start past EOF
        ("bytes=0-9,20-29", 100, None),  # multi-range unsupported
        ("items=0-9", 100, None),  # wrong unit
        ("bytes=abc", 100, None),  # malformed
        ("bytes=5-1", 100, None),  # start > end
    ],
)
def test_parse_byte_range(header, size, expected):
    from w2v_common.storage.service import _parse_byte_range

    assert _parse_byte_range(header, size) == expected


# get_object_stream: local ----------------------------------------------------


@pytest.mark.asyncio
async def test_local_stream_full_200(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("v/clip.bin", b"0123456789")
    stream = await svc.get_object_stream("v/clip.bin")
    assert stream.status == 200
    assert stream.headers["content-length"] == "10"
    assert stream.headers["accept-ranges"] == "bytes"
    assert "etag" in stream.headers and "last-modified" in stream.headers
    assert await _drain(stream, 3) == b"0123456789"
    await stream.aclose()


@pytest.mark.asyncio
async def test_local_stream_range_206(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("v/clip.bin", b"0123456789")
    stream = await svc.get_object_stream("v/clip.bin", request_headers={"Range": "bytes=2-5"})
    assert stream.status == 206
    assert stream.headers["content-range"] == "bytes 2-5/10"
    assert stream.headers["content-length"] == "4"
    assert await _drain(stream, 3) == b"2345"


@pytest.mark.asyncio
async def test_local_stream_missing_404(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    stream = await svc.get_object_stream("nope.bin")
    assert stream.status == 404
    assert await _drain(stream) == b""


@pytest.mark.asyncio
async def test_local_stream_if_none_match_304(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("v/clip.bin", b"0123456789")
    first = await svc.get_object_stream("v/clip.bin")
    etag = first.headers["etag"]
    await first.aclose()
    second = await svc.get_object_stream("v/clip.bin", request_headers={"If-None-Match": etag})
    assert second.status == 304
    assert await _drain(second) == b""


@pytest.mark.asyncio
async def test_stream_async_context_manager(tmp_path):
    svc = StorageService(_local_settings(tmp_path))
    await svc.put_bytes("v/clip.bin", b"hello")
    async with await svc.get_object_stream("v/clip.bin") as stream:
        assert stream.status == 200
        assert await _drain(stream) == b"hello"


# get_object_stream: minio ----------------------------------------------------


@pytest.mark.asyncio
async def test_minio_stream_forwards_range_and_releases():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    fake = _FakeResponse(b"abcdef", status=206, headers={"content-range": "bytes 0-5/6"})
    svc._minio_client.get_object.return_value = fake
    stream = await svc.get_object_stream("k.mp4", request_headers={"Range": "bytes=0-5"})
    assert stream.status == 206
    assert stream.headers["content-range"] == "bytes 0-5/6"
    assert await _drain(stream, 4) == b"abcdef"
    await stream.aclose()
    assert fake.closed is True
    assert fake.released is True
    bucket, key = svc._minio_client.get_object.call_args.args
    assert bucket == "w2v-data"
    assert key == "k.mp4"
    assert svc._minio_client.get_object.call_args.kwargs["request_headers"] == {
        "Range": "bytes=0-5"
    }


@pytest.mark.asyncio
async def test_minio_stream_bucket_override_no_range():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.get_object.return_value = _FakeResponse(b"x", status=200)
    await svc.get_object_stream("k", bucket="w2v-audiobooks")
    assert svc._minio_client.get_object.call_args.args[0] == "w2v-audiobooks"
    assert svc._minio_client.get_object.call_args.kwargs["request_headers"] is None


# --- resign tenant confinement ----------------------------------------------


@pytest.fixture
def _identity_reset():
    from w2v_common.auth.identity_context import clear_identity_context

    clear_identity_context()
    yield
    clear_identity_context()


def _bind_tenant(tenant_id: str) -> None:
    from w2v_common.auth.identity_context import set_identity_context

    set_identity_context(tenant_id=tenant_id, license_id="lic", subject_id="sub")


def _resigning_service() -> StorageService:
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.presigned_get_object.return_value = (
        "http://minio.example.com:9000/w2v-data/k.bin?X-Amz-Signature=fresh"
    )
    return svc


@pytest.mark.parametrize(
    ("key", "tenant"),
    [
        (
            "media-cache/tenants/0195f48b-0000-7000-8000-0000000001aa/ab/abcd.mp3",
            "0195f48b-0000-7000-8000-0000000001aa",
        ),
        (
            "sound_library/tenants/0195f48b-0000-7000-8000-0000000001aa/whoosh.wav",
            "0195f48b-0000-7000-8000-0000000001aa",
        ),
        (
            "tenants/0195f48b-0000-7000-8000-0000000001aa/projects/p1/video-sources/s1/clip.mp4",
            "0195f48b-0000-7000-8000-0000000001aa",
        ),
        (
            "uploads/0195f48b-0000-7000-8000-0000000001aa/job-1/source.mp4",
            "0195f48b-0000-7000-8000-0000000001aa",
        ),
    ],
)
def test_key_tenant_id_reads_every_tenant_bearing_layout(key, tenant):
    from w2v_common.storage.service import key_tenant_id

    assert key_tenant_id(key) == tenant


@pytest.mark.parametrize(
    "key",
    [
        "artifacts/run-1/compile/final.mp4",
        "videos/job-1/final.mp4",
        "decks/job-1/deck.pdf",
        "content_generation/job-1/out.json",
        "audio/job-1.mp3",
        "run-1/thumb-0.png",
        "k.bin",
    ],
)
def test_key_tenant_id_is_none_for_untenanted_layouts(key):
    from w2v_common.storage.service import key_tenant_id

    assert key_tenant_id(key) is None


@pytest.mark.asyncio
async def test_resign_refuses_another_tenants_key(_identity_reset):
    """The T3-1 hole: `w2v-*` bucket ownership let ANY tenant's key re-sign."""
    svc = _resigning_service()
    _bind_tenant("0195f48b-0000-7000-8000-0000000001aa")

    assert (
        await svc.resign_presigned_url(
            "http://minio.example.com:9000/w2v-data/"
            "media-cache/tenants/0195f48b-0000-7000-8000-0000000001bb/ab/abcd.mp3?X-Amz-Signature=stale"
        )
        is None
    )
    assert (
        await svc.resign_presigned_url(
            "http://minio.example.com:9000/w2v-data/"
            "uploads/0195f48b-0000-7000-8000-0000000001bb/job-1/source.mp4?X-Amz-Signature=stale"
        )
        is None
    )


@pytest.mark.asyncio
async def test_resign_allows_own_tenants_key(_identity_reset):
    """Confinement must not be a blanket denial — my own key still re-signs."""
    svc = _resigning_service()
    _bind_tenant("0195f48b-0000-7000-8000-0000000001aa")

    out = await svc.resign_presigned_url(
        "http://minio.example.com:9000/w2v-data/"
        "media-cache/tenants/0195f48b-0000-7000-8000-0000000001aa/ab/abcd.mp3?X-Amz-Signature=stale"
    )
    assert out is not None
    assert "X-Amz-Signature=fresh" in out


@pytest.mark.asyncio
async def test_resign_fails_closed_when_no_tenant_is_bound(_identity_reset):
    svc = _resigning_service()

    assert (
        await svc.resign_presigned_url(
            "http://minio.example.com:9000/w2v-data/"
            "media-cache/tenants/0195f48b-0000-7000-8000-0000000001bb/ab/abcd.mp3?X-Amz-Signature=stale"
        )
        is None
    )


@pytest.mark.asyncio
async def test_resign_segment_boundary_not_string_prefix(_identity_reset):
    """One tenant UUID must not prefix-match another tenant's key."""
    svc = _resigning_service()
    _bind_tenant("0195f48b-0000-7000-8000-0000000001aa")

    assert (
        await svc.resign_presigned_url(
            "http://minio.example.com:9000/w2v-data/"
            "media-cache/tenants/0195f48b-0000-7000-8000-000000000abc/ab/abcd.mp3?X-Amz-Signature=stale"
        )
        is None
    )


@pytest.mark.asyncio
async def test_resign_untenanted_key_is_unchanged(_identity_reset):
    """Blast-radius guard: the ~majority key shape keeps working with no tenant bound."""
    svc = _resigning_service()

    out = await svc.resign_presigned_url(
        "http://minio.example.com:9000/w2v-data/artifacts/run-1/compile/final.mp4?X-Amz-Signature=old"
    )
    assert out is not None
    assert "X-Amz-Signature=fresh" in out


# --- listing failures are distinguishable from empty prefixes ----------------


def _obj(name: str):
    """Minimal stand-in for a miniopy_async listing entry."""
    return type("Obj", (), {"object_name": name})()


def _async_iter(items, *, fail_after: int | None = None, exc: BaseException | None = None):
    """Build an async generator that yields ``items``, optionally dying partway.

    ``fail_after=N`` raises ``exc`` after N items have been yielded, which is
    the mid-iteration failure the old handler discarded silently.
    """

    async def _gen(*args, **kwargs):
        for i, item in enumerate(items):
            if fail_after is not None and i == fail_after:
                raise exc or OSError("connection reset")
            yield item
        if fail_after is not None and fail_after >= len(items):
            raise exc or OSError("connection reset")

    return _gen


@pytest.mark.asyncio
async def test_minio_list_failure_is_not_an_empty_prefix():
    """A degraded listing must NOT read as 'there is nothing here'.

    This is the core assertion: the old code returned ``[]``, so a consumer
    (app_inspector ``list_screenshots``, bdd package builder) reported zero
    delivered artifacts, cleanly, for a job that produced N.
    """
    from w2v_common.storage import StorageListingError

    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.list_objects = _async_iter([], fail_after=0, exc=OSError("minio down"))

    with pytest.raises(StorageListingError) as excinfo:
        await svc.list_prefix("job-1/screenshots/")
    err = excinfo.value
    assert err.prefix == "job-1/screenshots/"
    assert err.bucket == "w2v-data"
    assert err.partial == []
    assert "job-1/screenshots/" in str(err)
    # Must not be an OSError: a broad `except OSError` around storage reads
    # would swallow this straight back into a silent zero.
    assert not isinstance(err, OSError)


@pytest.mark.asyncio
async def test_minio_list_partial_failure_does_not_report_the_partial_set():
    """A listing that dies mid-iteration must not pass off what it got as complete."""
    from w2v_common.storage import StorageListingError

    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.list_objects = _async_iter(
        [_obj("j/a.png"), _obj("j/b.png"), _obj("j/c.png")],
        fail_after=2,
        exc=OSError("reset mid-stream"),
    )

    with pytest.raises(StorageListingError) as excinfo:
        await svc.list_prefix("j/")
    # The collected-so-far keys are preserved, but only on the ERROR -- they are
    # never returned as if they were the whole listing.
    assert excinfo.value.partial == ["j/a.png", "j/b.png"]
    assert "2 key(s)" in str(excinfo.value)


@pytest.mark.asyncio
async def test_minio_list_genuinely_empty_prefix_still_reads_as_empty():
    """Regression guard that matters most: an empty prefix must stay empty."""
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.list_objects = _async_iter([])

    assert await svc.list_prefix("nothing-here/") == []


@pytest.mark.asyncio
async def test_minio_list_success_returns_every_key():
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.list_objects = _async_iter([_obj("j/a.png"), _obj("j/b.png")])

    assert await svc.list_prefix("j/") == ["j/a.png", "j/b.png"]


# --- _retrieve_minio: an absent object is not a degraded MinIO ----------------
#
# Orchestrator callers use ``get_bytes`` as an existence probe for keys that are
# legitimately absent (sound-library manifests, TTS clip sidecars, captions), so
# a clean run used to print several "Failed to retrieve from MinIO" WARNINGs --
# byte-identical to the line a real transport/auth failure produced.

_STORAGE_LOGGER = "w2v_common.storage.service"


def _s3_error(code: str, message: str):
    """A real ``miniopy_async`` ``S3Error``, built the way the client builds it."""
    from miniopy_async.error import S3Error

    return S3Error(
        code,
        message,
        "/w2v-data/k.bin",
        "req-1",
        "host-1",
        AsyncMock(),
        bucket_name="w2v-data",
        object_name="k.bin",
    )


def _minio_failing_get(exc: BaseException) -> StorageService:
    svc = StorageService(_minio_settings())
    svc._minio_client = AsyncMock()
    svc._minio_client.get_object.side_effect = exc
    return svc


async def test_minio_get_missing_key_is_debug_not_warning(caplog):
    svc = _minio_failing_get(_s3_error("NoSuchKey", "The specified key does not exist."))

    with caplog.at_level(logging.DEBUG, logger=_STORAGE_LOGGER):
        assert await svc.get_bytes("sound-library/manifest.json") is None

    records = [r for r in caplog.records if r.name == _STORAGE_LOGGER]
    assert [r for r in records if r.levelno >= logging.WARNING] == []
    debug = [r for r in records if r.levelno == logging.DEBUG]
    assert len(debug) == 1
    assert "sound-library/manifest.json" in debug[0].getMessage()


@pytest.mark.parametrize(
    ("exc_factory", "cls_name", "code"),
    [
        # transport: the socket dies mid-request (an OSError, as aiohttp surfaces it)
        (
            lambda: ConnectionResetError(54, "Connection reset by peer"),
            "ConnectionResetError",
            "54",
        ),
        # timeout
        (lambda: TimeoutError("timed out"), "TimeoutError", "None"),
        # auth
        (lambda: _s3_error("AccessDenied", "Access Denied."), "S3Error", "AccessDenied"),
        # bucket missing is NOT "object absent" -- it is a misconfiguration
        (lambda: _s3_error("NoSuchBucket", "no bucket"), "S3Error", "NoSuchBucket"),
    ],
)
async def test_minio_get_real_failure_is_one_warning_naming_the_error(
    caplog, exc_factory, cls_name, code
):
    svc = _minio_failing_get(exc_factory())

    with caplog.at_level(logging.DEBUG, logger=_STORAGE_LOGGER):
        assert await svc.get_bytes("j/clip.wav") is None

    warnings = [
        r for r in caplog.records if r.name == _STORAGE_LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "j/clip.wav" in message
    assert cls_name in message
    assert f"code={code}" in message


async def test_minio_get_log_never_carries_a_presigned_query_string(caplog):
    presigned = "j/clip.wav?X-Amz-Signature=deadbeef&X-Amz-Credential=AKIA%2Fx"

    for exc in (_s3_error("NoSuchKey", "absent"), ConnectionResetError(54, "reset")):
        svc = _minio_failing_get(exc)
        with caplog.at_level(logging.DEBUG, logger=_STORAGE_LOGGER):
            assert await svc.get_bytes(presigned) is None

    logged = " ".join(r.getMessage() for r in caplog.records if r.name == _STORAGE_LOGGER)
    assert "j/clip.wav" in logged
    assert "X-Amz" not in logged
    assert "deadbeef" not in logged


# --- aclose: the MinIO aiohttp session is released ---------------------------


async def test_aclose_closes_the_real_minio_session_and_is_idempotent() -> None:
    import aiohttp
    from miniopy_async import Minio

    svc = StorageService(_minio_settings())
    client = Minio("minio.example.com:9000", access_key="k", secret_key="s", secure=False)
    session = aiohttp.ClientSession()
    client.set_session(session)
    svc._minio_client = client

    await svc.aclose()
    assert session.closed
    assert client._session is None
    assert svc._minio_client is None
    await svc.aclose()  # idempotent


async def test_aclose_closes_minios_own_default_session() -> None:
    from miniopy_async import Minio

    svc = StorageService(_minio_settings())
    client = Minio("minio.example.com:9000", access_key="k", secret_key="s", secure=False)
    client._ensure_session()  # the session Minio opens on its first request
    from aiohttp_retry import RetryClient

    retry = client._session
    assert isinstance(retry, RetryClient)
    inner = retry._client  # RetryClient -> aiohttp.ClientSession
    assert not inner.closed
    svc._minio_client = client

    await svc.aclose()
    assert inner.closed


async def test_aclose_local_backend_is_noop(tmp_path) -> None:
    svc = StorageService(_local_settings(tmp_path))
    await svc.aclose()
    await svc.aclose()
    assert svc.backend == "local"
    assert svc._minio_client is None


async def test_async_with_initializes_and_closes(tmp_path) -> None:
    async with StorageService(_local_settings(tmp_path / "store")) as svc:
        assert (tmp_path / "store").is_dir()
        assert svc.backend == "local"
