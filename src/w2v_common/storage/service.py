"""Shared object-storage service with local and MinIO/S3 backends.

This is the fleet-wide consolidation of the ~12 hand-rolled per-service
``storage_service.py`` / ``minio_client.py`` modules. It is constructed from a
:class:`w2v_common.config.StorageSettings` and exposes generic object-key
primitives; each service keeps a thin domain wrapper (its own key conventions)
over this facade. No service routes through a separate storage *service* — every
service keeps its own bucket + credentials (per-service isolation); only the
client/presign/rewrite code is deduplicated here.

The MinIO client is ``miniopy-async`` and is imported lazily, so this module
(and ``w2v_common``) carry no hard MinIO dependency — a consumer that uses the
``minio`` backend declares ``miniopy-async`` itself; a ``local``-backend
consumer needs nothing extra.

The presign / resign / object-exists logic is lifted verbatim from the
orchestrator's ``storage_service.py`` (the superset), which encodes hard-won
fixes: SigV4 GET-only presign (no HEAD probe), ``%2520`` double-encoding of
keys with literal spaces/commas, NoSuchKey-only fail-closed in object_exists,
and non-OSError MinIO errors degrading instead of flapping readiness.
"""

from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse, urlunparse

from w2v_common.auth.identity_context import get_tenant_id_from_context

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from w2v_common.config.base import StorageSettings

logger = logging.getLogger(__name__)

# Default streaming chunk size (64 KiB) — matches the orchestrator artifact proxy.
_STREAM_CHUNK_BYTES = 64 * 1024


def _same_path(src: Path, dest: Path) -> bool:
    """True when ``src`` and ``dest`` denote the same file on disk.

    Guards :meth:`StorageService.put_file`'s local branch: a caller that composes
    an artifact directly under ``local_path`` and then "stores" it in place would
    otherwise hit ``shutil.SameFileError``. When both paths exist we use
    ``os.path.samefile`` (handles hardlinks / distinct-but-equal paths); otherwise
    (the usual case — ``dest`` not yet written) we fall back to a resolved-path
    compare.
    """
    try:
        if src.exists() and dest.exists():
            return src.samefile(dest)
    except OSError:
        return False
    return src.resolve() == dest.resolve()


def _parse_byte_range(range_header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single ``Range: bytes=start-end`` header into inclusive offsets.

    Returns ``(start, end)`` for a satisfiable single range, or ``None`` for no
    range / a blank, malformed, multi-range, or unsatisfiable spec (the caller
    then serves the whole object as ``200``). Supports open-ended (``start-``) and
    suffix (``-N``) forms per RFC 7233.
    """
    if not range_header or size <= 0:
        return None
    units, _, spec = range_header.partition("=")
    if units.strip().lower() != "bytes" or "," in spec:
        return None
    start_s, sep, end_s = spec.strip().partition("-")
    if not sep:
        return None
    try:
        if start_s == "":
            suffix = int(end_s)
            if suffix <= 0:
                return None
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    if start > end or start >= size:
        return None
    return start, min(end, size - 1)


#: The literal segment that introduces a tenant id in every tenant-scoped w2v key
#: layout: ``media-cache/tenants/{tenant}/…`` (orchestrator tts_storage),
#: ``sound_library/tenants/{tenant}/…`` (sound_catalog) and
#: ``tenants/{tenant}/projects/{project}/video-sources/{source}/`` (uploads_video).
_TENANT_MARKER = "tenants"

#: First-segment roots whose SECOND segment is a tenant id — the one layout that
#: does not use the ``tenants/`` marker: ``uploads/{tenant}/{job}/…``
#: (video_analyzer_service).
_TENANT_ROOTS: tuple[str, ...] = ("uploads",)


def key_tenant_id(key: str) -> str | None:
    """The tenant id encoded in object ``key``, or ``None`` when it encodes none.

    Only the layouts the fleet actually writes are recognised — see
    ``_TENANT_MARKER`` / ``_TENANT_ROOTS``. The majority of w2v keys
    (``artifacts/{run}/{stage}/…``, ``videos/{job}/…``, ``decks/{job}/…``,
    ``{run_id}/…``) carry **no** tenant at all; their isolation lives in the
    owning run/job row, not in the key, so this returns ``None`` for them and
    callers must not read that as "public".

    Matching is per **segment**, never ``str.startswith`` — a prefix compare
    lets ``…/tenants/{tenant}evil/x`` pass for tenant ``{tenant}``.
    """
    segments = [seg for seg in key.split("/") if seg]
    for index, segment in enumerate(segments[:-1]):
        if segment == _TENANT_MARKER or (index == 0 and segment in _TENANT_ROOTS):
            return segments[index + 1]
    return None


def _safe_stat(path: Path) -> os.stat_result | None:
    """``path.stat()`` or ``None`` if the path is absent/unreadable."""
    try:
        return path.stat()
    except OSError:
        return None


async def _read_file_range(
    path: Path, start: int, length: int, chunk_size: int
) -> AsyncIterator[bytes]:
    """Yield ``length`` bytes of ``path`` starting at offset ``start``, in
    ``chunk_size`` slices, doing the blocking file I/O off the event loop."""

    def _open() -> Any:
        handle = path.open("rb")
        if start:
            handle.seek(start)
        return handle

    handle = await asyncio.to_thread(_open)
    try:
        remaining = length
        while remaining > 0:
            data = await asyncio.to_thread(handle.read, min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
    finally:
        await asyncio.to_thread(handle.close)


class StreamedObject:
    """A range-aware, streaming read handle returned by
    :meth:`StorageService.get_object_stream`.

    Exposes the upstream ``status`` (``200`` full / ``206`` partial / ``404``
    absent), response ``headers`` (content-range, accept-ranges, etag,
    last-modified, content-length, content-type), and :meth:`iter_chunks` — an
    async byte iterator that never buffers the whole object. The caller MUST
    :meth:`aclose` it (or use it as an ``async with``) to release the underlying
    connection / file handle.
    """

    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        chunk_source: Callable[[int], AsyncIterator[bytes]],
        closer: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers
        self._chunk_source = chunk_source
        self._closer = closer

    async def iter_chunks(self, chunk_size: int = _STREAM_CHUNK_BYTES) -> AsyncIterator[bytes]:
        async for chunk in self._chunk_source(chunk_size):
            yield chunk

    async def aclose(self) -> None:
        if self._closer is not None:
            await self._closer()
            self._closer = None

    async def __aenter__(self) -> StreamedObject:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _minio_exc_types() -> tuple[type[BaseException], ...]:
    """MinIO failure types that are NOT ``OSError`` subclasses.

    ``miniopy_async`` raises ``S3Error`` / ``MinioException`` (e.g. on a missing
    bucket, expired creds, or a transport error). These do not derive from
    ``OSError``, so a handler that only catches ``OSError`` lets them propagate —
    turning a degraded-MinIO condition into a readiness flap (``check_health``)
    or an unhandled 500 (``presign_get``). Catch these alongside ``OSError`` so
    both degrade to a sane fallback.
    """
    try:
        from miniopy_async.error import MinioException, S3Error
    except ImportError:  # pragma: no cover -- a minio-backend consumer always has it
        return ()
    return (S3Error, MinioException)


# Computed once at import: the MinIO exception classes to catch alongside OSError.
_MINIO_EXC: tuple[type[BaseException], ...] = _minio_exc_types()


class StorageListingError(RuntimeError):
    """Raised when a prefix listing could not be completed.

    A listing that fails is NOT an empty prefix. Callers read ``[]`` as "there
    is nothing here" and report zero delivered artifacts, cleanly -- the fleet's
    #1 defect class (fail-open on a measurement gap). This makes the gap
    expressible and loud.

    Deliberately **not** an ``OSError`` subclass: the failures being reported
    are ``OSError``/``S3Error``, and existing broad ``except OSError`` handlers
    around storage reads would swallow this straight back into a silent zero.

    Attributes:
        prefix: The prefix whose listing failed.
        bucket: The bucket that was being listed.
        partial: Keys collected before the failure. Non-empty means the listing
            died mid-iteration -- an INCOMPLETE set, never a complete one. Only
            a caller that can act correctly on a known-incomplete set should
            read it.
    """

    def __init__(
        self,
        message: str,
        *,
        prefix: str,
        bucket: str,
        partial: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.prefix = prefix
        self.bucket = bucket
        self.partial: list[str] = list(partial or [])


class StorageService:
    """Manages object storage (local filesystem or MinIO/S3).

    Construct from a :class:`StorageSettings`; call :meth:`initialize` once at
    startup. All methods operate on object *keys* (the MinIO backend) or paths
    relative to ``local_path`` (the local backend); ``put_*`` returns the
    reference (``key`` for MinIO, absolute path for local) that ``get_bytes``
    reads back.
    """

    def __init__(self, settings: StorageSettings) -> None:
        self._backend = settings.backend
        self._local_path = Path(settings.local_path)
        self._minio_endpoint = settings.minio_endpoint
        self._minio_access_key = settings.minio_access_key
        self._minio_secret_key = settings.minio_secret_key
        self._minio_bucket = settings.minio_bucket
        self._minio_secure = settings.minio_secure
        self._presign_expiry = settings.presign_expiry_seconds
        self._minio_external_endpoint: str | None = settings.minio_external_endpoint
        self._minio_client: Any = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def bucket(self) -> str:
        return self._minio_bucket

    async def initialize(self) -> None:
        """Initialize the storage backend (create local dir / ensure bucket)."""
        if self._backend == "local":
            self._local_path.mkdir(parents=True, exist_ok=True)
            logger.info("Local storage initialized at %s", self._local_path)
        elif self._backend == "minio":
            await self._init_minio()
        else:
            raise ValueError(f"Unsupported storage backend: {self._backend}")

    async def aclose(self) -> None:
        """Close the MinIO client's aiohttp session. Idempotent; a no-op for ``local``.

        ``miniopy_async.Minio`` opens one ``aiohttp.ClientSession`` lazily and keeps it
        until ``close_session()``; without this hook every ``StorageService`` leaked it
        (consumers reached into ``_minio_client`` to close it). After ``aclose`` the
        service holds no client: call :meth:`initialize` again to reuse it.
        """
        client = self._minio_client
        self._minio_client = None
        if client is not None:
            await client.close_session()

    async def __aenter__(self) -> StorageService:
        """``async with StorageService(settings) as storage:`` — initialize, then aclose on exit."""
        await self.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _init_minio(self) -> None:
        try:
            from miniopy_async import Minio
        except ImportError as exc:
            raise RuntimeError(
                "miniopy-async is required for the MinIO storage backend. "
                "Install with: pip install miniopy-async"
            ) from exc

        self._minio_client = Minio(
            self._minio_endpoint,
            access_key=self._minio_access_key,
            secret_key=self._minio_secret_key,
            secure=self._minio_secure,
        )
        if not await self._minio_client.bucket_exists(self._minio_bucket):
            await self._minio_client.make_bucket(self._minio_bucket)
            logger.info("Created MinIO bucket: %s", self._minio_bucket)
        logger.info("MinIO storage initialized: %s/%s", self._minio_endpoint, self._minio_bucket)

    async def check_health(self) -> bool:
        """Storage backend health (local dir present / bucket reachable)."""
        if self._backend == "local":
            return self._local_path.exists()
        if self._backend == "minio":
            if self._minio_client is None:
                return False
            try:
                return await self._minio_client.bucket_exists(self._minio_bucket)
            except (OSError, *_MINIO_EXC):
                return False
        return False

    # --- object primitives ---------------------------------------------------

    async def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Store ``data`` at ``key``. Returns the reference ``get_bytes`` reads.

        local → writes ``local_path/key`` and returns the absolute path string.
        minio → ``put_object`` and returns ``key``.
        """
        if self._backend == "local":
            return await self._store_local(key, data)
        if self._backend == "minio":
            await self._minio_client.put_object(
                self._minio_bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            logger.info("Stored to MinIO: %s (%d bytes)", key, len(data))
            return key
        raise ValueError(f"Unsupported backend: {self._backend}")

    async def put_file(
        self,
        key: str,
        local_path: str | Path,
        *,
        content_type: str = "application/octet-stream",
        bucket: str | None = None,
    ) -> str:
        """Store a file that is already on disk at ``key`` — WITHOUT reading it
        into memory. Use this (not :meth:`put_bytes`) for large artifacts
        (compiled videos, audiobooks) so a multi-GB file never lands in RAM.

        local → copies ``local_path`` to ``local_path/key`` (buffered copy) and
        returns the absolute destination path. minio → streams via
        ``fput_object`` (chunked multipart, constant memory) and returns ``key``.
        ``bucket`` defaults to the service's own bucket; pass an explicit bucket
        to write into another bucket this service owns (mirrors
        :meth:`presign_get`).
        """
        src = Path(local_path)
        if self._backend == "local":
            dest = self._local_path / key
            # In-place store: the caller's source already IS the destination key
            # under local_path (e.g. a compile pipeline that writes the artifact
            # straight into the store). shutil.copyfile would raise SameFileError,
            # so treat it as a no-op and return the existing path.
            if await asyncio.to_thread(_same_path, src, dest):
                logger.info("put_file no-op (source is destination): %s", dest)
                return str(dest)
            await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, src, dest)
            logger.info("Stored file (local): %s -> %s", src, dest)
            return str(dest)
        if self._backend == "minio":
            await self._minio_client.fput_object(
                bucket or self._minio_bucket,
                key,
                str(src),
                content_type=content_type,
            )
            logger.info("Streamed file to MinIO: %s", key)
            return key
        raise ValueError(f"Unsupported backend: {self._backend}")

    async def get_bytes(self, ref: str, *, bucket: str | None = None) -> bytes | None:
        """Read raw bytes back by the reference a ``put_*`` returned.

        ``ref`` is a MinIO object key, or an absolute/relative path for the local
        backend (read by key, never by an arbitrary URL — no SSRF surface).
        ``bucket`` defaults to the service's own configured bucket; pass an
        explicit bucket to read an object in another bucket this service owns
        (mirrors :meth:`put_file` / :meth:`presign_get`). ``bucket`` is ignored by
        the local backend. Returns ``None`` if the object is absent.
        """
        if self._backend == "local":
            path = Path(ref)
            if not path.is_absolute():
                path = self._local_path / ref
            if not await asyncio.to_thread(path.is_file):
                return None
            return await asyncio.to_thread(path.read_bytes)
        if self._backend == "minio":
            return await self._retrieve_minio(ref, bucket=bucket)
        return None

    # Back-compat alias for the orchestrator's ``retrieve(ref)`` naming.
    retrieve = get_bytes

    async def get_object_stream(
        self,
        key: str,
        *,
        bucket: str | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> StreamedObject:
        """Open a streaming, range-aware read of ``key`` for byte-range delivery
        (seekable ``<video>`` / ``<audio>`` / large images).

        Unlike :meth:`get_bytes`, this NEVER buffers the whole object in memory —
        it returns a :class:`StreamedObject` exposing the upstream ``status``
        (``200`` / ``206`` / ``404``), the response ``headers``, and an async
        chunk iterator. ``request_headers`` (``Range`` / ``If-Range`` /
        ``If-None-Match`` / ``If-Modified-Since``) are forwarded to the backend so
        conditional and partial GETs work. ``bucket`` defaults to the service's
        own bucket. The caller MUST ``aclose()`` the result (or use ``async
        with``) to release the connection.

        This is what the orchestrator artifact proxy needs; it replaces reaching
        into the private ``_minio_client``.
        """
        if self._backend == "local":
            return await self._open_local_stream(key, request_headers=request_headers)
        if self._backend == "minio":
            if self._minio_client is None:
                raise RuntimeError("MinIO backend not initialized")
            response = await self._minio_client.get_object(
                bucket or self._minio_bucket,
                key,
                request_headers=dict(request_headers) if request_headers else None,
            )

            def _chunks(chunk_size: int) -> AsyncIterator[bytes]:
                return response.content.iter_chunked(chunk_size)

            async def _close() -> None:
                response.close()
                await response.release()

            return StreamedObject(
                status=response.status,
                headers=dict(response.headers),
                chunk_source=_chunks,
                closer=_close,
            )
        raise ValueError(f"Unsupported backend: {self._backend}")

    async def delete_prefix(self, prefix: str) -> bool:
        """Delete every object under ``prefix``. Returns True on success."""
        if self._backend == "local":
            dir_path = self._local_path / prefix
            if await asyncio.to_thread(dir_path.exists):
                await asyncio.to_thread(shutil.rmtree, dir_path)
                return True
            return False
        if self._backend == "minio":
            return await self._delete_minio_prefix(prefix)
        return False

    async def delete_object(self, key: str, *, bucket: str | None = None) -> bool:
        """Delete a SINGLE object by ``key`` (unlike :meth:`delete_prefix`, which
        removes everything under a prefix). Returns True on success, False on a
        storage error; a missing object is treated as success (idempotent).

        local → unlinks ``local_path/key`` (``missing_ok``). minio →
        ``remove_object`` on ``bucket`` (defaults to the service's own bucket).
        """
        if self._backend == "local":
            path = Path(key)
            if not path.is_absolute():
                path = self._local_path / key
            await asyncio.to_thread(path.unlink, True)
            return True
        if self._backend == "minio":
            try:
                await self._minio_client.remove_object(bucket or self._minio_bucket, key)
                return True
            except (OSError, *_MINIO_EXC):
                logger.warning("Failed to delete MinIO object: %s", key)
                return False
        return False

    async def list_prefix(self, prefix: str) -> list[str]:
        """List object keys / relative paths under ``prefix``.

        An empty list means the prefix is genuinely empty. A listing that could
        not be completed raises :class:`StorageListingError` rather than
        degrading to ``[]`` -- the two are not the same answer, and the local
        backend has always raised (its ``rglob`` ``OSError`` propagates), so
        this also makes the two backends agree.
        """
        if self._backend == "local":
            dir_path = self._local_path / prefix
            if not await asyncio.to_thread(dir_path.exists):
                return []

            def _scan() -> list[str]:
                return [
                    str(p.relative_to(self._local_path)) for p in dir_path.rglob("*") if p.is_file()
                ]

            return await asyncio.to_thread(_scan)
        if self._backend == "minio":
            return await self._list_minio(prefix)
        return []

    async def presign_get(
        self, key: str, *, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        """Generate a presigned GET URL (MinIO only); ``None`` otherwise.

        ``bucket`` defaults to the service's own configured bucket; pass an
        explicit bucket to presign an object the service can read but did not
        write. When ``minio_external_endpoint`` is configured the in-cluster host
        is rewritten so clients outside Docker/k8s can reach MinIO directly.
        """
        if self._backend != "minio" or self._minio_client is None:
            return None
        from datetime import timedelta

        expiry = timedelta(seconds=expires or self._presign_expiry)
        try:
            url = await self._minio_client.presigned_get_object(
                bucket or self._minio_bucket,
                key,
                expires=expiry,
            )
            if url and self._minio_external_endpoint:
                url = self._rewrite_presigned_url(url, self._minio_external_endpoint)
            return url
        except (OSError, *_MINIO_EXC):
            logger.warning("Failed to generate presigned URL for: %s", key)
            return None

    async def object_exists(self, url: str) -> bool | None:
        """Probe whether the artifact behind ``url`` exists in storage.

        Returns ``True`` (present), ``False`` (definitively absent — bucket
        exists, key does not; the only signal a caller may fail on), or ``None``
        (indeterminate: foreign host, missing bucket, permissions/transport
        error, unprobeable URL — the caller MUST treat as "don't know").

        Does NOT HTTP ``HEAD`` the presigned URL: SigV4 presigns a single method
        (GET), so HEAD returns 403 even when the object exists. Instead it
        re-derives ``{bucket}/{key}`` and ``stat_object`` with our own creds.
        """
        scheme = urlparse(url).scheme
        if self._backend == "local" or scheme == "file":
            return await self._object_exists_local(url)
        if self._backend == "minio":
            return await self._object_exists_minio(url)
        return None

    async def object_exists_key(self, key: str, *, bucket: str | None = None) -> bool | None:
        """Probe existence by a bare object ``key`` (not a URL — see
        :meth:`object_exists`, which re-derives the key from a presigned URL).

        For a service checking its own key space (``storage.exists(rel_key)``):
        no scheme/host/bucket parsing, no ambiguity between a bucket prefix and a
        key that contains ``/``. Same tri-state contract as :meth:`object_exists`
        (True present / False definitively absent / None indeterminate).
        """
        if self._backend == "local":
            path = Path(key)
            if not path.is_absolute():
                path = self._local_path / key
            return await asyncio.to_thread(path.is_file)
        if self._backend != "minio" or self._minio_client is None:
            return None
        try:
            from miniopy_async.error import S3Error
        except ImportError:  # pragma: no cover -- a minio-backend consumer always has it
            return None
        try:
            await self._minio_client.stat_object(bucket or self._minio_bucket, key)
            return True
        except S3Error as exc:
            if getattr(exc, "code", None) == "NoSuchKey":
                return False
            logger.warning(
                "object_exists_key: indeterminate S3Error for %s (code=%s)",
                key,
                getattr(exc, "code", "?"),
            )
            return None
        except Exception as exc:  # noqa: BLE001 -- any probe failure is "don't know", never a verdict
            logger.warning("object_exists_key: transport error probing %s: %s", key, exc)
            return None

    async def resign_presigned_url(self, url: str) -> str | None:
        """Re-mint a fresh presigned GET URL for a (possibly expired) URL we own.

        Returns ``None`` (caller keeps the original) unless the URL is
        unambiguously ours: minio backend, http(s) scheme, a host we presign for,
        a bucket we own, and — for a tenant-scoped key — the CALLING tenant's own
        object (see :meth:`_may_resign_key`). The guard is load-bearing —
        ``presigned_get_object`` only *signs*; re-signing a foreign URL would mint
        a valid-looking pointer into our MinIO for an absent object (a guaranteed
        404), and re-signing another tenant's key would mint a live, working one.
        """
        if self._backend != "minio" or self._minio_client is None:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        if not self._is_own_minio_host(parsed.netloc):
            return None
        bucket, _, key = parsed.path.lstrip("/").partition("/")
        bucket, key = unquote(bucket), unquote(key)
        if not key or not self._is_own_bucket(bucket):
            return None
        if not self._may_resign_key(key):
            return None
        return await self.presign_get(key, bucket=bucket)

    def _may_resign_key(self, key: str) -> bool:
        """True when the CALLING tenant is allowed a fresh signature over ``key``.

        ``_is_own_bucket`` bounds re-signing to our MinIO but NOT to our tenant —
        it deliberately accepts every ``w2v-*`` bucket because our restricted user
        can read them all. So for a tenant-scoped key that check alone let any
        caller re-mint a live URL over ANOTHER tenant's object. Here the tenant
        baked into the key must match the tenant on the request (identity
        ContextVar, set by ``AuthMiddleware``), and an unbound tenant fails CLOSED.

        Keys whose layout carries no tenant (:func:`key_tenant_id` → ``None``) are
        untouched: the library cannot tell whose they are, their isolation is the
        owning run/job row, and gating them here would break every consumer that
        re-signs ``artifacts/{run}/{stage}/…``.
        """
        key_tenant = key_tenant_id(key)
        if key_tenant is None:
            return True
        caller_tenant = get_tenant_id_from_context()
        if caller_tenant is None:
            logger.warning("resign refused: tenant-scoped key with no tenant bound to the request")
            return False
        # ``key_tenant`` comes out of the object KEY (a string path segment);
        # compare on the canonical UUID rendering so "0195F48B-..." and
        # "0195f48b-..." are the same tenant, not two.
        if key_tenant.strip().lower() != str(caller_tenant):
            logger.warning("resign refused: key belongs to a different tenant")
            return False
        return True

    # --- presigned-URL helpers (pure) ----------------------------------------

    @staticmethod
    def _rewrite_presigned_url(url: str, external_endpoint: str) -> str:
        """Replace the host (and optionally scheme) of a presigned URL.

        ``external_endpoint`` may be a bare ``host:port``, ``host``, or a full
        ``scheme://host:port`` value.
        """
        parsed = urlparse(url)
        ext = urlparse(
            external_endpoint if "://" in external_endpoint else f"scheme://{external_endpoint}"
        )
        new_scheme = ext.scheme if ext.scheme and ext.scheme != "scheme" else parsed.scheme
        new_netloc = ext.netloc if ext.netloc else ext.path
        return urlunparse(
            (new_scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )

    @staticmethod
    def _endpoint_netloc(endpoint: str) -> str:
        """Normalise a configured endpoint to a bare ``host[:port]``."""
        if "://" in endpoint:
            return urlparse(endpoint).netloc
        return urlparse(f"scheme://{endpoint}").netloc

    def _is_own_minio_host(self, netloc: str) -> bool:
        """True when ``netloc`` is a MinIO host this service presigns for."""
        if not netloc:
            return False
        own = {self._endpoint_netloc(self._minio_endpoint)}
        if self._minio_external_endpoint:
            own.add(self._endpoint_netloc(self._minio_external_endpoint))
        return netloc in own

    def _is_own_bucket(self, bucket: str) -> bool:
        """True for a bucket this service can presign objects in.

        Its own configured bucket, or any ``w2v-*`` artifact bucket — every w2v
        service's restricted user can read every w2v bucket, so re-signing any of
        them with our own creds is safe; the host check bounds this to objects
        that actually live in our MinIO.

        This is a BUCKET check, not a TENANT check — on its own it confines
        nothing per-tenant. :meth:`_may_resign_key` adds the tenant confinement.
        """
        return bool(bucket) and (bucket == self._minio_bucket or bucket.startswith("w2v-"))

    # --- local backend -------------------------------------------------------

    async def _store_local(self, path: str, data: bytes) -> str:
        file_path = self._local_path / path
        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(file_path.write_bytes, data)
        logger.info("Stored artifact: %s (%d bytes)", file_path, len(data))
        return str(file_path)

    async def _object_exists_local(self, url: str) -> bool | None:
        parsed = urlparse(url)
        raw_path = parsed.path if parsed.scheme == "file" else url
        if not raw_path:
            return None
        return await asyncio.to_thread(Path(raw_path).is_file)

    async def _open_local_stream(
        self, key: str, *, request_headers: Mapping[str, str] | None = None
    ) -> StreamedObject:
        """Local-backend counterpart to the MinIO streaming get: honours a single
        ``Range`` header and ``If-None-Match`` so the same range-aware contract
        holds without a live MinIO (makes :meth:`get_object_stream` unit-testable).
        """
        path = Path(key)
        if not path.is_absolute():
            path = self._local_path / key

        async def _empty(_chunk_size: int) -> AsyncIterator[bytes]:
            return
            yield b""  # pragma: no cover -- makes this an async generator

        stat = await asyncio.to_thread(_safe_stat, path)
        if stat is None:
            return StreamedObject(status=404, headers={}, chunk_source=_empty)

        from email.utils import formatdate

        size = stat.st_size
        etag = f'"{int(stat.st_mtime)}-{size}"'
        headers: dict[str, str] = {
            "accept-ranges": "bytes",
            "content-type": mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            "etag": etag,
            "last-modified": formatdate(stat.st_mtime, usegmt=True),
        }

        req = {k.lower(): v for k, v in (request_headers or {}).items()}
        if_none_match = req.get("if-none-match")
        if if_none_match and (if_none_match.strip() == "*" or etag in if_none_match):
            return StreamedObject(status=304, headers=headers, chunk_source=_empty)

        rng = _parse_byte_range(req.get("range"), size)
        if rng is None:
            start, end, status = 0, size - 1, 200
        else:
            start, end = rng
            status = 206
            headers["content-range"] = f"bytes {start}-{end}/{size}"
        length = end - start + 1
        headers["content-length"] = str(length)

        def _chunks(chunk_size: int) -> AsyncIterator[bytes]:
            return _read_file_range(path, start, length, chunk_size)

        return StreamedObject(status=status, headers=headers, chunk_source=_chunks)

    # --- MinIO backend -------------------------------------------------------

    async def _retrieve_minio(self, key: str, *, bucket: str | None = None) -> bytes | None:
        try:
            response = await self._minio_client.get_object(bucket or self._minio_bucket, key)
            try:
                data = await response.read()
            finally:
                response.close()
                await response.release()
            return data
        except (OSError, *_MINIO_EXC) as exc:
            # Callers probe for objects that are legitimately absent (sound-library
            # manifests, TTS clip sidecars, captions). "Not there" is an answer, not
            # a degraded MinIO, so it must not share a WARNING with a real failure.
            # ``code`` is the S3 error code (``S3Error.code``) or, for a transport
            # ``OSError``, its errno. Only the key path, class and code are logged --
            # never ``str(exc)`` or a query string (presigned ``X-Amz-*`` material).
            code = getattr(exc, "code", None) or getattr(exc, "errno", None)
            safe_key = key.split("?", 1)[0]
            if code == "NoSuchKey":
                logger.debug("MinIO object absent: %s", safe_key)
            else:
                logger.warning(
                    "Failed to retrieve from MinIO: %s (%s, code=%s)",
                    safe_key,
                    type(exc).__name__,
                    code,
                )
            return None

    async def _delete_minio_prefix(self, prefix: str) -> bool:
        try:
            async for obj in self._minio_client.list_objects(
                self._minio_bucket, prefix=prefix, recursive=True
            ):
                await self._minio_client.remove_object(self._minio_bucket, obj.object_name)
            return True
        except (OSError, *_MINIO_EXC):
            logger.warning("Failed to delete MinIO prefix: %s", prefix)
            return False

    async def _list_minio(self, prefix: str) -> list[str]:
        """List every key under ``prefix``; raise if the listing is incomplete.

        The old handler returned ``[]`` on failure, so "MinIO is degraded" and
        "this prefix is empty" were the same value to every caller -- and
        because the ``except`` sits outside the ``async for``, a failure
        *mid-iteration* discarded the partially-collected keys too, reporting a
        partial listing as a clean zero. That is how a job that captured N
        screenshots reports 0 screenshots, cleanly (observed 2026-09-03).
        """
        result: list[str] = []
        try:
            async for obj in self._minio_client.list_objects(
                self._minio_bucket, prefix=prefix, recursive=True
            ):
                result.append(obj.object_name)
        except (OSError, *_MINIO_EXC) as exc:
            logger.warning(
                "Failed to list MinIO prefix %s after %d key(s): %s", prefix, len(result), exc
            )
            raise StorageListingError(
                f"listing bucket {self._minio_bucket!r} prefix {prefix!r} failed after "
                f"{len(result)} key(s): {type(exc).__name__}: {exc}",
                prefix=prefix,
                bucket=self._minio_bucket,
                partial=result,
            ) from exc
        return result

    async def _object_exists_minio(self, url: str) -> bool | None:
        if self._minio_client is None:
            return None
        path = urlparse(url).path.lstrip("/")
        bucket, _, key = path.partition("/")
        bucket, key = unquote(bucket), unquote(key)
        if not bucket or not key:
            return None
        try:
            from miniopy_async.error import S3Error
        except ImportError:  # pragma: no cover -- a minio-backend consumer always has it
            return None
        try:
            await self._minio_client.stat_object(bucket, key)
            return True
        except S3Error as exc:
            if getattr(exc, "code", None) == "NoSuchKey":
                return False
            logger.warning(
                "object_exists: indeterminate S3Error for %s (code=%s)",
                url,
                getattr(exc, "code", "?"),
            )
            return None
        except Exception as exc:  # noqa: BLE001 -- any probe failure is "don't know", never a verdict
            logger.warning("object_exists: transport error probing %s: %s", url, exc)
            return None
