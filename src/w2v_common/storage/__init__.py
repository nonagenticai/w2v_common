"""Shared object-storage abstraction (local + MinIO/S3 backends).

Construct :class:`StorageService` from a
:class:`w2v_common.config.StorageSettings`. See ``service.py`` for the rationale
(fleet-wide consolidation of per-service storage modules).
"""

from __future__ import annotations

from w2v_common.storage.service import StorageListingError, StorageService, StreamedObject

__all__ = ["StorageListingError", "StorageService", "StreamedObject"]
