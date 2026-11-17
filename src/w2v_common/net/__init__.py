"""Network helpers shared across the w2v fleet."""

from __future__ import annotations

from w2v_common.net.safe_fetch import (
    DEFAULT_MAX_FETCH_BYTES,
    DEFAULT_MAX_REDIRECTS,
    ResponseTooLargeError,
    SSRFError,
    safe_get,
    safe_get_sync,
    safe_get_to_path,
    safe_post,
    safe_post_sync,
    safe_request,
    safe_request_sync,
    validate_public_url,
)

__all__ = [
    "DEFAULT_MAX_FETCH_BYTES",
    "DEFAULT_MAX_REDIRECTS",
    "ResponseTooLargeError",
    "SSRFError",
    "safe_get",
    "safe_get_sync",
    "safe_get_to_path",
    "safe_post",
    "safe_post_sync",
    "safe_request",
    "safe_request_sync",
    "validate_public_url",
]
