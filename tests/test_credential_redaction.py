"""Credentials must not survive into an error message, a log line, or an audit row.

Two leaks, one class. Both were found by reading, not by an incident — but the
fleet has already had the same class land for real (a DB password reached stdout
the moment logging was turned on), so these are the cheap half of that lesson.

1. `_redact_for_log` masked only `Bearer`, `token=` and `Authorization:`. The
   orchestrator PUTs a tenant's BYO provider **API key** to worker_management;
   a pydantic 422 there echoes the offending value back in its `input` field,
   and that body is embedded verbatim into `HTTPStatusError`, logged by
   `logger.error`, and stored in the provisioning audit row as `error=str(exc)`.
   `"api_key": "sk-..."` matched none of the three patterns.

2. `SSRFError` messages interpolated the **full** URL. A presigned MinIO/S3 URL
   carries `X-Amz-Signature` in its query string, and `_pin` shows `user:pass@`
   userinfo is an expected shape here — so refusing a credentialed URL published
   the credential.
"""

from __future__ import annotations

import pytest

from w2v_common.clients.base import _redact_for_log
from w2v_common.net.safe_fetch import _safe_url_for_message

# Not a real key. `pragma: allowlist secret` is this repo's convention for a
# literal that only exists so a leak assertion has something to look for.
_SECRET = "sk-live-51H8xQwErTyUiOpAsDfGhJkL"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "body",
    [
        f'{{"detail":[{{"loc":["body","api_key"],"input":"{_SECRET}"}}]}}',
        f'{{"apiKey": "{_SECRET}"}}',
        f"api-key={_SECRET}",
        f'{{"client_secret": "{_SECRET}"}}',
        f'{{"password": "{_SECRET}"}}',
        f'{{"access_key": "{_SECRET}"}}',
        f'{{"refresh_token": "{_SECRET}"}}',
    ],
)
def test_credential_bearing_upstream_bodies_are_masked(body: str) -> None:
    redacted = _redact_for_log(body)

    assert _SECRET not in redacted, f"the secret survived redaction: {redacted}"
    assert "REDACTED" in redacted


def test_presign_signatures_are_masked() -> None:
    body = (
        "AccessDenied: https://minio.example.com/b/o?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Signature=deadbeefcafebabe0123456789abcdef"  # pragma: allowlist secret
    )

    redacted = _redact_for_log(body)

    assert "deadbeefcafebabe0123456789abcdef" not in redacted


def test_the_bearer_lane_still_works() -> None:
    """The original three patterns must not regress while widening."""
    assert _SECRET not in _redact_for_log(f"Authorization: Bearer {_SECRET}")


def test_redaction_does_not_eat_the_useful_part() -> None:
    """Over-redaction makes logs useless — the diagnosis must survive."""
    redacted = _redact_for_log('{"detail":"model gpt-5 is not enabled","api_key":"sk-x"}')

    assert "model gpt-5 is not enabled" in redacted


def test_ssrf_message_drops_the_presign_signature() -> None:
    url = (
        "https://minio.example.com:9000/tenant-a/master.mp4"
        "?X-Amz-Signature=deadbeefcafebabe&X-Amz-Expires=3600"  # pragma: allowlist secret
    )

    msg = _safe_url_for_message(url)

    assert "deadbeefcafebabe" not in msg
    # Still actionable: the operator needs the host and path to diagnose.
    assert "minio.example.com:9000" in msg
    assert "/tenant-a/master.mp4" in msg


def test_ssrf_message_drops_userinfo() -> None:
    url = "http://admin:hunter2@169.254.169.254/latest/meta-data/"  # pragma: allowlist secret
    msg = _safe_url_for_message(url)

    assert "hunter2" not in msg
    assert "admin" not in msg
    assert "169.254.169.254" in msg


def test_ssrf_message_survives_an_unparseable_url() -> None:
    """A guard that crashes while refusing a URL fails OPEN in the caller's face."""
    assert _safe_url_for_message("http://[::1") == "<unparseable url>"
