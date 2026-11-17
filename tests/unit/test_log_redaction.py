"""Log-redaction helper for error-body logging in ``BaseServiceClient``.

Upstream services occasionally echo the bearer/license token they rejected
in their error body. We log up to 1000 chars of the body to aid debugging,
which captures those secrets in log aggregators. ``_redact_for_log`` masks
the common patterns before logging.
"""

from __future__ import annotations

import pytest

from w2v_common.clients.base import _redact_for_log


class TestRedactBearer:
    @pytest.mark.parametrize(
        "raw, must_not_contain",
        [
            ("Bearer abc123xyz", "abc123xyz"),
            ('error="Bearer eyJhbGc.payload.sig"', "eyJhbGc.payload.sig"),
            ("bearer lowercase-ok", "lowercase-ok"),
            ("BEARER UPPERCASE", "UPPERCASE"),
        ],
    )
    def test_bearer_token_redacted(self, raw: str, must_not_contain: str) -> None:
        out = _redact_for_log(raw)
        assert must_not_contain not in out
        assert "[REDACTED]" in out


class TestRedactTokenParam:
    @pytest.mark.parametrize(
        "raw, must_not_contain",
        [
            ("token=abc.def.ghi", "abc.def.ghi"),
            ("?token=verysecret&x=1", "verysecret"),
            ('"token":"abc123"', "abc123"),
        ],
    )
    def test_token_param_redacted(self, raw: str, must_not_contain: str) -> None:
        out = _redact_for_log(raw)
        assert must_not_contain not in out
        assert "[REDACTED]" in out


class TestRedactAuthorizationHeader:
    def test_authorization_header_redacted(self) -> None:
        raw = "Authorization: Bearer my-secret-token"
        out = _redact_for_log(raw)
        assert "my-secret-token" not in out
        assert "[REDACTED]" in out


class TestPassthrough:
    def test_no_secrets_no_change(self) -> None:
        raw = '{"error":"upstream timeout","status":504}'
        assert _redact_for_log(raw) == raw

    def test_empty(self) -> None:
        assert _redact_for_log("") == ""
