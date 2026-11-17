"""0.94.0: a key whose name ENDS in a credential name is masked.

Up to 0.93.x the credential-name rule was ``\\b(access[_-]?key|...)``. ``_`` is a
word character, so ``\\b`` never fired inside ``browserstack_access_key`` /
``minio_access_key`` / ``aws_secret_access_key`` and their values were logged
verbatim — the orchestrator installed its own log filter
(``src/clients/app_inspector_client.py`` ``_DeviceFarmSecretLogFilter``) to
cover exactly that gap.
"""

from __future__ import annotations

import json

import pytest

from w2v_common.clients.base import _redact_for_log

_SECRET = "Zx9-fake-credential-value-Qw7"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "key",
    [
        "access_key",
        "accessKey",
        "minio_access_key",
        "aws_access_key",
        "secret_access_key",
        "aws_secret_access_key",
        "browserstack_access_key",
        "MINIO-ACCESS-KEY",
        "access_key_id",
        "AccessKeyId",
        "aws_access_key_id",
        "secret",
        "client_secret",
        "minio_secret",
        "webhook-secret",
        "db_password",
        "openai_api_key",
        "access_token",
        "accessToken",
        "auth_token",
        "id_token",
        "google_id_token",
        "bearer_token",
        # A prefix longer than the 64-char bound still gets its value masked.
        "x" * 80 + "_access_key",
    ],
)
@pytest.mark.parametrize("shape", ["json", "query"])
def test_every_key_form_is_masked(key: str, shape: str) -> None:
    body = json.dumps({key: _SECRET}) if shape == "json" else f"{key}={_SECRET}&next=1"

    redacted = _redact_for_log(body)

    assert _SECRET not in redacted, redacted
    assert "[REDACTED]" in redacted


def test_the_key_name_survives_so_the_log_stays_diagnosable() -> None:
    redacted = _redact_for_log(json.dumps({"minio_access_key": _SECRET}))

    assert "minio_access_key" in redacted


def test_nested_dicts_are_masked_at_every_depth() -> None:
    body = json.dumps(
        {
            "detail": "storage config rejected",
            "config": {
                "storage": {"minio_access_key": _SECRET + "A", "bucket": "media"},
                "device_farm": {"browserstack_access_key": _SECRET + "B", "user": "ci"},
                "aws": {"credentials": {"aws_secret_access_key": _SECRET + "C"}},
            },
        }
    )

    redacted = _redact_for_log(body)

    assert _SECRET not in redacted, redacted
    for kept in ("storage config rejected", '"bucket": "media"', '"user": "ci"'):
        assert kept in redacted


def test_pydantic_422_naming_a_suffixed_key_masks_the_input() -> None:
    """The orchestrator's actual leak: app_inspector 422s the device_farm body,
    and the credential sits under ``input`` with the key only in ``loc``."""
    body = json.dumps(
        {
            "detail": [
                {
                    "loc": ["body", "device_farm", "browserstack_access_key"],
                    "msg": "String should have at most 64 characters",
                    "input": _SECRET,
                }
            ]
        }
    )

    redacted = _redact_for_log(body)

    assert _SECRET not in redacted, redacted
    assert "String should have at most 64 characters" in redacted


@pytest.mark.parametrize(
    "body",
    [
        {"secret_name": "media-creds"},  # pragma: allowlist secret
        {"password_policy": "min-12"},  # pragma: allowlist secret
        {"access_key_rotation_days": "90"},
        {"max_tokens": "512"},
        {"token_count": "12"},
        {"tokens_used": "40"},
        {"bucket": "media"},
        {"keyword": "travel"},
    ],
)
def test_non_secret_keys_are_untouched(body: dict[str, str]) -> None:
    raw = json.dumps(body)

    assert _redact_for_log(raw) == raw


def test_a_non_credential_422_keeps_its_input() -> None:
    raw = json.dumps({"detail": [{"loc": ["body", "max_tokens"], "input": "lots"}]})

    assert _redact_for_log(raw) == raw
