"""``WMClient.message()`` served by inference-router (``LLM_BACKEND=ir``, the default).

Owner decision 2026-10-20: w2v LLM traffic goes to inference-router (IR) directly;
worker_management is dropped for w2v. This backend reuses the vLLM-direct machinery
(:mod:`w2v_common.clients.vllm_direct`: seed prompts, xgrammar-stripped
``response_format``, Draft 2020-12 validation + re-ask, retry policy, error types)
and changes only what IR's contract requires:

* ``POST {LLM_IR_BASE_URL}/chat/completions`` — default
  ``service_url("inference_router") + "/v1"`` (so ``W2V_INFERENCE_ROUTER_URL``
  overrides it the discovery way). An explicit ``LLM_IR_BASE_URL`` must resolve to
  the host discovery knows as ``inference_router``: delegation is keyed on that
  host, and a URL outside it would silently bill the service account instead of
  the tenant (refused at construction).
* auth ``X-Tenant-Id: Bearer <LLM_IR_TOKEN>`` — a licence whose app is
  ``inference-router`` (an app-``w2v`` ``DOWNSTREAM_LICENSE_TOKEN`` is refused by
  IR). ``LLM_IR_TOKEN_FILE`` names a file holding it instead; the env var wins.
* ``identity_event_hook`` on the pooled client: inference_router is a METERED,
  DELEGATING destination (``auth.outbound.METERED_DESTINATION_SERVICES``), so a
  bound + VOUCHED tenant goes out as ``X-Effective-Tenant-Id``, an unvouched one
  fails closed (``UnvouchedTenantError``) before anything is sent, and the subject,
  licence id, correlation id (= run id), stage, actor and w2v job/project ride
  along. Provenance follows the repo rule — nothing is threaded by hand.
* attribution headers per call: ``X-App-Id`` (licences registry key, ``w2v``),
  ``X-Worker-Name`` (the worker), ``X-Job-Id`` (the w2v run, when bound — omitted
  otherwise, IR's omit-never-forge rule), ``X-Project-Id`` (set by ``WMClient``).
* ``enable_reasoning`` as a TOP-LEVEL bool; ``chat_template_kwargs`` is never sent
  (IR ignores it). ``max_tokens`` clamped to IR's 16384 ceiling (it 422s above).

IR meters the call itself (cost event → billing → the orchestrator's run mirror),
so this client emits nothing to the cost bus and, by default, sends no run-usage
report (see :mod:`w2v_common.clients.vllm_usage_report`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx

from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.auth.outbound import is_metered_destination
from w2v_common.clients.vllm_direct import (
    BACKEND_IR,
    DEFAULT_MODEL,
    VLLM_MAX_OUTPUT_TOKENS,
    JsonObject,
    LLMBackend,
    VLLMDirectClient,
)
from w2v_common.clients.vllm_usage_report import PROVIDER_IR
from w2v_common.discovery import service_url
from w2v_common.observability.request_context import trace_event_hook

logger = logging.getLogger(__name__)

SERVICE_NAME: Final = "inference_router"
#: The licences app an IR caller's licence must belong to.
IR_APP_KEY: Final = "inference-router"
#: IR's ``max_tokens`` ceiling (``ChatCompletionRequest.max_tokens le=16384``).
IR_MAX_OUTPUT_TOKENS: Final = VLLM_MAX_OUTPUT_TOKENS
WORKER_NAME_HEADER: Final = "X-Worker-Name"

BASE_URL_ENV: Final = "LLM_IR_BASE_URL"
TOKEN_ENV: Final = "LLM_IR_TOKEN"  # noqa: S105 - an env var NAME, not a secret
TOKEN_FILE_ENV: Final = "LLM_IR_TOKEN_FILE"  # noqa: S105 - an env var NAME
MODEL_ENV: Final = "LLM_IR_MODEL_DEFAULT"


class IRNotConfiguredError(RuntimeError):
    """``LLM_BACKEND`` defaulted to ``ir`` but no IR licence is configured.

    Raised from the first call (and reported by ``health_check``), not from
    construction: a defaulted backend must not crash every consumer's test suite
    that builds a ``WMClient`` without LLM env. An EXPLICIT ``LLM_BACKEND=ir``
    without a token still fails construction.
    """


def default_base_url() -> str:
    """``service_url("inference_router")`` + ``/v1`` (honours ``W2V_INFERENCE_ROUTER_URL``)."""
    return service_url("inference_router").rstrip("/") + "/v1"


def _read_token(env: Mapping[str, str]) -> str:
    token = (env.get(TOKEN_ENV) or "").strip()
    if token:
        return token
    path = (env.get(TOKEN_FILE_ENV) or "").strip()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        # The path is config, not a secret; the content never reaches a message.
        raise ValueError(
            f"{TOKEN_FILE_ENV}={path!r} is not readable: {type(exc).__name__}"
        ) from None


@dataclass(frozen=True, slots=True)
class IRSettings:
    """``LLM_IR_*`` env. ``token`` must be a licence for app ``inference-router``."""

    base_url: str
    token: str = field(repr=False)
    model_default: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError(
                f"LLM_BACKEND=ir requires {TOKEN_ENV} (or {TOKEN_FILE_ENV}): a licence "
                f"token for app `{IR_APP_KEY}`"
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"{BASE_URL_ENV} must be an http(s) URL")
        if not is_metered_destination(self.base_url):
            raise ValueError(
                f"{BASE_URL_ENV}={self.base_url!r} is not the inference_router host "
                f"discovery resolves ({default_base_url()!r}); tenant delegation keys on "
                "that host, so this URL would bill the service account. Point "
                "W2V_INFERENCE_ROUTER_URL at it instead."
            )
        if not self.model_default.strip():
            raise ValueError(f"{MODEL_ENV} must not be empty")

    @classmethod
    def from_env(cls) -> IRSettings:
        env = os.environ
        return cls(
            base_url=(env.get(BASE_URL_ENV) or "").strip() or default_base_url(),
            token=_read_token(env),
            model_default=env.get(MODEL_ENV, DEFAULT_MODEL).strip(),
        )

    def model_for(self, worker_name: str) -> str:
        """IR routes by model; every w2v worker asks for the same served model."""
        return self.model_default


class IRDirectClient(VLLMDirectClient):
    """Pooled httpx client for inference-router, owned by one ``WMClient``."""

    service_name: str = SERVICE_NAME
    backend: LLMBackend = BACKEND_IR
    auth_hint: str = (
        f"{TOKEN_ENV} must be a licence for app `{IR_APP_KEY}`, and a delegated tenant "
        "needs this licence's subject in inference-router's AUTH_DELEGATING_SUBJECTS"
    )

    def _build_client(self, transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
        """IR's credential shape: ``X-Tenant-Id: Bearer`` + the S2S identity hooks."""
        return httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
            follow_redirects=False,
            headers={"X-Tenant-Id": f"Bearer {self.settings.token}"},
            event_hooks={"request": [identity_event_hook, trace_event_hook]},
        )

    def _usage_labels(self, model: str) -> JsonObject:
        """IR meters this call itself: no ``funding`` claim here, only who did."""
        return {
            "served_provider": PROVIDER_IR,
            "provider": PROVIDER_IR,
            "metered_by": PROVIDER_IR,
        }


__all__ = [
    "IR_APP_KEY",
    "IR_MAX_OUTPUT_TOKENS",
    "WORKER_NAME_HEADER",
    "IRDirectClient",
    "IRNotConfiguredError",
    "IRSettings",
    "default_base_url",
]
