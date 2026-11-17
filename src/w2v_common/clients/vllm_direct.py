"""``WMClient.message()`` served straight by the pgx vLLM (``LLM_BACKEND=vllm``).

Since 0.109.0 the DEFAULT backend is ``ir`` (:mod:`w2v_common.clients.ir_direct`,
owner decision 2026-10-20: w2v LLM traffic goes to inference-router directly and
worker_management is dropped). ``vllm`` stays as the rollback path. The OpenAI-
compatible body, schema validation, re-ask and retry machinery below are shared by
both backends; :class:`VLLMDirectClient` is the base the IR client extends.

Owner decision 2026-10-17: the WM → inference-router hop mints a per-tenant
``workers`` bundle token on every call, and licences answering 409
``ambiguous_bundle`` took every w2v LLM stage down. This backend makes ONE
OpenAI-compatible Chat Completions call to the vLLM with a static ``vllm`` licence
(``LLM_VLLM_TOKEN``) instead. It is a lift of the orchestrator's
``src/services/llm/vllm_direct.py`` (w2v_orchestrator #1115) so the seven services
that construct :class:`~w2v_common.clients.WMClient` themselves get the same
behaviour with zero code change (the ``wm`` backend it was the alternative to is gone).

What it keeps from the WM path, because callers depend on it:

* the worker's SEED prompt + temperature (``vllm_worker_defaults.yaml``, vendored
  from WM's ``seed/w2v_workers.yaml``), then the per-call ``system_prompt`` as a
  system message — the order WM applies;
* ``WM_MIN_OUTPUT_TOKENS`` as a floor, clamped to WM's 16384 ceiling for this vLLM;
* ``chat_template_kwargs.enable_thinking = bool(enable_reasoning)`` — what
  inference-router forwarded (thinking OFF unless a caller opts in);
* structured output as vLLM guided decoding (``response_format=json_schema``) over
  the xgrammar-STRIPPED schema, validated (Draft 2020-12) against that same stripped
  schema and re-asked with the violations (``OUTPUT_RETRIES``). ``prefixItems``
  survives the strip, so ``non_empty_array`` stays a grammar-level guarantee.
  Count/length bounds the grammar cannot carry are checked against the ORIGINAL
  schema and reported in ``usage_metrics.bound_violations``, never fatal;
* the WM path's HTTP retry policy (``BaseServiceClient._request``'s constants), and
  its ERROR TYPES: a non-2xx after retries is ``httpx.HTTPStatusError`` built by
  :func:`~w2v_common.clients.base.raise_for_status_with_body`; an exhausted schema
  re-ask or a ``finish_reason == "length"`` reply is an ``HTTPStatusError`` over a
  synthetic 422 (WM's verdict for both); a network failure is ``httpx.TransportError``.

The token is never part of an error, a response body we hand back, or a log line.
No metering is emitted (billing is out of this lane): usage is returned in
``usage_metrics`` with ``funding="platform_direct"``. Usage ATTRIBUTION is: after a
successful call inside a w2v run context, a fire-and-forget report goes to the
orchestrator's ``POST /api/runs/{run_id}/usage`` (:mod:`w2v_common.clients.vllm_usage_report`,
switch ``LLM_VLLM_USAGE_REPORT_ENABLED``; never sent from the orchestrator itself).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import httpx
import yaml
from jsonschema import Draft202012Validator

from w2v_common.clients import base as _wm_retry
from w2v_common.clients._xgrammar_schema import BOUND_KEYWORDS, strip_xgrammar_unsupported
from w2v_common.clients.vllm_usage_report import new_call_id

if TYPE_CHECKING:
    from w2v_common.clients.vllm_usage_report import VLLMUsageReporter

logger = logging.getLogger(__name__)

LLMBackend = Literal["ir", "vllm"]
BACKEND_IR: Final = "ir"
BACKEND_VLLM: Final = "vllm"
#: ``LLM_BACKEND`` when unset (0.109.0; was ``wm``, which no longer exists).
DEFAULT_BACKEND: Final = BACKEND_IR
SERVICE_NAME: Final = "vllm"

DEFAULT_BASE_URL: Final = "http://localhost:8001/v1"
DEFAULT_MODEL: Final = "local-llm"

#: Re-asks after a reply that is not valid JSON / violates the schema (WM: 3).
OUTPUT_RETRIES: Final = 3
#: WM's ``max_tokens`` ceiling (``workers/models.py``) protecting the self-hosted
#: vLLM from the FP4-MoE kernel crash near the full context (incident 2026-05-13).
VLLM_MAX_OUTPUT_TOKENS: Final = 16_384
#: WM's auto-create default (``message_endpoints.py``: temperature=0.1, no prompt).
AUTO_CREATE_TEMPERATURE: Final = 0.1
#: Key a non-object root is wrapped under, so the grammar can carry it.
_WRAP_KEY: Final = "response"
_AUTH_HTTP: Final = frozenset({401, 403})
#: Request headers that carry a credential and must never ride an exception.
_CREDENTIAL_HEADERS: Final = frozenset({"authorization", "x-tenant-id"})
_DEFAULTS_PATH: Final = Path(__file__).with_name("vllm_worker_defaults.yaml")
_FRAMING_HEADERS: Final = frozenset({"content-length", "content-encoding", "transfer-encoding"})

#: Anything key-shaped in an upstream body (``sk-…``, ``Bearer …``).
_KEYISH: Final = re.compile(r"(?i)(bearer\s+\S+|\b(?:sk|pk|rk|key)-[A-Za-z0-9_\-*.]{4,})")
_MAX_DETAIL: Final = 300

#: Patched by tests; every retry sleep goes through here.
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

JsonObject = dict[str, object]


# ── configuration ───────────────────────────────────────────────────────────


def llm_backend_is_explicit(value: str | None = None) -> bool:
    """True when the backend was CHOSEN (kwarg or a non-blank ``LLM_BACKEND``)."""
    raw = value if value is not None else os.environ.get("LLM_BACKEND", "")
    return bool(str(raw).strip())


def resolve_llm_backend(value: str | None = None) -> LLMBackend:
    """``value`` if given, else env ``LLM_BACKEND``, else ``ir``. Unknown → ``ValueError``.

    A typo must not silently mean the default: the operator would believe the flip
    happened while every call kept going to the old backend. ``wm`` is refused by
    name: the worker_management backend was deleted in 0.109.0.
    """
    raw = value if value is not None else os.environ.get("LLM_BACKEND", "")
    raw = str(raw).strip().lower() or DEFAULT_BACKEND
    if raw == BACKEND_IR:
        return BACKEND_IR
    if raw == BACKEND_VLLM:
        return BACKEND_VLLM
    if raw == "wm":
        raise ValueError(
            "LLM_BACKEND=wm was removed in w2v_common 0.109.0 (owner decision 2026-10-20: "
            "w2v LLM traffic goes to inference-router directly); use ir, or vllm to roll back"
        )
    raise ValueError(f"LLM_BACKEND={raw!r} is not one of: {BACKEND_IR}, {BACKEND_VLLM}")


def _parse_fast_workers(value: str | None) -> frozenset[str]:
    """Comma-separated worker names (the orchestrator's ``LLM_VLLM_FAST_WORKERS``)."""
    if not value:
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class VLLMSettings:
    """``LLM_VLLM_*`` env, named and validated exactly as the orchestrator's.

    ``token`` must be a licence for app ``vllm``: both vLLM front doors bind it
    (the configured vLLM host is a license-authz STRICT_APP_HOST) — a w2v
    ``DOWNSTREAM_LICENSE_TOKEN`` is an app ``w2v`` licence and gets 403.
    """

    base_url: str
    token: str = field(repr=False)
    model_default: str = DEFAULT_MODEL
    model_fast: str = ""
    fast_workers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("LLM_BACKEND=vllm requires LLM_VLLM_TOKEN (a `vllm` licence token)")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_VLLM_BASE_URL must be an http(s) URL")
        if not self.model_default.strip():
            raise ValueError("LLM_VLLM_MODEL_DEFAULT must not be empty")

    @classmethod
    def from_env(cls) -> VLLMSettings:
        env = os.environ
        return cls(
            base_url=env.get("LLM_VLLM_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            token=env.get("LLM_VLLM_TOKEN", "").strip(),
            model_default=env.get("LLM_VLLM_MODEL_DEFAULT", DEFAULT_MODEL).strip(),
            model_fast=env.get("LLM_VLLM_MODEL_FAST", "").strip(),
            fast_workers=_parse_fast_workers(env.get("LLM_VLLM_FAST_WORKERS")),
        )

    def model_for(self, worker_name: str) -> str:
        """The served model for ``worker_name``: fast when listed and configured."""
        if self.model_fast and worker_name in self.fast_workers:
            return self.model_fast
        return self.model_default


# ── worker defaults (WM seed) ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WorkerDefaults:
    worker_name: str
    temperature: float = AUTO_CREATE_TEMPERATURE
    system_prompt: str | None = None
    #: False for names WM never seeded (it auto-created them with no prompt).
    seeded: bool = False


def _parse_defaults(raw: object) -> dict[str, WorkerDefaults]:
    name_of_file = _DEFAULTS_PATH.name
    if not isinstance(raw, dict):
        raise ValueError(f"{name_of_file}: expected a mapping with a 'workers' list")
    workers = cast("dict[str, object]", raw).get("workers")
    if not isinstance(workers, list):
        raise ValueError(f"{name_of_file}: expected a top-level 'workers' list")
    out: dict[str, WorkerDefaults] = {}
    for item in cast("list[object]", workers):
        entry = cast("dict[str, object]", item) if isinstance(item, dict) else None
        name = entry.get("worker_name") if entry is not None else None
        if entry is None or not isinstance(name, str):
            raise ValueError(f"{name_of_file}: every worker needs a worker_name")
        if name in out:
            raise ValueError(f"{name_of_file}: duplicate worker {name!r}")
        temperature = entry.get("temperature", AUTO_CREATE_TEMPERATURE)
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ValueError(f"{name_of_file}: {name}: temperature must be a number")
        prompt = entry.get("system_prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"{name_of_file}: {name}: system_prompt must be a string")
        out[name] = WorkerDefaults(
            worker_name=name,
            temperature=float(temperature),
            system_prompt=prompt or None,
            seeded=bool(entry.get("seeded", False)),
        )
    return out


@lru_cache(maxsize=1)
def _load_defaults() -> dict[str, WorkerDefaults]:
    return _parse_defaults(yaml.safe_load(_DEFAULTS_PATH.read_text(encoding="utf-8")))


def known_workers() -> frozenset[str]:
    return frozenset(_load_defaults())


def get_worker_defaults(worker_name: str) -> WorkerDefaults:
    """Seeded defaults for ``worker_name``; WM's auto-create defaults otherwise."""
    found = _load_defaults().get(worker_name)
    if found is not None:
        return found
    logger.debug(
        "vllm backend: worker %r has no vendored seed; using WM auto-create defaults "
        "(temperature=%s, no seed prompt)",
        worker_name,
        AUTO_CREATE_TEMPERATURE,
    )
    return WorkerDefaults(worker_name=worker_name)


# ── output contract ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OutputContract:
    """What the decoder is told, and how to read its answer back."""

    grammar: JsonObject | None  # the response_format schema; None ⇒ plain text
    wrapped: bool  # root was not an object: the answer lives under ``_WRAP_KEY``


def output_contract(schema: Mapping[str, object] | None) -> OutputContract:
    """Decoder-visible schema for ``schema`` (xgrammar-stripped, object-rooted)."""
    if not schema:
        return OutputContract(grammar=None, wrapped=False)
    stripped = cast("JsonObject", strip_xgrammar_unsupported(dict(schema)))
    if schema.get("type") == "object" or (
        "type" not in schema and isinstance(schema.get("properties"), Mapping)
    ):
        return OutputContract(grammar=stripped, wrapped=False)
    return OutputContract(
        grammar={
            "type": "object",
            "properties": {_WRAP_KEY: stripped},
            "required": [_WRAP_KEY],
            "additionalProperties": False,
        },
        wrapped=True,
    )


def contract_violations(grammar: Mapping[str, object], output: object) -> list[str]:
    """Draft 2020-12 errors of ``output`` against the DECODER-visible schema."""
    validator = Draft202012Validator(dict(grammar))
    problems: list[str] = []
    for err in sorted(validator.iter_errors(output), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        problems.append(f"{where}: {err.message}")
    return problems[:10]


def bound_violations(schema: Mapping[str, object] | None, output: object) -> list[dict[str, str]]:
    """Advisory: count/length bounds of the ORIGINAL schema the grammar could not carry."""
    if not schema:
        return []
    out: list[dict[str, str]] = []
    for err in Draft202012Validator(dict(schema)).iter_errors(output):
        if err.validator in BOUND_KEYWORDS:
            out.append(
                {
                    "path": "/" + "/".join(str(p) for p in err.absolute_path),
                    "keyword": str(err.validator),
                    "message": err.message[:200],
                }
            )
    return out


def output_type_label(output: object, schema: Mapping[str, object] | None) -> str:
    """WM's ``output_type`` derivation (``message_endpoints.py``)."""
    if isinstance(output, bool):
        return "boolean"
    if isinstance(output, str):
        return "enum" if schema and "enum" in schema else "string"
    if isinstance(output, dict):
        return "object"
    if isinstance(output, list):
        return "array"
    return "other"


# ── request body ────────────────────────────────────────────────────────────


def user_content(
    message: str | None, content_blocks: Sequence[Mapping[str, object]] | None
) -> str | list[JsonObject]:
    """``message`` as-is, or ``content_blocks`` as OpenAI chat content parts."""
    if message is not None and content_blocks is not None:
        raise ValueError("Pass exactly one of 'message' or 'content_blocks', not both")
    if message is not None:
        return message
    if content_blocks is None:
        raise ValueError("Either 'message' or 'content_blocks' must be provided")
    parts: list[JsonObject] = []
    for block in content_blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            media_type = str(block.get("media_type") or "image/jpeg")
            data = str(block.get("data", ""))
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
            )
        else:
            raise ValueError(f"unsupported content block type {kind!r}")
    return parts


def build_body(
    *,
    worker_name: str,
    message: str | None,
    content_blocks: Sequence[Mapping[str, object]] | None,
    system_prompt: str | None,
    max_tokens: int,
    enable_reasoning: bool | None,
    model: str,
    contract: OutputContract,
    backend: LLMBackend = BACKEND_VLLM,
) -> JsonObject:
    """The Chat Completions body for the first attempt (re-asks append messages).

    ``backend`` decides how ``enable_reasoning`` is spelled. The vLLM takes it as
    ``chat_template_kwargs.enable_thinking``; inference-router takes a top-level
    ``enable_reasoning`` bool and IGNORES ``chat_template_kwargs`` — consumers
    implement IR's contract, never the reverse. Either way thinking is OFF unless
    the caller opts in.
    """
    defaults = get_worker_defaults(worker_name)
    messages: list[JsonObject] = []
    instructions = [p for p in (defaults.system_prompt, system_prompt) if p]
    if instructions:
        messages.append({"role": "system", "content": "\n\n".join(instructions)})
    messages.append({"role": "user", "content": user_content(message, content_blocks)})
    body: JsonObject = {
        "model": model,
        "messages": messages,
        "temperature": defaults.temperature,
        "max_tokens": min(max_tokens, VLLM_MAX_OUTPUT_TOKENS),
    }
    if backend == BACKEND_IR:
        body["enable_reasoning"] = bool(enable_reasoning)
    else:
        body["chat_template_kwargs"] = {"enable_thinking": bool(enable_reasoning)}
    if contract.grammar is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "OutputData", "schema": contract.grammar, "strict": False},
        }
    return body


# ── errors ──────────────────────────────────────────────────────────────────


def sanitise(text: str, secret: str | None) -> str:
    """Upstream text safe for users and logs: no token, nothing key-shaped, bounded."""
    if secret:
        text = text.replace(secret, "<redacted>")
    text = _KEYISH.sub("<redacted>", text)
    text = " ".join(text.split())
    return text[:_MAX_DETAIL] + ("…" if len(text) > _MAX_DETAIL else "")


def _bare_request(request: httpx.Request) -> httpx.Request:
    """``request`` without its credential headers, for an exception to carry.

    ``Authorization`` (the vLLM bearer) and ``X-Tenant-Id`` (inference-router's
    ``Bearer <licence>``) are both dropped.
    """
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _CREDENTIAL_HEADERS}
    return httpx.Request(request.method, request.url, headers=headers)


def _raise_upstream_status(
    response: httpx.Response,
    token: str,
    service_name: str = SERVICE_NAME,
    auth_hint: str = "LLM_VLLM_TOKEN must be a licence for app `vllm`",
) -> None:
    """``raise_for_status_with_body`` over a token-free copy of ``response``.

    The copy keeps status + headers; its body and request carry no token, so
    neither ``str(exc)`` nor ``exc.response.text`` / ``exc.request.headers`` can
    leak it into a consumer's log line. A 401/403 additionally logs WHY (the
    upstream's reason) next to what the credential must be.
    """
    status = response.status_code
    if status in _AUTH_HTTP:
        logger.error(
            "%s rejected the call (HTTP %d): %s; %s",
            service_name,
            status,
            sanitise(response.text, token),
            auth_hint,
        )
    safe = httpx.Response(
        status,
        # The body below is already DECODED text: drop the framing headers that
        # describe the wire bytes, or httpx would try to gunzip plain text.
        headers={k: v for k, v in response.headers.items() if k.lower() not in _FRAMING_HEADERS},
        # An empty token (a keyless tenant provider) must not be "replaced": ``str.replace("")``
        # would insert the marker between every character.
        content=(response.text.replace(token, "<redacted>") if token else response.text).encode(),
        request=_bare_request(response.request),
    )
    _wm_retry.raise_for_status_with_body(safe, service_name)


def _synthetic_status(
    status: int, url: str, detail: str, service_name: str = SERVICE_NAME
) -> httpx.HTTPStatusError:
    """An unacceptable reply as the ``HTTPStatusError`` the WM path would raise."""
    request = httpx.Request("POST", url)
    response = httpx.Response(status, json={"detail": detail}, request=request)
    logger.error("%s returned %d: %s", service_name, status, detail)
    return httpx.HTTPStatusError(
        f"{service_name} returned {status}: {detail}", request=request, response=response
    )


def _synthetic_422(
    url: str, detail: str, service_name: str = SERVICE_NAME
) -> httpx.HTTPStatusError:
    """WM's verdict (422) for a reply that cannot be accepted, as an HTTPStatusError."""
    return _synthetic_status(422, url, detail, service_name)


# ── transport ───────────────────────────────────────────────────────────────


def _retry_sleep_seconds(response: httpx.Response, attempt: int) -> float:
    """``BaseServiceClient``'s backoff: linear per status, a 429's Retry-After may only lengthen it."""
    _, base_delay = _wm_retry._retry_budget(response.status_code)
    default_delay = base_delay * attempt
    if response.status_code in _wm_retry._RETRY_AFTER_STATUS_CODES:
        honoured = _wm_retry._parse_retry_after(response.headers.get("Retry-After"))
        if honoured is not None:
            return max(honoured, default_delay)
    return default_delay


class RetryTransport(httpx.AsyncBaseTransport):
    """The WM path's retry policy (``BaseServiceClient._request``) for the vLLM call.

    429 up to 6 attempts (linear 5s, ``Retry-After`` honoured, capped), 500/502/503/504
    up to 3 attempts (linear 1s), connect/timeout errors up to 3 attempts (linear 1s),
    all inside the 75s cumulative sleep budget. The constants are read at call time
    from :mod:`w2v_common.clients.base`, so the two paths cannot drift. Any other
    status (4xx auth/validation included) is returned untouched.

    ``retry_read_timeout=False`` (a TENANT's paid provider, ``w2v_common.providers``)
    re-sends only ``ConnectError``/``ConnectTimeout`` — the request provably never
    reached the provider. A read/write timeout may mean the POST is already
    executing (and billing) upstream, so it propagates on the first failure.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, *, retry_read_timeout: bool = True) -> None:
        self._inner = inner
        self._retryable_errors: tuple[type[Exception], ...] = (
            (httpx.TimeoutException, httpx.ConnectError)
            if retry_read_timeout
            else (httpx.ConnectError, httpx.ConnectTimeout)
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        slept = 0.0
        attempt = 1
        while True:
            try:
                response = await self._inner.handle_async_request(request)
            except self._retryable_errors as exc:
                if attempt >= _wm_retry._MAX_RETRIES:
                    raise
                delay = _wm_retry._RETRY_DELAY * attempt
                slept += delay
                logger.warning(
                    "vllm transport error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    _wm_retry._MAX_RETRIES,
                    delay,
                    type(exc).__name__,
                )
                await _sleep(delay)
                attempt += 1
                continue
            status = response.status_code
            max_attempts, _ = _wm_retry._retry_budget(status)
            if status not in _wm_retry._RETRYABLE_STATUS_CODES or attempt >= max_attempts:
                return response
            delay = _retry_sleep_seconds(response, attempt)
            if slept + delay > _wm_retry._RETRY_SLEEP_BUDGET:
                return response
            slept += delay
            await response.aclose()
            logger.warning(
                "vllm returned %d (attempt %d/%d), retrying in %.1fs",
                status,
                attempt,
                max_attempts,
                delay,
            )
            await _sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        await self._inner.aclose()


# ── the call ────────────────────────────────────────────────────────────────


def _parse_reply(content: str, contract: OutputContract) -> tuple[object, list[str]]:
    """``(output, problems)`` for one completion's ``content``."""
    if contract.grammar is None:
        return content, []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"reply is not valid JSON ({exc.msg})"]
    problems = contract_violations(contract.grammar, parsed)
    if problems:
        return None, problems
    if contract.wrapped:
        return cast("JsonObject", parsed)[_WRAP_KEY], []
    return parsed, []


def _reask(content: str, problems: list[str]) -> list[JsonObject]:
    """The assistant's rejected reply plus the violations, for the next attempt."""
    return [
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": "Your reply does not match the required JSON schema: "
            + "; ".join(problems)
            + ". Reply again with ONLY JSON that matches the schema.",
        },
    ]


class DirectSettings(Protocol):
    """What :class:`VLLMDirectClient` reads from its settings (vLLM or IR)."""

    @property
    def base_url(self) -> str: ...

    @property
    def token(self) -> str: ...

    def model_for(self, worker_name: str) -> str: ...


class VLLMDirectClient:
    """Pooled httpx client for the vLLM, owned by one :class:`WMClient`.

    The base of :class:`~w2v_common.clients.ir_direct.IRDirectClient`: a subclass
    changes the credential shape (:meth:`_build_client`), the body's reasoning
    field (:attr:`backend`) and the labels — never the call/validate/re-ask loop.
    """

    #: Label in errors, logs and ``usage_metrics``.
    service_name: str = SERVICE_NAME
    #: Which body dialect :func:`build_body` speaks.
    backend: LLMBackend = BACKEND_VLLM
    #: What a 401/403 means for THIS credential, appended to the upstream's reason.
    auth_hint: str = "LLM_VLLM_TOKEN must be a licence for app `vllm`"
    #: Whether schema-violation messages (which quote the MODEL's reply) may appear
    #: in WARNING logs and the final 422. ``False`` for a TENANT's provider: its
    #: response content is the tenant's data and is never echoed.
    echo_reply_problems: bool = True

    def __init__(
        self,
        settings: DirectSettings,
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        usage_reporter: VLLMUsageReporter | None = None,
    ) -> None:
        self.settings = settings
        self.timeout = timeout
        self._inner_transport = transport
        self._client: httpx.AsyncClient | None = None
        #: Optional run-usage reporter; ``None`` = no report (never affects the call).
        self.usage_reporter = usage_reporter

    @property
    def _completions_url(self) -> str:
        return self.settings.base_url.rstrip("/") + "/chat/completions"

    def _build_client(self, transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
        """The pooled client: the static licence as the ONLY credential."""
        return httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
            follow_redirects=False,
            # Nothing identity-shaped: the static licence is the ONLY credential.
            headers={"Authorization": f"Bearer {self.settings.token}"},
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            inner = self._inner_transport or httpx.AsyncHTTPTransport()
            self._client = self._build_client(RetryTransport(inner))
        return self._client

    def _usage_labels(self, model: str) -> JsonObject:
        """Provider/funding keys of ``usage_metrics`` (the vLLM: platform GPU, unmetered)."""
        return {
            "served_provider": self.backend,
            "provider": self.backend,
            "funding": "platform_direct",
        }

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if self.usage_reporter is not None:
            await self.usage_reporter.close()

    async def health_check(self) -> JsonObject:
        """``GET {base}/models`` with the credential; reports, never raises."""
        url = self.settings.base_url.rstrip("/") + "/models"
        token = self.settings.token
        try:
            response = await self._get_client().get(url)
            if response.status_code >= 400:
                return {
                    "status": "unhealthy",
                    "service": self.service_name,
                    "error": f"HTTP {response.status_code}: " + sanitise(response.text, token),
                }
            data = response.json()
            raw_models = data.get("data") if isinstance(data, dict) else None
            models = [
                str(cast("JsonObject", m).get("id"))
                for m in cast("list[object]", raw_models or [])
                if isinstance(m, dict)
            ]
            return {"status": "healthy", "service": self.service_name, "models": models}
        except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises
            return {
                "status": "unhealthy",
                "service": self.service_name,
                "error": f"{type(exc).__name__}: {sanitise(str(exc), token)}",
            }

    async def _complete_once(
        self, body: JsonObject, headers: Mapping[str, str] | None = None
    ) -> tuple[str, int, int]:
        """One Chat Completions round trip: ``(content, prompt_tokens, completion_tokens)``."""
        token = self.settings.token
        try:
            response = await self._get_client().post(
                self._completions_url, json=body, headers=dict(headers or {})
            )
        except httpx.TransportError as exc:
            # Same type, but carrying a token-free request: httpx re-attaches the
            # original request, whose credential header is the licence.
            raise type(exc)(sanitise(str(exc), token), request=_bare_request(exc.request)) from None
        if response.status_code != 200:
            _raise_upstream_status(response, token, self.service_name, self.auth_hint)
        try:
            data = cast("JsonObject", response.json())
        except ValueError:
            raise _synthetic_status(
                502,
                self._completions_url,
                f"{self.service_name} returned a 200 whose body is not JSON",
                self.service_name,
            ) from None
        usage = cast("JsonObject", data.get("usage") or {})
        choices = cast("list[JsonObject]", data.get("choices") or [{}])
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise _synthetic_422(
                self._completions_url,
                f"Model token limit ({body['max_tokens']}) reached before the reply finished "
                "(finish_reason=length)",
                self.service_name,
            )
        message = cast("JsonObject", choice.get("message") or {})
        content = str(message.get("content") or "")
        return (
            content,
            int(cast("int", usage.get("prompt_tokens") or 0)),
            int(cast("int", usage.get("completion_tokens") or 0)),
        )

    async def message(
        self,
        worker_name: str,
        *,
        message: str | None,
        content_blocks: Sequence[Mapping[str, object]] | None,
        system_prompt: str | None,
        output_schema: Mapping[str, object] | None,
        max_tokens: int,
        enable_reasoning: bool | None,
        context: Mapping[str, object] | None,
        overrides_ignored: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
    ) -> JsonObject:
        """Run one call; return WM's ``MessageResponse`` payload shape.

        ``headers`` are sent on every attempt of this call (re-asks included).
        """
        model = self.settings.model_for(worker_name)
        schema = dict(output_schema) if output_schema else None
        contract = output_contract(schema)
        body = build_body(
            worker_name=worker_name,
            message=message,
            content_blocks=content_blocks,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
            model=model,
            contract=contract,
            backend=self.backend,
        )
        in_tok = out_tok = requests = 0
        problems: list[str] = []
        output: object = None
        for attempt in range(OUTPUT_RETRIES + 1):
            content, used_in, used_out = await self._complete_once(body, headers)
            requests += 1
            in_tok += used_in
            out_tok += used_out
            output, problems = _parse_reply(content, contract)
            if not problems:
                break
            logger.warning(
                "%s reply violates the schema worker=%s attempt=%d/%d: %s",
                self.service_name,
                worker_name,
                attempt + 1,
                OUTPUT_RETRIES + 1,
                "; ".join(problems)[:500]
                if self.echo_reply_problems
                else f"{len(problems)} problem(s), content withheld",
            )
            # A fresh list each attempt: the transport serialised the previous one.
            body["messages"] = [*cast("list[object]", body["messages"]), *_reask(content, problems)]
        if problems:
            raise _synthetic_422(
                self._completions_url,
                f"Exceeded maximum output retries ({OUTPUT_RETRIES}) for worker "
                f"{worker_name!r}: "
                + (
                    "; ".join(problems)[:300]
                    if self.echo_reply_problems
                    else f"{len(problems)} schema problem(s), content withheld"
                ),
                self.service_name,
            )

        usage_metrics: JsonObject = {
            # WM's ``TokenUsageMetrics`` key names — what every WMResponse reader
            # uses (content_service llm_retry, html_render's near-limit check).
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "served_model": model,
            "requested_model": model,
            # Kept alongside: same values, the names this backend shipped with.
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "requests": requests,
            "model": model,
            **self._usage_labels(model),
        }
        bounds = bound_violations(schema, output)
        if bounds:
            usage_metrics["bound_violations"] = bounds
        metadata: JsonObject = {
            "source": "platform",
            "backend": self.backend,
            "served_model": model,
            "worker_name": worker_name,
            "seed_prompt_applied": get_worker_defaults(worker_name).system_prompt is not None,
        }
        if overrides_ignored:
            metadata["overrides_ignored"] = dict(overrides_ignored)
        if context is not None:
            metadata["context"] = dict(context)
        logger.info(
            "%s call worker=%s model=%s requests=%d input_tokens=%d output_tokens=%d",
            self.service_name,
            worker_name,
            model,
            requests,
            in_tok,
            out_tok,
        )
        if self.usage_reporter is not None:
            # One idempotency key per LLM call (no upstream request id is kept);
            # the reporter's single retry reuses it. Never raises.
            self.usage_reporter.schedule(
                call_id=new_call_id(),
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        return {
            "output": output,
            "output_type": output_type_label(output, schema),
            "output_schema_used": schema,
            "metadata": metadata,
            "usage_metrics": usage_metrics,
        }


__all__ = [
    "BACKEND_IR",
    "BACKEND_VLLM",
    "DEFAULT_BACKEND",
    "OUTPUT_RETRIES",
    "VLLM_MAX_OUTPUT_TOKENS",
    "DirectSettings",
    "LLMBackend",
    "RetryTransport",
    "VLLMDirectClient",
    "VLLMSettings",
    "WorkerDefaults",
    "bound_violations",
    "build_body",
    "contract_violations",
    "get_worker_defaults",
    "known_workers",
    "llm_backend_is_explicit",
    "output_contract",
    "resolve_llm_backend",
    "sanitise",
]
