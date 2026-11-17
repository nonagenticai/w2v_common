"""The w2v LLM client: ``WMClient.message()`` → inference-router (default) or the vLLM.

Name kept: ~10 consumer repos construct ``WMClient`` and the orchestrator subclasses
it, so a rename is a fleet fan-out with no behavioural payoff. No ``LLMClient``
alias either: bdd_playwright_service already has its own, different ``LLMClient``
shim, and two meanings of one name in the fleet is worse than a stale one.

Backend (``LLM_BACKEND``):

* ``ir`` (default since 0.109.0, owner decision 2026-10-20) — inference-router's
  OpenAI-compatible ``/v1/chat/completions`` (:mod:`w2v_common.clients.ir_direct`)
  with an app-``inference-router`` licence (``LLM_IR_TOKEN`` / ``LLM_IR_TOKEN_FILE``),
  tenant delegation + attribution headers. IR meters the call.
* ``vllm`` — the pgx vLLM directly with an app-``vllm`` licence
  (:mod:`w2v_common.clients.vllm_direct`). Kept as the rollback path.

The worker_management backend (``POST /workers/{name}/message``) was DELETED in
0.109.0; ``LLM_BACKEND=wm`` now fails construction by name. What remains of WM here
is the pooled client this class still inherits from ``BaseServiceClient`` (base URL
``WORKER_MANAGEMENT_URL`` / ``service_url("worker_management")`` and
``WM_AUTH_MODE``): no ``message()`` call uses it, but the orchestrator's subclass
(``src/services/llm/dispatch.py``) mirrors tenant provider config to WM's
``/tenant-config/...`` through it. Delete it with that mirror.

Both backends keep the WM-era surface callers depend on: the same ``message()``
signature, the same :class:`WMResponse`, ``WM_MIN_OUTPUT_TOKENS`` as a floor, the
same exception types (``httpx.HTTPStatusError`` / ``httpx.TransportError``).

Usage::

    llm = WMClient()
    result = await llm.message(
        worker_name="course-curriculum-generator",
        message=user_text,
        system_prompt="You are a curriculum architect …",
        output_schema={"type": "object", "properties": {…}},
    )
    structured_output = result.output  # dict-shaped per output_schema
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from prometheus_client import REGISTRY, Counter

from w2v_common.auth.bundle_tokens import WMAuthMode, resolve_wm_auth_mode
from w2v_common.auth.identity_context import (
    PROJECT_ID_HEADER,
    get_tenant_id_from_context,
    get_w2v_project_id_from_context,
    identity_event_hook,
    parse_attribution_uuid,
    tenant_level_project_id,
    tenant_licence_event_hook,
)
from w2v_common.clients.base import BaseServiceClient, _env_float
from w2v_common.clients.ir_direct import SERVICE_NAME as IR_SERVICE_NAME
from w2v_common.clients.ir_direct import (
    WORKER_NAME_HEADER,
    IRDirectClient,
    IRNotConfiguredError,
    IRSettings,
)
from w2v_common.clients.vllm_direct import (
    BACKEND_VLLM,
    LLMBackend,
    VLLMDirectClient,
    VLLMSettings,
    llm_backend_is_explicit,
    resolve_llm_backend,
)
from w2v_common.clients.vllm_usage_report import (
    IR_ENABLED_ENV,
    PROVIDER_IR,
    VLLMUsageReporter,
    resolve_run_id,
    usage_report_enabled,
)
from w2v_common.discovery import service_url

logger = logging.getLogger(__name__)


# The smallest ``max_tokens`` this client will ever send, and the value it
# substitutes when a caller passes none. (History below is from the WM backend,
# deleted in 0.109.0; the reasoning-model mechanism is the same on IR / the vLLM.)
#
# ⚠️ OMITTING ``max_tokens`` IS NOT NEUTRAL. It handed the call to WM's own
# default of 6000 (``pydantic_ai_agent.py``), and 6000 is an OBSERVED PRODUCTION
# FAILURE, not a safe fallback. The model behind ``local-llm`` is a REASONING
# model: its thinking is charged against ``max_tokens``, and when the budget runs
# out mid-thought pydantic_ai raises — post-generation, at
# ``_agent_graph.py:~1112``, when ``finish_reason == 'length'`` and every emitted
# part is a ``ThinkingPart``:
#
#     422  Model token limit (6000) exceeded before any response was generated
#
# That is TERMINAL for the caller — 422 classifies as permanent, so it is not
# retried. Sampled from worker-management's own logs on 2026-08-19 alone:
# ``marketing-email-copywriter`` 12:47:43Z, ``promo_script`` x2, ``growth_bundle``,
# ``website_review_script``, ``segment_select`` — every one of them at N=6000,
# i.e. every one of them a caller that passed nothing.
#
# ⚠️ THE INTUITIVE FIX IS BACKWARDS, and two merged PRs shipped it that way. A
# stale comment described this 422 as a PRE-FLIGHT RESERVATION against a small
# context window, which would make LOWERING ``max_tokens`` the cure. There is no
# such reservation: the model is served at ``--max-model-len=131072`` and the
# raise happens AFTER generation. Refuted with a clock, not an argument — a
# pre-flight rejection is milliseconds, and the real worker at
# ``max_tokens=1760`` returned 422 after **153.3 seconds**. ``max_tokens`` is a
# CAP, not a reservation, so raising it can never make a call that previously
# succeeded fail.
#
# Why 8000 and not 6000: 6000 is precisely the failing default. A floor set to it
# would pass fleet-wide while the outages continued — a green gate over a live
# bug, which is worse than no gate. 8000 is the smallest value with no recorded
# failure across 45,605 orchestrator ``stage_outputs`` rows going back to
# 2026-06-03 (recorded failures sit at N = 1760, 4000 and 6000). Note what that
# is: an absence of counter-evidence, not a proof of sufficiency.
#
# A caller that knows its answer is larger should pass a bigger number, not rely
# on this. A caller that thinks it wants a SMALLER one is asking to reproduce the
# outage above; there is no legitimate reason to send less than the model needs
# to finish thinking, so this is a FLOOR and not merely a default — an explicit
# smaller value is raised to it, and ``_effective_max_tokens`` logs when it does.
WM_MIN_OUTPUT_TOKENS = 8_000

#: Caller attribution headers stamped on EVERY inference-router call. IR's
#: per-app accounting keys on ``X-App-Id`` = the caller's licences-registry app
#: key; owner decision (2026-10-07): every w2v service's registry key is ``w2v``.
#: ``X-Worker-Name`` carries the worker; ``X-Job-Id`` the w2v RUN (bound w2v job,
#: else the correlation id) and is OMITTED when there is none — IR records it as
#: billing ``metadata.job_id`` and forbids a forged default.
APP_ID_HEADER = "X-App-Id"
JOB_ID_HEADER = "X-Job-Id"
DEFAULT_APP_ID = "w2v"

# Set after the first "no tenant, no project" warning so a tenant-less caller
# (startup task, CronJob) does not log once per LLM call.
_warned_no_project = False


def _project_fallback_counter() -> Counter:
    """Register (or re-use) the project-fallback counter (reload-safe)."""
    try:
        return Counter(
            _PROJECT_FALLBACK_NAME,
            "LLM calls that carried no W2V project. reason=tenant_level: the "
            "tenant-level project was sent instead; reason=no_tenant: nothing was sent.",
            ["reason", "worker"],
        )
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(_PROJECT_FALLBACK_NAME)
        if not isinstance(existing, Counter):  # pragma: no cover - defensive
            raise
        return existing


_PROJECT_FALLBACK_NAME = "w2v_wm_project_fallback_total"
wm_project_fallback_total = _project_fallback_counter()


def _project_header(project_id: uuid.UUID | str | None, worker_name: str) -> dict[str, str]:
    """``X-Project-Id`` for one inference-router call: the W2V project of the work.

    ⛔ W2V projects (orchestrator ``projects.id``) and fr projects are SEPARATE
    concepts (owner ruling 2026-10-07). The fr project bound from an inbound
    ``X-Project-Id`` is NEVER a source here. In order:

    1. ``project_id`` passed by the caller (strict canonical UUID).
    2. The W2V project bound on this context — by the orchestrator's run
       dispatcher (``set_w2v_project_context``) or, on a downstream hop, from
       the inbound ``X-W2V-Project-Id``.
    3. No W2V project: the TENANT-LEVEL project of the bound tenant
       (:func:`tenant_level_project_id`), counted as ``reason=tenant_level``.
    4. No tenant either: nothing, counted as ``reason=no_tenant`` and logged once.

    The header is set on the REQUEST, and ``identity_event_hook`` never
    overwrites a caller-set ``X-Project-Id``, so an fr project bound with
    ``AUTH_ATTRIBUTION_FORWARD_HEADERS`` on cannot displace it.
    """
    global _warned_no_project
    if project_id is not None:
        canonical = parse_attribution_uuid(str(project_id))
        if canonical is None:
            raise ValueError(f"project_id is not a canonical non-nil uuid: {project_id!r}")
        return {PROJECT_ID_HEADER: canonical}
    bound = get_w2v_project_id_from_context()
    if bound:
        return {PROJECT_ID_HEADER: bound}
    project = tenant_level_project_id(get_tenant_id_from_context())
    if project:
        wm_project_fallback_total.labels(reason="tenant_level", worker=worker_name).inc()
        return {PROJECT_ID_HEADER: project}
    wm_project_fallback_total.labels(reason="no_tenant", worker=worker_name).inc()
    if not _warned_no_project:
        _warned_no_project = True
        logger.warning(
            "LLM call without a bound tenant: no X-Project-Id sent (logged once per process)"
        )
    return {}


def _effective_max_tokens(max_tokens: int | None, worker_name: str) -> int:
    """Apply ``WM_MIN_OUTPUT_TOKENS`` as a floor, loudly.

    ``None`` (the overwhelmingly common case — 22 call sites across
    ``analysis_service``, ``content_service`` and ``video_analyzer_service` passed
    nothing at the time this landed) becomes the floor silently: that is the new
    default, not an override of anyone's intent.

    An explicit value BELOW the floor is a caller that has reasoned its way to the
    refuted mechanism, so it is raised and logged at WARNING with both numbers —
    silently honouring it would reproduce the 422, and silently ignoring it would
    hide the disagreement.
    """
    if max_tokens is None:
        return WM_MIN_OUTPUT_TOKENS
    if max_tokens < WM_MIN_OUTPUT_TOKENS:
        logger.warning(
            "LLM call to %s asked for max_tokens=%d, below the %d floor; raising it. "
            "A reasoning model spends this budget THINKING before it answers, and a "
            "budget it cannot finish thinking in returns a terminal 422.",
            worker_name,
            max_tokens,
            WM_MIN_OUTPUT_TOKENS,
        )
        return WM_MIN_OUTPUT_TOKENS
    return max_tokens


def _has_image_part(content_blocks: list[dict[str, Any]] | None) -> bool:
    """True when any typed content block is an image (→ the ``vlm`` slot)."""
    return any(block.get("type") == "image" for block in content_blocks or ())


@dataclass
class WMResponse:
    """Parsed WM `MessageResponse`."""

    output: Any
    output_type: str
    output_schema_used: dict[str, Any] | None
    metadata: dict[str, Any] | None
    usage_metrics: dict[str, Any] | None


class WMClient(BaseServiceClient):
    """Async w2v LLM client (module doc): inference-router by default, the vLLM on rollback.

    Construct once per service (e.g. on app startup) and reuse — the pooled
    ``httpx.AsyncClient`` of the backend reads identity ContextVars per request,
    so one client serves concurrent requests from different end-users correctly.

    Still a ``BaseServiceClient`` pointed at worker_management for ONE consumer:
    the orchestrator's subclass sends ``/tenant-config/...`` through the inherited
    ``_get_client`` / ``auth_mode``. ``message()`` and ``health_check()`` never
    touch WM. Two inherited things are overridden: the 600s ``_clamp_timeout``
    ceiling (the orchestrator's stages run 660s) and ``_get_client`` (WM's licence
    always travels as ``X-Tenant-Id: Bearer``, never ``Authorization``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        license_token: str | None = None,
        app_id: str = DEFAULT_APP_ID,
        *,
        auth_mode: str | WMAuthMode | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        llm_backend: str | None = None,
    ) -> None:
        # Resolved first: an unknown/removed LLM_BACKEND, or an EXPLICIT backend
        # without its licence, fails construction, loudly — never the first call.
        self.llm_backend: LLMBackend = resolve_llm_backend(llm_backend)
        explicit_backend = llm_backend_is_explicit(llm_backend)
        vllm_settings = VLLMSettings.from_env() if self.llm_backend == BACKEND_VLLM else None
        ir_settings: IRSettings | None = None
        #: Why the DEFAULTED ir backend cannot run (no licence configured), or None.
        self._llm_unconfigured: str | None = None
        if vllm_settings is None:
            try:
                ir_settings = IRSettings.from_env()
            except ValueError as exc:
                if explicit_backend:
                    raise
                # LLM_BACKEND unset: every consumer test that builds a WMClient
                # with no LLM env would crash here. Fail the CALL instead.
                self._llm_unconfigured = str(exc)
                logger.warning(
                    "LLM_BACKEND unset (default ir) and inference-router is not configured: "
                    "%s. LLM calls will raise IRNotConfiguredError.",
                    exc,
                )
        self.auth_mode = resolve_wm_auth_mode(auth_mode)
        self._transport = transport
        if not app_id:
            # An empty X-App-Id is the exact unattributed traffic this field
            # exists to end; refuse it at construction, not on the wire.
            raise ValueError("WMClient app_id must be a non-empty licences registry key")
        self.app_id = app_id
        # WM base URL (orchestrator tenant-config mirror only, see class doc):
        # explicit kwarg > WORKER_MANAGEMENT_URL env > discovery.
        resolved = (
            base_url or os.environ.get("WORKER_MANAGEMENT_URL") or service_url("worker_management")
        )
        # Seed every BaseServiceClient attribute through super() so this stays in
        # sync if the base grows one, then OVERWRITE self.timeout below. The
        # placeholder is required because super().__init__ runs _clamp_timeout(),
        # whose 600s ceiling would raise on the orchestrator's 660s stages.
        super().__init__(
            base_url=resolved,
            service_name="worker_management",
            timeout=30.0,  # placeholder, replaced verbatim on the next line
            license_token=license_token or os.environ.get("DOWNSTREAM_LICENSE_TOKEN", ""),
        )
        # The LLM is slow; a call that legitimately runs for minutes must NOT be
        # clamped like a normal downstream. Default is env-configurable
        # (WM_CLIENT_TIMEOUT_S, 600s) but an explicit caller value — e.g. the
        # orchestrator's 660s stages — is used VERBATIM (no _clamp_timeout).
        # timeout=None here means "use the env/600 default", NOT "no timeout".
        self.timeout = timeout if timeout is not None else _env_float("WM_CLIENT_TIMEOUT_S", 600.0)
        self._llm: VLLMDirectClient | None = None
        if vllm_settings is not None:
            self._llm = VLLMDirectClient(
                vllm_settings,
                timeout=self.timeout,
                transport=transport,
                usage_reporter=VLLMUsageReporter(
                    license_token=self.license_token, transport=transport
                ),
            )
        elif ir_settings is not None:
            # IR meters the call and its cost event reaches the run through the
            # orchestrator's billing mirror, so no run-usage report by default
            # (it would double count the tokens) — vllm_usage_report module doc.
            ir_report = usage_report_enabled(IR_ENABLED_ENV, default=False)
            self._llm = IRDirectClient(
                ir_settings,
                timeout=self.timeout,
                transport=transport,
                usage_reporter=VLLMUsageReporter(
                    license_token=self.license_token,
                    transport=transport,
                    provider=PROVIDER_IR,
                    enabled=True,
                )
                if ir_report
                else None,
            )

    async def _get_client(self) -> httpx.AsyncClient:
        """The inherited WM client: verbatim timeout + X-Tenant-Id-only auth.

        Not used by ``message()``; kept for the orchestrator's tenant-config
        mirror (class doc). In ``WM_AUTH_MODE=tenant_licence`` nothing
        credential-shaped is baked in: the caller sets the minted licence per
        request and ``tenant_licence_event_hook`` refuses a hop without one.
        """
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            hook = identity_event_hook
            if self.auth_mode is WMAuthMode.TENANT_LICENCE:
                hook = tenant_licence_event_hook
            elif self.license_token:
                headers["X-Tenant-Id"] = f"Bearer {self.license_token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                event_hooks={"request": [hook]},
                transport=self._transport,
            )
        return self._client

    async def health_check(self) -> dict[str, Any]:
        """The LLM backend's probe (``GET {base}/models``); reports, never raises.

        WM is not on the LLM path, so its health says nothing about this client.
        """
        if self._llm is None:
            return {
                "status": "unhealthy",
                "service": IR_SERVICE_NAME,
                "error": f"not configured: {self._llm_unconfigured}",
            }
        return await self._llm.health_check()

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if self._llm is not None:
            await self._llm.aclose()

    async def _tenant_slot_message(
        self,
        slot_name: Literal["llm", "vlm"],
        worker_name: str,
        *,
        message: str | None,
        content_blocks: list[dict[str, Any]] | None,
        context: dict[str, Any] | None,
        system_prompt: str | None,
        output_schema: dict[str, Any] | None,
        max_tokens: int | None,
        enable_reasoning: bool | None,
        ignored: dict[str, object],
    ) -> WMResponse | None:
        """The run's ``slot_name`` on its TENANT provider, or ``None`` for a deployment slot."""
        from w2v_common.providers.adapters import TenantLLMClient
        from w2v_common.providers.slots import resolve_provider_slot

        resolved = await resolve_provider_slot(slot_name)
        if not resolved.is_tenant:
            return None
        tenant = TenantLLMClient(resolved, timeout=self.timeout)
        try:
            payload = await tenant.message(
                worker_name,
                message=message,
                content_blocks=content_blocks,
                system_prompt=system_prompt,
                output_schema=output_schema,
                max_tokens=_effective_max_tokens(max_tokens, worker_name),
                enable_reasoning=enable_reasoning,
                context=context,
                overrides_ignored=ignored,
            )
        finally:
            await tenant.aclose()
        return WMResponse(
            output=payload.get("output"),
            output_type=str(payload.get("output_type", "string")),
            output_schema_used=output_schema,
            metadata=cast("dict[str, Any] | None", payload.get("metadata")),
            usage_metrics=cast("dict[str, Any] | None", payload.get("usage_metrics")),
        )

    def _ir_headers(self, worker_name: str, project_id: uuid.UUID | str | None) -> dict[str, str]:
        """Per-call attribution for inference-router (identity rides the pooled hook)."""
        headers = {
            APP_ID_HEADER: self.app_id,
            WORKER_NAME_HEADER: worker_name,
            **_project_header(project_id, worker_name),
        }
        run_id = resolve_run_id()
        if run_id:
            headers[JOB_ID_HEADER] = run_id
        return headers

    async def message(
        self,
        worker_name: str,
        message: str | None = None,
        *,
        content_blocks: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model_override: str | None = None,
        max_tokens: int | None = None,
        work_tier: Literal["small", "medium", "high"] | None = None,
        product: str | None = None,
        enable_reasoning: bool | None = None,
        disable_state_storage: bool = True,
        project_id: uuid.UUID | str | None = None,
        slot: Literal["llm", "vlm"] | None = None,
    ) -> WMResponse:
        """One LLM call for ``worker_name``; the structured response.

        ``worker_name`` selects the vendored seed prompt + temperature
        (``vllm_worker_defaults.yaml``); ``system_prompt`` follows it as a system
        message. ``output_schema`` is enforced as guided decoding + validation +
        re-ask. ``max_tokens`` is floored at ``WM_MIN_OUTPUT_TOKENS`` and clamped to
        16384. ``enable_reasoning`` turns thinking on (default off).

        ``model_override`` / ``work_tier`` / ``product`` were WM routing knobs; a
        single served model has no use for them, so they are recorded in
        ``metadata.overrides_ignored``, not silently dropped.
        ``disable_state_storage`` is accepted for signature compatibility only
        (nothing stores conversation state any more).

        Pass either ``message`` (plain string) OR ``content_blocks`` (typed
        ``{"type": "text", "text": …}`` / ``{"type": "image", "data": <base64>,
        "media_type": "image/jpeg"}`` blocks). Exactly one must be provided.

        ``project_id`` is the W2V project (orchestrator ``projects.id``) sent to
        inference-router as ``X-Project-Id``. Omit it only when the context already
        carries one; with neither, the tenant-level project is sent and
        ``w2v_wm_project_fallback_total`` counts it. Never pass an fr project.

        ``slot`` (0.113.0) names the run provider slot: ``llm``, or ``vlm`` — the
        default when ``content_blocks`` carry an image part. Only read when
        ``W2V_PROVIDER_SLOTS=enforce``: the run's slot is resolved and a TENANT slot
        is called through ``w2v_common.providers.TenantLLMClient`` (the tenant's
        secret only, no identity headers); a DEPLOYMENT slot takes today's
        inference-router path unchanged. ``off`` (default) never resolves anything.
        """
        del disable_state_storage  # WM-era knob; see docstring
        if message is None and content_blocks is None:
            raise ValueError("Either 'message' or 'content_blocks' must be provided")
        if message is not None and content_blocks is not None:
            raise ValueError("Pass exactly one of 'message' or 'content_blocks', not both")
        from w2v_common.providers.slots import provider_slots_enforced

        if provider_slots_enforced():
            tenant_response = await self._tenant_slot_message(
                slot or ("vlm" if _has_image_part(content_blocks) else "llm"),
                worker_name,
                message=message,
                content_blocks=content_blocks,
                context=context,
                system_prompt=system_prompt,
                output_schema=output_schema,
                max_tokens=max_tokens,
                enable_reasoning=enable_reasoning,
                ignored={
                    k: v
                    for k, v in (
                        ("model_override", model_override),
                        ("work_tier", work_tier),
                        ("product", product),
                    )
                    if v
                },
            )
            if tenant_response is not None:
                return tenant_response
        if self._llm is None:
            raise IRNotConfiguredError(
                f"no LLM backend is configured for this WMClient: {self._llm_unconfigured}"
            )

        ignored = {
            k: v
            for k, v in (
                ("model_override", model_override),
                ("work_tier", work_tier),
                ("product", product),
            )
            if v
        }
        headers = (
            self._ir_headers(worker_name, project_id)
            if isinstance(self._llm, IRDirectClient)
            else None
        )
        payload = await self._llm.message(
            worker_name,
            message=message,
            content_blocks=content_blocks,
            system_prompt=system_prompt,
            output_schema=output_schema,
            max_tokens=_effective_max_tokens(max_tokens, worker_name),
            enable_reasoning=enable_reasoning,
            context=context,
            overrides_ignored=ignored,
            headers=headers,
        )
        return WMResponse(
            output=payload.get("output"),
            output_type=str(payload.get("output_type", "string")),
            output_schema_used=output_schema,
            metadata=cast("dict[str, Any] | None", payload.get("metadata")),
            usage_metrics=cast("dict[str, Any] | None", payload.get("usage_metrics")),
        )
