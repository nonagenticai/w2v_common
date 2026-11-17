"""Report vLLM-direct token usage to the orchestrator's per-run ``run_usage`` table.

``LLM_BACKEND=vllm`` (:mod:`w2v_common.clients.vllm_direct`) bypasses worker
management, so no WM/inference-router usage row is ever written for the call and a
run that spent every token on the vLLM shows 0 tokens. After each SUCCESSFUL call
that runs inside a w2v run context, :class:`VLLMUsageReporter` fires one
``POST {orchestrator}/api/runs/{run_id}/usage`` in the background:

``{"call_id", "stage", "model", "provider": "vllm", "input_tokens",
"output_tokens", "total_tokens", "service"}`` → 202 ``{"recorded": bool}``.

* **Run id**: the bound w2v job (``X-W2V-Job-Id``), else the cost-correlation id
  (the orchestrator sets ``correlation_id = run_id``). Both must parse as a w2v job
  id (:func:`~w2v_common.auth.identity_context.parse_w2v_job_id`); no run id → no
  report. ``stage`` is the bound ``X-Stage`` (``null`` when unbound).
* **Idempotent**: one ``call_id`` (uuid7) per LLM call, minted by the caller and
  reused by the single retry, so the orchestrator dedupes a retried report.
* **Never fails or slows the LLM call**: scheduled as a task held by a strong
  reference (``cost_bus.schedule_cost_event``'s pattern), 5 s timeout per attempt,
  at most ONE retry (transport error / timeout / 5xx), every failure swallowed and
  logged at WARNING without the URL (and so without any userinfo) or credential,
  counted in ``w2v_vllm_usage_report_total{outcome}``.
* **Auth**: the canonical S2S shape of :class:`~w2v_common.clients.base.BaseServiceClient`
  — the service-account licence (``DOWNSTREAM_LICENSE_TOKEN``) in
  ``X-Tenant-Id: Bearer`` plus ``identity_event_hook`` (X-License-Id, X-Subject-Id,
  X-Correlation-Id, X-Stage, X-W2V-Job-Id/Project-Id, cell token) and
  ``trace_event_hook``.
* **Never inside the orchestrator**: it records its own calls directly, so a report
  from itself would double count. Off when ``LLM_VLLM_USAGE_REPORT_ENABLED`` is
  false, and ALWAYS off when this process's own service name
  (``CELL_SERVICE_NAME``, else ``OTEL_SERVICE_NAME``) is the orchestrator's k8s
  Service name.

This is usage ATTRIBUTION for the run view, not metering: nothing is emitted to
the billing cost bus, and the orchestrator records every reported row at cost 0
(``funding = local_gpu``) — it never prices a report.

``LLM_BACKEND=ir`` (0.109.0): inference-router METERS the call itself and its cost
event carries ``X-Correlation-Id = run_id`` + ``X-Stage``, so the orchestrator's
billing mirror (``RunUsageReconciler``) already puts the tokens AND the priced cost
into ``run_usage``. A report on top would count the same tokens twice, so the IR
backend does not report unless ``LLM_IR_USAGE_REPORT_ENABLED=true`` (default
false); when it does, ``provider`` is ``inference_router``. The report body carries
no ``funding`` key on purpose: the orchestrator's ``RunUsageReport`` is
``extra="forbid"``, so one would 422 every report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Final
from urllib.parse import quote

import httpx
from prometheus_client import REGISTRY, Counter

from w2v_common.auth.identity_context import (
    get_correlation_id_from_context,
    get_stage_from_context,
    get_tenant_id_from_context,
    get_w2v_job_id_from_context,
    identity_event_hook,
    is_tenant_vouched,
    parse_w2v_job_id,
)
from w2v_common.auth.keycloak import _looks_like_jwt
from w2v_common.auth.outbound import EFFECTIVE_TENANT_HEADER
from w2v_common.clients.base import BaseServiceClient
from w2v_common.discovery import cell_services, service_url
from w2v_common.observability.request_context import trace_event_hook

logger = logging.getLogger(__name__)

#: Kill switch (default on). The orchestrator is skipped regardless, see module doc.
ENABLED_ENV: Final = "LLM_VLLM_USAGE_REPORT_ENABLED"
#: The IR backend's switch (default OFF: IR's own cost event already reaches the run).
IR_ENABLED_ENV: Final = "LLM_IR_USAGE_REPORT_ENABLED"
PROVIDER_VLLM: Final = "vllm"
PROVIDER_IR: Final = "inference_router"
_TRUE: Final = frozenset({"1", "true", "yes", "on"})
ORCHESTRATOR: Final = "orchestrator"
#: Per-attempt budget; a report is worth far less than the LLM call it follows.
REPORT_TIMEOUT_S: Final = 5.0
#: Attempts in total (one retry).
MAX_ATTEMPTS: Final = 2
RETRY_DELAY_S: Final = 0.5
UNKNOWN_SERVICE: Final = "unknown"
_FALSE: Final = frozenset({"0", "false", "no", "off"})

#: Strong references: an un-referenced task may be garbage-collected mid-POST.
_background_tasks: set[asyncio.Task[None]] = set()


def _report_counter() -> Counter:
    """Register (or re-use) the report-outcome counter (reload-safe)."""
    try:
        return Counter(
            _COUNTER_NAME,
            "vLLM-direct usage reports to the orchestrator run_usage endpoint. "
            "outcome=sent|rejected|failed|dropped_no_loop|dropped_no_tenant|dropped_unvouched.",
            ["outcome"],
        )
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(_COUNTER_NAME)
        if not isinstance(existing, Counter):  # pragma: no cover - defensive
            raise
        return existing


_COUNTER_NAME = "w2v_vllm_usage_report_total"
vllm_usage_report_total = _report_counter()


def own_service_name() -> str:
    """This process's service name: ``CELL_SERVICE_NAME``, else ``OTEL_SERVICE_NAME``."""
    for name in ("CELL_SERVICE_NAME", "OTEL_SERVICE_NAME"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _orchestrator_service_names() -> frozenset[str]:
    return frozenset({cell_services()[ORCHESTRATOR], ORCHESTRATOR})


def usage_report_enabled(env: str = ENABLED_ENV, *, default: bool = True) -> bool:
    """False when switched off by ``env``, or when this process IS the orchestrator.

    ``default`` is the answer for an unset/blank ``env`` (vLLM: on, IR: off).
    """
    if own_service_name() in _orchestrator_service_names():
        return False
    raw = (os.environ.get(env) or "").strip().lower()
    if not raw:
        return default
    if default:
        return raw not in _FALSE
    return raw in _TRUE


def resolve_run_id() -> str | None:
    """The run this call belongs to: bound w2v job, else correlation id; ``None`` if neither."""
    return parse_w2v_job_id(get_w2v_job_id_from_context()) or parse_w2v_job_id(
        get_correlation_id_from_context()
    )


def new_call_id() -> str:
    """One idempotency key per LLM call (uuid7, the w2v id standard)."""
    return str(uuid.uuid7())


class VLLMUsageReporter(BaseServiceClient):
    """Fire-and-forget usage reports to the orchestrator. Never raises from :meth:`schedule`."""

    def __init__(
        self,
        *,
        license_token: str | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        enabled: bool | None = None,
        provider: str = PROVIDER_VLLM,
    ) -> None:
        super().__init__(
            base_url=base_url or service_url(ORCHESTRATOR),
            service_name=ORCHESTRATOR,
            timeout=REPORT_TIMEOUT_S,
            license_token=license_token,
        )
        self.enabled = usage_report_enabled() if enabled is None else enabled
        self.provider = provider
        self._transport = transport

    async def _get_client(self) -> httpx.AsyncClient:
        """``BaseServiceClient``'s headers + hooks, plus the test transport seam."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self.license_token:
                if _looks_like_jwt(self.license_token):
                    headers["Authorization"] = f"Bearer {self.license_token}"
                else:
                    headers["X-Tenant-Id"] = f"Bearer {self.license_token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                verify=self.ssl_context,
                follow_redirects=False,
                event_hooks={"request": [identity_event_hook, trace_event_hook]},
                transport=self._transport,
            )
        return self._client

    def schedule(
        self,
        *,
        call_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        """Schedule one report for the current run context. True when a task was created."""
        try:
            if not self.enabled:
                return False
            run_id = resolve_run_id()
            if run_id is None:
                return False
            # The orchestrator accepts delegation on THIS route only, keyed on
            # X-Effective-Tenant-Id = the run's tenant. identity_event_hook sends it
            # only to METERED_DESTINATION_SERVICES (the orchestrator is not one, and
            # must not be added globally), so it is set here, under the same rule the
            # hook applies before delegating: bound AND vouched tenant, else no report.
            tenant = get_tenant_id_from_context()
            if tenant is None or not is_tenant_vouched():
                vllm_usage_report_total.labels(
                    outcome="dropped_no_tenant" if tenant is None else "dropped_unvouched"
                ).inc()
                return False
            headers = {EFFECTIVE_TENANT_HEADER: str(tenant)}
            body: dict[str, object] = {
                "call_id": call_id,
                "stage": get_stage_from_context() or None,
                "model": model,
                "provider": self.provider,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens) + int(output_tokens),
                "service": own_service_name() or UNKNOWN_SERVICE,
            }
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                vllm_usage_report_total.labels(outcome="dropped_no_loop").inc()
                return False
            # create_task copies the context: the identity hooks see this call's tenant.
            task = loop.create_task(self._send(run_id, body, headers))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            return True
        except Exception as exc:  # noqa: BLE001 — a usage report must never fail the LLM call
            logger.warning("vllm usage report not scheduled: %s", type(exc).__name__)
            return False

    async def _send(self, run_id: str, body: dict[str, object], headers: dict[str, str]) -> None:
        path = f"/api/runs/{quote(run_id, safe='')}/usage"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                client = await self._get_client()
                response = await client.post(path, json=body, headers=headers)
            except Exception as exc:  # noqa: BLE001 — fire-and-forget: every failure is swallowed
                retryable = isinstance(exc, httpx.TransportError)
                if retryable and attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(RETRY_DELAY_S)
                    continue
                vllm_usage_report_total.labels(outcome="failed").inc()
                logger.warning(
                    "vllm usage report failed run=%s call_id=%s attempt=%d: %s",
                    run_id,
                    body["call_id"],
                    attempt,
                    type(exc).__name__,
                )
                return
            status = response.status_code
            if status < 300:
                vllm_usage_report_total.labels(outcome="sent").inc()
                return
            if status >= 500 and attempt < MAX_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY_S)
                continue
            vllm_usage_report_total.labels(outcome="rejected").inc()
            logger.warning(
                "vllm usage report rejected run=%s call_id=%s status=%d",
                run_id,
                body["call_id"],
                status,
            )
            return

    async def close(self) -> None:
        """Let in-flight reports finish (bounded), then close the pooled client."""
        await drain_pending_usage_reports(timeout=REPORT_TIMEOUT_S)
        await super().close()


async def drain_pending_usage_reports(timeout: float = REPORT_TIMEOUT_S) -> None:  # noqa: ASYNC109 — bounded drain budget, mirrors cost_bus.drain_pending
    """Wait up to ``timeout`` seconds for scheduled reports; never raises."""
    pending = [t for t in _background_tasks if not t.done()]
    if not pending:
        return
    try:
        await asyncio.wait(pending, timeout=timeout)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
        logger.warning("vllm usage report drain failed: %s", type(exc).__name__)


__all__ = [
    "ENABLED_ENV",
    "IR_ENABLED_ENV",
    "PROVIDER_IR",
    "PROVIDER_VLLM",
    "VLLMUsageReporter",
    "drain_pending_usage_reports",
    "new_call_id",
    "own_service_name",
    "resolve_run_id",
    "usage_report_enabled",
    "vllm_usage_report_total",
]
