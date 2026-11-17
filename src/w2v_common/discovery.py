"""Service discovery for w2v peer services.

Peer services are discovered by a single canonical logical name. In-cluster
URLs use the Kubernetes short form ``http://<service>.<namespace>:<port>``,
which resolves inside the cluster. Deployments that need the fully-qualified
form set ``W2V_DNS_SUFFIX`` to their cluster's DNS suffix; the suffix
is deployment configuration, not library code.

For peer services that live OUTSIDE the cluster, the namespace is ``None``
and the URL builder returns ``http://<host>:<port>``. Set
``W2V_<NAME>_HOST`` (or the per-peer ``W2V_<NAME>_URL``) to point at the
real host.

Service names, namespaces and ports are deployment conventions; this module
is the single source of truth for peer-service addressing — individual
services must not hardcode peer URLs.
"""

from __future__ import annotations

import os
from typing import Final

# Canonical peer service catalog: logical name -> (host, namespace, port).
# ``namespace=None`` means the peer is a host service outside the cluster; set
# ``W2V_<NAME>_HOST`` (or ``W2V_<NAME>_URL``) per deployment to address it.
_SERVICES: Final[dict[str, tuple[str, str | None, int]]] = {
    "orchestrator": ("w2v-orchestrator", "w2v", 6300),
    # The license-introspecting gateway. Registered here so
    # ``auth.outbound.is_metered_destination`` can match a caller's BASE URL and
    # not only the bare name -- callers pass URLs, so a name-only entry in
    # METERED_DESTINATION_SERVICES would still classify it as a plain backend.
    "modality_gateway": ("modality-gateway", "modality-gateway", 8890),
    "analysis": ("analysis-service", "w2v", 6303),
    # Web-design capability (HTML generation + Playwright screenshots) lifted out
    # of analysis_service's redesign domain into its own service. Serves the
    # canonical /api/designs surface; consumed by orchestrator promo_html +
    # website_redesign stages.
    "html_render": ("html-render-service", "w2v", 6319),
    "content": ("content-service", "w2v", 6305),
    "video_production": ("video-production-service", "w2v", 6308),
    # Text-to-speech host service. It runs outside the cluster, so the
    # namespace is None; deployments set W2V_TTS_HOST (or W2V_TTS_URL) to the
    # real host.
    "tts": ("tts", None, 9202),
    # The narration gateway in front of the TTS hosts. Registered here for the
    # same reason as ``modality_gateway`` above: it is a METERED destination and
    # ``auth.outbound.is_metered_destination`` matches by HOST, resolving each
    # metered name through ``service_url``. While this entry was missing,
    # ``service_url("tts_gateway")`` raised KeyError, the host loop hit
    # ``continue``, and no request to the gateway could ever be stamped with
    # ``X-Effective-Tenant-Id`` — 1082 tts_jobs/tts_batch_jobs rows filed under
    # the orchestrator's service-account tenant as a result.
    "tts_gateway": ("tts-gateway", "w2v", 6320),
    "ai_content_generation": ("ai-content-generation-service", "w2v", 6310),
    "app_inspector": ("app-inspector-service", "w2v", 6311),
    "slide_service": ("slide-service", "w2v", 6312),
    "bdd_playwright": ("bdd-playwright-service", "w2v", 6313),
    "course": ("course-service", "w2v", 6314),
    "youtube_upload": ("youtube-upload-service", "w2v", 6315),
    "frontend_test_runner": ("frontend-test-runner-service", "w2v", 6316),
    "video_analyzer": ("video-analyzer-service", "w2v", 6317),
    # Transcription host service. It runs outside the cluster, so the namespace
    # is None; deployments set W2V_TRANSCRIPTION_WORKER_HOST (or ..._URL).
    "transcription_worker": ("asr", None, 9201),
    # Cross-namespace scrape + media peers, called DIRECTLY (KAG removal, 2026-10-17).
    # KAG was only ever a gateway: its scrape routes forwarded to webscraper and
    # its /media/* routes to media-discovery verbatim, so w2v now calls them
    # itself (``clients.webscraper.WebscraperClient``; the orchestrator's media
    # client). Both authenticate the caller's licence in ``X-Tenant-Id: Bearer``.
    # discovery-engine (:8807) and scientific-processor (:8808) stay unaddressable
    # (still DENY'd, scaled to 0). Enforced by
    # ``tests/unit/test_discovery.py::TestDirectScrapeAndMediaPeers``.
    "webscraper": ("webscraper", "webscraper", 9119),
    "media_discovery": ("media-discovery", "media-discovery", 8809),
    # Cross-namespace platform services. These were previously hardcoded as
    # DEFAULT_BASE_URL literals on each client / module-level env-default; now
    # consolidated here so discovery is the only source of in-cluster DNS
    # construction (config-externalization-policy: a single source of truth
    # for service URLs).
    "worker_management": ("worker-management", "workers", 8887),
    "artifacts": ("artifacts", "artifacts", 8556),
    "licenses": ("licenses", "licenses", 8544),
    # Cross-namespace inference + media services consumed by w2v_orchestrator
    # stage handlers. Previously hardcoded in orchestrator clients with custom
    # env-var fallbacks (INFERENCE_ROUTER_URL, VIDEO_COMPILE_URL); centralised
    # here so the single per-peer override is the discovery-standard
    # W2V_<NAME>_URL.
    "inference_router": ("inference-router", "inference-router", 8150),
    "video_compile": ("video-compiler", "video-compiler", 8200),
}


def _dns_suffix() -> str:
    """Return the configured DNS suffix, with a leading dot, or ``""``.

    Empty is the Kubernetes short form (``service.namespace``), which resolves
    inside the cluster. A deployment that addresses peers across clusters sets
    ``W2V_DNS_SUFFIX`` to its cluster's DNS suffix (leading dot optional).
    """
    suffix = os.environ.get("W2V_DNS_SUFFIX", "")
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def service_url(name: str) -> str:
    """Return the HTTP URL for a w2v peer service.

    For in-cluster peers (most entries), the URL is the Kubernetes short form
    ``http://<host>.<namespace>:<port>``, plus ``W2V_DNS_SUFFIX`` when set.

    For ``namespace=None`` entries (host services outside the cluster), the
    URL is ``http://<host>:<port>``, where the host can be set per deployment
    with ``W2V_<UPPER_NAME>_HOST``.

    In local dev, set the env var ``W2V_LOCAL=1`` to use
    ``http://localhost:<port>`` instead, and ``W2V_<UPPER_NAME>_URL`` to
    override an individual peer's URL (tests, port-forwards, etc.).
    """
    try:
        host, namespace, port = _SERVICES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SERVICES))
        raise KeyError(f"Unknown peer service {name!r}. Known: {known}") from exc

    override = os.environ.get(f"W2V_{name.upper()}_URL")
    if override:
        return override

    if os.environ.get("W2V_LOCAL"):
        return f"http://localhost:{port}"

    if namespace is None:
        host = os.environ.get(f"W2V_{name.upper()}_HOST", host)
        return f"http://{host}:{port}"
    return f"http://{host}.{namespace}{_dns_suffix()}:{port}"


def service_port(name: str) -> int:
    """Return the fixed port for a peer service."""
    try:
        return _SERVICES[name][2]
    except KeyError as exc:
        known = ", ".join(sorted(_SERVICES))
        raise KeyError(f"Unknown peer service {name!r}. Known: {known}") from exc


def list_services() -> list[str]:
    """Return the list of known peer-service logical names."""
    return sorted(_SERVICES)


#: The Kubernetes namespace that IS the w2v cell (P7 intra-cell auth). Peers
#: registered in this namespace are in-cell: they receive a per-request cell
#: token (``X-W2V-Cell-Token``) and nothing outside it ever does.
CELL_NAMESPACE: Final[str] = "w2v"

#: In-cell peers that live OUTSIDE ``CELL_NAMESPACE``, admitted BY NAME. The
#: namespace rule is not widened, so no other cross-namespace service (WM,
#: licences, inference-router, ...) is ever pulled into the cell.
#:
#: * ``video_compile`` — video-compiler runs in its own ``video-compiler``
#:   namespace but is a w2v cell member (the orchestrator calls it directly for
#:   renders). Without this entry it never received a cell token, so its calls
#:   stayed on the legacy lane and it could never go ``CELL_AUTH_MODE=enforce``.
#:   Cell identity (``aud`` / its ``CELL_SERVICE_NAME``): ``video-compiler``.
CELL_EXTRA_MEMBERS: Final[frozenset[str]] = frozenset({"video_compile"})


def cell_services() -> dict[str, str]:
    """In-cell peers: logical discovery name -> the peer's Kubernetes Service name.

    The Service name (e.g. ``analysis-service``) is the peer's cell IDENTITY:
    it is the ``aud`` a caller mints for it and the ``CELL_SERVICE_NAME`` the
    peer itself is configured with. Derived from the catalog so a new w2v peer
    becomes a cell destination by being registered here, nowhere else; a peer
    in another namespace is in-cell only if listed in ``CELL_EXTRA_MEMBERS``.
    """
    return {
        name: host
        for name, (host, namespace, _port) in _SERVICES.items()
        if namespace == CELL_NAMESPACE or name in CELL_EXTRA_MEMBERS
    }
