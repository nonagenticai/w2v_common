# W2V Common

Shared middleware, config, auth, database, storage, and service clients for all w2v Python services.

## What W2V Common Provides

Importing `w2v_common` gives a w2v service its whole boilerplate spine in one dependency: it does not run — it is composed into a service. The `config` module ships a family of reusable Pydantic `*Settings` sub-configs (`DatabaseSettings`, `RedisSettings`, `APISettings`, `AuthSettings`, `KeycloakSettings`, `StorageSettings`, `LLMSettings`, `CacheSettings`, and the idempotency/metrics/audit/tracing/ffmpeg/logging blocks) that each service composes into its own `Settings` class rather than re-deriving. `APISettings` defaults the listen port to 8080 and a field validator hard-rejects `API_PORT=8000` (reserved for the user's local mother service) along with any out-of-range port, so a misconfigured port fails at settings-load time instead of at bind.

On the request path, `w2v_common` owns the canonical identity model. `auth.AuthMiddleware` is the inbound validator: license tokens travel only in `X-Tenant-Id: Bearer <license_token>` (with `Authorization: Bearer <jwt>` reserved for the caller's Keycloak JWT), tokens are introspected against the `licenses` service with a TTL cache plus a short negative cache to absorb revocation storms, and in-cluster trust is fail-closed by default. Outbound service-to-service identity rides ContextVars in `auth.identity_context`: the inbound middleware sets the context and an `identity_event_hook` registered on every service client stamps `X-Tenant-Id` / `X-License-Id` / `X-Subject-Id` as plain headers on each cross-service call, so the three-piece tenant/subject/license invariant is preserved end-to-end without per-service plumbing.

The `asgi` factory (`create_app`) wires the FastAPI app together with the shared error envelope and the opt-in middleware stack (request-id, security headers, idempotency, metrics, audit — each toggled by its matching `*Settings.enabled` flag), and exposes the standard health probes. The `database` module supplies the async SQLAlchemy engine, session factory, and health check over psycopg3 / Postgres 18 only — never asyncpg, never sqlite — plus the row-level-security helpers keyed on the per-request `app.current_tenant` GUC. `discovery.service_url()` is the single source of truth for peer-service addressing, building Kubernetes service URLs or configured host URLs so no service hardcodes a peer address, and the typed clients in `clients` (webscraper, artifact, worker-management, plus the `BaseServiceClient` base) inherit retries, redirect guards, timeout clamps, and the identity/trace event hooks automatically.

Rounding out the surface are `storage.StorageService` (a local or MinIO/S3 backend selected from `StorageSettings`), `metering` (a Redis cost-bus emitter for per-tenant cost events), `observability` (structured JSON logging, Prometheus metrics, and request/trace context), `net.safe_fetch` (SSRF-safe fetching of tenant-supplied URLs), `stages` (bounded-concurrency stage-execution primitives with typed timeout/retry errors for orchestrator handlers), and a small `testing` helper for injecting fakeredis into unit tests.

The library is consumed by the w2v services — locally as a path dependency (`w2v-common = { path = "../w2v_common" }`) and in CI/prod as a pinned git dependency on the public repository (a tag per release). Because every consumer pins a version, bumping `w2v_common` is fleet-wide: it triggers a re-lock and republish cascade across all consuming repos, so a behavior change here is a multi-repo change, not a single PR.

## Features

- **Composable settings with a port guard** — a family of Pydantic `*Settings` sub-configs that services compose into their own `Settings`, where `APISettings` rejects `API_PORT=8000` and out-of-range ports at load time so misconfiguration fails fast rather than at bind
- **Inbound license-token auth** — `AuthMiddleware` validates `X-Tenant-Id: Bearer <license_token>` by introspecting against the `licenses` service with a TTL + negative cache, fail-closed on in-cluster trust, so every service enforces tenancy identically
- **Outbound S2S identity forwarding** — ContextVar-based identity context plus a client event hook that stamps `X-Tenant-Id` / `X-License-Id` / `X-Subject-Id` on every cross-service call, preserving the three-piece tenant/subject/license invariant without per-service code
- **ASGI app factory** — `create_app` assembles FastAPI with the shared error envelope, opt-in middleware stack, and standard health probes so services boot from one canonical wiring
- **Postgres-only database layer** — async SQLAlchemy engine, session factory, and health check over psycopg3 / Postgres 18 (never asyncpg, never sqlite) plus RLS policy helpers keyed on the `app.current_tenant` GUC for tenant isolation
- **Service discovery** — `service_url()` builds Kubernetes service URLs or configured host URLs from a single source of truth, with `W2V_LOCAL` and per-peer overrides, so no service hardcodes a peer address
- **Typed service clients** — `BaseServiceClient` plus webscraper and artifact clients that inherit retries, redirect guards, timeout clamps, and identity/trace hooks
- **LLM client** — `WMClient.message()` calls inference-router's `/v1/chat/completions` by default (`LLM_BACKEND=ir`, `LLM_IR_TOKEN`/`LLM_IR_TOKEN_FILE` = an app-`inference-router` licence, tenant delegation + `X-App-Id`/`X-Worker-Name`/`X-Job-Id`/`X-Project-Id`), or the pgx vLLM directly (`LLM_BACKEND=vllm`, rollback); same `WMResponse`, schema validation + re-ask either way. The worker_management backend was removed in 0.109.0
- **Object storage abstraction** — `StorageService` with local and MinIO/S3 backends selected from `StorageSettings`, consolidating what used to be per-service storage modules; `await storage.aclose()` (or `async with`) releases the MinIO aiohttp session
- **Opt-in middleware stack** — request-id, security-headers, idempotency, metrics, and audit middleware each gated by its `*Settings.enabled` flag so services enable only what they need
- **Observability** — structured JSON logging, Prometheus HTTP/downstream metrics, and request/trace context propagation ready to mount
- **SSRF-safe fetching** — `net.safe_fetch` validates tenant-supplied URLs and re-checks every redirect hop against private/reserved/metadata ranges, the one fleet-wide guard for URL-fetching services. A hop that changes origin is reduced to an allow-list of request-describing headers (so no credential, cookie or estate identity header is replayed to a redirector-chosen host) and response bodies are capped at `DEFAULT_MAX_FETCH_BYTES` (64 MiB) unless the caller states its own `max_fetch_bytes`
- **vLLM-direct run usage** — with `LLM_BACKEND=vllm`, `WMClient` reports each successful call's tokens to the orchestrator (`POST /api/runs/{run_id}/usage`, idempotent `call_id`) in the background when a run context is bound; never fails or slows the call, never sent from the orchestrator itself, switch `LLM_VLLM_USAGE_REPORT_ENABLED` (default on). Attribution only — not billing metering. Off for `LLM_BACKEND=ir` (IR's own cost event reaches the run via the billing mirror; opt in with `LLM_IR_USAGE_REPORT_ENABLED=true`)
- **Cost metering** — a Redis cost-bus emitter that publishes per-tenant cost events to a canonical stream for downstream billing/metering
- **Stage-execution primitives** — bounded-concurrency helpers with typed `StageTimeoutError` / `StageRetriesExhaustedError` for orchestrator stage handlers that fan out downstream calls

## Modules

| Module | What It Provides |
|--------|-------------------|
| `asgi` | `create_app` FastAPI factory, error-envelope handlers, and middleware registration |
| `auth` | `AuthMiddleware` (inbound license-token validation), `User` model, and ContextVar identity context with S2S event hooks |
| `cache` | `RedisClient` plus `init_redis` / `get_redis_client` / `close_redis` lifecycle helpers |
| `clients` | `BaseServiceClient` and typed clients: webscraper, artifact, the LLM client (`WMClient`: inference-router / vLLM), sitemap/URL-graph helpers |
| `config` | Composable Pydantic `*Settings` sub-configs (API, Database, Redis, Auth, Keycloak, Storage, LLM, Cache, audit/metrics/tracing/ffmpeg) plus `DownstreamSettings` |
| `database` | Async SQLAlchemy engine, session factory, health check, and RLS policy helpers over psycopg3 / Postgres 18 |
| `discovery` | `service_url()` / `service_port()` — single source of truth for peer-service addressing |
| `metering` | Redis cost-bus emitter (`emit_cost_event`, `schedule_cost_event`, `drain_pending`) publishing per-tenant cost events |
| `middleware` | Opt-in ASGI middleware: request-id, security headers, idempotency, metrics, audit |
| `net` | `safe_fetch` — SSRF-safe fetching of caller-supplied URLs with per-redirect-hop re-validation |
| `observability` | Structured JSON logging, Prometheus metrics, and request/trace context propagation |
| `stages` | Bounded-concurrency stage-execution primitives with typed timeout/retry error types for orchestrator handlers |
| `storage` | `StorageService` object-storage abstraction over local and MinIO/S3 backends |
| `testing` | Shared test helper — `inject_fakeredis` for wiring a fake Redis into unit tests |

## Installation / Usage

```toml
# pyproject.toml
w2v-common = { path = "../w2v_common" }   # local path dep
```

```bash
# or pin the public repository at a release tag
uv add "w2v-common @ git+https://github.com/nonagenticai/w2v_common@v0.115.4"
```

```python
from w2v_common.asgi import create_app
from w2v_common.config import APISettings, DatabaseSettings
from w2v_common.discovery import service_url

# services compose the shared sub-configs into their own Settings,
# then build the app from the shared factory
app = create_app(...)
tts = service_url("tts")  # resolves the peer's k8s/Tailnet URL
```

## Architecture

```
src/w2v_common/
├── __init__.py            # exposes __version__ (from package metadata)
├── discovery.py           # service_url() / service_port() peer addressing
├── testing.py             # inject_fakeredis test helper
├── asgi/                  # create_app factory, error envelope, middleware registration
├── auth/                  # AuthMiddleware, identity_context (S2S hooks), keycloak, User model
├── cache/                 # RedisClient + lifecycle helpers
├── clients/               # BaseServiceClient + webscraper/artifact/wm/sitemap/url_graph clients
├── config/                # base.py (*Settings sub-configs) + downstream.py
├── database/              # connection, session, rls (SQLAlchemy + psycopg3 / Postgres 18)
├── metering/              # cost_bus Redis emitter
├── middleware/            # audit, idempotency, metrics, request_id, security
├── net/                   # safe_fetch SSRF guard
├── observability/         # logging, metrics, request_context
├── stages/                # execution primitives + typed stage errors
└── storage/               # StorageService (local + MinIO/S3)
```

## Testing

```bash
uv sync --all-extras
uv run pytest tests/ -v --cov=src                       # All tests with coverage
uv run ruff check src/ tests/                           # Lint
uv run ruff format --check src/ tests/                  # Format check
```

Tests run against a real Postgres 18 (psycopg3) plus fakeredis — no sqlite anywhere. Coverage threshold: 80% (enforced in `pyproject.toml`). `./scripts/ci-local.sh` reproduces the full CI `pre-build-checks` order (`ruff check`, `ruff format --check`, `ty check`, `pytest`, `detect-secrets`, `pip-audit`, `bandit`).

## Dependencies

- **FastAPI** — web framework the ASGI factory builds on
- **Uvicorn** (`[standard]`) — ASGI server used by consuming services
- **Pydantic** + **pydantic-settings** — settings sub-configs and schema validation
- **psycopg** (`[binary,pool]`) — Postgres 18 wire driver (no asyncpg)
- **SQLAlchemy** (`[asyncio]`) — async ORM / engine
- **redis** — cache client and cost-bus transport
- **httpx** — async HTTP client for the service clients
- **PyJWT** (`[crypto]`) + **cryptography** — Keycloak JWT validation
- **prometheus-client** — metrics export
- **cachetools** — token-introspection TTL caching
- **defusedxml** — safe XML parsing (sitemap/URL-graph clients)
- **python-dotenv** — env-file loading
- **miniopy-async** (dev/optional) — MinIO backend, imported lazily by `storage`; consumers declare it

## Licensing

This is a shared library: it contains no licence enforcement of its own and calls no
licence server. Consumers run with no licence switch set, and unset means off.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
