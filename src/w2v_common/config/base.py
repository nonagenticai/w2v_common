"""Reusable Pydantic Settings sub-configurations for w2v services.

Services compose these into their own Settings class:

    from w2v_common.config import DatabaseSettings, RedisSettings, ...

    class Settings(BaseSettings):
        database: DatabaseSettings = Field(default_factory=DatabaseSettings)
        redis: RedisSettings = Field(default_factory=RedisSettings)
        # ... plus service-specific settings
"""

import json
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.exceptions import SettingsError

# Canonical uuid7 for the n8n-dev tenant. Aligns with kag, licenses,
# worker_management, requirements_management, scientific_processor — see
# /Users/mini1/Movies/worker_management/src/api/tenant_constants.py for the
# org-wide name->uuid mapping. The legacy human-readable string "n8n-dev"
# was retired in 2026-04-29 (kag commit f89aae3); w2v migrated 2026-04-30.
DEFAULT_TENANT_UUID = "0195f48b-d000-7000-8000-000000000001"

# REMOVED 2026-08-26 — ``UNATTRIBUTED_TENANT_UUID`` (the nil-UUID sentinel).
# Product-owner ruling: a ``tenant_id`` may NEVER hold a sentinel. Every row
# carries a real registered tenant forwarded from the actual requestor. The
# nil UUID was the most damaging shape possible: it is a *valid* UUID, so it
# survived every type check and every dirty-row audit while silently
# mis-attributing cost, and it is registered in ``licenses.tenants`` as
# ``default-anon`` so it also LOOKED legitimate. There is no replacement
# constant on purpose — when the requestor's tenant is absent the correct
# behaviour is to fail loud (401 at the edge, ``MissingTenantError`` in the
# cost bus), never to invent an identity.


class _CsvFriendlyEnvSource(EnvSettingsSource):
    """EnvSettingsSource that falls back to the raw string when JSON decode fails.

    Pydantic-settings v2.4+ calls json.loads() on list fields in decode_complex_value
    *before* field_validators run. This subclass catches SettingsError and returns
    the raw value, allowing field_validators (mode="before") to handle CSV-style strings.
    """

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        try:
            return super().decode_complex_value(field_name, field, value)
        except SettingsError:
            return value  # field_validator will normalise it


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    url: str = Field(description="PostgreSQL database connection URL")
    pool_size: int = Field(default=10, description="Database connection pool size")
    pool_min_size: int = Field(default=2, description="Minimum connections in pool")
    max_overflow: int = Field(default=20, description="Maximum connections beyond pool_size")
    pool_timeout: int = Field(
        default=30, description="Timeout in seconds for getting a connection from the pool"
    )
    pool_recycle: int = Field(
        default=3600, description="Time in seconds after which connections are recycled"
    )
    echo: bool = Field(default=False, description="Echo SQL statements for debugging")

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class AuthSettings(BaseSettings):
    """Authentication configuration settings."""

    enabled: bool = Field(default=True, description="Enable authentication middleware")
    ssl_verify: bool = Field(default=True, description="SSL verification for service connections")
    # REMOVED 2026-08-26 — ``default_tenant_id`` (env AUTH_DEFAULT_TENANT_ID /
    # DEFAULT_TENANT_ID). A configurable default tenant is a sentinel with a
    # config knob: whatever it is set to, requests that carry no real tenant
    # get attributed to it. The trusted-network synthetic-identity path (the
    # only reader) now rejects a request with no ``X-Tenant-Id`` instead.
    cache_maxsize: int = Field(default=1000, description="License cache max entries")
    cache_ttl: int = Field(default=300, description="License cache TTL in seconds")
    introspect_timeout: float = Field(default=2.0, description="License introspection timeout")
    trust_incluster_identity: bool = Field(
        default=False,
        description=(
            "When true, unauthenticated requests whose source IP is in "
            "TRUSTED_CIDRS (default EMPTY since 0.89.0 — no peer is trusted "
            "unless the networks are named explicitly) are granted a synthetic "
            "'internal' identity. Default OFF (fail-closed) as of 0.6.9 — phases 1 "
            "(X-Tenant-Id routing) and 2 (bare-client license-token plumbing) "
            "are shipped, so the trusted-network fallback is no longer load- "
            "bearing for any caller. Operators that genuinely need the legacy "
            "fail-open behavior can opt back in via "
            "AUTH_TRUST_INCLUSTER_IDENTITY=true. As of 0.6.10 this field is "
            "the single source of truth; the env var still feeds it via "
            "Pydantic settings."
        ),
    )
    delegating_subjects: str = Field(
        default="",
        description=(
            "Comma-separated subject IDs (UUIDs) of trusted service accounts "
            "permitted to assert an effective tenant via the X-Effective-Tenant-Id "
            "header (env AUTH_DELEGATING_SUBJECTS). When an introspected license "
            "token's subject is in this set, AuthMiddleware re-scopes the request to "
            "the header's tenant, so cost attribution + BYO resolution follow the "
            "run's real tenant instead of the service-account tenant. Empty "
            "(default) disables the side-channel entirely — a customer token's "
            "subject is never in this set, so a customer can never spoof another "
            "tenant. Configure on internal RECEIVER services only (worker_management, "
            "html_render, content, slide, bdd, video_*, analysis); the orchestrator "
            "only STAMPS the header and need not honour it. Kept a plain CSV string "
            "(not list[str]) to dodge pydantic-settings 2.14's broken complex-type "
            "env decode; split at read time in middleware._delegating_subjects()."
        ),
    )
    actor_asserting_subjects: str = Field(
        default="",
        description=(
            "Comma-separated introspected licence subject IDs (env "
            "AUTH_ACTOR_ASSERTING_SUBJECTS) permitted to assert WHO ACTED via the "
            "X-Actor-Id header on the licence lane. Used ONLY for actor trust — "
            "DELIBERATELY SEPARATE from delegating_subjects, which lets a subject "
            "re-scope the request's TENANT: an app subject that may name its actor "
            "must not thereby gain tenant re-scoping, and vice versa. A subject on "
            "AUTH_DELEGATING_SUBJECTS is NOT actor-trusted unless also listed here "
            "(no back-compat inheritance: before 0.82.0 no licence-lane caller could "
            "assert an actor at all). Empty (default) = nobody; X-Actor-Id from a "
            "licence caller is ignored. Plain CSV string for the same reason as "
            "delegating_subjects."
        ),
    )
    attribution_forward_headers: bool = Field(
        default=False,
        description=(
            "When true (env AUTH_ATTRIBUTION_FORWARD_HEADERS), OUTBOUND calls carry "
            "the bound fr project as X-Project-Id (identity_context."
            "get_identity_headers and identity_event_hook). Default OFF. Since "
            "0.86.0 this flag gates X-Project-Id ONLY: a bound UUID actor is "
            "always forwarded as X-Actor-Id (receivers honour it only on a "
            "trusted lane). Inbound binding and cost-event project_id/actor_id "
            "are unaffected by this flag; it gates the wire only."
        ),
    )
    effective_tenant_unbilled_prefixes: str = Field(
        default="",
        description=(
            "Comma-separated request-path prefixes (env "
            "AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES) of routes that NEVER bill — "
            "e.g. a worker poll like '/api/tts/claim'. A delegating caller (subject "
            "in AUTH_DELEGATING_SUBJECTS) that omits X-Effective-Tenant-Id on such a "
            "path is not warned about and not counted in "
            "w2v_auth_effective_tenant_missing_total, because no cost is being "
            "mis-attributed. Matched at a path-SEGMENT boundary "
            "(middleware.matches_public_prefix): '/api/tts/' covers '/api/tts/claim' "
            "but never '/api/ttsx/claim'. Observability only — authentication and "
            "tenant resolution are unchanged, and a header that IS sent on such a "
            "path is still honoured. Empty (default) keeps the warning on every "
            "path. Plain CSV string for the same reason as delegating_subjects."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="AUTH_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class KeycloakSettings(BaseSettings):
    """Keycloak JWT validation configuration.

    JWT validation is only enabled when ``url`` is set. All fields read from
    ``KEYCLOAK_*`` env vars and mirror what :mod:`w2v_common.auth.keycloak`
    reads directly (kept here so services wanting typed access can depend on
    this instead of :func:`os.getenv`).
    """

    url: str | None = Field(
        default=None, description="Keycloak base URL, e.g. https://auth.nonagentic.ai"
    )
    realm: str = Field(default="portal", description="Keycloak realm name")
    service_realm: str = Field(
        default="",
        description=(
            "KEYCLOAK_SERVICE_REALM — realm the w2v-backend S2S token is minted from "
            "(realm-per-app P7: secure-apps). Empty = KEYCLOAK_REALM."
        ),
    )
    audiences: list[str] = Field(
        default=["w2v-backend", "w2v-frontend"],
        description="Accepted JWT audiences (comma-separated in env var)",
    )
    jwks_ttl: int = Field(default=900, description="JWKS cache TTL in seconds")
    jwks_timeout: float = Field(default=2.0, description="JWKS HTTP fetch timeout in seconds")
    trusted_issuers: str = Field(
        default="",
        description=(
            "KEYCLOAK_TRUSTED_ISSUERS — CSV of full issuer URLs; empty = the single "
            "KEYCLOAK_ISSUER (unchanged behaviour). Parsed by auth.keycloak."
        ),
    )
    service_only_realms: str = Field(
        default="secure-apps",
        description=(
            "KEYCLOAK_SERVICE_ONLY_REALMS — CSV of realms never honoured as a user "
            "session (applies only with an explicit KEYCLOAK_TRUSTED_ISSUERS)."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="KEYCLOAK_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    url: str = Field(description="Redis connection URL")
    max_connections: int = Field(default=10, description="Maximum Redis connections in pool")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout in seconds")
    decode_responses: bool = Field(
        default=True, description="Decode responses from bytes to strings"
    )
    health_check_interval: int = Field(
        default=0,
        description=(
            "Seconds between health-checks of idle pool connections; 0 disables. "
            "Recycles stale connections after a Redis restart."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="REDIS_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class APISettings(BaseSettings):
    """API server configuration settings."""

    host: str = Field(default="0.0.0.0", description="API server host")  # nosec B104  # noqa: S104 -- containerized service requires 0.0.0.0
    port: int = Field(
        default=8080,
        description=(
            "API server port. Default 8080 (NOT 8000 — port 8000 is reserved "
            "by the user's local mother service and is rejected by the post-init "
            "validator)."
        ),
    )
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:*",
            "http://127.0.0.1:*",
        ],
        description="CORS allowed origins (JSON array in env vars)",
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: list[str] = Field(
        default=["*"], description="Allowed HTTP methods for CORS"
    )
    cors_allow_headers: list[str] = Field(
        default=["*"], description="Allowed HTTP headers for CORS"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload on code changes")

    model_config = SettingsConfigDict(
        env_prefix="API_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Reject port 8000 (reserved for the user's local mother service).

        The port=8000 rejection is the load-bearing rule documented at
        ``port`` Field above and in ``CLAUDE.md``. The 1..65535 range check
        is a defensive sanity tightening — pydantic accepts any int by
        default and a bad config like ``API_PORT=-1`` should fail loudly
        rather than reaching ``uvicorn.run``.
        """
        if v == 8000:
            raise ValueError(
                "API_PORT 8000 is reserved (user's local mother service); pick another port"
            )
        if not (1 <= v <= 65535):
            raise ValueError(f"API_PORT {v} is out of range (must be 1..65535)")
        return v

    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> list[str]:
        """Parse comma-separated or JSON strings into lists for CORS fields."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError, ValueError:
                pass
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(item) for item in v]
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use lenient env source so CSV-format list fields don't break on JSON parse."""
        return (
            init_settings,
            _CsvFriendlyEnvSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


class StorageSettings(BaseSettings):
    """File storage configuration."""

    backend: str = Field(default="local", description="Storage backend: local or minio")
    local_path: str = Field(default="/app/data", description="Local storage path")
    minio_endpoint: str = Field(
        default="",
        description="MinIO endpoint (host:port). Required when backend=='minio'.",
    )
    minio_access_key: str = Field(default="", description="MinIO access key")
    minio_secret_key: str = Field(default="", description="MinIO secret key")
    minio_bucket: str = Field(default="w2v-data", description="MinIO bucket name")
    minio_secure: bool = Field(default=False, description="Use HTTPS for MinIO")
    minio_external_endpoint: str | None = Field(
        default=None,
        description=(
            "External MinIO endpoint (host[:port] or scheme://host[:port]). When set, the "
            "in-cluster host in a presigned URL is rewritten to this so clients outside "
            "Docker/k8s can reach MinIO directly. Optional."
        ),
    )
    presign_expiry_seconds: int = Field(
        default=3600, description="MinIO presigned URL expiry in seconds"
    )

    @model_validator(mode="after")
    def _check_minio_endpoint_when_backend_is_minio(self) -> StorageSettings:
        if self.backend == "minio" and not self.minio_endpoint:
            raise ValueError("minio_endpoint is required when STORAGE_BACKEND=minio")
        return self

    model_config = SettingsConfigDict(
        env_prefix="STORAGE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Date format for logs")
    json_logs: bool = Field(default=False, description="Output logs in JSON format")

    @field_validator("level", mode="before")
    @classmethod
    def validate_level(cls, v: Any) -> str:
        """Validate and uppercase the logging level string."""
        if isinstance(v, str):
            v = v.upper()
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v not in valid_levels:
                raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return str(v)

    model_config = SettingsConfigDict(
        env_prefix="LOG_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CacheSettings(BaseSettings):
    """In-memory and Redis cache configuration."""

    job_ttl: int = Field(default=3600, description="Job cache TTL in seconds")
    job_maxsize: int = Field(default=1000, description="Job cache max entries")
    license_ttl: int = Field(default=300, description="License introspection cache TTL in seconds")
    license_maxsize: int = Field(
        default=1000, description="License introspection cache max entries"
    )
    enabled: bool = Field(default=True, description="Enable caching")
    key_prefix: str = Field(default="", description="Redis key prefix")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")

    model_config = SettingsConfigDict(
        env_prefix="CACHE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class IdempotencySettings(BaseSettings):
    """Idempotency-Key middleware configuration.

    Opt-in: a service composes this onto its Settings and sets
    ``IDEMPOTENCY_ENABLED=true`` to have :func:`w2v_common.asgi.create_app`
    register :class:`w2v_common.middleware.IdempotencyMiddleware`. The service
    must also assign a connected cache to ``app.state.redis`` (e.g. in an
    ``on_startup`` hook); the middleware falls open when it is absent.
    """

    enabled: bool = Field(
        default=False, description="Register the Idempotency-Key POST-replay middleware"
    )
    ttl_seconds: int = Field(
        default=86400, description="TTL for cached POST responses keyed by ``Idempotency-Key``"
    )

    model_config = SettingsConfigDict(
        env_prefix="IDEMPOTENCY_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class MetricsSettings(BaseSettings):
    """Prometheus HTTP-metrics configuration.

    Opt-in: a service composes this onto its Settings and sets
    ``METRICS_ENABLED=true`` to have :func:`w2v_common.asgi.create_app`
    register :class:`w2v_common.middleware.MetricsMiddleware` and mount the
    auth-exempt ``GET /metrics`` endpoint.
    """

    enabled: bool = Field(
        default=False, description="Register MetricsMiddleware + mount GET /metrics"
    )

    model_config = SettingsConfigDict(
        env_prefix="METRICS_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class AuditSettings(BaseSettings):
    """Security audit-logging middleware configuration.

    Opt-in: a service composes this onto its Settings and sets
    ``AUDIT_ENABLED=true`` to have :func:`w2v_common.asgi.create_app` register
    :class:`w2v_common.middleware.AuditMiddleware` (outer than auth, so 401/403
    are captured). ``logger_name`` lets each service route audit records to its
    own logger.
    """

    enabled: bool = Field(default=False, description="Register the AuditMiddleware")
    logger_name: str = Field(
        default="w2v.audit", description="Logger name for audit/auth-failure records"
    )

    model_config = SettingsConfigDict(
        env_prefix="AUDIT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class TracingSettings(BaseSettings):
    """OpenTelemetry tracing configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    endpoint: str = Field(default="", description="OTLP exporter endpoint")
    sample_rate: float = Field(default=1.0, description="Trace sampling rate (0.0-1.0)")

    model_config = SettingsConfigDict(
        env_prefix="TRACING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class LLMSettings(BaseSettings):
    """Provider-agnostic LLM configuration.

    Code uses an OpenAI-compatible client (base_url + api_key + model). To
    switch providers (local llama.cpp/vLLM/ollama, OpenAI, Anthropic, groq,
    ...), edit the gitops configmap + secret. w2v_common holds no provider
    identity — that lives in deployment configuration only.
    """

    # Nullable: local providers may not require a key.
    api_key: str | None = Field(default=None, description="LLM API key (None for local)")
    # NO default — fail loud if env var is missing in production.
    base_url: str = Field(description="OpenAI-compatible API base URL")
    # NO default — fail loud if env var is missing in production.
    model: str = Field(description="Default LLM model")
    # 4000 leaves 4192 tokens of prompt headroom against vLLM's
    # --max-model-len 8192 on pgx-local (Nemotron). Override via LLM_MAX_TOKENS
    # for deployments that target a non-vLLM provider with a larger context.
    max_tokens: int = Field(default=4000, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="LLM temperature")
    timeout: int = Field(default=60, description="LLM API timeout in seconds")

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class FFmpegSettings(BaseSettings):
    """FFmpeg operation configuration."""

    timeout: int = Field(default=300, description="FFmpeg command timeout in seconds")
    threads: int = Field(default=1, description="FFmpeg thread count")
    preset: str = Field(default="medium", description="H.264 encoding preset")
    ffprobe_timeout: int = Field(default=10, description="ffprobe timeout in seconds")
    default_crf: int = Field(default=23, description="Default CRF for H.264 encoding")
    error_truncate_length: int = Field(
        default=500, description="Max chars for FFmpeg error messages"
    )

    model_config = SettingsConfigDict(
        env_prefix="FFMPEG_", env_file=".env", case_sensitive=False, extra="ignore"
    )
