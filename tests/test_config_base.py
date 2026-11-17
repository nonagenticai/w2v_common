"""Tests for w2v_common.config.base settings classes."""

import pytest

from w2v_common.config.base import (
    APISettings,
    AuthSettings,
    CacheSettings,
    DatabaseSettings,
    LoggingSettings,
    RedisSettings,
    StorageSettings,
    TracingSettings,
)


class TestDatabaseSettings:
    """Tests for DatabaseSettings."""

    def test_pool_defaults(self):
        # url is required (no localhost fallback); pass via kwarg.
        db_url = "postgresql+psycopg://u:p@h:5432/d"  # pragma: allowlist secret
        settings = DatabaseSettings(url=db_url)
        assert settings.pool_size == 10
        assert settings.max_overflow == 20
        assert settings.pool_timeout == 30
        assert settings.pool_recycle == 3600
        assert settings.echo is False

    def test_url_required(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="url"):
            DatabaseSettings()  # ty: ignore[missing-argument]

    def test_custom_values(self):
        db_url = "postgresql+psycopg://user:pass@host:5432/db"  # pragma: allowlist secret
        settings = DatabaseSettings(
            url=db_url,
            pool_size=5,
            max_overflow=10,
        )
        assert settings.url == db_url
        assert settings.pool_size == 5
        assert settings.max_overflow == 10


class TestAuthSettings:
    """Tests for AuthSettings."""

    def test_defaults(self):
        settings = AuthSettings()
        assert settings.enabled is True
        assert settings.ssl_verify is True

    def test_no_default_tenant_knob_exists(self):
        """A configurable default tenant IS a sentinel with a config knob.

        ``AuthSettings.default_tenant_id`` (env AUTH_DEFAULT_TENANT_ID /
        DEFAULT_TENANT_ID) defaulted to the nil-UUID and was applied whenever no
        real tenant resolved. Removed in 0.67.0 — assert it cannot come back,
        including via env (``extra="ignore"`` must not resurrect it as an attr).
        """
        assert not hasattr(AuthSettings(), "default_tenant_id")
        assert "default_tenant_id" not in AuthSettings.model_fields

    def test_middleware_has_no_default_tenant_fallback(self):
        """middleware.DEFAULT_TENANT_ID must not exist at all.

        It used to resolve AUTH_DEFAULT_TENANT_ID / DEFAULT_TENANT_ID / the
        nil-UUID and was substituted on the trusted-network lane. That lane now
        rejects a request with no usable X-Tenant-Id.
        """
        import importlib

        from w2v_common.auth import middleware as _mw

        importlib.reload(_mw)
        assert not hasattr(_mw, "DEFAULT_TENANT_ID")


class TestRedisSettings:
    """Tests for RedisSettings."""

    def test_pool_defaults(self, monkeypatch):
        monkeypatch.delenv("REDIS_HEALTH_CHECK_INTERVAL", raising=False)
        # url is required (no localhost fallback); pass via kwarg.
        settings = RedisSettings(url="redis://r:6379/0")
        assert settings.max_connections == 10
        assert settings.socket_timeout == 5
        assert settings.socket_connect_timeout == 5
        assert settings.decode_responses is True
        assert settings.health_check_interval == 0

    def test_url_required(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(ValueError, match="url"):
            RedisSettings()  # ty: ignore[missing-argument]

    def test_custom_values(self):
        settings = RedisSettings(url="redis://redis:6380/1", max_connections=20)
        assert settings.url == "redis://redis:6380/1"
        assert settings.max_connections == 20

    def test_health_check_interval_from_env(self, monkeypatch):
        """Services opt in to periodic pool health checks via env var."""
        monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "30")
        monkeypatch.setenv("REDIS_URL", "redis://r:6379/0")
        settings = RedisSettings()  # ty: ignore[missing-argument]
        assert settings.health_check_interval == 30


class TestAPISettings:
    """Tests for APISettings."""

    def test_defaults(self):
        settings = APISettings()
        assert settings.host == "0.0.0.0"
        # Default port is 8080, NOT 8000 — port 8000 is reserved by the
        # user's local mother service and is rejected at validation time.
        assert settings.port == 8080
        assert settings.debug is False
        assert settings.reload is False
        assert settings.cors_allow_credentials is True
        assert isinstance(settings.cors_origins, list)
        assert isinstance(settings.cors_allow_methods, list)
        assert isinstance(settings.cors_allow_headers, list)

    def test_cors_origins_list(self):
        origins = ["http://localhost:3000", "http://localhost:8080"]
        settings = APISettings(cors_origins=origins)
        assert settings.cors_origins == origins

    def test_cors_origins_single(self):
        origins = ["http://localhost:3000"]
        settings = APISettings(cors_origins=origins)
        assert settings.cors_origins == origins

    def test_cors_origins_from_env_json(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("API_CORS_ORIGINS", '["http://a.com","http://b.com"]')
        settings = APISettings()  # env file disabled; monkeypatch injects API_CORS_ORIGINS
        assert settings.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_methods_list(self):
        settings = APISettings(cors_allow_methods=["GET", "POST", "PUT"])
        assert settings.cors_allow_methods == ["GET", "POST", "PUT"]

    def test_cors_headers_list(self):
        settings = APISettings(cors_allow_headers=["Content-Type", "Authorization"])
        assert settings.cors_allow_headers == ["Content-Type", "Authorization"]

    def test_port_8000_rejected(self):
        # Port 8000 is reserved by the user's local mother service. The
        # field description has long claimed it is rejected; this validator
        # makes that promise enforceable. Pydantic v2 wraps validator
        # ValueErrors in ValidationError (a ValueError subclass), so
        # matching on ValueError + message text mirrors the existing
        # ``test_validate_level_invalid`` style.
        with pytest.raises(ValueError, match="API_PORT 8000 is reserved"):
            APISettings(port=8000)

    def test_port_non_8000_accepted(self):
        # Sanity: alternative ports (incl. tts_service's 6303) succeed.
        assert APISettings(port=8080).port == 8080
        assert APISettings(port=6303).port == 6303


class TestStorageSettings:
    """Tests for StorageSettings."""

    def test_field_defaults(self, monkeypatch):
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)
        monkeypatch.delenv("STORAGE_MINIO_ENDPOINT", raising=False)
        settings = StorageSettings()
        assert settings.backend == "local"
        assert settings.local_path == "/app/data"
        assert settings.minio_endpoint == ""
        assert settings.minio_access_key == ""
        assert settings.minio_secret_key == ""
        assert settings.minio_bucket == "w2v-data"
        assert settings.minio_secure is False

    def test_local_backend_without_minio_endpoint_constructs(self, monkeypatch):
        """Local-only consumers (e.g. slide-service) must not require minio_endpoint."""
        monkeypatch.delenv("STORAGE_MINIO_ENDPOINT", raising=False)
        settings = StorageSettings(backend="local")
        assert settings.backend == "local"
        assert settings.minio_endpoint == ""

    def test_minio_backend_requires_minio_endpoint(self, monkeypatch):
        monkeypatch.delenv("STORAGE_MINIO_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="minio_endpoint is required"):
            StorageSettings(backend="minio")

    def test_minio_backend_with_endpoint_ok(self):
        settings = StorageSettings(backend="minio", minio_endpoint="minio.example.com:8389")
        assert settings.backend == "minio"
        assert settings.minio_endpoint == "minio.example.com:8389"


class TestLoggingSettings:
    """Tests for LoggingSettings."""

    def test_defaults(self):
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert "asctime" in settings.format
        assert settings.json_logs is False

    def test_validate_level_lowercase(self):
        settings = LoggingSettings(level="debug")
        assert settings.level == "DEBUG"

    def test_validate_level_invalid(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingSettings(level="VERBOSE")


class TestCacheSettings:
    """Tests for CacheSettings."""

    def test_defaults(self):
        settings = CacheSettings()
        assert settings.job_ttl == 3600
        assert settings.job_maxsize == 1000
        assert settings.license_ttl == 300
        assert settings.license_maxsize == 1000


class TestTracingSettings:
    """Tests for TracingSettings."""

    def test_defaults(self):
        settings = TracingSettings()
        assert settings.enabled is False
        assert settings.endpoint == ""
        assert settings.sample_rate == 1.0

    def test_custom_values(self):
        settings = TracingSettings(enabled=True, endpoint="http://jaeger:4317", sample_rate=0.5)
        assert settings.enabled is True
        assert settings.endpoint == "http://jaeger:4317"
        assert settings.sample_rate == 0.5
