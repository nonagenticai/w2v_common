"""Tests for w2v_common package imports."""

import inspect


class TestPackageImports:
    """Tests for top-level and subpackage imports."""

    def test_version(self):
        from importlib.metadata import version

        import w2v_common

        # __version__ is read from package metadata, so it should match the
        # installed distribution's version (driven by pyproject.toml).
        assert w2v_common.__version__ == version("w2v-common")
        assert w2v_common.__version__ != "unknown"

    def test_auth_imports(self):
        from w2v_common.auth import AuthMiddleware, User

        assert inspect.isclass(AuthMiddleware), "AuthMiddleware should be a class"
        assert inspect.isclass(User), "User should be a class"
        # Verify they're the correct classes (not accidentally re-exported wrong things)
        assert issubclass(User, __import__("pydantic").BaseModel)
        assert hasattr(AuthMiddleware, "dispatch"), "AuthMiddleware should have dispatch method"

    def test_cache_imports(self):
        from w2v_common.cache import RedisClient, close_redis, get_redis_client, init_redis

        assert inspect.isclass(RedisClient), "RedisClient should be a class"
        assert hasattr(RedisClient, "connect"), "RedisClient should have connect method"
        assert hasattr(RedisClient, "disconnect"), "RedisClient should have disconnect method"
        assert inspect.iscoroutinefunction(close_redis), "close_redis should be async"
        assert inspect.iscoroutinefunction(init_redis), "init_redis should be async"
        assert callable(get_redis_client), "get_redis_client should be callable"

    def test_database_imports(self):
        from w2v_common.database import (
            close_engine,
            close_session_maker,
            get_db,
            get_db_session,
            get_engine,
            get_session_maker,
            health_check,
        )

        assert inspect.iscoroutinefunction(close_engine), "close_engine should be async"
        assert inspect.iscoroutinefunction(close_session_maker), (
            "close_session_maker should be async"
        )
        assert inspect.isasyncgenfunction(get_db), "get_db should be an async generator"
        assert callable(get_db_session), "get_db_session should be callable"
        assert inspect.iscoroutinefunction(get_engine), "get_engine should be async"
        assert inspect.iscoroutinefunction(get_session_maker), "get_session_maker should be async"
        assert inspect.iscoroutinefunction(health_check), "health_check should be async"

    def test_config_imports(self):
        from pydantic_settings import BaseSettings

        from w2v_common.config import (
            APISettings,
            AuthSettings,
            CacheSettings,
            DatabaseSettings,
            LoggingSettings,
            RedisSettings,
            StorageSettings,
            TracingSettings,
        )

        for cls in (
            APISettings,
            AuthSettings,
            CacheSettings,
            DatabaseSettings,
            LoggingSettings,
            RedisSettings,
            StorageSettings,
            TracingSettings,
        ):
            assert inspect.isclass(cls), f"{cls.__name__} should be a class"
            assert issubclass(cls, BaseSettings), (
                f"{cls.__name__} should be a BaseSettings subclass"
            )

    def test_middleware_imports(self):
        from starlette.middleware.base import BaseHTTPMiddleware

        from w2v_common.middleware import RequestIdMiddleware, SecurityHeadersMiddleware

        assert inspect.isclass(RequestIdMiddleware)
        assert inspect.isclass(SecurityHeadersMiddleware)
        assert issubclass(RequestIdMiddleware, BaseHTTPMiddleware)
        assert issubclass(SecurityHeadersMiddleware, BaseHTTPMiddleware)
        assert hasattr(RequestIdMiddleware, "dispatch")
        assert hasattr(SecurityHeadersMiddleware, "dispatch")
