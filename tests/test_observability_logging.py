"""Tests for `w2v_common.observability.logging.configure_logging`."""

from __future__ import annotations

import io
import json
import logging
from unittest.mock import MagicMock

from w2v_common.config.base import LoggingSettings
from w2v_common.observability.logging import (
    JsonFormatter,
    configure_logging,
    database_endpoint,
    redis_endpoint,
)


class TestConfigureLogging:
    """Behavioral tests for the centralized logging configurator."""

    def test_no_args_defaults_to_info(self):
        """`configure_logging()` with no args should not raise and sets INFO."""
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_settings_level_debug(self):
        """An explicit `LoggingSettings(level="DEBUG")` sets root to DEBUG."""
        configure_logging(LoggingSettings(level="DEBUG"))
        assert logging.getLogger().level == logging.DEBUG

    def test_keyword_level_override(self):
        """`level=` kwarg overrides the settings value."""
        configure_logging(LoggingSettings(level="INFO"), level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_json_logs_emits_json(self):
        """`json_logs=True` produces a JSON-shaped log line on the handler."""
        configure_logging(json_logs=True, level="INFO")
        root = logging.getLogger()
        # Replace stdout handler stream with a capturable buffer.
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert isinstance(handler.formatter, JsonFormatter)

        buffer = io.StringIO()
        handler.stream = buffer

        logger = logging.getLogger("w2v_common.tests.json")
        logger.info("hello-json")
        handler.flush()

        line = buffer.getvalue().strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "w2v_common.tests.json"
        assert payload["message"] == "hello-json"
        assert "timestamp" in payload

    def test_idempotent_no_duplicate_handlers(self):
        """Calling `configure_logging` multiple times does not stack handlers."""
        configure_logging()
        configure_logging()
        configure_logging(LoggingSettings(level="DEBUG"))
        root = logging.getLogger()
        assert len(root.handlers) == 1


# An OBVIOUSLY fake password. It must never be a value that resembles a real
# credential: this string is committed, printed by pytest on failure, and ends up
# in CI logs — the exact places this whole change exists to keep secrets out of.
_FAKE_PASSWORD = "NOT_A_REAL_PASSWORD_fake123"  # pragma: allowlist secret
_FAKE_DSN = f"postgresql+psycopg://svc_app:{_FAKE_PASSWORD}@postgres.example:5432/svc_db"  # pragma: allowlist secret
_FAKE_REDIS_URL = (
    f"redis://default:{_FAKE_PASSWORD}@redis.example:6379/0"  # pragma: allowlist secret
)


class TestDatabaseEndpoint:
    """`database_endpoint` renders a DSN for logging without its userinfo.

    The bug being closed: nine services log `settings.database.url` raw at INFO
    on the connect path, so the live app password lands in every pod's stdout and
    in log aggregation indefinitely.
    """

    def test_password_is_absent_from_captured_info_log(self):
        """THE regression test: the DSN password must not reach INFO log output.

        Asserted against real captured handler output rather than the helper's
        return value, because the return value is not what leaks — the *log line*
        is. Mutating the `logger.info` call back to `%s`, settings.database.url`
        (i.e. dropping the `database_endpoint(...)` wrapper) must fail HERE.
        """
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        logger = logging.getLogger("w2v_common.tests.dsn")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("Database connected: %s", database_endpoint(_FAKE_DSN))
        handler.flush()

        captured = buffer.getvalue()
        assert _FAKE_PASSWORD not in captured
        assert "svc_app" not in captured
        # Still USEFUL, not merely safe — an operator must be able to tell which
        # database was reached, or the line would just be deleted instead.
        assert "postgres.example:5432/svc_db" in captured

    def test_renders_scheme_host_port_dbname(self):
        """The driver assertion and the endpoint survive; the userinfo does not."""
        assert database_endpoint(_FAKE_DSN) == "postgresql+psycopg://postgres.example:5432/svc_db"

    def test_omits_port_when_absent(self):
        """A DSN with no explicit port renders without a dangling colon."""
        assert database_endpoint("postgresql://db.example/svc") == "postgresql://db.example/svc"

    def test_unparseable_fails_closed_not_open(self):
        """A bad DSN renders a sentinel, never the raw URL.

        A redactor that falls back to the original on a parse error leaks
        precisely when it is least expected.
        """
        assert database_endpoint("postgresql://host:99999/db") == "<unparseable>"
        assert database_endpoint("") == "<unparseable>"
        assert database_endpoint("not a dsn at all") == "<unparseable>"

    def test_never_returns_the_input_on_failure(self):
        """Totality check across malformed inputs: the input is never echoed."""
        for bad in (f"://{_FAKE_PASSWORD}@", f"postgresql://u:{_FAKE_PASSWORD}@:99999/d"):
            assert _FAKE_PASSWORD not in database_endpoint(bad)


class TestRedisEndpoint:
    """`redis_endpoint` renders a Redis DSN for logging without its userinfo.

    The bug being closed: eleven services log `settings.redis.url` raw at INFO on
    the connect path, one function BELOW the `database_endpoint`-guarded DSN log
    fixed in 0.67.24 — guarded beside unguarded. A Redis URL carries its password
    in exactly the same `user:pass@` slot a Postgres DSN does.
    """

    def test_password_is_absent_from_captured_info_log(self):
        """THE regression test: the Redis password must not reach INFO output.

        Asserted against real captured handler output, not the helper's return
        value, because the return value is not what leaks — the *log line* is.
        Mutating the call back to `logger.info("...", settings.redis.url)` (i.e.
        dropping the `redis_endpoint(...)` wrapper) must fail HERE.

        `setLevel` is used deliberately, NOT `monkeypatch.setattr(logger,
        "level", INFO)`: `logging.Logger` caches `isEnabledFor`, and only
        `setLevel()` invalidates that cache. A stale "INFO disabled" entry makes
        the record never reach the handler, and then `assert password not in
        captured` passes on EMPTY output with the leak fully intact. The
        `assert "redis.example:6379" in captured` below is the positive control
        that makes such a vacuous pass impossible.
        """
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        logger = logging.getLogger("w2v_common.tests.redis_dsn")
        previous_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info("Redis connected: %s", redis_endpoint(_FAKE_REDIS_URL))
            handler.flush()
        finally:
            logger.setLevel(previous_level)
            logger.handlers = []

        captured = buffer.getvalue()
        # POSITIVE CONTROL — proves a record actually reached the handler, so
        # none of the negative assertions below can pass on empty output.
        assert "redis.example:6379" in captured
        assert _FAKE_PASSWORD not in captured
        assert "default" not in captured

    def test_renders_host_and_port(self):
        """The endpoint survives; the userinfo, scheme and db index do not."""
        assert redis_endpoint(_FAKE_REDIS_URL) == "redis.example:6379"

    def test_omits_port_when_absent(self):
        """A URL with no explicit port renders without a dangling colon."""
        assert redis_endpoint("redis://redis.example/0") == "redis.example"

    def test_unparseable_fails_closed_not_open(self):
        """A bad URL renders a sentinel, never the raw URL."""
        assert redis_endpoint("redis://host:99999/0") == "<unparseable>"
        assert redis_endpoint("") == "<unparseable>"
        assert redis_endpoint("not a url at all") == "<unparseable>"

    def test_non_str_input_does_not_raise(self):
        """A mocked-settings test hands this a `MagicMock`; boot must survive.

        This is why the `except` is `Exception` and not `ValueError` — the narrow
        form broke six lifespan tests in `ai_content_generation_service`.
        """
        assert redis_endpoint(MagicMock()) == "<unparseable>"

    def test_never_returns_the_input_on_failure(self):
        """Totality check across malformed inputs: the input is never echoed."""
        for bad in (f"://{_FAKE_PASSWORD}@", f"redis://u:{_FAKE_PASSWORD}@:99999/0"):
            assert _FAKE_PASSWORD not in redis_endpoint(bad)

    def test_matches_the_private_cost_bus_alias(self):
        """`cost_bus._redis_endpoint` is now a thin alias — behaviour is identical.

        Guards the re-point: if the public helper's SHAPE ever drifts (a scheme
        or `/db` suffix added to mirror `database_endpoint`), metering's boot log
        line changes silently. This fails first.
        """
        from w2v_common.metering import cost_bus

        for candidate in (_FAKE_REDIS_URL, "redis://redis.example/0", "", "redis://h:99999/0"):
            assert cost_bus._redis_endpoint(candidate) == redis_endpoint(candidate)
