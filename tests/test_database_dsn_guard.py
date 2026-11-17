"""An unconfigured DSN must fail loud -- there is no localhost/superuser default DSN."""

from __future__ import annotations

import importlib

import pytest


def _reloaded(monkeypatch: pytest.MonkeyPatch, **env: str | None):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    import w2v_common.database.connection as conn

    return importlib.reload(conn)


@pytest.mark.asyncio
async def test_unset_database_url_refuses_to_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _reloaded(monkeypatch, DATABASE_URL=None)
    assert conn._DATABASE_URL_IS_UNSET is True
    assert conn.DATABASE_URL == ""
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        await conn.create_engine_with_retry(max_retries=1, retry_delay=0.0)


def test_the_old_insecure_opt_in_no_longer_unlocks_a_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ALLOW_INSECURE_LOCAL_DB`` used to unlock a localhost superuser DSN; it is inert now."""
    conn = _reloaded(monkeypatch, DATABASE_URL=None, ALLOW_INSECURE_LOCAL_DB="1")
    assert "localhost" not in conn.DATABASE_URL
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        conn._assert_dsn_is_configured()


def test_an_empty_database_url_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _reloaded(monkeypatch, DATABASE_URL="")
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        conn._assert_dsn_is_configured()


def test_a_configured_dsn_is_never_treated_as_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _reloaded(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg://app:pw@db.example:5432/svc",  # pragma: allowlist secret
    )
    assert conn._DATABASE_URL_IS_UNSET is False
    conn._assert_dsn_is_configured()  # must not raise
