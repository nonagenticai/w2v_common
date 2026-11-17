"""Shared test utilities for w2v microservices.

Import only in test code. Currently exposes the fakeredis injector helper.
SQLite/aiosqlite helpers were removed in 0.8.0 — every w2v service now
runs its tests against a real postgres database (per the cluster
standard "no SQLite anywhere"). Each service's `tests/conftest.py`
inlines `create_async_engine(_TEST_DATABASE_URL)` + `Base.metadata.create_all`
+ TRUNCATE-RESTART-IDENTITY-CASCADE for per-test isolation; see
`tts_service/tests/conftest.py` for the canonical shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fakeredis.aioredis

if TYPE_CHECKING:
    from w2v_common.cache.redis_client import RedisClient


def inject_fakeredis(client: RedisClient) -> fakeredis.aioredis.FakeRedis:
    """Inject a fakeredis async client into a RedisClient.

    Sets both _client and _pool so is_available returns True and
    disconnect() can clean up without errors.
    """
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    client._client = fake
    client._pool = fake.connection_pool
    return fake
