"""0.94.0: optional top-level ``funding`` on cost events.

billing-service ``cost_flusher`` reads ``c.get("funding")`` off the wire and
honours a value in its ``FUNDING_VALUES`` verbatim (never re-derived), so the
emitter must (a) only send it when the caller says so and (b) never send a
value outside the set.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, cast

import fakeredis.aioredis
import pytest

from w2v_common.metering import (
    FUNDING_VALUES,
    InvalidFundingError,
    emit_cost_event,
    resolve_stream,
    schedule_cost_event,
)

_TENANT = uuid.UUID("0195f48b-d000-7000-8000-0000000000aa")


@pytest.fixture
async def redis_conn():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        service_key="w2v-test",  # pragma: allowlist secret -- routing slug
        cost_source="w2v-test",
        tenant_id=_TENANT,
        provider="openai",
        model="gpt-x",
        input_tokens=1,
        output_tokens=2,
        latency_ms=3,
    )
    base.update(overrides)
    return base


async def _only_event(conn: fakeredis.aioredis.FakeRedis) -> dict[str, str]:
    entries = await conn.xrange(resolve_stream())
    assert len(entries) == 1
    return entries[0][1]


def test_the_taxonomy_matches_billing() -> None:
    assert frozenset({"local_gpu", "platform_external", "tenant_byo"}) == FUNDING_VALUES


@pytest.mark.parametrize("funding", sorted(FUNDING_VALUES))
async def test_funding_is_emitted_top_level_when_given(
    redis_conn: fakeredis.aioredis.FakeRedis, funding: str
) -> None:
    await emit_cost_event(redis_conn, **_kwargs(funding=funding))

    fields = await _only_event(redis_conn)
    assert fields["funding"] == funding
    assert "funding" not in fields["usage_extra"]


async def test_funding_key_is_omitted_when_not_given(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    await emit_cost_event(redis_conn, **_kwargs())

    assert "funding" not in await _only_event(redis_conn)


async def test_schedule_forwards_funding(redis_conn: fakeredis.aioredis.FakeRedis) -> None:
    assert schedule_cost_event(redis_conn, **_kwargs(funding="tenant_byo"))
    for _ in range(20):
        await asyncio.sleep(0)
        if await redis_conn.xlen(resolve_stream()):
            break

    assert (await _only_event(redis_conn))["funding"] == "tenant_byo"


@pytest.mark.parametrize("bad", ["byo", "", "LOCAL_GPU", "unknown"])
async def test_invalid_funding_raises_and_emits_nothing(
    redis_conn: fakeredis.aioredis.FakeRedis, bad: str
) -> None:
    with pytest.raises(InvalidFundingError):
        await emit_cost_event(redis_conn, **_kwargs(funding=cast(Any, bad)))

    assert await redis_conn.xlen(resolve_stream()) == 0


async def test_schedule_rejects_invalid_funding_synchronously(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    with pytest.raises(InvalidFundingError):
        schedule_cost_event(redis_conn, **_kwargs(funding=cast(Any, "byo")))
    await asyncio.sleep(0)

    assert await redis_conn.xlen(resolve_stream()) == 0


def test_invalid_funding_is_a_value_error() -> None:
    assert issubclass(InvalidFundingError, ValueError)
