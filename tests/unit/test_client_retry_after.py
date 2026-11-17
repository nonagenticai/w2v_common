"""``Retry-After`` honouring + the dedicated 429 retry budget.

Production context: 22 orchestrator runs died in a 429 burst on
``POST /api/test-runner/execute``. Retry was NOT missing -- 429 is in
``_RETRYABLE_STATUS_CODES`` and the POST carried an idempotency key -- but the
budget (3 attempts, linear 1s => 1s + 2s = 3s of waiting) could never outlive
a limiter with a 60-second sliding window, and the server's ``Retry-After``
was never read.

Every test here patches ``asyncio.sleep`` and asserts the delay ARGUMENT, so
the suite stays fast and the assertions are about the *decision*, not wall
clock. Both retry code paths (``_request`` and ``_request_raw``) are covered.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from w2v_common.clients import BaseServiceClient
from w2v_common.clients.base import (
    _MAX_RETRIES,
    _RATE_LIMIT_MAX_RETRIES,
    _RATE_LIMIT_RETRY_DELAY,
    _RETRY_AFTER_MAX_DELAY,
    _RETRY_DELAY,
    _RETRY_SLEEP_BUDGET,
    _parse_retry_after,
)

_BASE = "http://test"


def _client_with_transport(transport: httpx.MockTransport) -> BaseServiceClient:
    client = BaseServiceClient(base_url=_BASE, service_name="svc")
    client._client = httpx.AsyncClient(
        transport=transport,
        base_url=_BASE,
        follow_redirects=False,
    )
    return client


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture every ``asyncio.sleep`` delay without actually sleeping."""
    recorded: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr("w2v_common.clients.base.asyncio.sleep", _fake_sleep)
    return recorded


def _sequence_handler(
    responses: list[httpx.Response],
    calls: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    """Replay ``responses`` in order; the last one repeats forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index]

    return handler


def _429(retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(status_code=429, content=b'{"error":"rate_limited"}', headers=headers)


def _ok() -> httpx.Response:
    return httpx.Response(status_code=200, content=b'{"ok":true}')


class TestRetryAfterDeltaSeconds:
    """(a) ``Retry-After: 5`` must be honoured instead of the 1s backoff."""

    async def test_delta_seconds_header_drives_the_wait(self, sleeps: list[float]) -> None:
        # Two 429s so the assertion cannot be satisfied by the fallback: a
        # linear backoff would produce [5.0, 10.0] here, the header [30.0,
        # 30.0]. The value must sit ABOVE the ladder for the header to be the
        # thing being observed -- ``Retry-After`` is floored at the default
        # backoff (see TestRetryAfterNeverReducesBackoff), so a below-ladder
        # value would be indistinguishable from the fallback.
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("30"), _429("30"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 3
        assert sleeps == [30.0, 30.0]
        # The pre-fix behaviour was a flat 1s first sleep.
        assert sleeps[0] != _RETRY_DELAY

    async def test_delta_seconds_beats_the_default_backoff(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("7"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [7.0]
        assert sleeps[0] != _RETRY_DELAY
        assert sleeps[0] != _RATE_LIMIT_RETRY_DELAY


class TestRetryAfterHttpDate:
    """(b) The RFC 9110 HTTP-date form must resolve to the right delta."""

    async def test_http_date_header_is_parsed_to_a_delta(self, sleeps: list[float]) -> None:
        target = datetime.now(UTC) + timedelta(seconds=30)
        header = target.strftime("%a, %d %b %Y %H:%M:%S GMT")

        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(header), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert len(sleeps) == 1
        # Second-granularity header + clock skew during the test: allow 2s.
        assert 28.0 <= sleeps[0] <= 31.0

    async def test_http_date_in_the_past_floors_at_the_default_backoff(
        self, sleeps: list[float]
    ) -> None:
        """A past date must NOT collapse the wait to zero.

        Previously this floored at 0.0, which made a stale/skewed clock on the
        downstream strictly more aggressive than no header at all.
        """
        past = datetime.now(UTC) - timedelta(hours=1)
        header = past.strftime("%a, %d %b %Y %H:%M:%S GMT")

        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(header), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]


class TestMalformedRetryAfter:
    """(c) Garbage must fall back to the normal backoff and never raise."""

    @pytest.mark.parametrize("header", ["banana", "", "  ", "5.5", "-", "NaN", "Wed, 99 Xxx"])
    async def test_malformed_header_falls_back_to_backoff(
        self, sleeps: list[float], header: str
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(header), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert result == {"ok": True}
        # Falls back to the 429 budget's linear backoff: 5.0 * attempt(1).
        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]

    async def test_absent_header_falls_back_to_backoff(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]


class TestRetryAfterClamp:
    """(d) A far-future ``Retry-After`` must be clamped to the cap."""

    async def test_huge_delta_seconds_is_clamped(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("86400"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RETRY_AFTER_MAX_DELAY]

    async def test_far_future_http_date_is_clamped(self, sleeps: list[float]) -> None:
        far = datetime.now(UTC) + timedelta(days=30)
        header = far.strftime("%a, %d %b %Y %H:%M:%S GMT")

        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(header), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RETRY_AFTER_MAX_DELAY]


class TestRateLimitBudget:
    """(e) A 429 run that exhausted the OLD 3-attempt budget now succeeds."""

    async def test_four_429s_then_success(self, sleeps: list[float]) -> None:
        # Under the old budget the 4th call never happened: attempts 1 and 2
        # slept, attempt 3 raised. The whole burst was over in 3 seconds.
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(), _429(), _429(), _429(), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request(
                "POST", "/api/test-runner/execute", json={"x": 1}, idempotency_key="run-42"
            )
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 5
        assert len(calls) > _MAX_RETRIES
        assert sleeps == [5.0, 10.0, 15.0, 20.0]

    async def test_full_429_budget_outlasts_a_60s_window(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request(
                    "POST", "/api/test-runner/execute", json={"x": 1}, idempotency_key="run-43"
                )
        finally:
            await client.close()

        assert len(calls) == _RATE_LIMIT_MAX_RETRIES
        assert len(sleeps) == _RATE_LIMIT_MAX_RETRIES - 1
        # The whole point: total patience must comfortably exceed the
        # downstream limiter's 60-second sliding window.
        assert sum(sleeps) > 60.0


class Test5xxUnchanged:
    """(f) FLEET REGRESSION GUARD: 5xx retry count + delays must not move.

    ~20 repos consume this library. If any of these assertions change, every
    consumer's transient-failure timing changed with them.
    """

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    async def test_5xx_attempt_count_and_delays_are_historical(
        self, sleeps: list[float], status: int
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([httpx.Response(status_code=status)], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert len(calls) == _MAX_RETRIES == 3
        assert sleeps == [_RETRY_DELAY * 1, _RETRY_DELAY * 2] == [1.0, 2.0]

    async def test_5xx_ignores_retry_after(self, sleeps: list[float]) -> None:
        """A 503 carrying ``Retry-After`` keeps the historical 1s backoff.

        Deliberate: honouring the header on 5xx would silently re-time every
        consumer's retries against servers that already emit it.
        """
        calls: list[httpx.Request] = []
        handler = _sequence_handler(
            [httpx.Response(status_code=503, headers={"Retry-After": "42"}), _ok()], calls
        )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RETRY_DELAY * 1]

    async def test_5xx_after_429s_does_not_borrow_the_rate_limit_budget(
        self, sleeps: list[float]
    ) -> None:
        """A 5xx must never consume a 429's extra attempts."""
        calls: list[httpx.Request] = []
        handler = _sequence_handler(
            [_429(), _429(), _429(), httpx.Response(status_code=500)], calls
        )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert exc_info.value.response.status_code == 500
        # attempt 4 > _MAX_RETRIES, so the 500 is fatal immediately.
        assert len(calls) == 4
        # The three preceding sleeps are 429 sleeps on the 429 ladder; the 500
        # itself gets no sleep at all because attempt 4 is already past the
        # 5xx budget.
        assert sleeps == [5.0, 10.0, 15.0]

    async def test_transport_error_budget_is_unchanged(self, sleeps: list[float]) -> None:
        """The transport-error branch keeps ``_MAX_RETRIES`` / ``_RETRY_DELAY``."""
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ConnectError("boom")

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.ConnectError):
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert len(calls) == _MAX_RETRIES == 3
        assert sleeps == [1.0, 2.0]


class TestRequestRawPathParity:
    """(g) ``_request_raw`` is the SECOND retry path -- same behaviour."""

    async def test_raw_honours_retry_after(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("7"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            response = await client._request_raw("GET", "/api/audio")
        finally:
            await client.close()

        assert response.status_code == 200
        assert sleeps == [7.0]

    async def test_raw_clamps_retry_after(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("86400"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request_raw("GET", "/api/audio")
        finally:
            await client.close()

        assert sleeps == [_RETRY_AFTER_MAX_DELAY]

    async def test_raw_malformed_retry_after_falls_back(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("banana"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request_raw("GET", "/api/audio")
        finally:
            await client.close()

        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]

    async def test_raw_uses_the_rate_limit_budget(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(), _429(), _429(), _429(), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            response = await client._request_raw(
                "POST", "/api/audio", json={"x": 1}, idempotency_key="raw-1"
            )
        finally:
            await client.close()

        assert response.status_code == 200
        assert len(calls) == 5
        assert sleeps == [5.0, 10.0, 15.0, 20.0]

    async def test_raw_5xx_is_unchanged(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([httpx.Response(status_code=503)], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request_raw("GET", "/api/audio")
        finally:
            await client.close()

        assert len(calls) == _MAX_RETRIES == 3
        assert sleeps == [1.0, 2.0]


class TestHostileHeaderNeverRaises:
    """FINDING 1 (HIGH): ``_parse_retry_after`` must never raise.

    ``float(int(raw))`` raises ``OverflowError`` -- NOT a ``ValueError`` -- for
    an integer too large for a float. A header of 400 nines therefore escaped
    the ``except ValueError`` guard, propagated out of ``_retry_sleep_seconds``
    past ``_request``'s ``except httpx.TransportError`` and reached the caller
    in every consumer service. Attacker-controlled, one header, no retry.
    """

    # 400 nines: parses as an int (under CPython's 4300-digit
    # ``int_max_str_digits`` limit) but overflows float conversion.
    _OVERFLOW = "9" * 400

    @pytest.mark.parametrize(
        "header",
        [
            "9" * 400,  # OverflowError from float(int(...))
            "9" * 5000,  # ValueError: exceeds int_max_str_digits
            "-" + "9" * 400,  # negative overflow
            "Wed, 21 Oct 99999999 07:28:00 GMT",  # year out of range
            "Wed, 21 Oct 2026 07:28:00 +9999",  # offset beyond +/-24h
            "Wed, 21 Oct 2026 07:28:00 -9999",
            "Thu, 32 Dec 2026 25:99:99 GMT",  # day/time out of range
            "ééé",  # non-ASCII junk
            "Wed, 21 Oct 2026 07:28:00",  # naive: no zone at all
            "Wed, 21 Oct 2026 07:28:00 -0000",  # RFC "unknown zone" form
            "\x00\x01\x02",
            "1e999",
            "inf",
            "nan",
        ],
    )
    def test_parse_never_raises(self, header: str) -> None:
        """Direct unit-level proof, independent of the client plumbing."""
        result = _parse_retry_after(header)
        assert result is None or isinstance(result, float)

    async def test_overflow_header_does_not_escape_to_the_caller(self, sleeps: list[float]) -> None:
        """End-to-end: the call must SUCCEED, not raise ``OverflowError``."""
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(self._OVERFLOW), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert result == {"ok": True}
        # Falls all the way through to the default 429 backoff.
        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]

    async def test_overflow_header_does_not_escape_on_the_raw_path(
        self, sleeps: list[float]
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(self._OVERFLOW), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            response = await client._request_raw("GET", "/api/audio")
        finally:
            await client.close()

        assert response.status_code == 200
        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]

    def test_fuzz_never_raises(self) -> None:
        """Blind fuzz: no byte-string a downstream can emit may raise.

        Proves the "NEVER raises" docstring claim by execution rather than by
        reading the guard list.
        """
        rng = random.Random(20260806)  # noqa: S311 — why: deterministic test-corpus generator, seeded for reproducibility; no security role
        alphabet = string.printable + "é€中\x00"
        # Seed the corpus with shapes that hit each branch, then fuzz around it.
        corpus = [
            "",
            " ",
            "0",
            "9" * 400,
            "Wed, 21 Oct 2026 07:28:00 GMT",
            None,
        ]
        for _ in range(4000):
            length = rng.randint(0, 40)
            corpus.append("".join(rng.choice(alphabet) for _ in range(length)))
        for _ in range(400):
            corpus.append("9" * rng.randint(300, 4400))

        for candidate in corpus:
            result = _parse_retry_after(candidate)
            assert result is None or (
                isinstance(result, float) and result <= _RETRY_AFTER_MAX_DELAY
            ), f"bad result {result!r} for {candidate!r}"


class TestCumulativeSleepBudget:
    """FINDING 2 (HIGH): total sleep in one call must fit inside a job lease.

    Measured before the fix: a downstream repeating ``Retry-After: 120``
    produced sleeps ``[120, 120, 120, 120, 120]`` = 600s inside ONE
    ``_request`` call. The smallest ``job_lease_seconds`` in the fleet is
    ``video_compiler_service``'s 120s (tts_gateway and html_render_service are
    both 180s), so the call outlived its own lease and the job was reclaimed
    underneath it.
    """

    async def test_hostile_repeating_retry_after_stays_within_budget(
        self, sleeps: list[float]
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("120")], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client._request(
                    "POST", "/api/test-runner/execute", json={"x": 1}, idempotency_key="k"
                )
        finally:
            await client.close()

        # It TERMINATES (surfaces the 429) rather than running the full ladder.
        assert exc_info.value.response.status_code == 429
        assert sum(sleeps) <= _RETRY_SLEEP_BUDGET
        # The pre-fix measurement.
        assert sum(sleeps) < 600.0
        # Every individual sleep is also capped.
        assert all(delay <= _RETRY_AFTER_MAX_DELAY for delay in sleeps)
        # And the budget must stay under the smallest fleet job lease.
        assert _RETRY_SLEEP_BUDGET < 120.0

    async def test_budget_holds_on_the_raw_path(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("120")], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request_raw("POST", "/api/audio", json={"x": 1}, idempotency_key="k")
        finally:
            await client.close()

        assert sum(sleeps) <= _RETRY_SLEEP_BUDGET

    @pytest.mark.parametrize("header", ["30", "45", "60", "120", "3600"])
    async def test_budget_holds_for_any_hostile_value(
        self, sleeps: list[float], header: str
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(header)], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sum(sleeps) <= _RETRY_SLEEP_BUDGET

    async def test_budget_never_truncates_the_default_ladder(self, sleeps: list[float]) -> None:
        """The budget must be a no-op for a 429 burst with no header.

        The default ladder spends exactly ``_RETRY_SLEEP_BUDGET``; if the
        budget were smaller it would silently shorten the designed 429
        patience for all ~20 consumers.
        """
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert len(calls) == _RATE_LIMIT_MAX_RETRIES
        assert sleeps == [5.0, 10.0, 15.0, 20.0, 25.0]
        assert sum(sleeps) == _RETRY_SLEEP_BUDGET

    async def test_per_sleep_cap_is_reachable_under_the_budget(self, sleeps: list[float]) -> None:
        """``_RETRY_AFTER_MAX_DELAY`` must be small enough to actually be slept."""
        assert _RETRY_AFTER_MAX_DELAY <= _RETRY_SLEEP_BUDGET

        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("3600"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RETRY_AFTER_MAX_DELAY]


class TestRetryAfterNeverReducesBackoff:
    """FINDING 3 (MEDIUM): ``Retry-After`` may only lengthen the wait.

    ``max(seconds, 0.0)`` meant ``Retry-After: 0`` produced ``[0, 0, 0, 0, 0]``
    -- six back-to-back requests with zero backoff, strictly MORE aggressive
    than the pre-change 3 attempts at 1s/2s. A hostile upstream could turn the
    client into its own amplifier.
    """

    @pytest.mark.parametrize("header", ["0", "-5", "-0", "-999999"])
    async def test_zero_or_negative_still_sleeps_the_default(
        self, sleeps: list[float], header: str
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429(header)], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        # The full default 429 ladder, not a zero-backoff hammer.
        assert sleeps == [5.0, 10.0, 15.0, 20.0, 25.0]
        assert all(delay >= _RATE_LIMIT_RETRY_DELAY for delay in sleeps)
        assert min(sleeps) > 0.0

    async def test_small_retry_after_is_raised_to_the_default(self, sleeps: list[float]) -> None:
        """``Retry-After: 1`` is below the 429 base delay -- floor it."""
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("1"), _429("1"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1, _RATE_LIMIT_RETRY_DELAY * 2]

    async def test_zero_is_not_more_aggressive_than_the_raw_path_default(
        self, sleeps: list[float]
    ) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("0"), _ok()], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            await client._request_raw("GET", "/api/audio")
        finally:
            await client.close()

        assert sleeps == [_RATE_LIMIT_RETRY_DELAY * 1]


class TestRetryGateStillApplies:
    """A bare POST (no idempotency key) must still NOT retry on 429."""

    async def test_bare_post_429_is_not_retried(self, sleeps: list[float]) -> None:
        calls: list[httpx.Request] = []
        handler = _sequence_handler([_429("5")], calls)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("POST", "/api/foo", json={"x": 1})
        finally:
            await client.close()

        assert len(calls) == 1
        assert sleeps == []
