"""Bounds on attacker-reachable auth state.

Both caches here are keyed by a value an UNAUTHENTICATED caller chooses, so an
unbounded one is a remote memory-exhaustion primitive, not a tuning nit.
"""

from __future__ import annotations

import pytest

from w2v_common.auth import middleware as mw


@pytest.mark.asyncio
async def test_introspect_locks_are_bounded_under_a_distinct_token_flood() -> None:
    """A flood of DISTINCT bogus tokens must not grow the lock map without limit.

    ``_introspect_locks`` is keyed by a hash of the caller-supplied license token.
    Nothing authenticates before that key is minted, so an unauthenticated flood of
    distinct tokens used to add one ``asyncio.Lock`` per token, forever, on every
    pod. The bound is what makes it survivable; the exact ceiling is not the point.
    """
    mw._reset_introspect_locks()

    for i in range(5_000):
        mw._acquire_introspect_lock(f"token-hash-{i}")

    assert len(mw._introspect_locks) <= mw._INTROSPECT_LOCK_MAXSIZE


@pytest.mark.asyncio
async def test_a_held_lock_is_not_evicted_from_under_its_holder() -> None:
    """Eviction must never hand two waiters different lock objects for one key.

    A bounded map that evicts by insertion order would drop a lock that a slow
    introspection is still holding; the next caller for that same key would then
    mint a FRESH lock and both would enter the critical section together, which is
    exactly the upstream stampede the lock exists to prevent.
    """
    mw._reset_introspect_locks()

    key = "held-key"
    held = mw._acquire_introspect_lock(key)
    async with held:
        for i in range(5_000):
            mw._acquire_introspect_lock(f"filler-{i}")
        # Still the SAME object while held.
        assert mw._acquire_introspect_lock(key) is held


@pytest.mark.asyncio
async def test_unknown_kids_do_not_each_force_a_jwks_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attacker-chosen ``kid`` must not be an amplifier against Keycloak.

    Only the JWT *header* is parsed before key lookup, so the signature is
    irrelevant — anyone can mint ``{"kid": "<random>"}``. ``_get_signing_key`` used
    to call ``_refresh_jwks(force=True)`` unconditionally on a miss, so EVERY such
    request produced a real GET to Keycloak even when the JWKS was fresh, all of it
    serialized on ``_jwks_lock``. That is amplification against auth.nonagentic.ai
    and self-DoS at the same time: legitimate JWT traffic queues behind the flood.
    """
    from w2v_common.auth.keycloak import KeycloakJWTValidator

    v = KeycloakJWTValidator(
        keycloak_url="https://auth.example.invalid", realm="r", audiences=["a"]
    )
    # Warm, fresh cache holding one real kid.
    v._jwks_by_kid = {"real-kid": {"kid": "real-kid"}}
    v._jwks_fetched_at = __import__("time").monotonic()

    fetches = 0

    async def _counting_refresh(*, force: bool) -> None:
        nonlocal fetches
        if force:
            fetches += 1

    monkeypatch.setattr(v, "_refresh_jwks", _counting_refresh)

    for i in range(200):
        assert await v._get_signing_key(f"bogus-kid-{i}") is None

    # One forced probe is legitimate (a real rotation must still be discoverable);
    # 200 is an open amplifier.
    assert fetches <= 1, f"{fetches} forced JWKS fetches for 200 bogus kids"


def test_the_introspection_path_actually_uses_the_bounded_helper() -> None:
    """Pin the WIRING, not just the helper.

    The bound is worthless if the introspection path goes back to
    ``_introspect_locks.setdefault(...)``. Tests that exercise the helper directly
    cannot see that regression — verified by mutation: reverting the call site left
    them green. This asserts the call site itself.
    """
    import inspect

    source = inspect.getsource(mw._introspect_license)
    assert "_acquire_introspect_lock(" in source, (
        "the introspection path no longer uses the bounded lock helper"
    )
    assert "_introspect_locks.setdefault" not in source, (
        "the introspection path mints unbounded locks again"
    )
