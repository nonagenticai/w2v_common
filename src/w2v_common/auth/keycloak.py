"""Keycloak JWT validation with cached JWKS lookup.

Adapted from portal-bff/src/auth/keycloak.py. Configured entirely via env vars:

- ``KEYCLOAK_URL``              — e.g. ``https://auth.nonagentic.ai`` (required to enable JWT path).
                                  This is the URL the validator uses to FETCH the JWKS. In-cluster
                                  callers can point this at an internal Service URL (e.g.
                                  ``http://<keycloak-service>.<namespace>:8080``) to avoid
                                  egressing through the public ingress.
- ``KEYCLOAK_ISSUER``           — optional. Overrides the ``iss`` claim value the validator expects.
                                  Defaults to ``${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}`` so existing
                                  deploys keep working. Set this to the public issuer (e.g.
                                  ``https://auth.nonagentic.ai/realms/portal``) when ``KEYCLOAK_URL``
                                  points at an internal Service URL — the issuer recorded in tokens
                                  must still match the public URL Keycloak was minted with.
- ``KEYCLOAK_REALM``            — default ``portal``
- ``KEYCLOAK_AUDIENCES``        — comma-separated. When empty/unset (the default), audience
                                  validation is skipped and only the issuer is verified.
                                  This matches the fr backend, which trusts any ``aud`` from
                                  the configured realm. Set this to a non-empty list only if
                                  the realm has an ``oidc-audience-mapper`` emitting that aud.
- ``KEYCLOAK_JWKS_TTL``         — seconds, default 900 (15 min)
- ``KEYCLOAK_JWKS_TIMEOUT``     — seconds, default 2.0
- ``KEYCLOAK_JWT_LEEWAY``       — seconds, default 10. Tolerance for clock skew between
                                  pods and Keycloak when validating ``exp`` / ``nbf``.
- ``KEYCLOAK_TRUSTED_ISSUERS``  — realm-per-app P0. Comma-separated FULL issuer URLs whose
                                  tokens are accepted, each verified against ITS OWN realm's
                                  JWKS with ``iss`` pinned to that exact value. Unset/empty
                                  (the default) = exactly the single issuer above, i.e. no
                                  behaviour change. When set it REPLACES the default: list the
                                  ``KEYCLOAK_ISSUER`` value too if it must keep working.
- ``KEYCLOAK_SERVICE_ONLY_REALMS`` — comma-separated realm names, default ``secure-apps``.
                                  A token from one of these realms is never honoured as a
                                  USER session by :class:`AuthMiddleware` (``secure-apps`` holds
                                  human accounts too — billing-dashboard users). Applied only to
                                  issuers that came from an explicit ``KEYCLOAK_TRUSTED_ISSUERS``,
                                  so an unset env never refuses anything it accepts today.

Same contract as licenses ``auth/jwt_verify.py`` and CommonPractice
``KEYCLOAK_TRUSTED_ISSUERS`` / ``KEYCLOAK_SERVICE_ONLY_REALMS`` (#480).

If ``KEYCLOAK_URL`` is unset, :func:`get_jwt_validator` returns ``None`` and the
auth middleware skips the JWT branch entirely (backward compatible — services
that only use license tokens continue to work without any Keycloak config).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
import jwt
from jwt import PyJWK
from jwt.exceptions import InvalidTokenError

logger = logging.getLogger(__name__)


def _looks_like_jwt(token: str) -> bool:
    """Cheap structural check: a JWT has three base64url segments."""
    if not token:
        return False
    parts = token.split(".")
    return len(parts) == 3 and all(parts)


def realm_of_issuer(issuer: str | None) -> str | None:
    """``https://host/realms/<realm>`` -> ``<realm>``; ``None`` when not a realm issuer."""
    issuer = (issuer or "").strip().rstrip("/")
    if "/realms/" not in issuer:
        return None
    realm = issuer.rsplit("/realms/", 1)[-1]
    return realm if realm and "/" not in realm else None


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_trusted_issuers(raw: str | None) -> list[str]:
    """``KEYCLOAK_TRUSTED_ISSUERS`` -> issuer URLs (trailing ``/`` stripped, deduped, ordered).

    Keycloak's ``iss`` never ends in ``/``; a stray slash in the env value must not
    silently 401 a whole realm.
    """
    out: list[str] = []
    for item in _parse_csv(raw):
        issuer = item.rstrip("/")
        if issuer and issuer not in out:
            out.append(issuer)
    return out


#: ``KEYCLOAK_SERVICE_ONLY_REALMS`` default. ``secure-apps`` is the S2S realm, but
#: it is NOT SA-only (it holds human accounts), so it is never a user realm.
DEFAULT_SERVICE_ONLY_REALMS: tuple[str, ...] = ("secure-apps",)


def parse_service_only_realms(raw: str | None) -> list[str]:
    """``KEYCLOAK_SERVICE_ONLY_REALMS`` -> realm names; unset -> ``secure-apps``.

    Set to an empty string to name no realm at all (empty is distinct from unset).
    """
    if raw is None:
        return list(DEFAULT_SERVICE_ONLY_REALMS)
    return _parse_csv(raw)


#: Minimum seconds between two ROTATION-PROBE (forced) JWKS fetches. Bounds how
#: much upstream load an unknown-``kid`` flood can generate; see _get_signing_key.
_JWKS_FORCE_MIN_INTERVAL_S: float = 10.0


class KeycloakJWTValidator:
    """Validates RS256 JWTs issued by a Keycloak realm.

    Fetches the realm's JWKS on first use and caches keys by ``kid``. On a
    cache miss the JWKS is refetched once (keys may have rotated).
    """

    def __init__(
        self,
        keycloak_url: str,
        realm: str,
        audiences: list[str],
        jwks_ttl: int = 900,
        http_timeout: float = 2.0,
        issuer: str | None = None,
        leeway: float = 10.0,
        trusted_issuers: list[str] | None = None,
        service_only_realms: list[str] | None = None,
    ) -> None:
        """Build a validator.

        ``issuer`` / ``realm`` name the PRIMARY issuer, exactly as before.
        ``trusted_issuers`` (realm-per-app P0): ``None`` or empty = only the
        primary issuer is accepted (unchanged behaviour). Non-empty = EXACTLY
        that set is accepted; the primary is included only if listed. Each
        extra issuer gets its own JWKS cache, fetched through ``keycloak_url``
        from the realm named in the issuer (same as CP ``jwksUrlForIssuer``).
        ``service_only_realms``: realms :meth:`is_service_only_issuer` reports
        for issuers from an explicit ``trusted_issuers``; default ``secure-apps``.
        """
        self._base_url = keycloak_url.rstrip("/")
        self._realm = realm
        self._audiences = audiences
        self._jwks_ttl = jwks_ttl
        self._http_timeout = http_timeout
        self._leeway = leeway

        # JWKS is always fetched from KEYCLOAK_URL (may be an in-cluster Service URL).
        # Issuer (the value embedded in the token's ``iss`` claim) defaults to the same
        # base URL but can be overridden when the fetch URL differs from the public URL.
        self._jwks_uri = f"{self._base_url}/realms/{realm}/protocol/openid-connect/certs"
        self._issuer = issuer.rstrip("/") if issuer else f"{self._base_url}/realms/{realm}"

        self._jwks_by_kid: dict[str, dict[str, Any]] = {}
        self._jwks_fetched_at: float = 0.0
        self._jwks_forced_at: float = 0.0
        self._jwks_lock = asyncio.Lock()

        # Realm-per-app P0: the issuer allowlist. ``_explicit_trust`` records
        # whether it came from configuration; only then can the service-only
        # refusal apply (an unset env must never refuse what it accepts today).
        explicit = [i.rstrip("/") for i in (trusted_issuers or []) if i and i.strip()]
        self._explicit_trust = bool(explicit)
        self._trusted: tuple[str, ...] = tuple(dict.fromkeys(explicit)) or (self._issuer,)
        self._primary_trusted = self._issuer in self._trusted
        self._service_only_realms: tuple[str, ...] = tuple(
            DEFAULT_SERVICE_ONLY_REALMS if service_only_realms is None else service_only_realms
        )
        # One single-issuer engine (its own JWKS cache, lock and rotation floor)
        # per EXTRA trusted issuer. Bounded by the allowlist: an attacker-chosen
        # ``iss`` never creates an entry.
        self._extra: dict[str, KeycloakJWTValidator] = {}
        for extra_issuer in self._trusted:
            if extra_issuer == self._issuer:
                continue
            extra_realm = realm_of_issuer(extra_issuer)
            if extra_realm is None:
                raise ValueError(
                    f"KEYCLOAK_TRUSTED_ISSUERS entry {extra_issuer!r} is not a "
                    "Keycloak realm issuer (expected .../realms/<realm>)"
                )
            self._extra[extra_issuer] = KeycloakJWTValidator(
                keycloak_url=keycloak_url,
                realm=extra_realm,
                audiences=audiences,
                jwks_ttl=jwks_ttl,
                http_timeout=http_timeout,
                issuer=extra_issuer,
                leeway=leeway,
            )

    @property
    def issuer(self) -> str:
        """The PRIMARY issuer (``KEYCLOAK_ISSUER`` or derived from the realm)."""
        return self._issuer

    @property
    def trusted_issuers(self) -> tuple[str, ...]:
        """Every issuer whose tokens :meth:`validate` accepts."""
        return self._trusted

    @property
    def service_only_realms(self) -> tuple[str, ...]:
        return self._service_only_realms

    def is_service_only_issuer(self, issuer: str | None) -> bool:
        """True when ``issuer`` is a service-only realm that must not be a USER session.

        Only issuers from an explicit ``KEYCLOAK_TRUSTED_ISSUERS`` qualify, so the
        default configuration (one implicit issuer) never refuses anything.
        """
        if not self._explicit_trust:
            return False
        realm = realm_of_issuer(issuer)
        return realm is not None and realm in self._service_only_realms

    @property
    def audiences(self) -> list[str]:
        return list(self._audiences)

    async def validate(self, token: str) -> tuple[dict[str, Any] | None, str | None]:
        """Validate a JWT.

        Returns:
            ``(claims, None)`` on success, ``(None, reason)`` on failure where
            ``reason`` is a short human-readable diagnostic ("invalid issuer",
            "expired", "signature mismatch", etc.). The reason is safe to
            surface in an HTTP 401 response body for operator debugging —
            it does not leak token contents.

        Multi-issuer (realm-per-app P0): the unverified ``iss`` only SELECTS
        which trusted issuer's JWKS to use; the selected engine then pins
        ``issuer=`` to that exact value after the signature check, so
        "accept two realms" never degrades into "accept whichever key
        verifies". A token naming an extra trusted issuer goes to that
        issuer's engine; every other token takes the primary path exactly as
        before (and fails its issuer pin), unless the primary is not trusted.
        """
        if not _looks_like_jwt(token):
            return None, "not_jwt_shaped"

        if self._extra or not self._primary_trusted:
            try:
                unverified_iss = jwt.decode(token, options={"verify_signature": False}).get("iss")
            except InvalidTokenError as exc:
                logger.warning("JWT payload parse failed: %s", exc)
                return None, f"payload_parse_failed: {exc}"
            iss = str(unverified_iss or "").rstrip("/")
            engine = self._extra.get(iss)
            if engine is not None:
                return await engine._validate_pinned(token)
            if not self._primary_trusted:
                logger.warning("JWT issuer not trusted: %s", iss or "<missing>")
                return None, f"untrusted_issuer: {iss or '<missing>'}"
        return await self._validate_pinned(token)

    async def _validate_pinned(self, token: str) -> tuple[dict[str, Any] | None, str | None]:
        """Verify ``token`` against THIS engine's issuer + JWKS (single-issuer path)."""

        try:
            unverified_header = jwt.get_unverified_header(token)
        except InvalidTokenError as exc:
            logger.warning("JWT header parse failed: %s", exc)
            return None, f"header_parse_failed: {exc}"

        kid = unverified_header.get("kid")
        if not kid:
            logger.warning("JWT missing kid header")
            return None, "missing_kid"

        signing_key = await self._get_signing_key(kid)
        if signing_key is None:
            logger.warning("No signing key for kid=%s", kid)
            return None, f"unknown_kid: {kid}"

        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=self._audiences or None,
                issuer=self._issuer,
                leeway=self._leeway,
                options={
                    "require": ["exp", "iss", "sub"],
                    # Skip aud check entirely when no audiences are configured.
                    # PyJWT's default raises InvalidAudienceError if the token has
                    # an ``aud`` claim but none was passed — opt out to match fr/jose.
                    "verify_aud": bool(self._audiences),
                },
            )
        except InvalidTokenError as exc:
            logger.warning("JWT validation failed: %s", exc)
            return None, f"jwt_decode_failed: {exc}"

        return claims, None

    async def _get_signing_key(self, kid: str) -> Any | None:
        if self._is_jwks_fresh() and kid in self._jwks_by_kid:
            return PyJWK(self._jwks_by_kid[kid]).key

        await self._refresh_jwks(force=not self._is_jwks_fresh())
        if kid in self._jwks_by_kid:
            return PyJWK(self._jwks_by_kid[kid]).key

        # Possible key rotation — one more attempt with a forced refresh, RATE-FLOORED.
        #
        # ⚠️ This used to be unconditional, which made ``kid`` an amplifier: only the
        # JWT *header* is parsed before we get here, so the signature is irrelevant and
        # anyone can mint ``{"kid": "<random>"}``. Every such request produced a real GET
        # to Keycloak even when the JWKS was FRESH, and all of it serializes on
        # ``_jwks_lock`` — so a flood both amplifies against auth.nonagentic.ai and
        # queues legitimate JWT traffic behind itself.
        #
        # The floor keeps rotation discoverable (worst case: a genuinely new kid is
        # rejected for up to _JWKS_FORCE_MIN_INTERVAL_S, then picked up) while collapsing
        # an unbounded flood into at most one fetch per interval. A per-kid negative cache
        # would NOT be sufficient on its own — the keyspace is attacker-chosen and
        # unbounded, so each fresh random kid would still buy a fetch.
        now = time.monotonic()
        if (now - self._jwks_forced_at) < _JWKS_FORCE_MIN_INTERVAL_S:
            return None
        self._jwks_forced_at = now
        await self._refresh_jwks(force=True)
        if kid in self._jwks_by_kid:
            return PyJWK(self._jwks_by_kid[kid]).key
        return None

    def _is_jwks_fresh(self) -> bool:
        return (
            bool(self._jwks_by_kid) and (time.monotonic() - self._jwks_fetched_at) < self._jwks_ttl
        )

    async def _refresh_jwks(self, *, force: bool) -> None:
        async with self._jwks_lock:
            if not force and self._is_jwks_fresh():
                return
            try:
                async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                    resp = await client.get(self._jwks_uri)
                    resp.raise_for_status()
                    payload = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                # If we have keys cached, soft-fail: the cached set keeps
                # working until Keycloak rotates. If the cache is empty
                # (first-ever fetch failed), there is nothing to fall back
                # to — every JWT will 401 with ``unknown_kid``. Raise so the
                # caller (typically a startup prefetch) fails loud rather
                # than serving a broken auth path indefinitely.
                if self._jwks_by_kid:
                    logger.warning("JWKS fetch failed from %s: %s", self._jwks_uri, exc)
                    return
                logger.error("JWKS fetch failed from %s with empty cache: %s", self._jwks_uri, exc)
                raise

            keys = payload.get("keys", []) if isinstance(payload, dict) else []
            self._jwks_by_kid = {
                key["kid"]: key for key in keys if isinstance(key, dict) and key.get("kid")
            }
            self._jwks_fetched_at = time.monotonic()
            logger.debug("JWKS refreshed from %s: %d keys", self._jwks_uri, len(self._jwks_by_kid))

    async def prefetch_jwks(self) -> None:
        """Force a JWKS fetch. Intended for app startup lifespan.

        On first-ever failure (empty cache), raises so the service crashes
        loudly rather than serving 401s for every request. Subsequent
        background refreshes degrade silently when a valid cache exists.

        Extra trusted issuers are prefetched best-effort: a realm that does
        not exist YET (env set ahead of the realm-patch) logs a warning and is
        retried on first use instead of crash-looping the service.
        """
        if self._primary_trusted:
            await self._refresh_jwks(force=True)
        for extra_issuer, engine in self._extra.items():
            try:
                await engine._refresh_jwks(force=True)
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("JWKS prefetch failed for trusted issuer %s: %s", extra_issuer, exc)

    def is_ready(self) -> bool:
        """Return True iff the JWKS cache is non-empty and not severely stale.

        Wire to ``/readyz`` so operators see auth-path degradation before
        users do. Staleness threshold is ``2 * jwks_ttl`` to avoid flapping
        on transient JWKS-endpoint blips.

        Reports the PRIMARY issuer (unchanged); when the primary is not in the
        trusted set, every extra issuer must be ready instead.
        """
        if not self._primary_trusted:
            return bool(self._extra) and all(e.is_ready() for e in self._extra.values())
        if not self._jwks_by_kid:
            return False
        return (time.monotonic() - self._jwks_fetched_at) < (2 * self._jwks_ttl)


_validator: KeycloakJWTValidator | None = None
_validator_initialized: bool = False


def _parse_audiences(raw: str | None) -> list[str]:
    """Parse ``KEYCLOAK_AUDIENCES`` into a list. Empty/unset → ``[]`` (skip aud check)."""
    if not raw:
        return []
    return [a.strip() for a in raw.split(",") if a.strip()]


def get_jwt_validator() -> KeycloakJWTValidator | None:
    """Return the process-wide validator or ``None`` if Keycloak is not configured."""
    global _validator, _validator_initialized
    if _validator_initialized:
        return _validator
    _validator_initialized = True

    keycloak_url = os.getenv("KEYCLOAK_URL")
    if not keycloak_url:
        logger.debug("KEYCLOAK_URL not set — JWT validation disabled")
        return None

    realm = os.getenv("KEYCLOAK_REALM", "portal")
    audiences = _parse_audiences(os.getenv("KEYCLOAK_AUDIENCES"))
    jwks_ttl = int(os.getenv("KEYCLOAK_JWKS_TTL", "900"))
    http_timeout = float(os.getenv("KEYCLOAK_JWKS_TIMEOUT", "2.0"))
    leeway = float(os.getenv("KEYCLOAK_JWT_LEEWAY", "10"))
    issuer_override = os.getenv("KEYCLOAK_ISSUER") or None
    trusted = parse_trusted_issuers(os.getenv("KEYCLOAK_TRUSTED_ISSUERS"))
    service_only = parse_service_only_realms(os.getenv("KEYCLOAK_SERVICE_ONLY_REALMS"))

    _validator = KeycloakJWTValidator(
        keycloak_url=keycloak_url,
        realm=realm,
        audiences=audiences,
        jwks_ttl=jwks_ttl,
        http_timeout=http_timeout,
        issuer=issuer_override,
        leeway=leeway,
        trusted_issuers=trusted or None,
        service_only_realms=service_only,
    )
    logger.info(
        "Keycloak JWT validation enabled: issuer=%s trusted=%s service_only=%s audiences=%s",
        _validator.issuer,
        list(_validator.trusted_issuers),
        list(_validator.service_only_realms) if trusted else [],
        audiences,
    )
    return _validator


def reset_jwt_validator() -> None:
    """Reset the module-level validator singleton. Test hook only."""
    global _validator, _validator_initialized
    _validator = None
    _validator_initialized = False
