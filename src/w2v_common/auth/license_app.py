"""Which licence APP a service accepts (w2v_common#222).

A licence token authorises exactly one app (licenses ruling 2026-09-28: a
bundle licence included, ``/license/introspect`` never widens to ``apps[]``).
Before 0.102.0 ``AuthMiddleware`` checked only ``active`` and the tenant, so a
token minted for ``kag`` opened tts/asr: the APP the token was issued for was
never read.

``AUTH_LICENSE_APP_KEYS`` (comma-separated, e.g. ``w2v``) declares the apps
this service accepts. Read per call, like the middleware's other ``AUTH_*``
knobs, so a test or a restart-free rollout sees the change.

* **Set** → fail closed: a licence whose app key is not listed, or whose
  introspection names no app at all, is refused ``403 WRONG_APP``.
* **Unset** → the check is NOT armed. The request passes, but never silently:
  one startup WARNING, ``w2v_auth_license_app_check_armed`` = 0, and every
  licence request is counted ``w2v_auth_license_app_total{outcome="unchecked"}``
  with the app it carried — the measurement an operator needs before arming.

Why unset does not fail closed: ``bump-consumers`` pushes a new release
straight to 17 consumers' ``main``, and the app keys of the service-account
tokens those services receive are unmeasured. Refusing on unset would turn a
library bump into a fleet-wide 403 outage; arming per service, after its
``unchecked`` series shows only the expected apps, cannot.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Any

from prometheus_client import Counter, Gauge

__all__ = [
    "LICENSE_APP_KEYS_ENV",
    "WRONG_APP_CODE",
    "allowed_license_app_keys",
    "check_license_app",
    "introspected_app_key",
    "warn_if_license_app_unchecked",
]

logger = logging.getLogger(__name__)

#: The per-service declaration of which licence apps are accepted.
LICENSE_APP_KEYS_ENV = "AUTH_LICENSE_APP_KEYS"
#: Error code of the 403 a mismatched app gets.
WRONG_APP_CODE = "WRONG_APP"

#: Label value bound. App keys come from the licences ``app`` table (a small,
#: operator-owned set), but the mesh lane's ``x-app-key`` is only as trustworthy
#: as its vouch, so the label is shape-checked and capped all the same.
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MAX_LABELS = 32
_seen_labels: set[str] = set()

license_app_total = Counter(
    "w2v_auth_license_app_total",
    "Licence-authenticated requests by the app the licence was issued for, and "
    "whether the AUTH_LICENSE_APP_KEYS check allowed, refused or did not check it.",
    ["app_key", "outcome"],
)
license_app_check_armed = Gauge(
    "w2v_auth_license_app_check_armed",
    "1 when AUTH_LICENSE_APP_KEYS is set (wrong-app licences are refused), 0 when unset.",
)

_warned_unchecked = False


def allowed_license_app_keys() -> frozenset[str] | None:
    """The apps this service accepts, or ``None`` when the check is not armed."""
    raw = os.getenv(LICENSE_APP_KEYS_ENV) or ""
    keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
    return keys or None


def introspected_app_key(result: Mapping[str, Any]) -> str | None:
    """The app key an introspection names: ``app.key``, else a flat ``app_key``."""
    app = result.get("app")
    key = app.get("key") if isinstance(app, Mapping) else None
    if not key:
        key = result.get("app_key")
    return (key.strip() or None) if isinstance(key, str) else None


def _label(app_key: str | None) -> str:
    if not app_key:
        return "none"
    if not _LABEL_RE.match(app_key):
        return "invalid"
    if app_key not in _seen_labels:
        if len(_seen_labels) >= _MAX_LABELS:
            return "other"
        _seen_labels.add(app_key)
    return app_key


def check_license_app(app_key: str | None) -> bool:
    """Count the request and say whether its licence app may proceed.

    ``False`` only when the check is armed and ``app_key`` is absent or unlisted.
    """
    allowed = allowed_license_app_keys()
    if allowed is None:
        license_app_total.labels(app_key=_label(app_key), outcome="unchecked").inc()
        return True
    ok = app_key is not None and app_key in allowed
    license_app_total.labels(app_key=_label(app_key), outcome="allowed" if ok else "refused").inc()
    return ok


def warn_if_license_app_unchecked() -> None:
    """Startup hook: publish the armed gauge and WARN once when the check is off."""
    global _warned_unchecked
    allowed = allowed_license_app_keys()
    license_app_check_armed.set(0 if allowed is None else 1)
    if allowed is None and not _warned_unchecked:
        _warned_unchecked = True
        logger.warning(
            "%s is not set: licence tokens issued for ANY app are accepted by this "
            "service (w2v_common#222). Set it to this service's app key(s), e.g. "
            '%s=w2v, after w2v_auth_license_app_total{outcome="unchecked"} shows '
            "only the expected apps.",
            LICENSE_APP_KEYS_ENV,
            LICENSE_APP_KEYS_ENV,
        )
