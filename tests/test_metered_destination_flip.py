"""The owner flip, and the env var that could have undone it.

TWO THINGS ARE PINNED HERE.

1. **The flip itself.** The owner ruled 2026-09-13 that all usage bills to the
   LICENSED TENANT that caused it. Eight destinations that were measured filing
   customer work under the orchestrator service-account tenant
   ``e148e8af-3751-49bf-a0fa-56ab46d31ddc`` moved from
   ``MISFILING_PENDING_OWNER_DECISION`` to ``DELEGATING`` and into
   ``METERED_DESTINATION_SERVICES``. Every assertion below is about a
   destination BY NAME, and each one fails on origin/main.

2. **That the flip is not inert.** It is not enough for a name to be in a
   frozenset. ``is_metered_destination`` matches on HOSTS, so a name that
   discovery cannot resolve is in the set and stamps nothing — which is exactly
   why ``tts_gateway`` is still pending. The tests here therefore assert the
   PREDICATE and the HEADERS, never just set membership.

``tests/test_destination_tenancy_declared.py`` remains the guard that no
destination can exist undeclared; this file is about what the declarations now
SAY and what actually goes on the wire because of it.
"""

from __future__ import annotations

import contextlib
import re
import uuid
from collections.abc import Iterator

import pytest

from w2v_common.auth.destination_tenancy import (
    DESTINATION_TENANCY,
    DestinationTenancy,
    names_with_tenancy,
)
from w2v_common.auth.outbound import (
    EFFECTIVE_TENANT_HEADER,
    METERED_DESTINATION_SERVICES,
    TENANT_HEADER,
    MeteredDestinationConfigError,
    _parse_metered_override,
    build_downstream_tenant_headers,
    is_metered_destination,
    unresolvable_metered_names,
)
from w2v_common.discovery import list_services, service_url
from w2v_common.tenant import TenantSource, UnvouchedTenantError

TENANT = uuid.UUID("11111111-2222-4333-8444-555555555555")

#: The nine destinations the owner flipped — eight on 2026-09-13, and
#: ``tts_gateway`` on 2026-09-15 once discovery could resolve it. Written out by
#: hand rather than derived from the source, on purpose: a list derived from
#: ``names_with_tenancy(DELEGATING)`` would pass no matter what that function
#: returns, including an empty set. These are the names that were MEASURED
#: mis-filing, so these are the names that must now delegate.
FLIPPED = (
    "analysis",
    "video_analyzer",
    "slide_service",
    "app_inspector",
    "bdd_playwright",
    "ai_content_generation",
    "frontend_test_runner",
    "youtube_upload",
    # 1082 rows (tts_jobs=536 + tts_batch_jobs=546), 100% under the
    # orchestrator SA tenant, measured live 2026-09-15.
    "tts_gateway",
    # 12 w2v-video-compiler rows on 2026-09-16, 100% under the orchestrator SA
    # tenant, each lockstep with a merqi audiobook run: the orchestrator calls
    # the compiler DIRECTLY (audiobook/stitch.py), no video_production hop exists
    # to "account" the render, and the compiler emits its own cost event.
    "video_compile",
)


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test here reasons about the CODE default unless it says otherwise."""
    monkeypatch.delenv("W2V_METERED_DESTINATIONS", raising=False)
    # ``_parse_metered_override`` is @cache'd on the RAW string, so a value
    # parsed by an earlier test emits no second warning. A test asserting on
    # the warning would then pass or fail depending on test ORDER. Clear it.
    _parse_metered_override.cache_clear()
    yield
    _parse_metered_override.cache_clear()


# ---------------------------------------------------------------------------
# A. The register is closed, except where closing it would be a lie
# ---------------------------------------------------------------------------


def test_no_on_registry_destination_is_still_pending_the_owner_decision() -> None:
    """The defect register may only hold destinations discovery cannot resolve.

    The owner has ruled, so "waiting for the owner" is no longer an available
    state for anything this library can actually route to. The single permitted
    residue is an off-registry destination, where the flip would be inert — and
    that exception is asserted narrowly (``off_registry`` must be True) so it
    cannot quietly grow back into the parking lot it used to be.
    """
    pending = names_with_tenancy(DestinationTenancy.MISFILING_PENDING_OWNER_DECISION)
    routable_pending = sorted(
        name for name in pending if not DESTINATION_TENANCY[name].off_registry
    )
    assert not routable_pending, (
        "destination(s) still declared MISFILING_PENDING_OWNER_DECISION that "
        f"discovery CAN route to: {routable_pending}. The owner's ruling of "
        "2026-09-13 is that all usage bills to the licensed tenant that caused "
        "it, so these must move to DELEGATING and join "
        "METERED_DESTINATION_SERVICES."
    )


def test_tts_gateway_is_the_only_permitted_residue() -> None:
    """INVERTED 2026-09-15 — the residue is gone, so the register is EMPTY.

    ``tts_gateway`` used to be the single permitted entry here: the flip would
    have been inert because discovery could not resolve it. Now that it is
    registered as ``("tts-gateway", "w2v", 6320)`` the flip is real, so there
    is no permitted exception left at all. Kept as an EQUALITY against the
    empty set rather than deleted: an inequality ("not in") would go on passing
    if a DIFFERENT destination were parked here tomorrow, which is exactly the
    parking lot this register keeps growing back into.
    """
    assert names_with_tenancy(DestinationTenancy.MISFILING_PENDING_OWNER_DECISION) == frozenset()
    assert DESTINATION_TENANCY["tts_gateway"].tenancy is DestinationTenancy.DELEGATING
    assert DESTINATION_TENANCY["tts_gateway"].off_registry is False


# ---------------------------------------------------------------------------
# B. Each flipped destination actually delegates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FLIPPED)
def test_a_flipped_destination_is_declared_delegating(name: str) -> None:
    assert DESTINATION_TENANCY[name].tenancy is DestinationTenancy.DELEGATING
    assert name in METERED_DESTINATION_SERVICES


@pytest.mark.parametrize("name", FLIPPED)
def test_a_flipped_destination_is_metered_by_name_and_by_url(name: str) -> None:
    """Membership is not the fix; the PREDICATE answering True is the fix.

    Real callers pass a base URL (``LicensedServiceClient``) or a full request
    URL (``identity_event_hook``), never the bare discovery name, so the URL
    form is the one that decides what goes on the wire. Asserting only the
    name form is how an off-registry entry looks fixed while stamping nothing.
    """
    assert is_metered_destination(name) is True
    assert is_metered_destination(service_url(name)) is True
    assert is_metered_destination(service_url(name) + "/api/anything") is True


@pytest.mark.parametrize("name", FLIPPED)
def test_a_vouched_tenant_now_reaches_a_flipped_destination_as_itself(name: str) -> None:
    """The customer's tenant travels, instead of the caller's SA tenant."""
    headers = build_downstream_tenant_headers(
        TENANT,
        service_url(name),
        TenantSource.LICENSE_TOKEN,
        downstream_license_token="svc-token",
    )
    assert headers[EFFECTIVE_TENANT_HEADER] == str(TENANT)
    assert headers[TENANT_HEADER] == "Bearer svc-token"


@pytest.mark.parametrize("name", FLIPPED)
@pytest.mark.parametrize(
    "source",
    [
        TenantSource.UNVOUCHED_HEADER,
        TenantSource.TRUSTED_NETWORK_DEFAULT,
        TenantSource.AUTH_DISABLED,
    ],
)
def test_an_unvouched_tenant_now_raises_instead_of_mis_filing(
    name: str, source: TenantSource
) -> None:
    """THE DELIBERATE FAIL-CLOSED HALF — this is the intent, not a regression.

    Before the flip these hops sent a bare ``X-Tenant-Id`` and the receiver
    introspected the shared SA licence, so an unvouched provenance produced a
    row filed under the service account: wrong, silent, and permanent. Now the
    call refuses. A refused call is visible and recoverable; 15,663
    unattributable rows are neither.
    """
    with pytest.raises(UnvouchedTenantError) as exc:
        build_downstream_tenant_headers(
            TENANT,
            service_url(name),
            source,
            downstream_license_token="svc-token",
        )
    assert source.value in str(exc.value)


# ---------------------------------------------------------------------------
# C. tts_gateway: the ninth, and the trap that kept it pending
# ---------------------------------------------------------------------------

#: The live target, measured 2026-09-15 with ``kubectl -n w2v get svc
#: tts-gateway`` (NodePort, port 6320). Written as a literal, not derived from
#: ``service_url``: deriving it would make the assertions below pass against
#: ANY registration, including one pointing at the wrong port — and the wrong
#: port is precisely a host mismatch, i.e. the inert-wiring failure again.
TTS_GATEWAY_URL = "http://tts-gateway.w2v:6320"


def test_flipping_tts_gateway_would_be_inert_wiring() -> None:
    """INVERTED 2026-09-15 — the trap is now CLOSED, and this proves it.

    This test used to assert the trap: ``tts_gateway`` had no
    ``discovery._SERVICES`` entry, so ``service_url`` raised KeyError, the host
    loop in ``is_metered_destination`` hit ``continue``, and the match could
    never fire. Typing the name into ``METERED_DESTINATION_SERVICES`` would
    have satisfied every guard and stamped no header on any real request, while
    the 1082 SA-filed rows kept arriving.

    The fix registered the gateway, so the same three facts now read the other
    way round. Each assertion is the exact negation of the one it replaces —
    the test still asserts the MECHANISM (resolvability, then the predicate on
    a real base URL and a real request URL), never set membership.
    """
    assert "tts_gateway" in list_services()
    assert service_url("tts_gateway") == TTS_GATEWAY_URL
    assert unresolvable_metered_names(frozenset({"tts_gateway"})) == frozenset()
    # The shape a real caller passes: a base URL, which now MATCHES.
    assert is_metered_destination(TTS_GATEWAY_URL) is True
    assert is_metered_destination(TTS_GATEWAY_URL + "/api/tts/batches") is True
    # Matching is by HOSTNAME, port excluded (measured: the :9202 form matches
    # too), so the negative control has to be a DIFFERENT host. ``tts.example``
    # is the mlx-Kokoro host behind the gateway — it is a separate, declared,
    # non-delegating destination and must not be swept in by this change.
    assert is_metered_destination(service_url("tts")) is False


def test_a_vouched_tenant_now_reaches_tts_gateway_as_itself() -> None:
    """The end of the 1082 rows: the customer's tenant goes on the wire.

    The predicate is a means; the header is the end. Asserted on the full
    request URL shape a real narration call uses.
    """
    headers = build_downstream_tenant_headers(
        TENANT,
        TTS_GATEWAY_URL + "/api/tts/batches",
        TenantSource.LICENSE_TOKEN,
        downstream_license_token="svc-token",
    )
    assert headers[EFFECTIVE_TENANT_HEADER] == str(TENANT)
    assert headers[TENANT_HEADER] == "Bearer svc-token"


def test_no_metered_name_is_unresolvable() -> None:
    """The anti-inert invariant for the set as a whole.

    Every name in ``METERED_DESTINATION_SERVICES`` must be resolvable through
    discovery, or it is decoration. This is the assertion that stops the next
    session from "closing" the register by adding ``tts_gateway``.
    """
    assert unresolvable_metered_names() == frozenset(), (
        "metered destination(s) that discovery cannot resolve: "
        f"{sorted(unresolvable_metered_names())}. is_metered_destination "
        "matches on hosts, so such a name is inert — it stamps no "
        "X-Effective-Tenant-Id on any real request. Register the peer in "
        "discovery._SERVICES instead of listing it here."
    )


# ---------------------------------------------------------------------------
# D. The env var can no longer undo any of the above
# ---------------------------------------------------------------------------


def test_the_env_override_adds_instead_of_replacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE FOOTGUN, CLOSED.

    The low-effort mistake — naming the one peer you want to add — used to
    demote every other destination fleet-wide, with no error and no log line.
    It is now exactly what it looks like: one addition.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "course")
    assert is_metered_destination("course") is True
    for name in ("worker_management", "inference_router", *FLIPPED):
        assert is_metered_destination(name) is True, f"{name} was silently demoted"


@pytest.mark.parametrize("raw", ["", " ", ",", " , , "])
def test_an_empty_env_override_fails_loudly(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The value that used to demote the ENTIRE fleet in silence.

    There is no reading of "the operator set this variable" that means "meter
    nothing", so the only correct behaviour is to refuse.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", raw)
    with pytest.raises(MeteredDestinationConfigError, match="W2V_METERED_DESTINATIONS"):
        is_metered_destination("worker_management")


def test_an_unknown_added_name_is_inert_and_warned_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CHANGED ON PURPOSE 0.74.1. This test used to assert a raise.

    0.74.0 rejected any ADD that was not a discovery name. That was a mistake
    on two counts. It broke live config (consumers name HOSTS — see
    ``TestHostFormsAreValidConfig``), and it was redundant: the footgun was
    REPLACEMENT, and addition already defused it. Under addition an
    unrecognised name adds a string nothing matches — it meters nothing and
    demotes nothing. Turning that into a hard failure took the blast radius of
    a typo from zero to "the process cannot make any outbound call".

    So a typo is now LOUD but not FATAL: warned once, naming the token.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "workr_management")
    with caplog.at_level("WARNING", logger="w2v_common.auth.outbound"):
        assert is_metered_destination("inference_router") is True  # not demoted
        assert is_metered_destination("workr_management") is True  # inert-but-honoured literal

    assert any("workr_management" in record.getMessage() for record in caplog.records), (
        "an unrecognised name must be named in a warning, or a typo is invisible"
    )

    # The inertness is the actual safety property: it changes no REAL destination.
    for name in ("worker_management", "inference_router", *FLIPPED):
        assert is_metered_destination(service_url(name)) is True
    assert is_metered_destination(service_url("course")) is False


def test_a_deployment_can_still_demote_one_destination_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one legitimate use of the old semantics survives — by name.

    "Here, inference_router is NOT metered" is a real deployment need. It now costs one
    explicit token per destination, so the dangerous edit is the verbose one.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "-inference_router")
    assert is_metered_destination("inference_router") is False
    assert is_metered_destination("worker_management") is True
    assert is_metered_destination("analysis") is True


def test_adding_and_demoting_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "course, -inference_router")
    assert is_metered_destination("course") is True
    assert is_metered_destination("inference_router") is False
    assert is_metered_destination("analysis") is True


def test_kag_is_no_longer_a_metered_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    """W0b: the retired KAG gateway left the metered set with its discovery entry.

    A leftover ``-kag`` demotion is therefore STALE config and must raise like
    any other demotion of a destination that is not metered, rather than being
    swallowed while the operator believes it still does something.
    """
    monkeypatch.delenv("W2V_METERED_DESTINATIONS", raising=False)
    assert "kag" not in METERED_DESTINATION_SERVICES
    assert is_metered_destination("kag") is False
    assert is_metered_destination("http://kag.kag:8900/ingest") is False

    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "-kag")
    with pytest.raises(MeteredDestinationConfigError, match="kag"):
        is_metered_destination("worker_management")


def test_demoting_something_that_is_not_metered_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale demotion is a config line asserting something untrue."""
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "-course")
    with pytest.raises(MeteredDestinationConfigError, match="course"):
        is_metered_destination("worker_management")


def test_the_env_override_cannot_silently_empty_the_metered_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property, stated once: no value demotes a peer it did not name.

    Either the value names the demotion explicitly, or the peer stays metered,
    or the process raises. There is no fourth outcome — and the third is what
    every malformed value now produces.
    """
    for raw in ("", ",", "course", "workr_management", "-course", "worker_management"):
        monkeypatch.setenv("W2V_METERED_DESTINATIONS", raw)
        try:
            still_metered = is_metered_destination("analysis")
        except MeteredDestinationConfigError:
            continue
        assert still_metered is True, (
            f"W2V_METERED_DESTINATIONS={raw!r} demoted 'analysis' without naming it"
        )


# ---------------------------------------------------------------------------
# E. Host forms — the 0.74.0 regression, pinned
# ---------------------------------------------------------------------------

#: The two values MEASURED in live consumer CI when 0.74.0's strict validator
#: shipped. Written as literals, not derived from discovery: deriving them from
#: ``service_url`` would make the test pass against a validator that only
#: accepts discovery keys, which is the exact defect being pinned.
LIVE_HOST_FORMS = (
    # w2v_orchestrator
    "wm.example",
    # ai_content_generation_service
    "modality-gateway.modality-gateway",
)


class TestHostFormsAreValidConfig:
    """0.74.0 raised on production config. This is the regression that shipped.

    The validator checked every token against ``discovery.list_services()``, but
    consumers name HOSTS, and ``is_metered_destination`` matches on hostnames —
    so both values below were correct, working config that the validator
    declared illegal. The lock bumps were already on those consumers' mains, so
    the next deploy would have raised ``MeteredDestinationConfigError`` on the
    first outbound call in the process.
    """

    @pytest.mark.parametrize("value", LIVE_HOST_FORMS)
    def test_a_live_host_form_does_not_raise(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("W2V_METERED_DESTINATIONS", value)
        assert is_metered_destination("worker_management") is True

    def test_both_live_host_forms_together_do_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("W2V_METERED_DESTINATIONS", ", ".join(LIVE_HOST_FORMS))
        assert is_metered_destination("inference_router") is True

    def test_a_host_form_actually_meters_that_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NOT INERT WIRING — the entry has to change a real answer.

        Accepting the value without raising is only half the fix. If the entry
        validated but never matched, the config would be accepted and do
        nothing, which is this codebase's signature failure mode. ``wm.example``
        is not a discovery name and resolves through nothing, so the ONLY way it
        can work is direct hostname comparison — assert exactly that, on a full
        request URL with a path, which is the shape ``identity_event_hook``
        passes.
        """
        url = "http://wm.example:8887/workers/abc/message"
        assert is_metered_destination(url) is False  # baseline: not metered by default

        monkeypatch.setenv("W2V_METERED_DESTINATIONS", "wm.example")
        assert is_metered_destination(url) is True
        assert is_metered_destination("https://wm.example/") is True

    def test_a_host_form_produces_the_delegation_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The predicate is a means; the wire is the end."""
        monkeypatch.setenv("W2V_METERED_DESTINATIONS", "wm.example")
        headers = build_downstream_tenant_headers(
            TENANT,
            "http://wm.example:8887/workers/abc/message",
            TenantSource.LICENSE_TOKEN,
            downstream_license_token="svc-token",
        )
        assert headers[EFFECTIVE_TENANT_HEADER] == str(TENANT)
        assert headers[TENANT_HEADER] == "Bearer svc-token"

    def test_a_host_form_demotes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Additive stays additive — the 0.74.0 fix that was actually correct."""
        monkeypatch.setenv("W2V_METERED_DESTINATIONS", "wm.example")
        for name in ("worker_management", "inference_router", "modality_gateway", *FLIPPED):
            assert is_metered_destination(service_url(name)) is True, f"{name} demoted"

    def test_a_url_form_is_reduced_to_its_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator who pastes a URL gets the host they meant, not a dead literal."""
        monkeypatch.setenv("W2V_METERED_DESTINATIONS", "http://wm.example:8887/")
        assert is_metered_destination("http://wm.example:8887/workers/x") is True


# ---------------------------------------------------------------------------
# F. The demote direction is still strict, and the set can never be emptied
# ---------------------------------------------------------------------------


def test_an_unknown_demotion_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The asymmetry, stated as a test.

    An unknown ADD is inert — it meters nothing extra. An unknown DEMOTE is
    NOT inert in the operator's head: they believe a destination stopped being
    metered and it did not. Silent acceptance leaves live config asserting
    something false about billing, so this direction stays fatal.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "-workr_management")
    with pytest.raises(MeteredDestinationConfigError, match="workr_management"):
        is_metered_destination("worker_management")


def test_a_host_form_demotion_raises_because_it_is_not_metered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permissiveness on the ADD side must not leak into the DEMOTE side."""
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "-wm.example")
    with pytest.raises(MeteredDestinationConfigError, match=re.escape("wm.example")):
        is_metered_destination("worker_management")


def test_demoting_every_destination_by_name_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last door to the fleet-wide off switch, closed.

    Every other emptying path raises on the way in. This one is spelled
    correctly at every token — it is only the RESULT that is catastrophic — so
    it has to be caught on the result.
    """
    raw = ",".join(f"-{name}" for name in sorted(METERED_DESTINATION_SERVICES))
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", raw)
    with pytest.raises(MeteredDestinationConfigError, match="nothing metered"):
        is_metered_destination("worker_management")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        ",",
        "course",
        "workr_management",
        "-course",
        "worker_management",
        "wm.example",
        "-workr_management",
    ],
)
def test_no_value_of_the_env_var_can_demote_an_unnamed_destination(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property, restated to include the values 0.74.1 newly ACCEPTS.

    Either the value names a demotion explicitly, or the peer stays metered, or
    the process raises. Accepting host forms widened what parses — it must not
    have widened what can be demoted.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", raw)
    with contextlib.suppress(MeteredDestinationConfigError):
        assert is_metered_destination(service_url("analysis")) is True
