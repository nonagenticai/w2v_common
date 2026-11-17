"""``TENANT_RLS_MODE`` must fail LOUD on a value it does not understand.

Every consumer tests ``!= "enforce"`` and silently returns, so any typo or
plausible-looking synonym disables tenant isolation with no error and no log line.
The fail-closed default only ever covered a MISSING key, never a wrong-valued one.
"""

from __future__ import annotations

import pytest

from w2v_common.database.rls import DEFAULT_TENANT_RLS_MODE, tenant_rls_mode


def test_missing_key_still_defaults_to_enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENANT_RLS_MODE", raising=False)
    assert tenant_rls_mode() == DEFAULT_TENANT_RLS_MODE == "enforce"


@pytest.mark.parametrize("value", ["enforce", "off", "  ENFORCE  ", "Off"])
def test_recognised_values_are_accepted_case_and_space_insensitively(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TENANT_RLS_MODE", value)
    assert tenant_rls_mode() in {"enforce", "off"}


@pytest.mark.parametrize("value", ["true", "on", "1", "enforced", "yes", "enable", "strict"])
def test_a_plausible_but_wrong_value_raises_instead_of_disabling_isolation(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """These all read as "isolation ON" to a human and meant "OFF" to the code."""
    monkeypatch.setenv("TENANT_RLS_MODE", value)
    with pytest.raises(ValueError, match="TENANT_RLS_MODE"):
        tenant_rls_mode()
