"""The static guard that keeps the 0.67.7 fix from being undone.

Every case here is paired: something the guard MUST catch, and something it
must NOT. A guard is only worth having if both halves hold — one that fires on
its own documentation gets disabled within a week.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from w2v_common.auth.provenance_guard import (
    assert_no_contextvar_passthrough,
    find_contextvar_passthroughs,
)

# --------------------------------------------------------------------------
# MUST catch
# --------------------------------------------------------------------------

_DIRECT = """
from w2v_common.auth.identity_context import (
    get_tenant_source_from_context,
    set_identity_context,
)


async def _bg(tenant):
    set_identity_context(tenant, None, None, tenant_source=get_tenant_source_from_context())
"""

_QUALIFIED = """
from w2v_common.auth import identity_context


async def _bg(tenant):
    identity_context.set_identity_context(
        tenant,
        None,
        None,
        tenant_source=identity_context.get_tenant_source_from_context(),
    )
"""

_VIA_LOCAL = """
from w2v_common.auth.identity_context import (
    get_tenant_source_from_context,
    set_identity_context,
)


async def _bg(tenant):
    source = get_tenant_source_from_context()
    set_identity_context(tenant, None, None, tenant_source=source)
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [("direct", _DIRECT), ("qualified", _QUALIFIED), ("via_local", _VIA_LOCAL)],
)
def test_the_guard_catches_the_passthrough(label: str, source: str, tmp_path: Path) -> None:
    offender = tmp_path / f"{label}.py"
    offender.write_text(source)

    sites = find_contextvar_passthroughs(tmp_path)

    assert len(sites) == 1, f"{label}: expected exactly one site, got {sites}"
    assert sites[0].path == offender


def test_the_failure_names_the_file_and_line(tmp_path: Path) -> None:
    """A guard that only says "something is wrong" costs more than it saves."""
    offender = tmp_path / "routes.py"
    offender.write_text(_DIRECT)

    with pytest.raises(AssertionError) as excinfo:
        assert_no_contextvar_passthrough(tmp_path)

    message = str(excinfo.value)
    assert "routes.py:9" in message
    assert "bypasses the different-tenant drop" in message


def test_the_local_variable_hop_is_reported_as_such(tmp_path: Path) -> None:
    (tmp_path / "indirect.py").write_text(_VIA_LOCAL)

    (site,) = find_contextvar_passthroughs(tmp_path)

    assert site.via == "source"
    assert "via local `source`" in str(site)


# --------------------------------------------------------------------------
# MUST NOT catch
# --------------------------------------------------------------------------

_BARE_CALL = """
from w2v_common.auth.identity_context import set_identity_context


async def _bg(tenant):
    # The correct shape since 0.67.7: omission preserves for the same tenant
    # and drops on a tenant change.
    set_identity_context(tenant, None, None)
"""

_EXPLICIT_NONE = """
from w2v_common.auth.identity_context import set_identity_context


async def _reaper(rows):
    for row in rows:
        set_identity_context(row.tenant_id, None, None, tenant_source=None)
"""

_FROM_SNAPSHOT = """
from w2v_common.auth.identity_context import set_identity_context


async def _redispatch(job):
    # The endorsed pattern: provenance read from a PERSISTED row, not the
    # ambient ContextVar.
    set_identity_context(
        job.tenant_id, None, None, tenant_source=_restore_tenant_source(job.options)
    )
"""

_ONLY_IN_PROSE = '''
"""Explains the anti-pattern without committing it.

Never write::

    set_identity_context(tenant, None, None, tenant_source=get_tenant_source_from_context())

because an explicit value always wins.
"""

from w2v_common.auth.identity_context import set_identity_context


async def _bg(tenant):
    # set_identity_context(tenant, None, None, tenant_source=get_tenant_source_from_context())
    set_identity_context(tenant, None, None)
'''

_READ_BUT_NOT_PASSED = """
from w2v_common.auth.identity_context import (
    get_tenant_source_from_context,
    set_identity_context,
)


async def _bg(tenant):
    # Reading provenance to LOG it is fine; passing it back in is not.
    logger.info("inherited=%s", get_tenant_source_from_context())
    set_identity_context(tenant, None, None)
"""

_TAINT_DOES_NOT_LEAK_ACROSS_FUNCTIONS = """
from w2v_common.auth.identity_context import (
    get_tenant_source_from_context,
    set_identity_context,
)


def _audit():
    source = get_tenant_source_from_context()
    return source


def _bg(tenant, source):
    set_identity_context(tenant, None, None, tenant_source=source)
"""


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("bare_call", _BARE_CALL),
        ("explicit_none", _EXPLICIT_NONE),
        ("from_snapshot", _FROM_SNAPSHOT),
        ("only_in_prose", _ONLY_IN_PROSE),
        ("read_but_not_passed", _READ_BUT_NOT_PASSED),
        ("taint_scoped_to_function", _TAINT_DOES_NOT_LEAK_ACROSS_FUNCTIONS),
    ],
)
def test_the_guard_stays_quiet(label: str, source: str, tmp_path: Path) -> None:
    (tmp_path / f"{label}.py").write_text(source)

    assert find_contextvar_passthroughs(tmp_path) == [], label


# --------------------------------------------------------------------------
# Against the real tree
# --------------------------------------------------------------------------


def test_this_package_is_clean() -> None:
    """Non-vacuity on real code.

    ``provenance_guard.py``'s own docstring quotes the offending call verbatim,
    so a text-scanning implementation would fail right here. Passing proves the
    parse-don't-grep property against a genuine file rather than a fixture.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "w2v_common"
    assert src.is_dir(), src

    assert_no_contextvar_passthrough(src)


def test_a_file_that_cannot_be_parsed_is_not_silently_skipped(tmp_path: Path) -> None:
    """Silently ignoring unreadable files is how a guard reports a false all-clear."""
    (tmp_path / "broken.py").write_text("def (:\n")

    with pytest.raises(SyntaxError):
        find_contextvar_passthroughs(tmp_path)
