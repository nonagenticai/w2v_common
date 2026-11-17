"""Declared-floor regression test for known-CVE transitive dependencies.

Why this exists
---------------
`uv.lock` is a point-in-time resolution, not a constraint. Bumping a
transitive dependency in the lock alone fixes today's resolution and nothing
else: the next unrelated `uv lock` is free to walk an *undeclared* transitive
back DOWN below the CVE-safe version, silently, with green CI. The declared
floor in `pyproject.toml` is the only thing that binds it.

This test asserts both halves:

1. every known-CVE package that this repo actually resolves has a floor
   DECLARED in `pyproject.toml` — in whichever requirement list the package
   really resolves from — and that floor is `>=` the CVE-safe version;
2. the version actually INSTALLED in this environment also satisfies it.

Precedent: `analysis_service` (PRs #175/#176) introduced this floor block.

Adding the next CVE is a one-line addition to ``CVE_FLOORS``.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

# --- The table. One line per package with a known CVE floor. ---------------
# package name -> (minimum safe version, why)
CVE_FLOORS: dict[str, tuple[str, str]] = {
    "aiohttp": (
        "3.14.3",
        "CVE-2026-59881 + CVE-2026-69243 fixed in 3.14.2; CVE-2026-69244 fixed in 3.14.3",
    ),
    "cryptography": (
        "50.0.0",
        "CVE-2026-69247 / GHSA-g6cj-pr64-35w5, introduced in 44.0.0, fixed in 50.0.0",
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")
_SPEC_RE = re.compile(r">=\s*([0-9][0-9A-Za-z.\-_+!]*)")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_key(version: str) -> tuple[int, ...]:
    """Numeric-release tuple for a PEP 440 version (stdlib only)."""
    release = re.match(r"\d+(?:\.\d+)*", version.strip())
    if release is None:
        pytest.fail(f"unparseable version {version!r}")
    return tuple(int(part) for part in release.group(0).split("."))


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _as_table(container: dict[str, object], key: str) -> dict[str, object]:
    """Read a sub-table out of parsed TOML, tolerating its absence."""
    value = container.get(key)
    if value is None:
        return {}
    assert isinstance(value, dict), f"[{key}] is not a table"
    return {str(k): v for k, v in value.items()}


def _as_requirements(container: dict[str, object], key: str) -> list[str]:
    """Read a requirement list out of parsed TOML, tolerating its absence."""
    value = container.get(key)
    if value is None:
        return []
    assert isinstance(value, list), f"{key} is not an array"
    return [item for item in value if isinstance(item, str)]


def _requirement_lists() -> list[list[str]]:
    """Every requirement list a floor may legitimately be declared in.

    Not just ``[project].dependencies``: a package can reach this repo's lock
    through a dependency GROUP instead of the runtime closure. ``aiohttp`` is
    exactly that case here — it arrives only via the dev-only ``miniopy-async``.
    Declaring it in ``[project].dependencies`` to satisfy this test would add a
    real runtime dependency to all 17 consuming repos for a package none of them
    receive from w2v_common today, which is a worse outcome than the drift the
    floor is meant to prevent. So a floor counts as declared wherever the
    package actually resolves from.
    """
    data = _load_pyproject()
    project = _as_table(data, "project")

    lists: list[list[str]] = []

    runtime = _as_requirements(project, "dependencies")
    assert runtime, "[project].dependencies is empty — pyproject.toml parsed wrong"
    lists.append(runtime)

    for table_name, container in (
        ("optional-dependencies", project),
        ("dependency-groups", data),
    ):
        for value in _as_table(container, table_name).values():
            if isinstance(value, list):
                lists.append([item for item in value if isinstance(item, str)])

    return lists


def _declared_floors() -> dict[str, str]:
    """Map canonical package name -> the HIGHEST declared `>=` floor for it."""
    floors: dict[str, str] = {}
    for raw in (item for group in _requirement_lists() for item in group):
        name_match = _REQ_RE.match(raw)
        spec_match = _SPEC_RE.search(raw)
        if name_match is None or spec_match is None:
            continue
        name = _canonical(name_match.group(1))
        floor = spec_match.group(1)
        current = floors.get(name)
        if current is None or _version_key(floor) > _version_key(current):
            floors[name] = floor
    return floors


def _locked_packages() -> set[str]:
    """Canonical names of every package this repo actually resolves."""
    text = UV_LOCK.read_text(encoding="utf-8")
    return {_canonical(name) for name in re.findall(r'^name = "([^"]+)"$', text, re.MULTILINE)}


def _applicable() -> list[str]:
    locked = _locked_packages()
    return sorted(pkg for pkg in CVE_FLOORS if _canonical(pkg) in locked)


def test_lock_file_is_readable() -> None:
    """Guard the guard: an empty lock would make every case vacuously skip."""
    assert UV_LOCK.is_file(), f"{UV_LOCK} missing"
    assert len(_locked_packages()) > 10, "uv.lock parsed to implausibly few packages"


@pytest.mark.parametrize("package", CVE_FLOORS)
def test_cve_floor_is_declared_in_pyproject(package: str) -> None:
    """A package this repo resolves must carry an explicit, high-enough floor.

    Without the declaration the lock is the only thing holding the safe
    version, and the lock can silently drift back down.
    """
    if package not in _applicable():
        pytest.skip(f"{package} is not in this repo's uv.lock; no floor required")

    safe, reason = CVE_FLOORS[package]
    declared = _declared_floors().get(_canonical(package))

    assert declared is not None, (
        f"no declared floor for {package} in {PYPROJECT.name}, but it IS resolved in "
        f"uv.lock. Add `{package}>={safe}` to whichever requirement list it actually "
        f"resolves from ([project].dependencies, an extra, or a dependency group) "
        f"— {reason}. A lock-only fix drifts back on the next unrelated `uv lock`."
    )
    assert _version_key(declared) >= _version_key(safe), (
        f"declared floor {package}>={declared} is BELOW the CVE-safe {safe} — {reason}"
    )


@pytest.mark.parametrize("package", CVE_FLOORS)
def test_installed_version_satisfies_cve_floor(package: str) -> None:
    """The floor must also be true of what is actually installed right now."""
    if package not in _applicable():
        pytest.skip(f"{package} is not in this repo's uv.lock; nothing installed to check")

    safe, reason = CVE_FLOORS[package]
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise AssertionError(
            f"{package} is resolved in uv.lock but is not installed in this "
            "environment, so the CVE floor cannot be verified. A skip here would "
            "turn a real gap into a green run."
        ) from exc

    assert _version_key(installed) >= _version_key(safe), (
        f"installed {package}=={installed} is BELOW the CVE-safe {safe} — {reason}"
    )
