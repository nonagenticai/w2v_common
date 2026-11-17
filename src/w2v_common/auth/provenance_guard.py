"""Static guard against re-introducing the ContextVar provenance pass-through.

Since 0.67.7 ``set_identity_context`` distinguishes "the caller did not mention
provenance" from "the caller said None"::

    omitted, same tenant      -> provenance PRESERVED
    omitted, different tenant -> provenance DROPPED to None   <- the safety net
    tenant_source=None        -> cleared
    tenant_source=<value>     -> always wins

That makes ``tenant_source=get_tenant_source_from_context()`` actively harmful.
It reads as "preserve what I inherited", but because an explicit value *always
wins* it also **bypasses the different-tenant drop**: a background job that
rebinds some other tenant would carry the first tenant's proof onto it, which is
exactly the escalation the outbound guard exists to stop. Omitting the argument
is both shorter and strictly safer.

The shape was correct against <=0.67.6, where the assignment was unconditional,
so several services legitimately shipped it. This module exists so that fixing
them stays fixed.

Consumers wire it into their own suite with one test::

    from w2v_common.auth.provenance_guard import assert_no_contextvar_passthrough

    def test_no_contextvar_provenance_passthrough() -> None:
        assert_no_contextvar_passthrough("src")

Pure stdlib on purpose: no third-party import, so any consumer can call it
without taking on a test dependency.

⚠️ This is AST-based, never a text scan. The offending call shape is *quoted in
the prose that explains it* — including in this docstring — so a regex over the
source flags its own documentation. Parsing is what makes the difference between
a guard and a false-positive generator.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PassthroughSite",
    "assert_no_contextvar_passthrough",
    "find_contextvar_passthroughs",
]

#: The call being guarded.
_TARGET_CALL = "set_identity_context"
#: The keyword whose explicit value defeats the tenant-change drop.
_GUARDED_KEYWORD = "tenant_source"
#: Reading provenance back out of the ambient context.
_CONTEXT_READER = "get_tenant_source_from_context"


@dataclass(frozen=True)
class PassthroughSite:
    """One offending call site."""

    path: Path
    line: int
    #: How the value reached the keyword: "direct" or the local variable's name.
    via: str

    def __str__(self) -> str:
        detail = "" if self.via == "direct" else f" (via local `{self.via}`)"
        return f"{self.path}:{self.line}{detail}"


def _callee_name(node: ast.expr) -> str | None:
    """Final attribute of a call target, so `mod.f()` and `f()` both match."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_context_read(node: ast.expr | None) -> bool:
    """True for `get_tenant_source_from_context()` in either import style."""
    return isinstance(node, ast.Call) and _callee_name(node.func) == _CONTEXT_READER


class _Visitor(ast.NodeVisitor):
    """Find guarded calls, resolving one hop of local-variable indirection.

    Tracking assignments matters: splitting the read onto its own line is the
    obvious way around a naive check, and it has identical semantics.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.found: list[PassthroughSite] = []
        # Locals in the current function bound to a context read.
        self._tainted: set[str] = set()

    #: Nodes that open a new local scope, so taint cannot leak between them.
    _SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def visit(self, node: ast.AST) -> None:
        """Reset tracked locals when entering a new function scope.

        Dispatching here rather than defining ``visit_FunctionDef`` keeps the
        camelCase names ``NodeVisitor`` expects out of the class body.
        """
        if isinstance(node, self._SCOPES):
            outer, self._tainted = self._tainted, set()
            super().visit(node)
            self._tainted = outer
            return
        super().visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_context_read(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._tainted.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _is_context_read(node.value) and isinstance(node.target, ast.Name):
            self._tainted.add(node.target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _callee_name(node.func) == _TARGET_CALL:
            for kw in node.keywords:
                if kw.arg != _GUARDED_KEYWORD:
                    continue
                via: str | None = None
                if _is_context_read(kw.value):
                    via = "direct"
                elif isinstance(kw.value, ast.Name) and kw.value.id in self._tainted:
                    via = kw.value.id
                if via is not None:
                    self.found.append(
                        PassthroughSite(path=self.path, line=kw.value.lineno, via=via)
                    )
        self.generic_visit(node)


def find_contextvar_passthroughs(*roots: str | Path) -> list[PassthroughSite]:
    """Return every offending call site under ``roots``.

    Directories are walked for ``*.py``; a path to a file is checked directly.
    A file that does not parse raises ``SyntaxError`` rather than being skipped
    — a guard that silently ignores what it cannot read is not a guard.
    """
    sites: list[PassthroughSite] = []
    for root in roots:
        base = Path(root)
        paths = sorted(base.rglob("*.py")) if base.is_dir() else [base]
        for path in paths:
            visitor = _Visitor(path)
            visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            sites.extend(visitor.found)
    return sites


def assert_no_contextvar_passthrough(*roots: str | Path) -> None:
    """Fail naming every offending site, or return quietly."""
    sites = find_contextvar_passthroughs(*roots)
    if not sites:
        return
    listed = "\n".join(f"  - {site}" for site in sites)
    raise AssertionError(
        f"{len(sites)} call(s) pass provenance explicitly out of the ambient "
        f"context:\n{listed}\n\n"
        f"An explicit `{_GUARDED_KEYWORD}` ALWAYS wins, so this bypasses the "
        "different-tenant drop and lets a rebound tenant inherit another "
        "tenant's proof. Delete the keyword: since w2v-common 0.67.7, omitting "
        "it preserves provenance for the same tenant and drops it when the "
        "tenant changes, which is the behaviour these call sites want."
    )
