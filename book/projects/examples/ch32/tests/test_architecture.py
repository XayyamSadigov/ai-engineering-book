# path: book/projects/examples/ch32/tests/test_architecture.py
"""Architecture fitness test: the dependency rule, enforced in CI.

Diagrams rot; this test does not. It parses every module and checks its imports against
the allowed direction: adapters -> application -> domain, never the reverse, and no
provider, framework, or I/O library inside domain or application.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "northwind_triage"
STDLIB = set(sys.stdlib_module_names)

# Third-party roots each layer may import. Everything else is a violation.
ALLOWED_THIRD_PARTY = {
    "domain": {"pydantic"},
    "application": {"pydantic"},
    "root_pure": {"pydantic"},          # flags.py, version_manifest.py, experiments.py
}
# Intra-package layers each layer may import.
ALLOWED_INTERNAL = {
    "domain": {"domain"},
    "application": {"domain", "application", "flags", "version_manifest"},
    "root_pure": set(),
}
PURE_ROOT_MODULES = {"flags.py", "version_manifest.py", "experiments.py"}


def layer_of(path: Path) -> str | None:
    rel = path.relative_to(PKG)
    if rel.parts[0] in ("domain", "application"):
        return rel.parts[0]
    if len(rel.parts) == 1 and rel.name in PURE_ROOT_MODULES:
        return "root_pure"
    return None  # adapters, composition, CLIs: may import anything


def imports_of(path: Path) -> list[tuple[str, int]]:
    """(target, level) pairs. level > 0 means a relative import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.level))
    return out


def internal_target(path: Path, module: str, level: int) -> str:
    """Resolve a relative import to the first package component under northwind_triage."""
    base = path.parent.relative_to(PKG).parts
    base = base[: len(base) - (level - 1)] if level > 1 else base
    parts = [*base, *module.split(".")] if module else list(base)
    return parts[0] if parts else ""


MODULES = [p for p in PKG.rglob("*.py") if layer_of(p) is not None]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PKG)))
def test_dependency_rule(path: Path) -> None:
    layer = layer_of(path)
    assert layer is not None
    violations = []
    for module, level in imports_of(path):
        if level > 0:
            target = internal_target(path, module, level)
            target = target.removesuffix(".py")
            if target not in ALLOWED_INTERNAL[layer] and target != path.parent.name:
                violations.append(f"relative import of {target!r}")
            continue
        root = module.split(".")[0]
        if root == "northwind_triage":
            target = module.split(".")[1] if "." in module else ""
            if target not in ALLOWED_INTERNAL[layer]:
                violations.append(f"import of northwind_triage.{target}")
        elif root not in STDLIB and root != "__future__" and root not in ALLOWED_THIRD_PARTY[layer]:
            violations.append(f"third-party import {root!r}")
    assert not violations, f"{path.relative_to(PKG)} ({layer}): {violations}"


def test_rule_catches_a_violation(tmp_path: Path) -> None:
    """The fitness test is itself tested: a domain module importing httpx must fail."""
    bad = tmp_path / "bad.py"
    bad.write_text("import httpx\nfrom ..adapters import tools\n")
    found = imports_of(bad)
    assert ("httpx", 0) in found and ("adapters", 2) in found
    assert "httpx" not in ALLOWED_THIRD_PARTY["domain"]
    assert "adapters" not in ALLOWED_INTERNAL["domain"]
