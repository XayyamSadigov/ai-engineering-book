# path: book/projects/examples/ch37/code_search.py
"""Retrieval over code: symbols first, lexical second, structure always.

- A symbol index from Python's `ast`: every module, class, function, and method with its
  qualified name, location, signature, docstring, the names it calls, and its imports.
- Exact lookups an embedding cannot do reliably: definition of a name, callers of a function,
  modules that depend on a module.
- Identifier-aware lexical search: `complete_structured` and `RetryPolicy` are split into words,
  and matches in a symbol's name weigh more than matches in its docstring or body.
- AST-aware chunks: a symbol's full source plus the context a reader needs (module imports and
  the enclosing class header), never a fixed-size window that cuts a function in half.
"""
from __future__ import annotations

import ast
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

STOP = frozenset("self cls none true false return def class import from the a an of to and or in is for if".split())


def split_identifier(name: str) -> list[str]:
    """complete_structured -> [complete, structured]; RetryPolicy -> [retry, policy]; HTTPError -> [http, error]."""
    words: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", name):
        words += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
    return [w.lower() for w in words if w and w.lower() not in STOP]


@dataclass
class Symbol:
    qualname: str  # aie_core.llm.gateway.ModelGateway.complete
    name: str
    kind: Literal["class", "function", "method"]
    module: str
    path: Path
    lineno: int
    end_lineno: int
    signature: str
    doc: str
    parent: str | None = None  # qualname of the enclosing class
    calls: set[str] = field(default_factory=set)  # bare names called in the body
    body_terms: Counter[str] = field(default_factory=Counter)


class CodeIndex:
    def __init__(self) -> None:
        self.symbols: dict[str, Symbol] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.imports: dict[str, set[str]] = defaultdict(set)  # module -> internal modules it imports
        self.module_imports_src: dict[str, list[str]] = {}  # module -> its import lines (for chunks)
        self.sources: dict[str, list[str]] = {}  # module -> source lines
        self._idf: dict[str, float] = {}

    # ---------------------------------------------------------------------- build
    @classmethod
    def build(cls, package_dir: Path) -> "CodeIndex":
        idx = cls()
        root = package_dir.parent
        for path in sorted(package_dir.rglob("*.py")):
            rel = path.relative_to(root).with_suffix("")
            module = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            idx.sources[module] = src.splitlines()
            idx._index_imports(module, tree, is_package=rel.name == "__init__", package=package_dir.name)
            idx._index_defs(module, path, tree.body, parent=None)
        idx._compute_idf()
        return idx

    def _index_imports(self, module: str, tree: ast.Module, is_package: bool, package: str) -> None:
        lines: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = module if is_package else module.rpartition(".")[0]
                for _ in range(max(0, node.level - 1)):
                    base = base.rpartition(".")[0]
                absolute = (f"{base}.{node.module}" if node.module else base) if node.level else (node.module or "")
                targets = [absolute] + [f"{absolute}.{a.name}" for a in node.names]
            else:
                continue
            lines.append(ast.get_source_segment("\n".join(self.sources[module]), node) or "")
            for t in targets:
                if t.split(".")[0] == package:
                    self.imports[module].add(t)
        self.module_imports_src[module] = [line for line in lines if line]

    def _index_defs(self, module: str, path: Path, body: list[ast.stmt], parent: Symbol | None) -> None:
        for node in body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            prefix = parent.qualname if parent else module
            kind: Literal["class", "function", "method"] = (
                "class" if isinstance(node, ast.ClassDef) else "method" if parent and parent.kind == "class" else "function"
            )
            sym = Symbol(
                qualname=f"{prefix}.{node.name}", name=node.name, kind=kind, module=module, path=path,
                lineno=node.lineno, end_lineno=node.end_lineno or node.lineno,
                signature=self._signature(node), doc=(ast.get_docstring(node) or "").split("\n")[0],
                parent=parent.qualname if parent else None,
            )
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    if isinstance(f, ast.Name):
                        sym.calls.add(f.id)
                    elif isinstance(f, ast.Attribute):
                        sym.calls.add(f.attr)
                if isinstance(sub, ast.Name):
                    sym.body_terms.update(split_identifier(sub.id))
                elif isinstance(sub, ast.Attribute):
                    sym.body_terms.update(split_identifier(sub.attr))
            self.symbols[sym.qualname] = sym
            self.by_name[sym.name].append(sym.qualname)
            if isinstance(node, ast.ClassDef):
                self._index_defs(module, path, node.body, parent=sym)

    @staticmethod
    def _signature(node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            return f"class {node.name}({bases})" if bases else f"class {node.name}"
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"

    def _fields(self, s: Symbol) -> tuple[list[str], list[str], Counter[str]]:
        name_terms = split_identifier(s.name) + (split_identifier(s.parent.rsplit(".", 1)[-1]) if s.parent else [])
        return name_terms, split_identifier(s.doc), s.body_terms

    def _compute_idf(self) -> None:
        df: Counter[str] = Counter()
        for s in self.symbols.values():
            name, doc, body = self._fields(s)
            df.update(set(name) | set(doc) | set(body))
        n = len(self.symbols)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    # ---------------------------------------------------------------------- exact queries
    def find_definition(self, name: str) -> list[Symbol]:
        """By bare name ('complete') or qualified suffix ('ModelGateway.complete')."""
        if "." in name:
            return [s for q, s in self.symbols.items() if q.endswith("." + name)]
        return [self.symbols[q] for q in self.by_name.get(name, [])]

    def callers(self, name: str) -> list[Symbol]:
        """Name-based: fast and dependency-free, but it conflates different functions that share a
        name. A type-aware index (a language server, SCIP, or a compiler) resolves the real target."""
        return sorted(
            (s for s in self.symbols.values() if s.kind != "class" and name in s.calls and s.name != name),
            key=lambda s: s.qualname,
        )

    def dependents(self, module: str) -> list[str]:
        """Modules that import `module` or a name from it: the blast radius of changing it."""
        return sorted(m for m, targets in self.imports.items() if any(t == module or t.startswith(module + ".") for t in targets) and m != module)

    def dependencies(self, module: str) -> list[str]:
        mods = set()
        for t in self.imports.get(module, set()):
            if t in self.sources:  # a module
                mods.add(t)
            elif t.rpartition(".")[0] in self.sources and t.rpartition(".")[0] not in self.imports[module]:
                mods.add(t.rpartition(".")[0])  # a name imported from a module not listed itself
        return sorted(mods - {module})

    # ---------------------------------------------------------------------- lexical search
    def search(self, query: str, k: int = 5) -> list[tuple[Symbol, float]]:
        q = split_identifier(query)
        out = []
        for s in self.symbols.values():
            name, doc, body = self._fields(s)
            score = 0.0
            for t in q:
                idf = self._idf.get(t, 0.0)
                score += idf * (3.0 * (t in name) + 1.5 * (t in doc) + 0.3 * math.log1p(body[t]))
            if score > 0:
                out.append((s, round(score, 3)))
        out.sort(key=lambda x: (-x[1], x[0].qualname))
        return out[:k]

    def grep(self, pattern: str, max_hits: int = 20) -> list[str]:
        """The baseline every coding agent starts with: a regex over lines."""
        rx = re.compile(pattern)
        hits = [f"{m}:{i + 1}: {line.strip()}" for m, lines in self.sources.items() for i, line in enumerate(lines) if rx.search(line)]
        return hits[:max_hits]

    # ---------------------------------------------------------------------- chunks
    def chunk(self, sym: Symbol) -> str:
        """Symbol source plus the context needed to read it: path, imports, enclosing class header."""
        lines = self.sources[sym.module]
        parts = [f"# {sym.module} ({sym.path.name}:{sym.lineno}-{sym.end_lineno})"]
        parts += self.module_imports_src.get(sym.module, [])
        if sym.parent:
            cls = self.symbols[sym.parent]
            header = [cls.signature + ":"] + ([f'    """{cls.doc}"""'] if cls.doc else []) + ["    ..."]
            parts += [""] + header
        parts += [""] + lines[sym.lineno - 1 : sym.end_lineno]
        return "\n".join(parts)


def main() -> None:
    package = Path(__file__).resolve().parents[2] / "aie_core" / "aie_core"
    idx = CodeIndex.build(package)
    print(f"{len(idx.symbols)} symbols in {len(idx.sources)} modules")
    query = " ".join(sys.argv[1:]) or "retry backoff jitter"
    for s, score in idx.search(query):
        print(f"{score:6.2f}  {s.kind:8} {s.qualname}  {s.signature[:70]}")
    print("\ncallers of _prepare:", [s.qualname for s in idx.callers("_prepare")])
    print("dependents of aie_core.llm.errors:", idx.dependents("aie_core.llm.errors"))


if __name__ == "__main__":
    main()
