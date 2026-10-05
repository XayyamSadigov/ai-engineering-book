# path: book/projects/examples/ch36/sql_guard.py
"""SQL safety validator for an analytics assistant (Chapter 36, case B).

The model proposes SQL; this module authorizes it. It is the code-side half of
"the model proposes, code authorizes" applied to a warehouse. The guard is
deliberately strict: anything it cannot prove safe is rejected.

Checks, in order:
  1. no SQL comments (`--`, `/* */`) outside string literals
  2. exactly one statement
  3. read-only: a single SELECT (optionally with CTEs / set operations)
  4. every referenced table is on the allowlist (CTE names are not tables)
  5. no forbidden functions (sleep, file access, remote links ...)
  6. no blocked columns (PII), optionally no `SELECT *`
  7. a LIMIT exists and is <= the configured maximum (added or clamped)

Two engines implement the same contract. `sqlglot` parses the statement into
an AST and is the preferred engine. The regex engine is a conservative fallback
for environments where sqlglot is unavailable; it accepts fewer queries, never
more. Both engines are exercised by the same test suite.

The guard is one layer, not the whole defense. Run the resulting SQL under a
read-only database role with a statement timeout and a row cap of its own.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

try:  # pragma: no cover - exercised indirectly by the engine tests
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError

    HAVE_SQLGLOT = True
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]
    ParseError = Exception  # type: ignore[assignment,misc]
    HAVE_SQLGLOT = False

Engine = Literal["auto", "sqlglot", "regex"]

DEFAULT_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        # time-based probing / resource abuse
        "pg_sleep", "pg_sleep_for", "pg_sleep_until", "sleep", "benchmark", "waitfor",
        # file system and server access
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
        "lo_import", "lo_export", "load_file", "xp_cmdshell", "copy_from",
        # remote links and dynamic SQL
        "dblink", "dblink_exec", "dblink_connect", "openrowset", "opendatasource",
        # settings and session state
        "set_config", "current_setting",
    }
)

WRITE_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke",
    "merge", "call", "exec", "execute", "copy", "into", "lock", "vacuum", "analyze",
    "set", "show", "pragma", "attach", "detach", "begin", "commit", "rollback", "do",
    "replace", "upsert", "load", "import", "export", "reindex", "cluster", "listen", "notify",
)


class GuardConfig(BaseModel):
    """What the analytics assistant is allowed to ask the warehouse."""

    allowed_tables: set[str] = Field(default_factory=set)
    """Table names, lowercase. Either `table` or `schema.table` matches."""
    default_limit: int = 200
    max_limit: int = 1000
    forbidden_functions: set[str] = Field(
        default_factory=lambda: set(DEFAULT_FORBIDDEN_FUNCTIONS)
    )
    blocked_columns: set[str] = Field(default_factory=set)
    """Column names (lowercase) that must never be selected, filtered or joined on."""
    forbid_select_star: bool = False
    dialect: str | None = None
    """sqlglot dialect name used to parse and to render; None means generic SQL."""


class Violation(BaseModel):
    code: str
    message: str


class GuardResult(BaseModel):
    ok: bool
    sql: str
    """Normalized SQL safe to execute when ok; the input (trimmed) otherwise."""
    tables: list[str] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    engine: Literal["sqlglot", "regex"]

    @property
    def codes(self) -> set[str]:
        return {v.code for v in self.violations}


# --------------------------------------------------------------------------
# shared text-level helpers
# --------------------------------------------------------------------------
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", re.S)
_DOLLAR_QUOTE = re.compile(r"\$([A-Za-z_]*)\$.*?\$\1\$", re.S)


def strip_string_literals(sql: str) -> str:
    """Replace every string literal with an empty literal so that keyword and
    comment checks cannot be fooled by text such as `WHERE note = '; DROP'`."""
    sql = _DOLLAR_QUOTE.sub("''", sql)
    return _STRING_LITERAL.sub(lambda m: "''" if m.group(0)[0] == "'" else '""', sql)


def _comment_violations(stripped: str) -> list[Violation]:
    out: list[Violation] = []
    if "--" in stripped:
        out.append(Violation(code="comment", message="line comment `--` is not allowed"))
    if "/*" in stripped or "*/" in stripped:
        out.append(Violation(code="comment", message="block comment `/* */` is not allowed"))
    return out


def _statement_count(stripped: str) -> int:
    parts = [p for p in stripped.split(";") if p.strip()]
    return len(parts)


def _table_allowed(name: str, allowed: set[str]) -> bool:
    name = name.lower().strip('"`[]')
    if name in allowed:
        return True
    last = name.rsplit(".", 1)[-1]
    return last in allowed


def _normalize_table_name(name: str) -> str:
    return ".".join(part.strip('"`[]').lower() for part in name.split("."))


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------
class SqlGuard:
    def __init__(self, config: GuardConfig, engine: Engine = "auto") -> None:
        self.config = config
        if engine == "auto":
            engine = "sqlglot" if HAVE_SQLGLOT else "regex"
        if engine == "sqlglot" and not HAVE_SQLGLOT:
            raise RuntimeError("sqlglot engine requested but sqlglot is not installed")
        self.engine: Literal["sqlglot", "regex"] = engine  # type: ignore[assignment]

    # -- public -------------------------------------------------------------
    def check(self, sql: str) -> GuardResult:
        text = sql.strip()
        if not text:
            return GuardResult(
                ok=False, sql=text, engine=self.engine,
                violations=[Violation(code="empty", message="empty statement")],
            )

        stripped = strip_string_literals(text)
        violations = _comment_violations(stripped)
        count = _statement_count(stripped)
        if count != 1:
            violations.append(
                Violation(code="multiple_statements", message=f"expected 1 statement, found {count}")
            )
        if violations:
            # Do not bother parsing something we will reject anyway.
            return GuardResult(ok=False, sql=text, engine=self.engine, violations=violations)

        text = text.rstrip(";").strip()
        if self.engine == "sqlglot":
            return self._check_sqlglot(text)
        return self._check_regex(text, strip_string_literals(text))

    # -- sqlglot engine -------------------------------------------------------
    def _check_sqlglot(self, text: str) -> GuardResult:
        cfg = self.config
        violations: list[Violation] = []
        try:
            statements = sqlglot.parse(text, read=cfg.dialect)
        except ParseError as e:
            return GuardResult(
                ok=False, sql=text, engine="sqlglot",
                violations=[Violation(code="parse_error", message=str(e).splitlines()[0])],
            )
        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            return GuardResult(
                ok=False, sql=text, engine="sqlglot",
                violations=[Violation(code="multiple_statements", message=f"found {len(statements)} statements")],
            )
        root = statements[0]

        # 3. read-only. The root must be a query, and no write/DDL node may
        #    appear anywhere in the tree (e.g. a DELETE hidden inside a CTE).
        if not isinstance(root, exp.Query) or isinstance(root, exp.Subquery):
            violations.append(Violation(code="not_select", message=f"root is {type(root).__name__}, expected SELECT"))
        forbidden_types = tuple(
            t for t in (
                getattr(exp, n, None) for n in (
                    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge",
                    "TruncateTable", "Grant", "Command", "Transaction", "Commit", "Rollback",
                    "Set", "Lock", "Into", "Copy", "Use", "Describe", "Show", "Pragma", "Attach", "Detach",
                )
            ) if t is not None
        )
        for node in root.walk():
            if isinstance(node, forbidden_types):
                violations.append(Violation(code="not_read_only", message=f"{type(node).__name__} is not allowed"))
                break
        if root.args.get("locks"):
            violations.append(Violation(code="not_read_only", message="FOR UPDATE / FOR SHARE is not allowed"))

        # 4. tables. CTE aliases are virtual and never checked against the
        #    allowlist; the real tables inside them are.
        cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
        tables: list[str] = []
        for t in root.find_all(exp.Table):
            if not isinstance(t.this, exp.Identifier):
                violations.append(Violation(code="table_function", message=f"table-valued expression not allowed: {t.sql()}"))
                continue
            parts = [p for p in (t.catalog, t.db, t.name) if p]
            full = ".".join(p.lower() for p in parts)
            if len(parts) == 1 and full in cte_names:
                continue
            tables.append(full)
            if not _table_allowed(full, cfg.allowed_tables):
                violations.append(Violation(code="table_not_allowed", message=f"table `{full}` is not on the allowlist"))

        # 5. functions
        for f in root.find_all(exp.Func):
            name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
            if name in cfg.forbidden_functions:
                violations.append(Violation(code="forbidden_function", message=f"function `{name}` is not allowed"))

        # 6. columns
        for c in root.find_all(exp.Column):
            if c.name.lower() in cfg.blocked_columns:
                violations.append(Violation(code="blocked_column", message=f"column `{c.name}` is blocked"))
        if cfg.forbid_select_star and any(True for _ in root.find_all(exp.Star)):
            violations.append(Violation(code="select_star", message="SELECT * is not allowed; name the columns"))

        if violations:
            return GuardResult(ok=False, sql=text, engine="sqlglot", tables=sorted(set(tables)), violations=violations)

        # 7. LIMIT on the outermost query
        limit_node = root.args.get("limit")
        if limit_node is None:
            root = root.limit(cfg.default_limit)
        else:
            value = limit_node.expression if isinstance(limit_node, exp.Limit) else None
            if not isinstance(value, exp.Literal) or not value.is_int:
                return GuardResult(
                    ok=False, sql=text, engine="sqlglot", tables=sorted(set(tables)),
                    violations=[Violation(code="limit_not_literal", message="LIMIT must be an integer literal")],
                )
            if int(value.this) > cfg.max_limit:
                root = root.limit(cfg.max_limit)

        return GuardResult(ok=True, sql=root.sql(dialect=cfg.dialect), engine="sqlglot", tables=sorted(set(tables)))

    # -- regex engine -------------------------------------------------------
    _CTE_HEAD = re.compile(r"\bwith\s+(?:recursive\s+)?([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s+as\s*\(", re.I)
    _CTE_NEXT = re.compile(r"\)\s*,\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s+as\s*\(", re.I)
    _FROM_JOIN = re.compile(r"\b(?:from|join)\s+([A-Za-z_\"`\[][\w\"`\[\].]*)", re.I)
    _COMMA_TABLE = re.compile(r"\s*(?:(?:as\s+)?[A-Za-z_]\w*\s*)?,\s*([A-Za-z_\"`\[][\w\"`\[\].]*)", re.I)
    _TRAILING_LIMIT = re.compile(r"\blimit\s+(\d+)\s*(?:offset\s+\d+\s*)?$", re.I)
    _ANY_LIMIT = re.compile(r"\blimit\b", re.I)
    _IDENT_WORD = re.compile(r"[A-Za-z_]\w*")

    def _check_regex(self, text: str, stripped: str) -> GuardResult:
        cfg = self.config
        violations: list[Violation] = []
        lowered = stripped.lower()

        # 3. read-only. Leading keyword plus a blanket ban on write keywords.
        head = lowered.lstrip(" (\t\r\n")
        first = self._IDENT_WORD.match(head)
        if first is None or first.group(0) not in ("select", "with"):
            violations.append(Violation(code="not_select", message="statement must start with SELECT or WITH"))
        for kw in WRITE_KEYWORDS:
            if re.search(rf"\b{kw}\b", lowered):
                violations.append(Violation(code="not_read_only", message=f"keyword `{kw.upper()}` is not allowed"))
                break
        if re.search(r"\bfor\s+(update|share|no\s+key\s+update|key\s+share)\b", lowered):
            violations.append(Violation(code="not_read_only", message="FOR UPDATE / FOR SHARE is not allowed"))

        # 4. tables
        cte_names = {m.group(1).lower() for m in self._CTE_HEAD.finditer(stripped)}
        cte_names |= {m.group(1).lower() for m in self._CTE_NEXT.finditer(stripped)}
        tables: list[str] = []
        for m in self._FROM_JOIN.finditer(stripped):
            names = [m.group(1)]
            pos = m.end()
            # table-valued function: FROM generate_series(...)
            rest = stripped[pos:].lstrip()
            if rest.startswith("("):
                violations.append(Violation(code="table_function", message=f"table-valued expression not allowed: {m.group(1)}"))
                continue
            # comma-separated FROM list: FROM a x, b y
            while True:
                cm = self._COMMA_TABLE.match(stripped, pos)
                if cm is None:
                    break
                names.append(cm.group(1))
                pos = cm.end()
            for raw in names:
                full = _normalize_table_name(raw)
                if "." not in full and full in cte_names:
                    continue
                tables.append(full)
                if not _table_allowed(full, cfg.allowed_tables):
                    violations.append(Violation(code="table_not_allowed", message=f"table `{full}` is not on the allowlist"))

        # 5. functions
        for fn in sorted(cfg.forbidden_functions):
            if re.search(rf"\b{re.escape(fn)}\s*\(", lowered):
                violations.append(Violation(code="forbidden_function", message=f"function `{fn}` is not allowed"))

        # 6. columns. Crude on purpose: any bare occurrence of a blocked word.
        for col in sorted(cfg.blocked_columns):
            if re.search(rf"\b{re.escape(col)}\b", lowered):
                violations.append(Violation(code="blocked_column", message=f"column `{col}` is blocked"))
        if cfg.forbid_select_star and re.search(r"(\bselect\s+(?:distinct\s+)?\*|\.\*)", lowered):
            violations.append(Violation(code="select_star", message="SELECT * is not allowed; name the columns"))

        if violations:
            return GuardResult(ok=False, sql=text, engine="regex", tables=sorted(set(tables)), violations=violations)

        # 7. LIMIT. Only a trailing integer LIMIT counts; anything else that
        #    mentions LIMIT at the end is rejected, a missing LIMIT is added.
        tm = self._TRAILING_LIMIT.search(stripped)
        if tm:
            if int(tm.group(1)) > cfg.max_limit:
                start = tm.start(1)
                text = text[:start] + str(cfg.max_limit) + text[tm.end(1):]
        else:
            tail = lowered[-40:]
            if re.search(r"\blimit\b", tail) and not re.search(r"\)\s*$", tail):
                return GuardResult(
                    ok=False, sql=text, engine="regex", tables=sorted(set(tables)),
                    violations=[Violation(code="limit_not_literal", message="trailing LIMIT must be an integer literal")],
                )
            text = f"{text} LIMIT {cfg.default_limit}"

        return GuardResult(ok=True, sql=text, engine="regex", tables=sorted(set(tables)))


def guard_sql(sql: str, allowed_tables: set[str], engine: Engine = "auto", **overrides: object) -> GuardResult:
    """One-call convenience wrapper used by the chapter text."""
    cfg = GuardConfig(allowed_tables={t.lower() for t in allowed_tables}, **overrides)  # type: ignore[arg-type]
    return SqlGuard(cfg, engine=engine).check(sql)
