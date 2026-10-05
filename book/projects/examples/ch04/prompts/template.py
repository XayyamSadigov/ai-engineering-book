# path: book/projects/examples/ch04/prompts/template.py
"""Sandboxed, strict prompt templates that keep untrusted values inside labeled data blocks.

Three guarantees, each enforced by code rather than by convention:

1. Strictness. Every variable the template references must be declared, every required
   variable must be supplied, and nothing undeclared may be passed. A typo fails at load
   time (in CI), not as an empty string in production.
2. Sandboxing. Templates run in Jinja2's immutable sandbox, so a prompt file edited by a
   non-engineer cannot reach Python internals or mutate the values it is given.
3. Data labeling. Variables are untrusted unless declared `trusted`. Untrusted strings are
   wrapped so that, when rendered, they appear inside an <untrusted_data> block whose
   delimiter cannot be forged from inside the value. Labeling is a hint to the model, not
   a security boundary; Chapter 26 explains why the boundary lives in code.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from jinja2 import StrictUndefined, TemplateSyntaxError, meta, nodes
from jinja2.exceptions import SecurityError, UndefinedError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict

from aie_core.llm.types import Message, Role

DATA_TAG = "untrusted_data"
_FORGED_TAG = re.compile(r"<(\s*/?\s*)(" + DATA_TAG + r")", re.IGNORECASE)
_ATTR_UNSAFE = re.compile(r"[^A-Za-z0-9_.:@ \-]")
# Filters that may be applied to an untrusted value in an output position: `data` delimits,
# `length`/`count` produce integers that cannot carry instructions.
_SAFE_FILTERS = frozenset({"data", "length", "count"})


class PromptDefinitionError(ValueError):
    """The prompt file or template is malformed (raised at load time)."""


class PromptRenderError(ValueError):
    """Rendering failed: missing, unknown, or invalid variables."""


class TemplateSecurityError(PromptDefinitionError):
    """The template uses an untrusted variable in a way that would bypass delimiting."""


class VariableSpec(BaseModel):
    """Declaration of one template variable, written in the prompt file's front matter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trusted: bool = False  # secure default: anything not declared trusted is data
    required: bool = True
    max_chars: int | None = None
    description: str = ""


def sanitize_untrusted(text: str) -> str:
    """Normalize and neutralize a value that will sit inside a data block.

    NFKC folds look-alike characters (full-width '<' becomes '<') so the forged-tag check
    sees them; control and invisible format characters (zero-width, bidi overrides) are
    dropped because they hide text from human reviewers; any '<untrusted_data' or
    '</untrusted_data' sequence is escaped so the value cannot close its own block.
    """
    folded = unicodedata.normalize("NFKC", text)
    visible = "".join(ch for ch in folded if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf"))
    return _FORGED_TAG.sub(lambda m: "&lt;" + m.group(1) + m.group(2), visible)


def _attr(value: Any) -> str:
    return _ATTR_UNSAFE.sub("_", str(value.raw if isinstance(value, Untrusted) else value))[:64]


class Untrusted:
    """A string from outside the trust boundary. Not a `str` subclass on purpose: any code
    path that stringifies it gets the escaped form, never the raw text."""

    __slots__ = ("raw", "label", "max_chars")

    def __init__(self, raw: str, label: str, max_chars: int | None = None) -> None:
        self.raw = raw
        self.label = label
        self.max_chars = max_chars

    @property
    def escaped(self) -> str:
        text = sanitize_untrusted(self.raw)
        if self.max_chars is not None and len(text) > self.max_chars:
            dropped = len(text) - self.max_chars
            text = f"{text[: self.max_chars]}\n[truncated {dropped} chars]"
        return text

    def block(self, label: str | None = None, **attrs: Any) -> str:
        parts = [f'label="{_attr(label or self.label)}"']
        parts += [f'{_attr(k)}="{_attr(v)}"' for k, v in sorted(attrs.items())]
        return f"<{DATA_TAG} {' '.join(parts)}>\n{self.escaped}\n</{DATA_TAG}>"

    def __str__(self) -> str:
        return self.escaped

    def __bool__(self) -> bool:
        return bool(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def __eq__(self, other: object) -> bool:
        other_raw = other.raw if isinstance(other, Untrusted) else other
        return self.raw == other_raw

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:
        return f"Untrusted({self.label!r}, {len(self.raw)} chars)"


def taint(value: Any, label: str, max_chars: int | None = None) -> Any:
    """Recursively wrap every string inside `value`. Numbers, booleans and None pass through:
    they cannot carry instructions. Dict keys are left alone; render values, not keys."""
    if isinstance(value, Untrusted):
        return value
    if isinstance(value, str):
        return Untrusted(value, label, max_chars)
    if isinstance(value, Mapping):
        return {k: taint(v, f"{label}.{k}", max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [taint(v, f"{label}[{i}]", max_chars) for i, v in enumerate(value)]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise PromptRenderError(f"unsupported type for untrusted variable {label!r}: {type(value).__name__}")


def _data_filter(value: Any, label: str | None = None, **attrs: Any) -> str:
    """`{{ doc.text | data(source=doc.id) }}`: delimit explicitly, with extra attributes.
    Applied to a trusted string it still delimits (marking something as data is never unsafe)."""
    if isinstance(value, Untrusted):
        return value.block(label, **attrs)
    return Untrusted(str(value), label or "data").block(**attrs)


def _finalize(value: Any) -> Any:
    """Called by Jinja on every `{{ ... }}` result: bare untrusted values are delimited."""
    if isinstance(value, Untrusted):
        return value.block()
    if value is None:
        raise PromptRenderError("a template expression rendered None; guard optional variables with {% if %}")
    return value


def _make_env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(
        undefined=StrictUndefined,
        autoescape=False,  # HTML escaping is the wrong escaping for prompts
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        finalize=_finalize,
    )
    env.filters["data"] = _data_filter
    return env


_ENV = _make_env()


# ---------------------------------------------------------------- static taint check
def _root_name(expr: nodes.Node) -> str | None:
    while isinstance(expr, (nodes.Getattr, nodes.Getitem)):
        expr = expr.node
    return expr.name if isinstance(expr, nodes.Name) else None


def _names_in(expr: nodes.Node) -> set[str]:
    return {n.name for n in expr.find_all(nodes.Name) if n.ctx == "load"} | (
        {expr.name} if isinstance(expr, nodes.Name) and expr.ctx == "load" else set()
    )


def _tainted_names(ast: nodes.Template, untrusted: set[str]) -> set[str]:
    """Untrusted variables plus loop variables iterating over them (conservative: no scoping)."""
    tainted = set(untrusted)
    changed = True
    while changed:
        changed = False
        for loop in ast.find_all(nodes.For):
            if _names_in(loop.iter) & tainted:
                targets = [loop.target] if isinstance(loop.target, nodes.Name) else list(loop.target.find_all(nodes.Name))
                for target in targets:
                    if target.name not in tainted:
                        tainted.add(target.name)
                        changed = True
    return tainted


def check_template_taint(ast: nodes.Template, untrusted: set[str], where: str) -> None:
    """Reject templates that would render an untrusted value without its data block.

    Allowed in output: a bare reference (`{{ x }}`, `{{ doc.text }}`) or a safe filter
    (`{{ x | data("label") }}`, `{{ docs | length }}`). Rejected: any other expression that
    contains an untrusted name (`{{ x | upper }}`, `{{ "a" ~ x }}`), and untrusted names in
    `set`, macro definitions, or calls, where taint tracking would be lost.
    """
    tainted = _tainted_names(ast, untrusted)
    for node_type in (nodes.Assign, nodes.AssignBlock, nodes.Macro, nodes.CallBlock, nodes.FilterBlock):
        for node in ast.find_all(node_type):
            hit = _names_in(node) & tainted
            if hit:
                raise TemplateSecurityError(f"{where}: untrusted {sorted(hit)} used inside {node_type.__name__}")
    for output in ast.find_all(nodes.Output):
        for expr in output.nodes:
            if isinstance(expr, nodes.TemplateData):
                continue
            hit = _names_in(expr) & tainted
            if not hit:
                continue
            if _root_name(expr) in tainted:
                continue
            if (
                isinstance(expr, nodes.Filter)
                and expr.name in _SAFE_FILTERS
                and expr.node is not None
                and _root_name(expr.node) in tainted
            ):
                continue  # filter arguments may be tainted: block attributes are sanitized
            raise TemplateSecurityError(
                f"{where}: untrusted {sorted(hit)} rendered through an expression that bypasses the "
                f"data block (line {expr.lineno}); render it bare or with | data(...)"
            )


# ---------------------------------------------------------------- the template
class PromptTemplate:
    """An ordered list of (role, Jinja source) sections plus declared variables."""

    def __init__(
        self,
        sections: Sequence[tuple[Role, str]],
        variables: Mapping[str, VariableSpec] | None = None,
        *,
        name: str = "inline",
    ) -> None:
        if not sections:
            raise PromptDefinitionError(f"{name}: a prompt needs at least one message section")
        self.name = name
        self.variables: dict[str, VariableSpec] = dict(variables or {})
        self.sections: list[tuple[Role, str]] = [(Role(role), src) for role, src in sections]
        untrusted = {k for k, v in self.variables.items() if not v.trusted}
        self._compiled = []
        for i, (role, source) in enumerate(self.sections):
            where = f"{name} section {i} ({role.value})"
            try:
                ast = _ENV.parse(source)
            except TemplateSyntaxError as exc:
                raise PromptDefinitionError(f"{where}: {exc.message} (line {exc.lineno})") from exc
            undeclared = meta.find_undeclared_variables(ast) - set(self.variables)
            if undeclared:
                raise PromptDefinitionError(f"{where}: undeclared variables {sorted(undeclared)}")
            check_template_taint(ast, untrusted, where)
            self._compiled.append((role, _ENV.from_string(source)))

    def render(self, values: Mapping[str, Any]) -> list[Message]:
        unknown = set(values) - set(self.variables)
        if unknown:
            raise PromptRenderError(f"{self.name}: unknown variables {sorted(unknown)}")
        context: dict[str, Any] = {}
        for var, spec in self.variables.items():
            if var not in values or values[var] is None:
                if spec.required:
                    raise PromptRenderError(f"{self.name}: missing required variable {var!r}")
                context[var] = None
                continue
            value = values[var]
            context[var] = value if spec.trusted else taint(value, var, spec.max_chars)
        messages: list[Message] = []
        for role, template in self._compiled:
            try:
                text = template.render(context).strip()
            except (UndefinedError, SecurityError) as exc:
                raise PromptRenderError(f"{self.name} ({role.value}): {exc}") from exc
            if text:
                messages.append(Message(role=role, content=text))
        return messages


__all__ = [
    "DATA_TAG",
    "PromptDefinitionError",
    "PromptRenderError",
    "TemplateSecurityError",
    "VariableSpec",
    "Untrusted",
    "PromptTemplate",
    "sanitize_untrusted",
    "taint",
    "check_template_taint",
]
