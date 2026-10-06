# path: book/projects/guardrails/guardrails/output.py
"""Output guardrails. Model output is untrusted input to whatever consumes it.

* `SchemaCheck`: structured output must validate before anything downstream reads it.
* `UrlAllowlistCheck`: links and images may point only at allowlisted hosts; off-list images
  are removed (a rendered image is a request the victim's browser makes for the attacker).
* `CanaryCheck`: known canary strings (planted in the system prompt or sensitive records) must
  never appear in output.
* `CitationCheck`: answers on grounded paths must cite ids that were actually shown.
* `ActiveContentCheck`: flags script, event handlers and destructive SQL/shell text.
* `escape_html`, `ANSWER_PANE_CSP`: render safely.

A note that belongs in code, not just in a book: nothing in this module, or anywhere else in a
well-built system, passes model output to `eval`, `exec`, a shell, or a string-built SQL query.
Generated SQL goes through a parser and an allowlist of read-only statements over a semantic
layer (Chapter 36, Case B); generated code runs in a sandbox (Chapter 16) or not at all.
"""
from __future__ import annotations

import html
import html.parser
import json
import re
from typing import Callable, Iterable
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from aie_core.llm.structured import extract_json

from .pipeline import Action, BaseCheck, FailMode, Finding, GuardContext, Stage, Subject, Verdict

OUTPUT_ONLY = frozenset({Stage.OUTPUT})

# Content-Security-Policy for the pane that renders answers: images and fetches only from our
# own origin and the allowlisted intranet host. Defense in depth behind UrlAllowlistCheck.
ANSWER_PANE_CSP = (
    "default-src 'none'; img-src 'self' https://intranet.northwind.example; "
    "style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


def escape_html(text: str) -> str:
    """Escape for an HTML text node or a quoted attribute value."""
    return html.escape(text, quote=True)


# --------------------------------------------------------------------------- schema
class SchemaCheck(BaseCheck):
    """Validates JSON output against a pydantic model or an arbitrary validator callable.
    Fail-closed: if we cannot prove the shape, downstream code does not get the payload."""

    fail_mode = FailMode.CLOSED
    stages = OUTPUT_ONLY

    def __init__(self, schema: type[BaseModel] | None = None,
                 validator: Callable[[str], None] | None = None, name: str = "schema") -> None:
        if schema is None and validator is None:
            raise ValueError("SchemaCheck needs a schema or a validator")
        self.schema = schema
        self.validator = validator
        self.name = name

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        try:
            if self.schema is not None:
                obj = self.schema.model_validate(json.loads(extract_json(subject.text)))
                normalized = obj.model_dump_json()
            else:
                normalized = subject.text
            if self.validator is not None:
                self.validator(subject.text)
        except (ValueError, ValidationError) as exc:
            return Verdict.block(f"schema validation failed: {type(exc).__name__}")
        # Downstream code should consume `normalized` (the re-serialized, validated object), never
        # the raw text, so extra prose or fences around the JSON cannot ride along.
        return Verdict.allow("schema ok", normalized=normalized)


# --------------------------------------------------------------------------- URLs
_ALT = r"((?:[^\[\]]|\[[^\]]*\])*)"                                     # alt/link text, one level of brackets
_TITLE = r"""(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?"""                      # "t", 't' or (t)
MD_IMAGE = re.compile(r"!\[" + _ALT + r"\]\(\s*<?([^)\s>]+)>?" + _TITLE + r"\s*\)")
MD_LINK = re.compile(r"(?<!!)\[" + _ALT + r"\]\(\s*<?([^)\s>]+)>?" + _TITLE + r"\s*\)")
HTML_A = re.compile(r"(<a\b[^>]*>)(.{0,2000}?)</a\s*>", re.I | re.S)   # bounded: many unclosed <a> stay linear
# Any tag that can make a browser fetch or navigate. Every URL-bearing attribute must pass, so a
# decoy `data-src` or an `alt="src=..."` cannot stand in for the real `src`.
HTML_TAG = re.compile(r"<(?:img|a|source|video|audio|iframe|embed|object|link|form|input)\b[^>]*>", re.I)
_URL_ATTRS = {"src", "href", "srcset", "poster", "action", "formaction", "data", "background", "cite"}
REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+).*$", re.M)          # [x]: https://... reference links
BARE_URL = re.compile(r"(?i)\b(?:https?|ftp|data|javascript|vbscript|file):[^\s<>\"')\]]+|\bwww\.[^\s<>\"')\]]+"
                      r"|(?<![\w/\\:)\]])//[A-Za-z0-9\[][^\s<>\"')\]]*")   # scheme-less //host, not code like a//b
SAFE_SCHEMES = {"http", "https", "mailto"}


def host_allowed(url: str, allowed_hosts: Iterable[str], allow_subdomains: bool = True) -> bool:
    """Parse properly: userinfo tricks (`https://good.example@evil.example/`), trailing dots,
    case and lookalike suffixes (`good.example.evil.example`) all resolve to the real host."""
    url = url.strip().replace("\\", "/")   # browsers treat a backslash as a slash; urlsplit does not
    if url.startswith("//"):
        url = "https:" + url
    candidate = url if "://" in url or url.lower().startswith(("mailto:", "data:", "javascript:")) else "https://" + url
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return False
    scheme = parts.scheme.lower()
    if scheme not in SAFE_SCHEMES:
        return False
    if scheme == "mailto":
        host = parts.path.rsplit("@", 1)[-1]
    else:
        host = parts.hostname or ""
    host = host.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    for allowed in allowed_hosts:
        a = allowed.lower().rstrip(".")
        if host == a or (allow_subdomains and host.endswith("." + a)):
            return True
    return False


class _Attrs(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attrs: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.attrs = attrs


def tag_urls(tag: str) -> list[str]:
    """Every URL a browser could take from one HTML tag: src, href, srcset candidates, data-*."""
    parser = _Attrs()
    parser.feed(tag)
    urls: list[str] = []
    for name, value in parser.attrs:
        if value is None:
            continue
        name = name.lower()
        if name == "srcset":
            urls += [c.strip().split()[0] for c in value.split(",") if c.strip()]
        elif name in _URL_ATTRS or (name.startswith("data-") and re.match(r"(?i)\s*(?:[a-z][a-z0-9+.-]*:|//)", value)):
            urls.append(value.strip())
    return urls


class UrlAllowlistCheck(BaseCheck):
    """Removes or blocks links and images to hosts outside the allowlist.

    `mode="strip"` (default) rewrites the answer: off-list images disappear entirely, off-list
    links keep their text and lose the target. `mode="block"` refuses the whole answer, for
    surfaces where silently editing output is worse than failing.
    """

    fail_mode = FailMode.CLOSED
    stages = OUTPUT_ONLY

    def __init__(self, allowed_hosts: Iterable[str], mode: str = "strip", allow_subdomains: bool = True,
                 name: str = "url_allowlist") -> None:
        self.allowed_hosts = tuple(allowed_hosts)
        self.mode = mode
        self.allow_subdomains = allow_subdomains
        self.name = name

    def _ok(self, url: str) -> bool:
        return host_allowed(url, self.allowed_hosts, self.allow_subdomains)

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        text = subject.text
        findings: list[Finding] = []

        def img(m: re.Match[str]) -> str:
            if self._ok(m.group(2)):
                return m.group(0)
            findings.append(Finding("image", m.start(), m.end()))
            return "[image removed]"

        def link(m: re.Match[str]) -> str:
            if self._ok(m.group(2)):
                return m.group(0)
            findings.append(Finding("link", m.start(), m.end()))
            return f"{m.group(1)} [link removed]"

        def html_a(m: re.Match[str]) -> str:
            if all(self._ok(u) for u in tag_urls(m.group(1))):
                return m.group(0)
            findings.append(Finding("html_link", m.start(), m.end()))
            return f"{m.group(2)} [link removed]"

        def html_tag(m: re.Match[str]) -> str:   # images, unclosed anchors, media, frames, forms
            if all(self._ok(u) for u in tag_urls(m.group(0))):
                return m.group(0)
            is_img = m.group(0)[1:4].lower() == "img"
            findings.append(Finding("html_image" if is_img else "html_link", m.start(), m.end()))
            return "[image removed]" if is_img else "[link removed]"

        def ref_def(m: re.Match[str]) -> str:
            if self._ok(m.group(1)):
                return m.group(0)
            findings.append(Finding("reference_link", m.start(), m.end()))
            return "[link removed]"

        def bare(m: re.Match[str]) -> str:
            if self._ok(m.group(0)):
                return m.group(0)
            findings.append(Finding("url", m.start(), m.end()))
            return "[link removed]"

        text = MD_IMAGE.sub(img, text)
        text = MD_LINK.sub(link, text)
        text = HTML_A.sub(html_a, text)
        text = HTML_TAG.sub(html_tag, text)
        text = REF_DEF.sub(ref_def, text)
        text = BARE_URL.sub(bare, text)

        if not findings:
            return Verdict.allow("all links allowlisted")
        kinds = sorted({f.kind for f in findings})
        reason = f"{len(findings)} off-allowlist target(s): {', '.join(kinds)}"
        if self.mode == "block":
            return Verdict.block(reason, findings=findings)
        return Verdict.redact(text, reason, 1.0, findings)


# --------------------------------------------------------------------------- canaries
class CanaryCheck(BaseCheck):
    """Blocks output containing any known canary (a unique marker planted in the system prompt or
    in sensitive records). A canary in output is proof of a leak, not a suspicion."""

    fail_mode = FailMode.CLOSED

    def __init__(self, canaries: Iterable[str], stages: frozenset[Stage] = frozenset({Stage.OUTPUT, Stage.TOOL}),
                 name: str = "canary") -> None:
        self.canaries = tuple(c for c in canaries if c)
        self.stages = stages
        self.name = name

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        haystack = subject.text
        if subject.tool_call is not None:
            haystack += json.dumps(subject.tool_call.arguments, ensure_ascii=False)
        hits = [c for c in self.canaries if c in haystack]
        if hits:
            return Verdict.block(f"{len(hits)} canary marker(s) in outbound content",
                                 findings=[Finding("canary")] * len(hits))
        return Verdict.allow()


# --------------------------------------------------------------------------- citations
CITATION = re.compile(r"\[([A-Za-z0-9][\w.:#/-]{0,120})\](?!\()")


class CitationCheck(BaseCheck):
    """On grounded paths every answer must cite, and only cite ids from `ctx.evidence_ids`.
    The abstention token is an allowed answer without citations."""

    fail_mode = FailMode.CLOSED
    stages = OUTPUT_ONLY

    def __init__(self, require_citation: bool = True, abstain_token: str = "INSUFFICIENT_EVIDENCE",
                 name: str = "citations") -> None:
        self.require_citation = require_citation
        self.abstain_token = abstain_token
        self.name = name

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        if self.abstain_token and subject.text.strip().startswith(self.abstain_token):
            return Verdict.allow("abstained")
        cited = CITATION.findall(subject.text)
        cited = [c for c in cited if not c.startswith(("image removed", "link removed"))]
        if not cited:
            return Verdict.block("no citations") if self.require_citation else Verdict.flag("no citations")
        invalid = sorted({c for c in cited if c not in ctx.evidence_ids})
        if invalid:
            return Verdict.block(f"{len(invalid)} citation(s) to ids never shown",
                                 findings=[Finding("invalid_citation", detail=c) for c in invalid])
        return Verdict.allow(f"{len(set(cited))} valid citation(s)")


# --------------------------------------------------------------------------- active content
ACTIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "script_tag": re.compile(r"<\s*script\b", re.I),
    "event_handler": re.compile(r"<[^>]+\son[a-z]+\s*=", re.I),
    "javascript_url": re.compile(r"javascript\s*:", re.I),
    "iframe": re.compile(r"<\s*(iframe|object|embed)\b", re.I),
    "destructive_sql": re.compile(r"\b(drop|truncate)\s+table\b|\bdelete\s+from\s+\w+\s*(;|$)", re.I),
    "destructive_shell": re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\s+/|\bcurl\b[^|\n]*\|\s*(ba|z)?sh\b|\bmkfs\b", re.I),
}


class ActiveContentCheck(BaseCheck):
    """Flags (or blocks) text that would be dangerous if a consumer executed or rendered it raw.
    This is a sensor for a broken consumer, not a replacement for escaping and parameterizing."""

    fail_mode = FailMode.OPEN
    stages = OUTPUT_ONLY

    def __init__(self, block_kinds: Iterable[str] = ("script_tag", "event_handler", "javascript_url", "iframe"),
                 name: str = "active_content") -> None:
        self.block_kinds = set(block_kinds)
        self.name = name

    def evaluate(self, subject: Subject, ctx: GuardContext) -> Verdict:
        hits = [Finding(k, m.start(), m.end()) for k, rx in ACTIVE_PATTERNS.items()
                for m in [rx.search(subject.text)] if m]
        if not hits:
            return Verdict.allow()
        kinds = [h.kind for h in hits]
        reason = "active content: " + ", ".join(kinds)
        if self.block_kinds & set(kinds):
            return Verdict.block(reason, findings=hits)
        return Verdict.flag(reason, 0.8, hits)


__all__ = [
    "ANSWER_PANE_CSP", "escape_html", "SchemaCheck", "host_allowed", "UrlAllowlistCheck", "CanaryCheck",
    "CitationCheck", "ActiveContentCheck", "ACTIVE_PATTERNS",
]
