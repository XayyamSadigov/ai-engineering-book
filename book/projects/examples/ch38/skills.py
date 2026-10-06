# path: book/projects/examples/ch38/skills.py
"""Agent Skills: procedural knowledge packaged as folders, loaded by progressive disclosure.

A skill is a directory with a SKILL.md file whose front matter holds a name, a one-line
description, and a version, followed by the procedure itself. Optional files next to it
(reference tables, templates, scripts) are resources.

Three levels of disclosure keep the permanent context small:
  1. catalog: name + description of every installed skill (a few dozen tokens each),
  2. body: the SKILL.md procedure, loaded when the task matches the description,
  3. resources: individual files, loaded only when the procedure asks for them.

Skills are also a supply chain. `SkillLock` pins each skill to a content hash, and
`audit_skill` flags what a reviewer must look at before a new version is trusted:
executable files, network access, and instruction-override language.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentkit import ErrorClass, FunctionTool, ToolOutput

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or our the this to use when with what you your "
    "do does my me we can should".split())
EXECUTABLE_SUFFIXES = {".sh", ".py", ".js", ".ps1", ".bat", ".rb"}
RISKY_PATTERNS = {
    "network": re.compile(r"\b(curl|wget|https?://|requests\.|httpx\.|urllib)", re.I),
    "override": re.compile(r"ignore (all|any|previous|prior) (instructions|rules)|disregard the system", re.I),
    "secrets": re.compile(r"\b(api[_-]?key|password|token|secret)s?\b\s*[:=]", re.I),
    "shell": re.compile(r"\b(rm -rf|sudo|chmod \+x|eval\(|exec\()", re.I),
}


class SkillError(ValueError):
    pass


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """A deliberately small YAML subset: `key: value`, `key: [a, b]`, quoted strings."""
    m = FRONT_MATTER.match(text)
    if not m:
        raise SkillError("SKILL.md must start with a '---' front-matter block")
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise SkillError(f"bad front-matter line: {line!r}")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = value.strip("\"'")
    return meta, text[m.end():]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 2}


def skill_files(directory: Path) -> dict[str, bytes]:
    """Every file that belongs to a skill, by relative path. The hash, the resource list,
    and what the agent can read all use this one definition."""
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.relative_to(directory).parts}


def _hash_files(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(files[rel])
    return h.hexdigest()[:16]


def content_hash(directory: Path) -> str:
    """Hash of every file's relative path and bytes; any change produces a new hash."""
    return _hash_files(skill_files(directory))


@dataclass
class Skill:
    name: str
    description: str
    version: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    frozen: dict[str, bytes] | None = None      # set by freeze(): content served from memory, not disk
    _body: str | None = None

    def freeze(self) -> "Skill":
        """A copy holding the skill's bytes in memory. Verify the hash of these bytes and serve
        these bytes: otherwise a file rewritten after the lock check would be served as trusted."""
        return Skill(self.name, self.description, self.version, self.path, self.meta, skill_files(self.path))

    def _files(self) -> dict[str, bytes]:
        return self.frozen if self.frozen is not None else skill_files(self.path)

    @property
    def body(self) -> str:
        """Level 2: read only when the skill is activated."""
        if self._body is None:
            _, self._body = parse_front_matter(self._files()["SKILL.md"].decode("utf-8"))
        return self._body

    def resources(self) -> list[str]:
        return sorted(rel for rel in self._files() if rel != "SKILL.md")

    def read_resource(self, rel: str) -> str:
        """Level 3: one file of the skill, by the same definition the hash uses."""
        target = (self.path / rel).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise SkillError(f"{rel!r} is not a resource of skill {self.name}")
        data = self._files().get(target.relative_to(self.path.resolve()).as_posix())
        if data is None:
            raise SkillError(f"{rel!r} is not a resource of skill {self.name}")
        return data.decode("utf-8")

    def hash(self) -> str:
        return _hash_files(self._files())


def load_skills(root: str | Path) -> dict[str, Skill]:
    """Level 1: read only front matter. Invalid skills fail loudly instead of loading half-parsed."""
    skills: dict[str, Skill] = {}
    for md in sorted(Path(root).glob("*/SKILL.md")):
        head = md.read_text(encoding="utf-8")[:4000]
        meta, _ = parse_front_matter(head)
        name, desc, version = meta.get("name", ""), meta.get("description", ""), meta.get("version", "")
        if not NAME_RE.match(name) or name != md.parent.name:
            raise SkillError(f"{md}: name {name!r} must be kebab-case and equal the folder name")
        if not desc or len(desc) > 300:
            raise SkillError(f"{md}: description is required and must be at most 300 characters")
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise SkillError(f"{md}: version {version!r} must be semantic (x.y.z)")
        if name in skills:
            raise SkillError(f"duplicate skill {name}")
        skills[name] = Skill(name, desc, version, md.parent, meta)
    return skills


def catalog_prompt(skills: dict[str, Skill]) -> str:
    lines = ["Skills available (load one with load_skill when the task matches its description):"]
    lines += [f"- {s.name}: {s.description}" for s in skills.values()]
    return "\n".join(lines)


@dataclass(frozen=True)
class SkillMatch:
    name: str
    score: float
    matched: tuple[str, ...]


def select_skills(task: str, skills: dict[str, Skill], *, k: int = 2, min_score: float = 0.2) -> list[SkillMatch]:
    """Rank skills by how much of each description's vocabulary the task covers. Lexical on
    purpose: descriptions are short, written for matching, and auditable; swap in embeddings
    (Chapter 8) when the catalog grows past what keywords can separate."""
    task_terms = _tokens(task)
    out = []
    for s in skills.values():
        terms = _tokens(s.description + " " + s.name.replace("-", " ") + " " + " ".join(s.meta.get("triggers", [])))
        if not terms:
            continue
        matched = tuple(sorted(task_terms & terms))
        score = len(matched) / len(terms) ** 0.5
        if score >= min_score:
            out.append(SkillMatch(s.name, round(score, 3), matched))
    return sorted(out, key=lambda m: (-m.score, m.name))[:k]


# --------------------------------------------------------------------------- supply chain
@dataclass
class Finding:
    skill: str
    file: str
    kind: str
    detail: str


def audit_skill(skill: Skill) -> list[Finding]:
    findings: list[Finding] = []
    for rel in ["SKILL.md", *skill.resources()]:
        path = skill.path / rel
        if path.suffix in EXECUTABLE_SUFFIXES:
            findings.append(Finding(skill.name, rel, "executable", "runs code; review it and sandbox its execution"))
        text = path.read_text(encoding="utf-8", errors="replace")
        for kind, rx in RISKY_PATTERNS.items():
            m = rx.search(text)
            if m:
                findings.append(Finding(skill.name, rel, kind, m.group(0)))
    return findings


class SkillLock:
    """A lockfile: skill name -> version and content hash, approved by a reviewer."""

    def __init__(self, entries: dict[str, dict[str, str]] | None = None) -> None:
        self.entries = entries or {}

    @classmethod
    def from_skills(cls, skills: dict[str, Skill]) -> "SkillLock":
        return cls({s.name: {"version": s.version, "hash": s.hash()} for s in skills.values()})

    @classmethod
    def load(cls, path: str | Path) -> "SkillLock":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def verify(self, skills: dict[str, Skill]) -> list[str]:
        problems = []
        for s in skills.values():
            pin = self.entries.get(s.name)
            if pin is None:
                problems.append(f"{s.name}: not in lockfile (unreviewed)")
            elif pin["version"] != s.version:
                problems.append(f"{s.name}: version {s.version} but lockfile pins {pin['version']}")
            elif pin["hash"] != s.hash():
                problems.append(f"{s.name}: content changed without a version bump")
        return problems

    def trusted(self, skills: dict[str, Skill]) -> dict[str, Skill]:
        """Only skills whose version and hash match the lock are offered to the agent, frozen
        so that the content checked is exactly the content served."""
        frozen = {n: s.freeze() for n, s in skills.items()}
        bad = {p.split(":")[0] for p in self.verify(frozen)}
        return {n: s for n, s in frozen.items() if n not in bad}


# --------------------------------------------------------------------------- agent tools
def skill_tools(skills: dict[str, Skill]) -> list[FunctionTool]:
    def load_skill(name: str) -> ToolOutput:
        s = skills.get(name)
        if s is None:
            return ToolOutput.failure(f"unknown skill {name!r}; available: {sorted(skills)}", ErrorClass.VALIDATION)
        res = s.resources()
        extra = f"\n\nResources (read with read_skill_resource): {res}" if res else ""
        return ToolOutput(content=f"# skill {s.name} v{s.version}\n{s.body.strip()}{extra}",
                          artifacts={f"skill:{s.name}": s.version})

    def read_skill_resource(name: str, path: str) -> ToolOutput:
        s = skills.get(name)
        if s is None:
            return ToolOutput.failure(f"unknown skill {name!r}", ErrorClass.VALIDATION)
        try:
            return ToolOutput(content=s.read_resource(path))
        except SkillError as exc:
            return ToolOutput.failure(str(exc), ErrorClass.VALIDATION)

    obj = {"type": "object", "additionalProperties": False}
    return [
        FunctionTool("load_skill", "Load the full procedure of an installed skill by name.",
                     {**obj, "properties": {"name": {"type": "string"}}, "required": ["name"]}, fn=load_skill),
        FunctionTool("read_skill_resource", "Read one resource file of a loaded skill.",
                     {**obj, "properties": {"name": {"type": "string"}, "path": {"type": "string"}},
                      "required": ["name", "path"]}, fn=read_skill_resource),
    ]


__all__ = [
    "SkillError", "parse_front_matter", "skill_files", "content_hash", "Skill", "load_skills", "catalog_prompt", "SkillMatch",
    "select_skills", "Finding", "audit_skill", "SkillLock", "skill_tools",
]
