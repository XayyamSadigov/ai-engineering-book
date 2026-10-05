# path: book/projects/examples/ch32/northwind_triage/adapters/prompt_store.py
"""File-backed prompt store with an immutability lock.

Layout: prompt_files/<prompt id>/<version>.md with '## system' and '## user' sections.
prompts.lock maps 'id@version' to the content hash; a published version must never change.
Chapter 4 builds the full registry (aliases, front matter, rendering); this is the minimum
the triage example needs.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..application.ports import PromptVersion

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "prompt_files"


class PromptNotFound(LookupError):
    pass


class LockMismatch(Exception):
    pass


def parse_prompt_file(prompt_id: str, version: str, text: str) -> PromptVersion:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.strip().lower() in {"## system", "## user"}:
            current = line.strip().lower()[3:]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"{prompt_id}@{version}: needs '## system' and '## user' sections")
    return PromptVersion(
        id=prompt_id,
        version=version,
        system="\n".join(sections["system"]).strip(),
        user_template="\n".join(sections["user"]).strip(),
    )


class FilePromptStore:
    def __init__(self, root: str | Path = DEFAULT_DIR) -> None:
        self.root = Path(root)
        self._cache: dict[tuple[str, str], PromptVersion] = {}

    def get(self, prompt_id: str, version: str) -> PromptVersion:
        key = (prompt_id, version)
        if key not in self._cache:
            path = self.root / prompt_id / f"{version}.md"
            if not path.is_file():
                raise PromptNotFound(f"{prompt_id}@{version} not found under {self.root}")
            self._cache[key] = parse_prompt_file(prompt_id, version, path.read_text(encoding="utf-8"))
        return self._cache[key]

    def all_versions(self) -> list[PromptVersion]:
        out = []
        for path in sorted(self.root.glob("*/*.md")):
            out.append(self.get(path.parent.name, path.stem))
        return out

    # ------------------------------------------------------------------ lock
    def lock_path(self) -> Path:
        return self.root / "prompts.lock"

    def compute_lock(self) -> dict[str, str]:
        return {f"{p.id}@{p.version}": p.sha for p in self.all_versions()}

    def verify_lock(self) -> None:
        """Fail if a locked version changed or disappeared. New versions must be added to the
        lock deliberately (`python -m northwind_triage.adapters.prompt_store --write-lock`)."""
        locked: dict[str, str] = json.loads(self.lock_path().read_text(encoding="utf-8"))
        current = self.compute_lock()
        problems = []
        for key, sha in locked.items():
            if key not in current:
                problems.append(f"{key} is locked but missing")
            elif current[key] != sha:
                problems.append(f"{key} changed after publication ({sha} -> {current[key]}); "
                                f"create a new version instead")
        for key in sorted(set(current) - set(locked)):
            problems.append(f"{key} is not in prompts.lock")
        if problems:
            raise LockMismatch("; ".join(problems))

    def write_lock(self) -> None:
        self.lock_path().write_text(json.dumps(self.compute_lock(), indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    import sys

    store = FilePromptStore()
    if "--write-lock" in sys.argv:
        store.write_lock()
        print(f"wrote {store.lock_path()}")
    else:
        store.verify_lock()
        print("prompts.lock OK")
