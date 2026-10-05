# path: book/projects/p5-incident-agent/incident_agent/adapters/store.py
"""Investigation records: one JSON file each (or in memory for tests)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..domain.models import Investigation


class InvestigationStore(Protocol):
    def save(self, inv: Investigation) -> None: ...

    def get(self, inv_id: str) -> Investigation | None: ...


class InMemoryInvestigationStore:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def save(self, inv: Investigation) -> None:
        self._items[inv.id] = inv.model_dump_json()      # store a copy, as a database would

    def get(self, inv_id: str) -> Investigation | None:
        raw = self._items.get(inv_id)
        return Investigation.model_validate_json(raw) if raw else None


class FileInvestigationStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, inv: Investigation) -> None:
        tmp = self.root / f"{inv.id}.json.tmp"
        tmp.write_text(inv.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.root / f"{inv.id}.json")          # atomic on POSIX

    def get(self, inv_id: str) -> Investigation | None:
        path = self.root / f"{inv_id}.json"
        return Investigation.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


__all__ = ["InvestigationStore", "InMemoryInvestigationStore", "FileInvestigationStore"]
