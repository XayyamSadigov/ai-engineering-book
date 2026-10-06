# path: book/projects/agentkit/agentkit/store.py
"""Event stores: append-only logs keyed by run id.

The in-memory store is for tests and single-process tools. The JSONL store writes one file
per run and one event per line, flushed on every append, so a crash loses at most the event
being written; `load` drops that torn last line. Chapter 38 swaps in a database-backed store with the same protocol.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

from .events import Event, event_from_json, event_to_json


class EventStore(Protocol):
    def append(self, event: Event) -> None: ...

    def load(self, run_id: str) -> list[Event]: ...

    def runs(self) -> list[str]: ...


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: dict[str, list[Event]] = {}
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        with self._lock:
            log = self._events.setdefault(event.run_id, [])
            if event.seq != len(log):
                raise ValueError(f"run {event.run_id}: expected seq {len(log)}, got {event.seq}")
            log.append(event)

    def load(self, run_id: str) -> list[Event]:
        with self._lock:
            return list(self._events.get(run_id, []))

    def runs(self) -> list[str]:
        with self._lock:
            return list(self._events)


class JsonlEventStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_seq: dict[str, int] = {}

    def _path(self, run_id: str) -> Path:
        safe = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_.")
        if not safe or safe != run_id:
            raise ValueError(f"run id {run_id!r} is not a safe file name")
        return self.directory / f"{safe}.jsonl"

    def append(self, event: Event) -> None:
        path = self._path(event.run_id)
        with self._lock:
            expected = self._next_seq.get(event.run_id)
            if expected is None:
                expected = len(self.load(event.run_id)) if path.exists() else 0
            if event.seq != expected:
                raise ValueError(f"run {event.run_id}: expected seq {expected}, got {event.seq}")
            with path.open("ab") as f:                       # bytes: "\n" on every platform
                f.write((event_to_json(event) + "\n").encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            self._next_seq[event.run_id] = expected + 1

    def load(self, run_id: str) -> list[Event]:
        path = self._path(run_id)
        if not path.exists():
            return []
        events: list[Event] = []
        data = path.read_bytes()
        lines = data.split(b"\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(event_from_json(line.decode("utf-8")))
            except ValueError:                 # includes a UTF-8 sequence cut in half
                if i != len(lines) - 1:        # complete lines end with "\n"; this one did not
                    raise                      # corruption mid-file is not a torn write: fail loudly
                with path.open("r+b") as f:    # drop the torn tail so later appends stay valid
                    f.truncate(len(data) - len(line))
                return events
        if lines[-1]:                          # whole last event, newline lost: restore it
            with path.open("ab") as f:
                f.write(b"\n")
        return events

    def runs(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.jsonl"))


__all__ = ["EventStore", "InMemoryEventStore", "JsonlEventStore"]
