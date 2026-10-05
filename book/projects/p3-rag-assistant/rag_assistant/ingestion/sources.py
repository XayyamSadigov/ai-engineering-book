# path: book/projects/p3-rag-assistant/rag_assistant/ingestion/sources.py
"""Source connectors: where raw documents come from, addressed by (connector, uri).

The queue carries pointers, not documents. A worker re-reads the source when it runs, so a
job that waited in the queue indexes what the source holds *now*, and a retried job never
indexes a stale payload. Two connectors cover Northwind:

- `FolderConnector`: a directory tree (the shared-data docs folder, a synced share, a git
  checkout). `list()` enumerates it for full syncs.
- `BlobConnector`: originals uploaded through the admin API. This is a copy the assistant owns,
  so deleting a document must delete it too.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from ragkit.parsers import EXTENSIONS


class SourceNotFound(LookupError):
    """The source no longer has this item: treat as a deletion, not as a transient error."""


@runtime_checkable
class SourceConnector(Protocol):
    name: str

    def read(self, uri: str) -> bytes: ...

    def list(self) -> Iterator[str]: ...


class FolderConnector:
    name = "folder"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, uri: str) -> Path:
        p = (self.root / uri).resolve()
        if self.root not in p.parents and p != self.root:  # no "../../etc/passwd" through a uri
            raise ValueError(f"uri {uri!r} escapes the source root")
        return p

    def read(self, uri: str) -> bytes:
        p = self._path(uri)
        if not p.is_file():
            raise SourceNotFound(uri)
        return p.read_bytes()

    def list(self) -> Iterator[str]:
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                yield p.relative_to(self.root).as_posix()


class BlobConnector:
    """Uploaded originals keyed by uri. Directory-backed in deployments, in-memory in tests."""

    name = "blob"

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._mem: dict[str, bytes] = {}
        self._lock = threading.Lock()
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def uri_for(doc_id: str, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower() or ".md"
        return f"{doc_id}/{hashlib.sha256(content).hexdigest()[:16]}{suffix}"

    def _file(self, uri: str) -> Path:
        assert self.directory is not None
        if ".." in Path(uri).parts:
            raise ValueError(f"invalid blob uri {uri!r}")
        return self.directory / uri

    def put(self, uri: str, content: bytes) -> None:
        with self._lock:
            if self.directory:
                f = self._file(uri)
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f.suffix + ".tmp")
                tmp.write_bytes(content)
                tmp.replace(f)
            else:
                self._mem[uri] = content

    def read(self, uri: str) -> bytes:
        if self.directory:
            f = self._file(uri)
            if not f.is_file():
                raise SourceNotFound(uri)
            return f.read_bytes()
        try:
            return self._mem[uri]
        except KeyError:
            raise SourceNotFound(uri) from None

    def delete_doc(self, doc_id: str) -> int:
        """Remove every stored original of a document (all uploaded versions)."""
        with self._lock:
            if self.directory:
                folder = self.directory / doc_id
                files = list(folder.glob("*")) if folder.is_dir() else []
                for f in files:
                    f.unlink()
                if folder.is_dir():
                    folder.rmdir()
                return len(files)
            doomed = [u for u in self._mem if u.startswith(doc_id + "/")]
            for u in doomed:
                del self._mem[u]
            return len(doomed)

    def has_doc(self, doc_id: str) -> bool:
        if self.directory:
            return (self.directory / doc_id).is_dir()
        return any(u.startswith(doc_id + "/") for u in self._mem)

    def list(self) -> Iterator[str]:
        if self.directory:
            for p in sorted(self.directory.rglob("*")):
                if p.is_file() and not p.name.endswith(".tmp"):
                    yield p.relative_to(self.directory).as_posix()
        else:
            yield from sorted(self._mem)


__all__ = ["BlobConnector", "FolderConnector", "SourceConnector", "SourceNotFound"]
