# path: book/projects/examples/ch08/embedlab/corpus.py
"""Load Northwind documents and tickets, split documents into sections, and build the
embedding client from settings.

With EMBEDDING_PROVIDER=fake (the default) the client is FakeEmbeddings in vocabulary mode
over the corpus vocabulary: a bag-of-words model that is deterministic and offline, and whose
similarity tracks shared words. Set EMBEDDING_PROVIDER=openai (plus EMBEDDING_MODEL, LLM_BASE_URL
and a key) to run every example against a real model without changing any other code.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from aie_core.embeddings import EmbeddingClient, FakeEmbeddings
from aie_core.settings import Settings, make_embedding_client

SHARED_DATA = Path(__file__).resolve().parents[3] / "shared-data"
LOCAL_DATA = Path(__file__).resolve().parents[1] / "data"

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can cannot could did do does doing down during each either few for
    from further had has have having he her here hers him his how i if in into is it its itself just
    me more most my no nor not now of off on once only or other our ours out over own same she should
    so some such than that the their theirs them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you
    your yours yes per via etc get got one two use used using may must new see within without
    search query document passage""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    tenant: str
    acl_groups: tuple[str, ...]
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}"


@dataclass(frozen=True)
class Section:
    id: str
    doc_id: str
    heading: str
    text: str


@dataclass(frozen=True)
class Ticket:
    id: str
    tenant: str
    subject: str
    body: str
    category: str
    priority: str
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}"


def parse_front_matter(raw: str) -> tuple[dict[str, object], str]:
    """Minimal YAML front-matter reader for the flat `key: value` headers in shared-data."""
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta: dict[str, object] = {}
    for line in header.strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip("\"'") for v in value[1:-1].split(",")]
            meta[key.strip()] = [v for v in items if v]
        else:
            meta[key.strip()] = value.strip("\"'")
    return meta, body.strip()


def load_docs(directory: Path | None = None) -> list[Doc]:
    directory = directory or SHARED_DATA / "docs"
    docs = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(
            Doc(
                id=str(meta.get("id", path.stem)),
                title=str(meta.get("title", path.stem)),
                tenant=str(meta.get("tenant", "shared")),
                acl_groups=tuple(meta.get("acl_groups", ["all"])),  # type: ignore[arg-type]
                body=body,
            )
        )
    return docs


def load_tickets(path: Path | None = None) -> list[Ticket]:
    path = path or SHARED_DATA / "tickets.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            Ticket(
                id=row["id"],
                tenant=row.get("tenant", "shared"),
                subject=row["subject"],
                body=row["body"],
                category=row["category"],
                priority=row.get("priority", ""),
                extra={k: v for k, v in row.items() if k not in {"id", "tenant", "subject", "body", "category", "priority"}},
            )
        )
    return out


def load_jsonl(name: str) -> list[dict]:
    path = LOCAL_DATA / name
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def split_sections(doc: Doc) -> list[Section]:
    """One section per `## ` heading. Each section text carries the document title and heading,
    so a section about "Troubleshooting" still says which product it troubleshoots.
    Chapter 11 owns real chunking; this is the simplest structure-aware split."""
    parts = re.split(r"(?m)^## +", doc.body)
    sections: list[Section] = []
    preamble = parts[0].strip()
    if preamble:
        sections.append(Section(f"{doc.id}#s0", doc.id, "", f"{doc.title}\n\n{preamble}"))
    for n, part in enumerate(parts[1:], start=1):
        heading, _, content = part.partition("\n")
        sections.append(Section(f"{doc.id}#s{n}", doc.id, heading.strip(), f"{doc.title} > {heading.strip()}\n\n{content.strip()}"))
    return sections


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and not t.isdigit() and len(t) > 1]


def build_vocabulary(texts: Iterable[str], min_df: int = 2, max_size: int = 3000) -> list[str]:
    """Vocabulary for the bag-of-words fake, ordered by document frequency (most common first).

    The ordering matters for the Matryoshka experiment: the first coordinates then carry the
    words that occur in the most texts, so a prefix keeps the broadest signal.
    """
    df: Counter[str] = Counter()
    for t in texts:
        df.update(set(tokenize(t)))
    words = [w for w, c in df.items() if c >= min_df]
    words.sort(key=lambda w: (-df[w], w))
    return words[:max_size]


def make_client(settings: Settings | None = None, corpus_texts: Iterable[str] | None = None) -> EmbeddingClient:
    """The one place the chapter decides which embedding model it talks to."""
    settings = settings or Settings()
    if settings.embedding_provider == "fake":
        texts = list(corpus_texts) if corpus_texts is not None else default_corpus_texts()
        return FakeEmbeddings(vocabulary=build_vocabulary(texts), model="fake-bow-v1")
    return make_embedding_client(settings)


def default_corpus_texts() -> list[str]:
    docs = load_docs()
    texts = [d.text for d in docs]
    texts += [s.text for d in docs for s in split_sections(d)]
    texts += [t.text for t in load_tickets()]
    return texts


__all__ = [
    "Doc",
    "Section",
    "Ticket",
    "SHARED_DATA",
    "STOPWORDS",
    "build_vocabulary",
    "default_corpus_texts",
    "load_docs",
    "load_jsonl",
    "load_tickets",
    "make_client",
    "parse_front_matter",
    "split_sections",
    "tokenize",
]
