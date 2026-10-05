# path: book/projects/ragkit/ragkit/normalize.py
"""Cleaning, normalization, and duplicate detection.

Normalization makes equal content byte-equal (so hashes and caches work) and removes text that
would pollute retrieval (page furniture, invisible characters, repeated footers). It must not
change meaning, which is why code blocks are only touched for line endings and why the Unicode
form is configurable.

Near-duplicate detection uses word shingles and MinHash with banded locality-sensitive hashing:
each document gets a short signature whose agreement rate estimates Jaccard similarity of the
shingle sets, and the bands make candidate lookup sublinear in corpus size.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from .documents import Block, Document

# ----------------------------------------------------------------------------- text level
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)  # incl. soft hyphen
_SPACES = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"([a-z])-[ \t]*\n[ \t]*([a-z])")

DEFAULT_BOILERPLATE: tuple[str, ...] = (
    r"^\s*(page\s*)?\d+\s*(of|/)\s*\d+\s*$",
    r"^\s*(confidential|internal use only|do not distribute)\.?\s*$",
    r"^\s*(©|\(c\)|copyright)\s.*all rights reserved\.?\s*$",
    r"^\s*(click here to )?unsubscribe\b.*$",
    r"^\s*(this email|this message) (was|has been) sent to\b.*$",
)


def normalize_unicode(text: str, form: Literal["NFC", "NFKC"] = "NFKC") -> str:
    """NFKC folds ligatures and full-width forms; it also turns 'm²' into 'm2'. Choose per corpus."""
    return unicodedata.normalize(form, text).translate(_ZERO_WIDTH)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def dehyphenate(text: str) -> str:
    """Join words split across lines by a hyphen ('manage-\\nment'). Lowercase-only to spare 'COVID-\\n19'."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def remove_boilerplate(text: str, patterns: Sequence[str] = DEFAULT_BOILERPLATE) -> str:
    compiled = [re.compile(p, re.I) for p in patterns]
    kept = [ln for ln in text.split("\n") if not any(c.match(ln) for c in compiled)]
    return "\n".join(kept)


def strip_repeated_lines(pages: list[list[str]], *, min_fraction: float = 0.6, edge_lines: int = 2) -> list[list[str]]:
    """Drop running headers and footers: lines at a page edge that repeat on most pages.

    Digits are masked before counting so 'Northwind HR Policy, page 3' matches across pages.
    """

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter[str] = Counter()
    for lines in pages:
        non_blank = [ln for ln in lines if ln.strip()]
        edges = set(map(key, non_blank[:edge_lines] + non_blank[-edge_lines:]))
        counts.update(edges)
    threshold = max(2, int(min_fraction * len(pages) + 0.999))
    repeated = {k for k, c in counts.items() if c >= threshold and k}
    out = []
    for lines in pages:
        non_blank_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(non_blank_idx[:edge_lines] + non_blank_idx[-edge_lines:])
        out.append([ln for i, ln in enumerate(lines) if not (i in edge_idx and key(ln) in repeated)])
    return out


class NormalizationConfig(BaseModel):
    unicode_form: Literal["NFC", "NFKC"] = "NFKC"
    boilerplate_patterns: tuple[str, ...] = DEFAULT_BOILERPLATE
    dehyphenate: bool = True


def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    cfg = config or NormalizationConfig()
    text = normalize_unicode(text, cfg.unicode_form)
    if cfg.dehyphenate:
        text = dehyphenate(text)
    text = remove_boilerplate(text, cfg.boilerplate_patterns)
    return normalize_whitespace(text)


def _normalize_block(block: Block, cfg: NormalizationConfig) -> Block | None:
    if block.kind == "code":  # code is data: only line endings and invisible characters
        text = block.text.replace("\r\n", "\n").translate(_ZERO_WIDTH)
        return block.model_copy(update={"text": text})
    if block.kind == "table":
        header = [normalize_text(c, cfg) for c in block.meta.get("header", [])]
        rows = [[normalize_text(c, cfg) for c in row] for row in block.meta.get("rows", [])]
        from .parsers.base import render_table

        meta = {**block.meta, "header": header, "rows": rows}
        return block.model_copy(update={"text": render_table(header, rows), "meta": meta})
    if block.kind == "list":  # keep one item per line
        lines = [normalize_text(ln, cfg) for ln in block.text.split("\n")]
        text = "\n".join(ln for ln in lines if ln)
    else:
        text = normalize_text(block.text, cfg)
    if not text:
        return None
    return block.model_copy(update={"text": text})


def normalize_document(doc: Document, config: NormalizationConfig | None = None) -> Document:
    """Normalize block by block and rebuild text, offsets, and content hash."""
    cfg = config or NormalizationConfig()
    blocks = [b for b in (_normalize_block(b, cfg) for b in doc.blocks) if b is not None]
    meta = {**doc.metadata, "normalized": cfg.unicode_form}
    return doc.with_blocks(blocks, metadata=meta, title=normalize_text(doc.title, cfg))


# ----------------------------------------------------------------------------- near duplicates
_WORD = re.compile(r"\w+", re.UNICODE)
_MERSENNE = np.uint64((1 << 31) - 1)


def shingles(text: str, k: int = 5) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """MinHash over k-word shingles with universal hashing ((a*x + b) mod p), deterministic by seed."""

    def __init__(self, num_perm: int = 128, *, shingle_size: int = 5, seed: int = 7) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.RandomState(seed)
        self._a = rng.randint(1, int(_MERSENNE), size=num_perm).astype(np.uint64)
        self._b = rng.randint(0, int(_MERSENNE), size=num_perm).astype(np.uint64)

    @staticmethod
    def _base_hash(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") % int(_MERSENNE)

    def signature(self, text: str) -> np.ndarray:
        sh = shingles(text, self.shingle_size)
        if not sh:
            return np.full(self.num_perm, int(_MERSENNE), dtype=np.uint64)
        x = np.fromiter((self._base_hash(s) for s in sh), dtype=np.uint64, count=len(sh))
        hashed = (np.outer(x, self._a) + self._b) % _MERSENNE  # (shingles, num_perm), fits in uint64
        return hashed.min(axis=0)

    @staticmethod
    def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


class NearDuplicateIndex:
    """Banded LSH over MinHash signatures. With b bands of r rows, pairs with Jaccard s collide in
    at least one band with probability 1 - (1 - s**r)**b, an S-curve centred near (1/b)**(1/r)."""

    def __init__(self, *, threshold: float = 0.9, num_perm: int = 128, bands: int = 16, seed: int = 7) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self.hasher = MinHasher(num_perm, seed=seed)
        self._buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
        self._signatures: dict[str, np.ndarray] = {}

    def _band_keys(self, sig: np.ndarray) -> list[tuple[int, bytes]]:
        return [(i, sig[i * self.rows : (i + 1) * self.rows].tobytes()) for i in range(self.bands)]

    def query(self, text: str) -> list[tuple[str, float]]:
        sig = self.hasher.signature(text)
        return self._query_sig(sig)

    def _query_sig(self, sig: np.ndarray) -> list[tuple[str, float]]:
        candidates: set[str] = set()
        for key in self._band_keys(sig):
            candidates.update(self._buckets.get(key, ()))
        scored = [(c, MinHasher.similarity(sig, self._signatures[c])) for c in candidates]
        return sorted([s for s in scored if s[1] >= self.threshold], key=lambda s: (-s[1], s[0]))

    def add(self, key: str, text: str) -> list[tuple[str, float]]:
        """Insert and return existing entries that are near-duplicates of this text."""
        if key in self._signatures:
            raise KeyError(f"duplicate key {key!r}")
        sig = self.hasher.signature(text)
        matches = self._query_sig(sig)
        self._signatures[key] = sig
        for band_key in self._band_keys(sig):
            self._buckets[band_key].append(key)
        return matches

    def __len__(self) -> int:
        return len(self._signatures)


class DuplicateRecord(BaseModel):
    dropped_id: str
    kept_id: str
    similarity: float  # 1.0 for exact (hash-equal) duplicates
    exact: bool


class DedupResult(BaseModel):
    kept: list[Document]
    duplicates: list[DuplicateRecord] = Field(default_factory=list)


def dedupe_documents(
    docs: Iterable[Document],
    *,
    threshold: float = 0.9,
    near: bool = True,
    keep: Literal["first", "newest"] = "newest",
) -> DedupResult:
    """Drop exact and near duplicates, but only within the same permission scope.

    Two documents with the same text and different ACLs are not duplicates for retrieval: merging
    them would either leak the restricted copy or hide the public one. The scope key is
    (tenant, sorted acl_groups).
    """
    items = list(docs)
    order = {id(d): i for i, d in enumerate(items)}
    if keep == "newest":  # visit newest first so it is the one kept; stable on ties
        items = sorted(items, key=lambda d: d.updated_at or "", reverse=True)
    kept: list[Document] = []
    dups: list[DuplicateRecord] = []
    by_hash: dict[tuple[object, ...], str] = {}
    indexes: dict[tuple[object, ...], NearDuplicateIndex] = {}
    for d in items:
        scope = (d.tenant, tuple(sorted(d.acl_groups)))
        hkey = (*scope, d.content_hash)
        if hkey in by_hash:
            dups.append(DuplicateRecord(dropped_id=d.id, kept_id=by_hash[hkey], similarity=1.0, exact=True))
            continue
        if near:
            index = indexes.setdefault(scope, NearDuplicateIndex(threshold=threshold))
            matches = index.query(d.text)
            if matches:
                dups.append(DuplicateRecord(dropped_id=d.id, kept_id=matches[0][0], similarity=matches[0][1], exact=False))
                continue
            index.add(d.id, d.text)
        by_hash[hkey] = d.id
        kept.append(d)
    kept.sort(key=lambda d: order[id(d)])  # report survivors in input order
    return DedupResult(kept=kept, duplicates=dups)


__all__ = [
    "DEFAULT_BOILERPLATE",
    "DedupResult",
    "DuplicateRecord",
    "MinHasher",
    "NearDuplicateIndex",
    "NormalizationConfig",
    "dedupe_documents",
    "dehyphenate",
    "jaccard",
    "normalize_document",
    "normalize_text",
    "normalize_unicode",
    "normalize_whitespace",
    "remove_boilerplate",
    "shingles",
    "strip_repeated_lines",
]
