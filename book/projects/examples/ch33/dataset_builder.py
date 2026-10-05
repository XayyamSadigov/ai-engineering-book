# path: book/projects/examples/ch33/dataset_builder.py
"""Dataset builder for supervised fine-tuning (Chapter 33).

Turns raw labeled records (typically exported from production traces) into a training set
you can defend: cleaned, deduplicated, split without leakage, written as JSONL chat examples,
and described by a data card. Everything here is deterministic given a seed, so a build is
reproducible and the data card can be committed next to the model version it produced.

Dependencies: pydantic, numpy, scikit-learn (only for the TF-IDF fallback embedder).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------

Source = Literal["production_trace", "annotated", "synthetic"]


class RawExample(BaseModel):
    """One labeled record as it comes out of the trace store or the labeling tool."""

    id: str
    text: str
    label: str
    entity_id: str  # the grouping key for leakage-safe splits: account, customer, template...
    created_at: datetime
    source: Source = "production_trace"
    consent: bool = True  # False when the owner opted out of training use
    labeler_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatExample(BaseModel):
    """The JSONL row format accepted by most hosted fine-tuning APIs and by open-weights trainers."""

    messages: list[ChatMessage]
    meta: dict[str, str] = Field(default_factory=dict)  # kept out of the training payload


class CleanConfig(BaseModel):
    min_chars: int = 10
    max_chars: int = 4000
    allowed_labels: set[str] | None = None
    require_consent: bool = True
    scrub_pii: bool = True


class SplitConfig(BaseModel):
    strategy: Literal["entity", "time"] = "entity"
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 7
    # For the time strategy: everything at or after this instant becomes the test set, the
    # slice before it is split into train/val by entity. Mimics "train on the past, test on
    # the future", which is how the model will actually be used.
    time_cutoff: datetime | None = None


class DedupStats(BaseModel):
    exact_removed: int = 0
    near_removed: int = 0
    conflicting_pairs: list[tuple[str, str]] = Field(default_factory=list)


class Splits(BaseModel):
    train: list[RawExample]
    val: list[RawExample]
    test: list[RawExample]


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_WS = re.compile(r"\s+")


def scrub_pii(text: str) -> str:
    """Minimal PII scrubber: replaces emails, phone numbers, and IBAN-like tokens with placeholders.

    Production systems use a dedicated PII service (see Chapter 27). The point here is that the
    scrub happens *before* the text enters a training file, because weights cannot be redacted.
    """
    text = _EMAIL.sub("<EMAIL>", text)
    text = _IBAN.sub("<IBAN>", text)
    text = _PHONE.sub("<PHONE>", text)
    return text


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.strip())


def clean(examples: Iterable[RawExample], config: CleanConfig | None = None) -> tuple[list[RawExample], Counter]:
    """Drop examples that must not or cannot be trained on. Returns kept examples and drop reasons."""
    config = config or CleanConfig()
    kept: list[RawExample] = []
    reasons: Counter = Counter()
    for ex in examples:
        text = normalize_text(ex.text)
        if config.scrub_pii:
            text = scrub_pii(text)
        if config.require_consent and not ex.consent:
            reasons["no_consent"] += 1
            continue
        if len(text) < config.min_chars:
            reasons["too_short"] += 1
            continue
        if len(text) > config.max_chars:
            reasons["too_long"] += 1
            continue
        if not ex.label.strip():
            reasons["empty_label"] += 1
            continue
        if config.allowed_labels is not None and ex.label not in config.allowed_labels:
            reasons["unknown_label"] += 1
            continue
        kept.append(ex.model_copy(update={"text": text}))
    return kept, reasons


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------

EmbedFn = Callable[[list[str]], np.ndarray]


def _fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_text(text).lower().encode("utf-8")).hexdigest()


def dedupe_exact(examples: list[RawExample]) -> tuple[list[RawExample], int, list[tuple[str, str]]]:
    """Keep the earliest example per normalized text. Report pairs whose labels disagree."""
    by_time = sorted(examples, key=lambda e: (e.created_at, e.id))
    seen: dict[str, RawExample] = {}
    conflicts: list[tuple[str, str]] = []
    removed = 0
    for ex in by_time:
        fp = _fingerprint(ex.text)
        if fp in seen:
            removed += 1
            if seen[fp].label != ex.label:
                conflicts.append((seen[fp].id, ex.id))
            continue
        seen[fp] = ex
    return list(seen.values()), removed, conflicts


def tfidf_embed(texts: list[str]) -> np.ndarray:
    """Fallback embedder: character n-gram TF-IDF, L2-normalized so dot product == cosine.

    Catches paraphrases that differ by a few tokens, typos, and boilerplate reshuffles. It does
    not catch semantic paraphrases with different wording; plug in a real embedding client for
    that (``aie_core.embeddings`` in later chapters).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    matrix = vec.fit_transform([t.lower() for t in texts])
    dense = matrix.toarray().astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def dedupe_near(
    examples: list[RawExample],
    embed_fn: EmbedFn | None = None,
    threshold: float = 0.90,
) -> tuple[list[RawExample], int, list[tuple[str, str]]]:
    """Remove near-duplicates by cosine similarity, keeping the earliest of each cluster.

    O(n^2) in memory for the similarity matrix: fine for tens of thousands of rows. Beyond that,
    bucket by entity or use an approximate-nearest-neighbor index (Chapter 9).
    """
    if len(examples) < 2:
        return list(examples), 0, []
    embed = embed_fn or tfidf_embed
    ordered = sorted(examples, key=lambda e: (e.created_at, e.id))
    vectors = embed([e.text for e in ordered])
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sims = vectors @ vectors.T
    keep = np.ones(len(ordered), dtype=bool)
    conflicts: list[tuple[str, str]] = []
    for i in range(len(ordered)):
        if not keep[i]:
            continue
        later = np.where(sims[i, i + 1 :] >= threshold)[0] + i + 1
        for j in later:
            if keep[j]:
                keep[j] = False
                if ordered[i].label != ordered[j].label:
                    conflicts.append((ordered[i].id, ordered[j].id))
    kept = [e for e, k in zip(ordered, keep) if k]
    return kept, int((~keep).sum()), conflicts


# --------------------------------------------------------------------------------------
# Splitting without leakage
# --------------------------------------------------------------------------------------


def _bucket(entity_id: str, seed: int) -> float:
    """Deterministic uniform value in [0, 1) per entity. Same entity -> same split, every build."""
    digest = hashlib.sha256(f"{seed}:{entity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_by_entity(examples: list[RawExample], config: SplitConfig) -> Splits:
    """Assign whole entities to splits, filling train, then val, then test by example count.

    Entities are visited in a seeded pseudo-random order (their hash), so the assignment is
    stable across builds and independent of input order. Filling by example count rather than
    by entity count keeps the fractions close to target even when a few entities hold most of
    the rows, which is the normal shape of support traffic.
    """
    groups: dict[str, list[RawExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.entity_id].append(ex)
    order = sorted(groups, key=lambda eid: _bucket(eid, config.seed))
    total = len(examples)
    train_target = config.train_frac * total
    val_target = (config.train_frac + config.val_frac) * total
    train, val, test = [], [], []
    assigned = 0
    for eid in order:
        rows = groups[eid]
        if assigned < train_target:
            train.extend(rows)
        elif assigned < val_target:
            val.extend(rows)
        else:
            test.extend(rows)
        assigned += len(rows)
    return Splits(train=train, val=val, test=test)


def split_by_time(examples: list[RawExample], config: SplitConfig) -> Splits:
    if config.time_cutoff is None:
        raise ValueError("time strategy requires time_cutoff")
    cutoff = config.time_cutoff
    past = [e for e in examples if e.created_at < cutoff]
    future = [e for e in examples if e.created_at >= cutoff]
    # Inside the past slice we still group by entity so val does not see train's accounts.
    inner = SplitConfig(strategy="entity", train_frac=config.train_frac / (config.train_frac + config.val_frac),
                        val_frac=config.val_frac / (config.train_frac + config.val_frac), seed=config.seed)
    inner_split = split_by_entity(past, inner)
    return Splits(train=inner_split.train, val=inner_split.val + inner_split.test, test=future)


def split(examples: list[RawExample], config: SplitConfig | None = None) -> Splits:
    config = config or SplitConfig()
    if config.strategy == "time":
        return split_by_time(examples, config)
    return split_by_entity(examples, config)


def assert_no_entity_overlap(splits: Splits) -> None:
    """Fail loudly if an entity appears in more than one split. Cheap, and it has caught real leaks."""
    seen: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for ex in getattr(splits, name):
            prev = seen.setdefault(ex.entity_id, name)
            if prev != name:
                raise ValueError(f"entity {ex.entity_id!r} appears in both {prev} and {name}")


# --------------------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class ArtifactHit(BaseModel):
    token: str
    label: str
    support: int
    purity: float


def detect_label_artifacts(
    examples: list[RawExample], min_support: int = 20, min_purity: float = 0.98
) -> list[ArtifactHit]:
    """Find tokens that almost perfectly predict one label.

    A token present in >= min_support examples, of which >= min_purity share one label, is a
    shortcut the model will learn instead of the task. Typical sources: an agent macro that
    inserts a phrase when they pick a category, a template header, a ticket-system tag that
    leaked into the body. Treat every hit as a question, not a verdict: "refund" predicting
    the refund category is legitimate signal, "[auto-routed]" is not.
    """
    per_token: dict[str, Counter] = defaultdict(Counter)
    for ex in examples:
        for tok in set(tokenize(ex.text)):
            per_token[tok][ex.label] += 1
    hits: list[ArtifactHit] = []
    for tok, counts in per_token.items():
        support = sum(counts.values())
        if support < min_support:
            continue
        label, top = counts.most_common(1)[0]
        purity = top / support
        if purity >= min_purity:
            hits.append(ArtifactHit(token=tok, label=label, support=support, purity=round(purity, 4)))
    return sorted(hits, key=lambda h: (-h.support, h.token))


def length_distribution(examples: list[RawExample]) -> dict[str, float]:
    """Token-ish lengths (whitespace words). Inspect by tokens, not only rows: long rows dominate loss."""
    lengths = np.array([len(tokenize(e.text)) for e in examples] or [0])
    return {
        "count": int(len(examples)),
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "max": float(lengths.max()),
        "total_tokens": float(lengths.sum()),
    }


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Inter-annotator agreement corrected for chance. 1.0 = perfect, 0.0 = chance level."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1 - expected)


# --------------------------------------------------------------------------------------
# Output format and data card
# --------------------------------------------------------------------------------------


def to_chat_example(ex: RawExample, system_prompt: str, user_template: str = "{text}") -> ChatExample:
    """Render one example in the exact shape the model will see at inference time."""
    return ChatExample(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_template.format(text=ex.text)),
            ChatMessage(role="assistant", content=ex.label),
        ],
        meta={"id": ex.id, "entity_id": ex.entity_id, "source": ex.source},
    )


def write_jsonl(path: Path, examples: Iterable[ChatExample]) -> int:
    """Write one JSON object per line with only the ``messages`` key. Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({"messages": [m.model_dump() for m in ex.messages]}, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_card(
    splits: Splits,
    *,
    drop_reasons: Counter,
    dedup: DedupStats,
    split_config: SplitConfig,
    artifacts: list[ArtifactHit],
    system_prompt: str,
    files: dict[str, Path] | None = None,
) -> dict:
    """A dict you commit next to the model version: what is in the data, where it came from, how it was split."""
    all_examples = splits.train + splits.val + splits.test
    times = [e.created_at for e in all_examples]
    labels = sorted({e.label for e in all_examples})

    def label_hist(rows: list[RawExample]) -> dict[str, int]:
        return dict(sorted(Counter(e.label for e in rows).items()))

    card = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "single-label classification",
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest()[:16],
        "labels": labels,
        "label_count": len(labels),
        "sizes": {k: len(getattr(splits, k)) for k in ("train", "val", "test")},
        "label_distribution": {k: label_hist(getattr(splits, k)) for k in ("train", "val", "test")},
        "rare_labels_train": [l for l, c in Counter(e.label for e in splits.train).items() if c < 20],
        "sources": dict(Counter(e.source for e in all_examples)),
        "time_range": {
            "min": min(times).isoformat() if times else None,
            "max": max(times).isoformat() if times else None,
        },
        "split": split_config.model_dump(mode="json"),
        "cleaning": {"dropped": dict(drop_reasons)},
        "dedup": dedup.model_dump(),
        "length_tokens": {k: length_distribution(getattr(splits, k)) for k in ("train", "val", "test")},
        "suspected_label_artifacts": [h.model_dump() for h in artifacts],
        "files": {k: {"path": str(p), "sha256": file_sha256(p)} for k, p in (files or {}).items() if p.exists()},
        "known_limitations": [],
    }
    return card


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


class BuildResult(BaseModel):
    splits: Splits
    card: dict


def build_dataset(
    raw: list[RawExample],
    *,
    out_dir: Path,
    system_prompt: str,
    clean_config: CleanConfig | None = None,
    split_config: SplitConfig | None = None,
    embed_fn: EmbedFn | None = None,
    near_threshold: float = 0.90,
) -> BuildResult:
    """Clean -> dedupe (exact, near) -> split -> write JSONL -> data card. Order matters:
    dedupe before split, otherwise the same text can land on both sides of the holdout."""
    split_config = split_config or SplitConfig()
    kept, reasons = clean(raw, clean_config)
    kept, exact_removed, conflicts = dedupe_exact(kept)
    kept, near_removed, near_conflicts = dedupe_near(kept, embed_fn=embed_fn, threshold=near_threshold)
    dedup = DedupStats(exact_removed=exact_removed, near_removed=near_removed,
                       conflicting_pairs=conflicts + near_conflicts)
    splits = split(kept, split_config)
    if split_config.strategy == "entity":
        assert_no_entity_overlap(splits)
    for name in ("train", "val", "test"):
        if not getattr(splits, name):
            raise ValueError(f"{name} split is empty: too few entities or a time cutoff outside the data range")
    artifacts = detect_label_artifacts(splits.train)

    files = {
        "train": out_dir / "train.jsonl",
        "val": out_dir / "val.jsonl",
        "test": out_dir / "test.jsonl",
    }
    for name, path in files.items():
        write_jsonl(path, (to_chat_example(e, system_prompt) for e in getattr(splits, name)))
    card = data_card(splits, drop_reasons=reasons, dedup=dedup, split_config=split_config,
                     artifacts=artifacts, system_prompt=system_prompt, files=files)
    (out_dir / "data_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return BuildResult(splits=splits, card=card)
