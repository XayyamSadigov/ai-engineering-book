# path: book/projects/p2-semantic-search/semsearch/eval/run_eval.py
"""Evaluate retrieval against a gold file: recall@k, MRR, hit rate, ACL leakage.

Gold rows (shared-data/eval/retrieval_gold.jsonl, or the local fixture) look like:
  {"id": "RQ-001", "question": "...", "required_doc_ids": [...], "acceptable_doc_ids": [...],
   "user_groups": ["all"], "tenant": "shared", "tags": ["exact-fact"]}

Rows tagged "forbidden-doc" invert the expectation: the required document exists but the
user may not see it, so any appearance in the results is a leak, not a success.
Rows whose required documents are not in the index at all are reported separately as
ingestion gaps, so that a missing document is never blamed on the retriever.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from pydantic import BaseModel, Field

from ..domain.metrics import recall_at_k, reciprocal_rank, unique_in_order
from ..service import Principal, SearchService

PROJECT_DIR = Path(__file__).resolve().parents[2]
SHARED_GOLD = PROJECT_DIR.parent / "shared-data" / "eval" / "retrieval_gold.jsonl"
LOCAL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retrieval_fixture.jsonl"


class GoldRow(BaseModel):
    id: str
    question: str
    required_doc_ids: list[str]
    acceptable_doc_ids: list[str] = Field(default_factory=list)
    user_groups: list[str]
    tenant: str
    tags: list[str] = Field(default_factory=list)

    @property
    def forbidden(self) -> bool:
        return "forbidden-doc" in self.tags


def default_gold_path() -> Path:
    return SHARED_GOLD if SHARED_GOLD.exists() else LOCAL_FIXTURE


def load_gold(path: str | Path | None = None) -> list[GoldRow]:
    p = Path(path) if path else default_gold_path()
    return [GoldRow(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class EvalReport:
    gold_path: str
    ks: tuple[int, ...]
    rows: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    leaks: list[str] = field(default_factory=list)
    forbidden_rows: int = 0
    ingestion_gaps: list[str] = field(default_factory=list)
    misses: list[dict[str, Any]] = field(default_factory=list)
    recall_by_tag: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"ks": list(self.ks)}

    def format(self) -> str:
        lines = [f"gold: {self.gold_path}", f"scored rows: {self.rows}  forbidden rows: {self.forbidden_rows}"]
        lines += [f"recall@{k}: {self.recall[k]:.3f}   hit@{k}: {self.hit_rate[k]:.3f}" for k in self.ks]
        lines.append(f"MRR: {self.mrr:.3f}")
        lines.append(f"ACL leaks: {len(self.leaks)} {self.leaks}")
        lines.append(f"ingestion gaps: {self.ingestion_gaps}")
        kmax = max(self.ks)
        lines += [f"recall@{kmax} [{tag}]: {v:.3f}" for tag, v in sorted(self.recall_by_tag.items())]
        return "\n".join(lines)


def evaluate(service: SearchService, gold: Sequence[GoldRow], ks: tuple[int, ...] = (1, 3, 5, 10), gold_path: str = "") -> EvalReport:
    report = EvalReport(gold_path=gold_path, ks=ks)
    kmax = max(ks)
    indexed = set(service.store.doc_versions(service.namespace))
    recalls: dict[int, list[float]] = defaultdict(list)
    hits: dict[int, list[float]] = defaultdict(list)
    rrs: list[float] = []
    by_tag: dict[str, list[float]] = defaultdict(list)
    for row in gold:
        principal = Principal(user_id=f"eval:{row.id}", tenant=row.tenant, groups=tuple(row.user_groups))
        # Ask for more chunks than kmax so that kmax *documents* survive the collapse.
        result = service.search(row.question, principal, k=min(service.max_k, kmax * 4))
        ranked = unique_in_order(h.doc_id for h in result.hits)
        if row.forbidden:
            report.forbidden_rows += 1
            if set(row.required_doc_ids) & set(ranked):
                report.leaks.append(row.id)
            continue
        missing = [d for d in row.required_doc_ids if d not in indexed]
        if missing:
            report.ingestion_gaps.append(f"{row.id}:{','.join(missing)}")
            continue
        report.rows += 1
        for k in ks:
            r = recall_at_k(ranked, row.required_doc_ids, k)
            recalls[k].append(r)
            hits[k].append(1.0 if r > 0 else 0.0)
        rrs.append(reciprocal_rank(ranked, row.required_doc_ids))
        r_max = recalls[kmax][-1]
        for tag in row.tags:
            by_tag[tag].append(r_max)
        if r_max < 1.0:
            report.misses.append({"id": row.id, "required": row.required_doc_ids, "got": ranked[:kmax]})
    if report.rows:
        report.recall = {k: mean(recalls[k]) for k in ks}
        report.hit_rate = {k: mean(hits[k]) for k in ks}
        report.mrr = mean(rrs)
        report.recall_by_tag = {t: mean(v) for t, v in by_tag.items()}
    else:
        report.recall = {k: 0.0 for k in ks}
        report.hit_rate = {k: 0.0 for k in ks}
    return report


__all__ = ["GoldRow", "EvalReport", "load_gold", "evaluate", "default_gold_path"]
