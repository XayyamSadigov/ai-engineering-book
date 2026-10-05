# path: book/projects/p2-semantic-search/tests/test_loader.py
from __future__ import annotations

import pytest

from semsearch.ingest.loader import chunk_paragraphs, embedding_text, load_corpus, load_document, parse_front_matter

from .conftest import SHARED_DOCS

DOC = """---
id: hr-pto-policy
title: Paid Time Off (PTO) Policy
version: "3.0"
tenant: shared
acl_groups: ["all"]
tags: [hr, pto, "carry, over"]
owner: People Operations
---

# Paid Time Off (PTO) Policy

Intro paragraph.

## 3. Carryover

Up to 10 days carry over.

Must be used by 31 March.

## 4. Requesting time off

Submit 14 days in advance.
"""


def test_front_matter_subset():
    meta, body = parse_front_matter(DOC)
    assert meta["version"] == "3.0"
    assert meta["acl_groups"] == ["all"]
    assert meta["tags"] == ["hr", "pto", "carry, over"]
    assert body.lstrip().startswith("# Paid Time Off")


def test_load_document_fields_and_versioning(tmp_path):
    p = tmp_path / "pto.md"
    p.write_text(DOC, encoding="utf-8")
    doc = load_document(p)
    assert doc.doc_id == "hr-pto-policy"
    assert doc.acl_groups == ("all",)
    assert doc.metadata["owner"] == "People Operations"
    v1 = doc.index_version
    p.write_text(DOC.replace("10 days", "12 days"), encoding="utf-8")
    assert load_document(p).index_version != v1  # same declared version, new content


def test_missing_acl_fails_closed(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nid: x\ntitle: X\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="acl_groups"):
        load_document(p)


def test_chunks_follow_headings():
    _, body = parse_front_matter(DOC)
    chunks = chunk_paragraphs(body, max_chars=1000)
    sections = [c.section for c in chunks]
    assert sections == ["Paid Time Off (PTO) Policy", "3. Carryover", "4. Requesting time off"]
    assert "31 March" in chunks[1].text
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_size_budget_splits_long_sections():
    body = "## S\n\n" + "\n\n".join("p" * 300 for _ in range(5))
    chunks = chunk_paragraphs(body, max_chars=700)
    assert len(chunks) >= 3
    assert all(c.section == "S" for c in chunks)


def test_embedding_text_carries_title_and_section(tmp_path):
    p = tmp_path / "pto.md"
    p.write_text(DOC, encoding="utf-8")
    doc = load_document(p)
    chunk = chunk_paragraphs(doc.body)[1]
    assert embedding_text(doc, chunk).startswith("Paid Time Off (PTO) Policy > 3. Carryover")


@pytest.mark.skipif(not SHARED_DOCS.exists(), reason="shared-data not present")
def test_shared_corpus_loads_with_unique_ids():
    docs = load_corpus(SHARED_DOCS)
    assert len(docs) >= 10
    assert all(d.acl_groups for d in docs)
    assert {d.tenant for d in docs} <= {"shared", "retail", "logistics"}
