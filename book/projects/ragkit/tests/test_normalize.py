# path: book/projects/ragkit/tests/test_normalize.py
from ragkit.documents import Block, Document, SourceType
from ragkit.normalize import (
    MinHasher,
    NearDuplicateIndex,
    NormalizationConfig,
    dedupe_documents,
    jaccard,
    normalize_document,
    normalize_text,
    normalize_unicode,
    normalize_whitespace,
    remove_boilerplate,
    shingles,
    strip_repeated_lines,
)


def test_unicode_and_invisible_characters():
    assert normalize_unicode("ﬁle name​") == "file name"
    assert normalize_unicode("soft­hyphen") == "softhyphen"
    assert normalize_unicode("m²", "NFC") == "m²" and normalize_unicode("m²", "NFKC") == "m2"


def test_whitespace_and_boilerplate():
    assert normalize_whitespace("a  \t b\r\n\r\n\r\n\nc  ") == "a b\n\nc"
    text = "Body line\nPage 3 of 12\nCONFIDENTIAL\n© 2026 Northwind. All rights reserved."
    assert remove_boilerplate(text) == "Body line"
    assert normalize_text("exam-\nple text") == "example text"


def test_strip_repeated_lines_masks_digits():
    pages = [["Northwind HR, page 1", "Alpha"], ["Northwind HR, page 2", "Beta"], ["Northwind HR, page 3", "Gamma"]]
    assert strip_repeated_lines(pages) == [["Alpha"], ["Beta"], ["Gamma"]]


def test_normalize_document_preserves_code_and_rebuilds_offsets():
    blocks = [
        Block(kind="paragraph", text="Hello   world"),
        Block(kind="code", text="```\nx  =  1 # keep spacing\n```"),
        Block(kind="table", text="", meta={"header": ["A ", "B"], "rows": [["1", "two  words"]]}),
        Block(kind="paragraph", text="Page 2 of 9"),
    ]
    d = Document(id="d", version="1", source_uri="d", source_type=SourceType.MARKDOWN, parser="t", text="x",
                 blocks=blocks)
    n = normalize_document(d)
    assert [b.kind for b in n.blocks] == ["paragraph", "code", "table"]  # boilerplate-only block dropped
    assert n.blocks[0].text == "Hello world"
    assert n.blocks[1].text == "```\nx  =  1 # keep spacing\n```"
    assert n.blocks[2].text == "| A | B |\n|---|---|\n| 1 | two words |"
    assert all(n.text[b.char_start : b.char_end] == b.text for b in n.blocks)
    assert n.content_hash != d.content_hash and n.metadata["normalized"] == "NFKC"
    assert normalize_document(d, NormalizationConfig(unicode_form="NFC")).metadata["normalized"] == "NFC"


BASE = ("Employees may carry over up to ten unused paid time off days into the next calendar year. "
        "Carried over days must be used by the end of March or they are forfeited without payout. ") * 3


def test_minhash_estimates_jaccard():
    near = BASE.replace("end of March", "end of April")
    exact = jaccard(shingles(BASE), shingles(near))
    h = MinHasher(256)
    est = MinHasher.similarity(h.signature(BASE), h.signature(near))
    assert abs(est - exact) < 0.12
    assert MinHasher.similarity(h.signature(BASE), h.signature("completely different text about VPN errors")) < 0.1


def test_near_duplicate_index_finds_only_near_copies():
    idx = NearDuplicateIndex(threshold=0.7)
    assert idx.add("a", BASE) == []
    assert idx.add("b", "VPN error 809 means the local network blocks port 443; switch to TCP fallback.") == []
    matches = idx.add("c", BASE.replace("ten", "10"))
    assert [m[0] for m in matches] == ["a"]
    assert len(idx) == 3


def _d(i, text, tenant="retail", acl=("all",), updated="2026-01-01"):
    return Document(id=i, version="1", source_uri=i, source_type=SourceType.TEXT, parser="t", text=text,
                    tenant=tenant, acl_groups=list(acl), updated_at=updated)


def test_dedupe_exact_near_and_permission_scope():
    docs = [
        _d("old", BASE, updated="2025-01-01"),
        _d("new", BASE.replace("ten", "10"), updated="2026-01-01"),
        _d("copy", BASE.replace("ten", "10"), updated="2024-01-01"),
        _d("restricted", BASE, acl=("hr",)),  # same text, different ACL: not a duplicate
    ]
    r = dedupe_documents(docs, threshold=0.7)
    assert [d.id for d in r.kept] == ["new", "restricted"]
    by_id = {x.dropped_id: x for x in r.duplicates}
    assert by_id["copy"].exact and by_id["copy"].kept_id == "new"
    assert not by_id["old"].exact and by_id["old"].kept_id == "new"
