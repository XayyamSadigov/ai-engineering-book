# path: book/projects/ragkit/tests/test_chunkers.py
import pytest
from aie_core.embeddings import FakeEmbeddings
from conftest import POLICY_MD, big_table_md
from pdf_fixtures import make_pdf

from ragkit.chunking import (
    FixedTokenChunker,
    MarkdownSectionChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    expand_to_parents,
    split_roles,
    split_sentences,
)
from ragkit.documents import Document, SourceType
from ragkit.parsers import DocDefaults, MarkdownParser, PdfParser, TextParser
from ragkit.tokenizers import RegexTokenizer

TOK = RegexTokenizer()


def md(text: str, uri: str = "t.md") -> Document:
    return MarkdownParser().parse(text, source_uri=uri, defaults=DocDefaults(tenant="retail", acl_groups=["all"]))[0]


def plain(text: str) -> Document:
    return Document(id="p", version="1", source_uri="p.txt", source_type=SourceType.TEXT, parser="t", text=text,
                    tenant="retail", acl_groups=["all"])


LONG = " ".join(f"Sentence number {i} talks about refunds and returns." for i in range(60))


def tokens(text: str) -> list[str]:
    return [text[s:e] for s, e in TOK.spans(text)]


# ----------------------------------------------------------------------------- fixed
def test_fixed_overlap_is_exact_and_sizes_bounded():
    chunks = FixedTokenChunker(50, 10, tokenizer=TOK).chunk(plain(LONG))
    assert len(chunks) > 3
    assert all(c.token_count <= 50 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert tokens(a.text)[-10:] == tokens(b.text)[:10]


def test_fixed_covers_every_token_and_offsets_match():
    doc = plain(LONG)
    chunks = FixedTokenChunker(40, 0, tokenizer=TOK).chunk(doc)
    assert sum(len(tokens(c.text)) for c in chunks) == len(tokens(LONG))
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_fixed_validates_parameters_and_handles_empty():
    with pytest.raises(ValueError):
        FixedTokenChunker(10, 10)
    with pytest.raises(ValueError):
        FixedTokenChunker(0, 0)
    assert FixedTokenChunker(10, 2, tokenizer=TOK).chunk(plain("   ")) == []


# ----------------------------------------------------------------------------- recursive
def test_recursive_prefers_paragraph_boundaries():
    paras = [f"Paragraph {i}. " + "word " * 30 for i in range(6)]
    doc = plain("\n\n".join(p.strip() for p in paras))
    chunks = RecursiveChunker(80, tokenizer=TOK).chunk(doc)
    assert all(c.token_count <= 80 for c in chunks)
    assert all(c.text.startswith("Paragraph") for c in chunks)  # never starts mid-paragraph


def test_recursive_falls_back_to_tokens_for_unbreakable_text():
    doc = plain("x" * 10 + "-" * 300)  # punctuation tokens, no separators at all
    chunks = RecursiveChunker(50, tokenizer=TOK).chunk(doc)
    assert all(c.token_count <= 50 for c in chunks)
    assert "".join(c.text for c in chunks) == doc.text


def test_recursive_overlap_repeats_whole_atoms():
    doc = plain("\n".join(f"Line {i} has five tokens" for i in range(40)))
    chunks = RecursiveChunker(30, overlap=6, tokenizer=TOK).chunk(doc)
    for a, b in zip(chunks, chunks[1:]):
        assert b.text.split("\n")[0] == a.text.split("\n")[-1]


# ----------------------------------------------------------------------------- sentences
def test_sentence_splitter_handles_abbreviations_versions_and_newlines():
    text = "See e.g. the API v2.3 docs. Dr. Lee approved it! Next line\n- list item. Price is 4.50 USD."
    sents = [text[s:e] for s, e in split_sentences(text)]
    assert sents == ["See e.g. the API v2.3 docs.", "Dr. Lee approved it!", "Next line", "- list item.",
                     "Price is 4.50 USD."]


def test_sentence_chunker_never_cuts_inside_a_sentence_and_overlaps():
    doc = plain(LONG)
    sents = {doc.text[s:e] for s, e in split_sentences(doc.text)}
    chunks = SentenceChunker(60, overlap_sentences=1, tokenizer=TOK).chunk(doc)
    for c in chunks:
        assert c.token_count <= 60
        parts = [c.text[s:e] for s, e in split_sentences(c.text)]
        assert all(p in sents for p in parts)
    for a, b in zip(chunks, chunks[1:]):
        last = [a.text[s:e] for s, e in split_sentences(a.text)][-1]
        assert b.text.startswith(last)


def test_sentence_chunker_keeps_fitting_paragraphs_whole():
    doc = plain("Alpha one. Alpha two.\n\nBeta one. Beta two. Beta three.\n\nGamma one.")
    chunks = SentenceChunker(11, overlap_sentences=0, tokenizer=TOK).chunk(doc)  # 6 + 9 + 3 tokens
    assert [c.text for c in chunks] == ["Alpha one. Alpha two.", "Beta one. Beta two. Beta three.", "Gamma one."]


# ----------------------------------------------------------------------------- section-aware
def test_section_chunks_never_span_sections_and_include_heading(policy_doc):
    chunks = MarkdownSectionChunker(400, tokenizer=TOK).chunk(policy_doc)
    paths = [c.section_path for c in chunks]
    assert paths == [["Test Policy"], ["Test Policy", "Limits"], ["Test Policy", "Example code"],
                     ["Test Policy", "Example code", "Details"]]
    assert chunks[1].text.startswith("## Limits\n\n| Client type | Limit |")
    assert all(policy_doc.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_section_keeps_code_block_intact_even_when_oversize(policy_doc):
    code_block = next(b for b in policy_doc.blocks if b.kind == "code")
    for budget in (400, 12):
        chunks = MarkdownSectionChunker(budget, tokenizer=TOK).chunk(policy_doc)
        holders = [c for c in chunks if "def call(client):" in c.text]
        assert len(holders) == 1
        assert code_block.text in holders[0].text
        assert holders[0].text.count("```") % 2 == 0
    oversize = [c for c in MarkdownSectionChunker(12, tokenizer=TOK).chunk(policy_doc) if c.metadata.get("oversize")]
    assert len(oversize) == 1 and oversize[0].kind == "code"


def test_small_table_stays_whole(policy_doc):
    chunks = MarkdownSectionChunker(400, tokenizer=TOK).chunk(policy_doc)
    limits = chunks[1]
    assert "| Store Server | 200 requests/minute |" in limits.text and "| Internal tools |" in limits.text
    assert limits.kind == "mixed"  # heading + table + paragraph


def test_large_table_split_repeats_header_and_loses_no_rows():
    doc = md(big_table_md(40))
    chunks = MarkdownSectionChunker(120, tokenizer=TOK).chunk(doc)
    parts = [c for c in chunks if c.kind == "table"]
    assert len(parts) > 2
    for i, p in enumerate(parts, start=1):
        assert p.text.startswith("| Code | HTTP | Meaning |\n|---|---|---|\n")
        assert p.token_count <= 120
        assert p.metadata["table_part"] == i and p.metadata["table_parts"] == len(parts)
    rows = [ln for p in parts for ln in p.text.splitlines()[2:]]
    assert rows == [f"| E-{i:03d} | {400 + i % 100} | Meaning number {i} for the error code table |" for i in range(40)]
    assert all(p.metadata["header_repeated"] for p in parts[1:])


def test_large_table_without_header_repeat():
    parts = [c for c in MarkdownSectionChunker(120, repeat_table_header=False, tokenizer=TOK).chunk(md(big_table_md(40)))
             if c.kind == "table"]
    assert parts[0].text.startswith("| Code |")
    assert not any(p.text.startswith("| Code |") for p in parts[1:])


def test_section_chunker_falls_back_without_blocks():
    chunks = MarkdownSectionChunker(50, tokenizer=TOK).chunk(plain(LONG))
    assert chunks and all(c.token_count <= 50 for c in chunks)


def test_transcript_chunks_carry_speakers():
    raw = "\n".join(f"[00:0{i}:00] {'Dana' if i % 2 else 'Sam'}: Point {i} about the outage." for i in range(6))
    doc = TextParser().parse(raw, source_uri="call.txt", defaults=DocDefaults(tenant="retail", acl_groups=["all"]))[0]
    chunks = MarkdownSectionChunker(30, tokenizer=TOK).chunk(doc)
    assert chunks[0].metadata["speakers"] == ["Dana", "Sam"] and chunks[0].metadata["start_time"] == "00:00:00"


def test_pdf_chunks_record_pages():
    pdf = make_pdf([["Policy text on page one about refunds."] * 3, ["More text on page two about returns."] * 3])
    doc = PdfParser(min_chars_per_page=5).parse(pdf, source_uri="p.pdf")[0]
    chunks = FixedTokenChunker(20, 0, tokenizer=TOK).chunk(doc)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)


# ----------------------------------------------------------------------------- semantic
TOPIC_A = ["Refund policy covers returns within thirty days.", "Refund requests need the original receipt.",
           "Refund amounts go back to the original card."]
TOPIC_B = ["VPN access needs the NorthGate client.", "VPN error 809 means port 443 is blocked.",
           "VPN sessions expire after twelve hours."]
VOCAB = ["refund", "policy", "returns", "receipt", "card", "original", "vpn", "northgate", "client", "error", "port",
         "sessions", "access"]


def test_semantic_chunker_cuts_at_topic_shift():
    emb = FakeEmbeddings(vocabulary=VOCAB)
    doc = plain(" ".join(TOPIC_A + TOPIC_B))
    chunker = SemanticChunker(emb, window=0, breakpoint_percentile=80, max_tokens=200, min_tokens=5, tokenizer=TOK)
    chunks = chunker.chunk(doc)
    assert [c.text for c in chunks] == [" ".join(TOPIC_A), " ".join(TOPIC_B)]
    assert max(chunker.last_distances) == chunker.last_distances[2]  # the A->B boundary
    assert emb.calls and len(emb.calls[0]) == 6  # one embedding per sentence


def test_semantic_chunker_enforces_max_and_fingerprint_names_model():
    emb = FakeEmbeddings(vocabulary=VOCAB, model="vocab-test")
    chunks = SemanticChunker(emb, window=0, max_tokens=25, min_tokens=3, tokenizer=TOK).chunk(plain(" ".join(TOPIC_A * 4)))
    assert all(c.token_count <= 25 for c in chunks)
    other = SemanticChunker(FakeEmbeddings(vocabulary=VOCAB, model="other"), window=0, max_tokens=25, min_tokens=3,
                            tokenizer=TOK)
    assert other.fingerprint() != SemanticChunker(emb, window=0, max_tokens=25, min_tokens=3, tokenizer=TOK).fingerprint()


# ----------------------------------------------------------------------------- parent-child
def test_parent_child_linkage(policy_doc):
    chunker = ParentChildChunker(MarkdownSectionChunker(400, tokenizer=TOK), SentenceChunker(8, 0, tokenizer=TOK),
                                 tokenizer=TOK)
    chunks = chunker.chunk(policy_doc)
    parents, children = split_roles(chunks)
    by_id = {p.id: p for p in parents}
    assert parents and children
    assert all(c.role == "child" and c.parent_id in by_id for c in children)
    assert all(c.text in by_id[c.parent_id].text for c in children)
    assert all(policy_doc.text[c.char_start : c.char_end] == c.text for c in children)
    assert all(p.metadata["child_count"] == sum(1 for c in children if c.parent_id == p.id) >= 1 for p in parents)
    assert all(c.section_path == by_id[c.parent_id].section_path for c in children)
    # output order: each parent followed by its children
    assert chunks[0].role == "parent" and chunks[1].parent_id == chunks[0].id


def test_expand_to_parents_dedupes_in_rank_order(policy_doc):
    chunks = ParentChildChunker(MarkdownSectionChunker(400, tokenizer=TOK), SentenceChunker(8, 0, tokenizer=TOK),
                                tokenizer=TOK).chunk(policy_doc)
    parents, children = split_roles(chunks)
    by_id = {p.id: p for p in parents}
    first_parent_kids = [c for c in children if c.parent_id == parents[0].id]
    second_parent_kids = [c for c in children if c.parent_id == parents[1].id]
    hits = [second_parent_kids[0], first_parent_kids[0], second_parent_kids[-1]]
    assert [p.id for p in expand_to_parents(hits, by_id)] == [parents[1].id, parents[0].id]


# ----------------------------------------------------------------------------- identity
ALL_CHUNKERS = [
    lambda: FixedTokenChunker(40, 8, tokenizer=TOK),
    lambda: RecursiveChunker(60, tokenizer=TOK),
    lambda: SentenceChunker(40, tokenizer=TOK),
    lambda: MarkdownSectionChunker(60, tokenizer=TOK),
    lambda: ParentChildChunker(tokenizer=TOK),
    lambda: SemanticChunker(FakeEmbeddings(32), max_tokens=60, min_tokens=5, tokenizer=TOK),
]


@pytest.mark.parametrize("make", ALL_CHUNKERS)
def test_ids_are_deterministic_unique_and_inherit_acl(make):
    doc = md(POLICY_MD)
    a, b = make().chunk(doc), make().chunk(md(POLICY_MD))
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)
    assert all(c.id.startswith(f"{doc.id}:") for c in a)
    assert all(c.tenant == doc.tenant and c.acl_groups == doc.acl_groups for c in a)
    assert all(c.metadata["title"] == "Test Policy" and c.version == doc.version for c in a)


def test_section_ids_survive_insertions_and_version_bumps():
    original = md(POLICY_MD)
    edited_md = POLICY_MD.replace('version: "1.0"', 'version: "1.1"').replace(
        "# Test Policy\n", "# Test Policy\n\nA brand new opening paragraph inserted above everything else.\n", 1)
    edited = md(edited_md)
    chunker = MarkdownSectionChunker(400, tokenizer=TOK)
    before = {c.section_path[-1]: c.id for c in chunker.chunk(original)}
    after = {c.section_path[-1]: c.id for c in chunker.chunk(edited)}
    assert before["Test Policy"] != after["Test Policy"]  # the edited section changes
    for unchanged in ("Limits", "Example code", "Details"):
        assert before[unchanged] == after[unchanged]


def test_config_change_changes_ids_and_duplicates_get_distinct_ids():
    doc = md("# T\n\nSame paragraph.\n\n## A\n\nSame paragraph.\n\nSame paragraph.\n")
    c1 = MarkdownSectionChunker(3, include_headings=False, tokenizer=TOK).chunk(doc)
    c2 = MarkdownSectionChunker(4, include_headings=False, tokenizer=TOK).chunk(doc)
    assert {c.id for c in c1}.isdisjoint({c.id for c in c2})
    assert [c.text for c in c1] == ["Same paragraph."] * 3  # two of them share a section
    assert len({c.id for c in c1}) == 3
