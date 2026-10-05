# path: book/projects/ragkit/tests/test_documents.py
from ragkit.documents import Block, Chunk, Document, SourceType, assemble, doc_id_from_uri, sha256_text


def _doc(**kw):
    base = dict(id="d1", version="1", source_uri="a.md", source_type=SourceType.MARKDOWN, parser="t/1", text="hello")
    base.update(kw)
    return Document(**base)


def test_content_hash_is_filled_from_text():
    assert _doc().content_hash == sha256_text("hello")


def test_assemble_offsets_point_into_text():
    blocks = [Block(kind="heading", text="# T"), Block(kind="paragraph", text="One."), Block(kind="paragraph", text="")]
    text, out = assemble(blocks)
    assert text == "# T\n\nOne."
    assert len(out) == 2  # empty block dropped
    assert all(text[b.char_start : b.char_end] == b.text for b in out)


def test_with_blocks_rebuilds_text_and_hash():
    d = _doc()
    d2 = d.with_blocks([Block(kind="paragraph", text="new text")])
    assert d2.text == "new text"
    assert d2.content_hash == sha256_text("new text")
    assert d2.id == d.id


def test_acl_fail_closed_and_visibility():
    assert not _doc().has_acl
    d = _doc(tenant="retail", acl_groups=["hr"])
    assert d.has_acl
    assert d.visible_to(["hr"], tenant="retail")
    assert not d.visible_to(["hr"], tenant="logistics")
    assert not d.visible_to(["all"], tenant="retail")
    assert _doc(tenant="shared", acl_groups=["all"]).visible_to(["all"], tenant="logistics")


def test_doc_id_from_uri_is_stable():
    assert doc_id_from_uri("docs/a.md") == doc_id_from_uri("docs/a.md")
    assert doc_id_from_uri("docs/a.md") != doc_id_from_uri("docs/b.md")


def test_embedding_text_has_breadcrumb_but_text_stays_clean():
    c = Chunk(
        id="d1:x", doc_id="d1", version="1", text="Body.", content_hash="h", tenant="retail", acl_groups=["all"],
        metadata={"title": "Policy"}, section_path=["Policy", "Limits"], position=0, char_start=0, char_end=5,
        token_count=2, chunker="t",
    )
    assert c.embedding_text() == "Policy > Limits\n\nBody."
    assert c.text == "Body."
