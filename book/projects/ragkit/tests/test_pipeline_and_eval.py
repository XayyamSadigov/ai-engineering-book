# path: book/projects/ragkit/tests/test_pipeline_and_eval.py
import json

from conftest import SHARED_DOCS
from pdf_fixtures import make_pdf

from ragkit.chunking import FixedTokenChunker, MarkdownSectionChunker
from ragkit.eval.chunk_size import EvidenceQuestion, TfidfRetriever, canon, evaluate_chunker, main
from ragkit.parsers import DocDefaults
from ragkit.pipeline import chunk_documents, diff_chunks, load_documents
from ragkit.tokenizers import RegexTokenizer

TOK = RegexTokenizer()


def test_load_shared_corpus():
    report = load_documents(SHARED_DOCS, root=SHARED_DOCS)
    assert report.summary() == {"documents": 24, "rejected": 0, "duplicates": 0, "needs_ocr": 0}
    assert all(d.source_uri.endswith(".md") and "/" not in d.source_uri for d in report.documents)


def test_load_rejects_missing_acl_bad_files_and_reports_ocr(tmp_path):
    (tmp_path / "no_acl.md").write_text("# Notes\n\nNo front matter here.")
    (tmp_path / "ok.md").write_text("---\nid: ok\ntenant: retail\nacl_groups: [all]\n---\n\n# OK\n\nFine.")
    (tmp_path / "copy.md").write_text("---\nid: copy\ntenant: retail\nacl_groups: [all]\n---\n\n# OK\n\nFine.")
    (tmp_path / "broken.jsonl").write_text("{not json")
    (tmp_path / "scan.pdf").write_bytes(make_pdf([[], []]))
    report = load_documents(tmp_path, root=tmp_path)
    reasons = {r.source_uri: r.reason for r in report.rejected}
    assert reasons["no_acl.md"] == "missing tenant or acl_groups"
    assert "broken.jsonl" in reasons and "scan.pdf" in reasons  # no ACL for the PDF either
    assert [d.id for d in report.documents] == ["copy"] or [d.id for d in report.documents] == ["ok"]
    assert len(report.duplicates) == 1 and report.duplicates[0].exact

    scanned = load_documents([tmp_path / "scan.pdf"], root=tmp_path, defaults=DocDefaults(tenant="retail", acl_groups=["all"]))
    assert len(scanned.needs_ocr) == 1 and scanned.documents[0].needs_ocr


def test_diff_chunks_after_an_edit(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("---\nid: p\ntenant: retail\nacl_groups: [all]\n---\n\n# P\n\nIntro.\n\n## A\n\nAlpha.\n\n## B\n\nBeta.\n")
    chunker = MarkdownSectionChunker(100, tokenizer=TOK)
    old = chunk_documents(load_documents(path, root=tmp_path).documents, chunker)
    path.write_text(path.read_text().replace("Beta.", "Beta, revised."))
    new = chunk_documents(load_documents(path, root=tmp_path).documents, chunker)
    diff = diff_chunks([c.id for c in old], new)
    assert len(diff.unchanged) == 2 and len(diff.added) == 1 and len(diff.removed) == 1
    assert diff.added[0].section_path[-1] == "B"


def test_tfidf_retriever_ranks_matching_text_first():
    r = TfidfRetriever(["VPN error 809 blocked port", "refund window thirty days", "PTO carryover ten days"])
    assert r.search("how many PTO days carry over", 2)[0] == 2
    assert r.search("zzz unknown", 2) == []


def test_canon_ignores_markup_and_case():
    assert canon("**Within 30 days**  of `x`") == "within 30 days of x"


def test_evaluate_chunker_measures_integrity_and_recall():
    docs = load_documents(SHARED_DOCS, root=SHARED_DOCS).documents
    qs = [EvidenceQuestion(id="q", doc_id="prod-retail-returns-api",
                           question="What does error RET-005 mean?",
                           evidence=["Code | HTTP | Meaning", "RET-005 | 409 | Return already exists"])]
    good = evaluate_chunker("section", MarkdownSectionChunker(300, tokenizer=TOK), docs, qs, k=3)
    tiny = evaluate_chunker("fixed-16", FixedTokenChunker(16, 0, tokenizer=TOK), docs, qs, k=3)
    assert good.span_integrity == 1.0 and good.recall_at_k == 1.0
    assert tiny.span_integrity == 0.0
    assert good.index_overhead < 0.05


def test_cli_runs_offline_and_emits_json(capsys):
    gold = SHARED_DOCS.parents[1] / "ragkit" / "eval_data" / "chunk_eval_gold.jsonl"
    assert main(["--docs", str(SHARED_DOCS), "--gold", str(gold), "--k", "3", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    names = [r["name"] for r in rows]
    assert "section-300" in names and "parent-child-800/96" in names
    assert all(0.0 <= r["recall_at_k"] <= 1.0 for r in rows)
