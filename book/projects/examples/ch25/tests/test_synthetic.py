# path: book/projects/examples/ch25/tests/test_synthetic.py
from __future__ import annotations

from aie_core import FakeLLM
from shared_data import load_docs, load_retrieval_gold

from taskevals.synthetic import (
    Candidate,
    SourceDoc,
    SyntheticGenerator,
    SyntheticItem,
    bias_report,
    validate_candidates,
)

DOCS = {d.id: d.text for d in load_docs()}

GENERATED = {"items": [
    # good, but copies the document's words: easy for lexical retrieval
    {"question": "What was the previous limit under Policy 2.2?", "answer": "5 days",
     "answer_quote": "The previous limit under Policy 2.2 was 5 days."},
    # good paraphrase
    {"question": "Do bank holidays come out of my vacation allowance?", "answer": "No, they are separate",
     "answer_quote": "Public holidays observed by Northwind are separate and are not deducted from PTO."},
    # near-duplicate of the first
    {"question": "What was the previous limit under the Policy 2.2?", "answer": "5 days",
     "answer_quote": "The previous limit under Policy 2.2 was 5 days."},
    # unanswerable: the quote is invented
    {"question": "Can I sell unused PTO back to the company?", "answer": "Yes, at 50% of salary",
     "answer_quote": "Unused PTO may be sold back at half pay."},
    # answer number not backed by the quote
    {"question": "How do part-timers accrue time off?", "answer": "12 days a year",
     "answer_quote": "Part-time employees accrue PTO pro rata to their contracted hours."},
]}


def _generator() -> tuple[SyntheticGenerator, FakeLLM]:
    llm = FakeLLM(handler=lambda req: GENERATED)
    return SyntheticGenerator(llm, per_doc=5), llm


def test_generation_uses_structured_output_and_delimits_the_document() -> None:
    gen, llm = _generator()
    cands = gen.generate([SourceDoc(id="hr-pto-policy", text=DOCS["hr-pto-policy"])])
    assert len(cands) == 5 and all(c.doc_id == "hr-pto-policy" for c in cands)
    prompt = llm.requests[0].messages[-1].text
    assert '<document id="hr-pto-policy">' in prompt and llm.requests[0].temperature > 0


def test_filters_remove_duplicates_and_unanswerable_items_and_tag_difficulty() -> None:
    gen, _ = _generator()
    cases, report = validate_candidates(gen.generate([SourceDoc(id="hr-pto-policy", text=DOCS["hr-pto-policy"])]),
                                        DOCS)
    assert report.generated == 5 and report.kept == 2 and report.yield_rate == 0.4
    assert set(report.rejected) == {"duplicate_in_batch", "quote_not_in_source", "answer_number_not_in_quote"}
    tags = {c.id: [t for t in c.tags if t.startswith("difficulty:")][0] for c in cases}
    assert sorted(tags.values()) == ["difficulty:easy-lexical", "difficulty:hard-paraphrase"]
    assert all("origin:synthetic" in c.tags and c.metadata["group"] == "hr-pto-policy" for c in cases)


def test_synthetic_items_that_duplicate_the_existing_dataset_are_rejected() -> None:
    gold_q = [q.question for q in load_retrieval_gold()]
    cand = Candidate(doc_id="hr-pto-policy", item=SyntheticItem(
        question="How many unused PTO days can I carry over into next year, and by when must I use them?",
        answer="10 days", answer_quote="employees may carry over up to 10 unused PTO days into the next calendar year"))
    cases, report = validate_candidates([cand], DOCS, existing_questions=gold_q)
    assert not cases and "duplicate_of_existing_dataset" in report.rejected


def test_bias_report_shows_synthetic_questions_echo_source_wording() -> None:
    echo = [Candidate(doc_id=d, item=SyntheticItem(question=q, answer="x", answer_quote=quote))
            for d, q, quote in [
                ("hr-pto-policy", "What was the previous limit under Policy 2.2?",
                 "The previous limit under Policy 2.2 was 5 days."),
                ("hr-pto-policy", "Do part-time employees accrue PTO pro rata to their contracted hours?",
                 "Part-time employees accrue PTO pro rata to their contracted hours."),
            ]]
    synthetic, _ = validate_candidates(echo, DOCS)
    reference = [(q.question, q.required_doc_ids[0]) for q in load_retrieval_gold() if q.required_doc_ids]
    rep = bias_report(synthetic, reference, DOCS)
    assert rep.synthetic_mean_overlap > rep.reference_mean_overlap
    assert rep.overlap_gap > 0.1 and rep.docs_covered == 1
