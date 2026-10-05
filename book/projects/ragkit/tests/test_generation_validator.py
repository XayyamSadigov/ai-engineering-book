# path: book/projects/ragkit/tests/test_generation_validator.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from generation_fixtures import pto_carryover, pto_hits, hit

from ragkit.generation import (
    CitationValidator,
    Claim,
    EvidencePacker,
    GroundedAnswer,
    JudgeVerdict,
    LLMGroundednessJudge,
)

CARRY_10 = "Employees may carry over up to 10 unused PTO days into the next calendar year"
DEADLINE = "Carried-over days must be used by 31 March of the following year"


def pto_only():
    return EvidencePacker().pack([hit(pto_carryover(), 0.9, 1)])


def test_hallucinated_citation_is_detected_and_removed():
    packed = pto_only()
    answer = GroundedAnswer(
        status="answered",
        answer=f"{CARRY_10} [E1]. {DEADLINE} [E9].",
        claims=[Claim(text=CARRY_10, citations=["E1"]), Claim(text=DEADLINE, citations=["E9"])],
        confidence="high",
    )
    report = CitationValidator().validate(answer, packed)
    assert not report.ok
    assert report.codes().count("unknown_citation") == 2  # inline marker and claim citation
    assert report.dropped_claims == [1]
    assert report.repaired.status == "partial"
    assert "[E9]" not in report.repaired.answer and "[E1]" in report.repaired.answer
    assert report.repaired.confidence == "low"


def test_claim_without_valid_citation_is_dropped():
    packed = pto_only()
    answer = GroundedAnswer(status="answered", answer=f"{CARRY_10} [E1].",
                            claims=[Claim(text=CARRY_10, citations=["E1"]), Claim(text=DEADLINE, citations=[])])
    report = CitationValidator().validate(answer, packed)
    assert "uncited_claim" in report.codes()
    assert [c.text for c in report.repaired.claims] == [CARRY_10]


def test_invented_number_is_unsupported_even_with_a_valid_citation():
    packed = pto_only()
    claim = "Employees may carry over up to 15 unused PTO days into the next calendar year"
    answer = GroundedAnswer(status="answered", answer=f"{claim} [E1].", claims=[Claim(text=claim, citations=["E1"])])
    report = CitationValidator().validate(answer, packed)
    [issue] = [i for i in report.issues if i.code == "unsupported_claim"]
    assert "15" in issue.detail
    assert report.repaired.status == "insufficient_evidence"
    assert report.repaired.claims == []


def test_quote_must_be_verbatim():
    packed = pto_only()
    good = Claim(text=CARRY_10, citations=["E1"], quote="may carry over up to 10 unused PTO days")
    bad = Claim(text=DEADLINE, citations=["E1"], quote="must be used within 90 days")
    answer = GroundedAnswer(status="answered", answer="x [E1].", claims=[good, bad])
    report = CitationValidator().validate(answer, packed)
    assert [i.claim_index for i in report.issues if i.code == "quote_not_found"] == [1]
    assert report.repaired.claims == [good]


def test_judge_hook_drops_claims_it_marks_unsupported():
    packed = pto_only()

    def judge(answer: GroundedAnswer, packed_) -> JudgeVerdict:
        return JudgeVerdict(score=2, unsupported_claims=[DEADLINE])

    answer = GroundedAnswer(status="answered", answer=f"{CARRY_10} [E1]. {DEADLINE} [E1].",
                            claims=[Claim(text=CARRY_10, citations=["E1"]), Claim(text=DEADLINE, citations=["E1"])])
    report = CitationValidator(judge=judge).validate(answer, packed)
    assert "judge_unsupported" in report.codes()
    assert report.dropped_claims == [1]
    assert report.judge is not None and report.judge.score == 2


def test_llm_judge_uses_recipe_contract():
    llm = FakeLLM(responses=[{"score": 3, "unsupported_claims": []}])
    packed = pto_only()
    answer = GroundedAnswer(status="answered", answer=f"{CARRY_10} [E1].", claims=[Claim(text=CARRY_10, citations=["E1"])])
    report = CitationValidator(judge=LLMGroundednessJudge(llm)).validate(answer, packed)
    assert report.ok and report.judge.score == 3
    req = llm.last_request
    assert "Rubric" in req.messages[0].text and req.metadata["prompt.id"] == "judge.groundedness"
    assert 'source="E1"' in req.messages[1].text


def test_citations_resolve_to_stable_source_metadata():
    packed = pto_only()
    answer = GroundedAnswer(status="answered", answer=f"{CARRY_10} [E1].", claims=[Claim(text=CARRY_10, citations=["e1"])])
    report = CitationValidator().validate(answer, packed)
    assert report.ok
    [cit] = report.citations
    assert cit.eid == "E1" and cit.doc_id == "hr-pto-policy" and cit.version == "3.0"
    assert cit.source_uri == "docs/pto-policy.md"
    assert cit.chunk_ids == [pto_carryover().id]
    assert cit.section.startswith("3. Carryover")


def test_answer_from_stale_source_only_is_flagged_and_dropped():
    packed = EvidencePacker().pack(pto_hits())
    [note] = packed.conflicts
    claim = "You may carry over up to 5 unused days into the next calendar year"
    answer = GroundedAnswer(status="answered", answer=f"{claim} [{note.older}].",
                            claims=[Claim(text=claim, citations=[note.older])])
    report = CitationValidator().validate(answer, packed)
    assert "stale_source_preferred" in report.codes()
    assert report.repaired.status == "insufficient_evidence"
    assert any(note.newer in m for m in report.repaired.missing_info)


def test_citing_both_sides_without_conflict_status_is_a_warning():
    packed = EvidencePacker().pack(pto_hits())
    [note] = packed.conflicts
    answer = GroundedAnswer(
        status="answered",
        answer=f"{CARRY_10} [{note.newer}]. The HR FAQ version 1.4 says up to 5 unused days [{note.older}].",
        claims=[Claim(text=CARRY_10, citations=[note.newer]),
                Claim(text="The HR FAQ version 1.4 says up to 5 unused days", citations=[note.older])],
    )
    report = CitationValidator().validate(answer, packed)
    assert report.ok  # warnings only
    assert "conflict_unreported" in report.codes()


def test_uncited_factual_sentence_triggers_rebuild_from_claims():
    packed = pto_only()
    answer = GroundedAnswer(status="answered",
                            answer=f"{CARRY_10} [E1]. Managers usually approve carryover requests within two days.",
                            claims=[Claim(text=CARRY_10, citations=["E1"])])
    report = CitationValidator().validate(answer, packed)
    assert "uncited_sentence" in report.codes()
    assert "Managers" not in report.repaired.answer
