# path: book/projects/ragkit/tests/test_generation_stream.py
from __future__ import annotations

from aie_core.llm.providers import FakeLLM
from aie_core.observability import InMemoryTracer
from generation_fixtures import hit, pto_carryover, pto_hits

from ragkit.generation import EvidencePacker, GroundedGenerator, GroundedQA, GroundedStreamer, SentenceBuffer
from ragkit.generation.stream import STREAM_SYSTEM_PROMPT

CARRY = "Employees may carry over up to 10 unused PTO days into the next calendar year [E1]."
DEADLINE = "Carried-over days must be used by 31 March of the following year [E1]."


def pto_only():
    return EvidencePacker().pack([hit(pto_carryover(), 0.9, 1)])


def test_sentence_buffer_waits_for_markers_split_across_deltas():
    buf = SentenceBuffer()
    assert buf.feed("Carry over up to 10 days [E") == []
    assert buf.feed("1]. Requests") == ["Carry over up to 10 days [E1]."]
    assert buf.feed(" need 14 days. [E2] Done") == ["Requests need 14 days. [E2]"]
    assert buf.flush() == ["Done"]


def test_decimal_points_do_not_end_sentences():
    buf = SentenceBuffer()
    assert buf.feed("Accrual is about 1.83 days per month [E1]. Next") == ["Accrual is about 1.83 days per month [E1]."]


def test_valid_sentences_stream_with_citation_events_first():
    streamer = GroundedStreamer(FakeLLM())
    events = list(streamer.stream_text([CARRY[:20], CARRY[20:], " ", DEADLINE], pto_only()))
    types = [e.type for e in events]
    assert types == ["status", "citation", "text", "text", "done"]
    assert events[1].citation.doc_id == "hr-pto-policy"
    done = events[-1]
    assert done.status == "answered" and len(done.answer.claims) == 2


def test_unknown_citation_and_unsupported_sentences_are_withheld():
    bad_id = "Carried-over days must be used by 31 March of the following year [E7]."
    bad_number = "Employees may carry over up to 20 unused PTO days into the next calendar year [E1]."
    uncited = "Most employees carry over their full allowance every single year."
    events = list(GroundedStreamer(FakeLLM()).stream_text([f"{CARRY} {bad_id} {bad_number} {uncited}"], pto_only()))
    withheld = [e for e in events if e.type == "withheld"]
    assert [e.issue.code for e in withheld] == ["unknown_citation", "unsupported_claim", "uncited_sentence"]
    shown = "".join(e.text for e in events if e.type == "text")
    assert "E7" not in shown and "20 unused" not in shown and "Most employees" not in shown
    assert events[-1].status == "partial"


def test_insufficient_evidence_token_emits_no_text():
    events = list(GroundedStreamer(FakeLLM()).stream_text(
        ["INSUFFICIENT_EVIDENCE: the documents do not ", "cover sabbatical leave."], pto_only()))
    assert [e.type for e in events] == ["status", "done"]
    assert events[-1].answer.status == "insufficient_evidence"
    assert "sabbatical" in events[-1].answer.missing_info[0]


def test_conflict_prefix_sets_status():
    packed = EvidencePacker().pack(pto_hits())
    [note] = packed.conflicts
    text = (f"CONFLICT: Employees may carry over up to 10 unused PTO days [{note.newer}]. "
            f"The HR FAQ version 1.4 says up to 5 unused days [{note.older}].")
    events = list(GroundedStreamer(FakeLLM()).stream_text([text], packed))
    assert events[0].status == "conflict" and events[-1].status == "conflict"
    assert [e.eids for e in events if e.type == "citation"] == [[note.newer], [note.older]]


def test_end_to_end_stream_from_fake_model_with_small_chunks():
    llm = FakeLLM(responses=[f"{CARRY} {DEADLINE}"], chunk_size=5)
    events = list(GroundedStreamer(llm).stream("How much PTO carries over?", pto_only()))
    assert llm.last_request.messages[0].text == STREAM_SYSTEM_PROMPT
    assert llm.last_request.metadata["prompt.id"] == "rag.grounded_answer.stream"
    assert [e.type for e in events].count("text") == 2
    assert events[-1].answer.status == "answered"


def test_empty_evidence_streams_abstention_without_model_call():
    llm = FakeLLM(responses=[])
    events = list(GroundedStreamer(llm).stream("anything", EvidencePacker().pack([])))
    assert llm.requests == [] and events[-1].answer.status == "insufficient_evidence"


def test_pipeline_emits_trace_spans():
    tracer = InMemoryTracer()
    llm = FakeLLM(responses=[{"status": "insufficient_evidence", "answer": "n/a", "claims": []}])
    GroundedQA(GroundedGenerator(llm, tracer=tracer), tracer=tracer).answer("q", pto_hits(), request_id="r1")
    names = [s.name for s in tracer.spans]
    assert "rag.generate" in names and "rag.answer" in names
    answer_span = next(s for s in tracer.spans if s.name == "rag.answer")
    assert answer_span.attributes["decision"] == "abstain" and answer_span.attributes["request_id"] == "r1"
