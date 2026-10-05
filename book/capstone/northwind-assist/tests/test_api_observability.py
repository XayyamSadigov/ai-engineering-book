# path: book/capstone/northwind-assist/tests/test_api_observability.py
"""SSE contract, feedback, extraction, UI, and tracing: one tree per request, lineage, redaction."""
from __future__ import annotations

from conftest import auth, chat, parse_sse

from northwind_assist import _paths
from northwind_assist.domain.events import EVENT_NAMES
from northwind_assist.observability.tracing import spans_of

PTO = "How many unused PTO days can I carry over into next year?"


def test_sse_contract_order_and_ids(client, container):
    with client.stream("POST", "/v1/chat", json={"message": PTO, "session_id": "sse1"},
                       headers=auth(container, "ana")) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-request-id"].startswith("req_")
        events = parse_sse(r.read().decode())
    names = [e["event"] for e in events]
    assert names[0] == "meta" and names[-1] == "done" and names.count("done") == 1
    assert set(names) <= set(EVENT_NAMES)
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)          # monotonically increasing ids
    meta, done = events[0]["data"], events[-1]["data"]
    for key in ("request_id", "model", "prompt_version", "index_version", "manifest", "degrade_level"):
        assert key in meta
    first_delta = names.index("delta")
    cited_before = {e["data"]["eid"] for e in events[:first_delta] if e["event"] == "citation"}
    assert "E1" in cited_before                       # a citation precedes the first sentence citing it
    assert done["status"] == "answered" and done["cost_usd"] > 0 and done["lineage"]["evidence_chunk_ids"]


def test_non_streaming_returns_same_payload(client, container):
    r = client.post("/v1/chat?stream=false", json={"message": PTO}, headers=auth(container, "ana"))
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered" and body["events"][0]["event"] == "meta"


def test_feedback_is_bound_to_the_requester(client, container):
    body = client.post("/v1/chat?stream=false", json={"message": PTO}, headers=auth(container, "ana")).json()
    rid = body["request_id"]
    assert client.post("/v1/feedback", json={"request_id": rid, "rating": "up"},
                       headers=auth(container, "lee")).status_code == 404
    assert client.post("/v1/feedback", json={"request_id": rid, "rating": "up"},
                       headers=auth(container, "ana")).json() == {"recorded": True}
    fb = [s for s in spans_of(container.tracer) if s.name == "feedback"]
    assert fb and fb[-1].attributes["response.id"] == rid


def test_extraction_route_returns_schema_valid_data(client, container):
    import json

    invoice = json.loads((_paths.SHARED_DATA / "invoices.jsonl").read_text().splitlines()[0])
    r = client.post("/v1/extract", json={"text": invoice["text"]}, headers=auth(container, "ana"))
    assert r.status_code == 200
    body = r.json()
    assert body["doc_type"] == "invoice" and body["route"] in ("accept", "human_review")
    assert body["data"] is None or body["data"]["invoice_number"]


def test_ui_is_served_with_csp(client):
    r = client.get("/")
    assert r.status_code == 200 and "Northwind Assist" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]


def test_one_trace_per_request_with_stage_spans_and_lineage(container):
    r = chat(container, "ana", PTO)
    spans = [s for s in spans_of(container.tracer) if getattr(s, "trace_id", None) == r.trace_id]
    names = {s.name for s in spans}
    for stage in ("request", "router.decide", "retrieval.pipeline", "retrieval.rerank", "guardrail.input",
                  "guardrail.context", "guardrail.output", "context.build", "llm.generate", "llm.complete"):
        assert stage in names, stage
    root = next(s for s in spans if s.name == "request")
    assert root.attributes["tenant.id"] == "retail" and root.attributes["cost.usd"] > 0
    assert "version.fingerprint" in root.attributes and "version.index_version" in root.attributes
    assert root.attributes["evidence.ids"]
    llm = next(s for s in spans if s.name == "llm.complete")
    assert llm.parent_span_id is not None and llm.attributes["llm.model"] == "general-2026-02"
    # streamed provider attempts carry the request's lineage as attributes (Chapter 31 export())
    assert llm.attributes["tenant.id"] == "retail"
    assert llm.attributes["prompt.id"] == "rag.grounded_answer.stream"


def test_agent_and_tool_spans_join_the_request_trace(container):
    r = chat(container, "ana", "create ticket: VPN drops every hour at store 0412")
    names = {s.name for s in spans_of(container.tracer) if getattr(s, "trace_id", None) == r.trace_id}
    assert {"agent.run", "agent.step", "agent.tool", "tool.execute", "guardrail.tool"} <= names


def test_pii_is_redacted_before_the_model_and_in_traces(container):
    r = chat(container, "ana", "My card 4111 1111 1111 1111 was charged twice, email ana.silva@northwind.example "
                               "about the expense policy refund rules")
    prompts = "\n".join(m.text for req in container.models.raw.requests for m in req.messages)
    assert "4111 1111 1111 1111" not in prompts and "ana.silva@northwind.example" not in prompts
    dumped = repr([s.attributes for s in spans_of(container.tracer)])
    assert "4111 1111 1111 1111" not in dumped and "ana.silva@northwind.example" not in dumped
    assert r.events[-1].event == "done"


def test_otel_backend_never_receives_pii():
    """RedactingTracer scrubs before OTelAITracer copies attributes into OpenTelemetry spans.
    The input guard already tokenizes user PII, so the probe writes raw values directly, the way a
    careless component would; the unwrapped tracer shows what the wrapper prevents."""
    from northwind_assist.observability.tracing import build_tracer

    raw = ("4111 1111 1111 1111", "ana.silva@northwind.example")

    def probe(tracer):
        with tracer.span("probe", **{"customer.email": raw[1]}) as span:
            span.set_attribute("note", f"card {raw[0]}")
        return repr([dict(s.attributes or {}) for s in tracer_exporter(tracer).get_finished_spans()])

    def tracer_exporter(tracer):
        return getattr(tracer, "inner", tracer).exporter

    wrapped = build_tracer({"TRACE_SINK": "otel", "OTEL_EXPORTER": "memory"})
    assert not any(v in probe(wrapped) for v in raw)
    unwrapped = build_tracer({"TRACE_SINK": "otel", "OTEL_EXPORTER": "memory"}).inner
    assert all(v in probe(unwrapped) for v in raw)
