# path: book/projects/p5-incident-agent/tests/test_dod.py
from __future__ import annotations

from incident_agent.agent import round_score
from incident_agent.domain.dod import check_report
from incident_agent.domain.report import claims, parse_sections

EVIDENCE = {"metric:trackline.p95_ms", "deploy:CHG-2026-0907", "inc-2026-02-tracking-latency",
            "it-incident-response-runbook"}
CATALOG = {"it-incident-response-runbook", "it-database-failover-runbook"}

GOOD = """# Incident report
## Summary
Trackline p95 is 3.6 s [metric:trackline.p95_ms]. A migration is the likely cause [deploy:CHG-2026-0907].
## Impact
- p95 rose fifteenfold [metric:trackline.p95_ms]
## Timeline
- 05:15 UTC migration applied [deploy:CHG-2026-0907]
## Likely cause
Same pattern as INC-2026-0217 [inc-2026-02-tracking-latency].
## Recommended runbook
Follow [it-incident-response-runbook].
## Next steps
1. Recreate the index concurrently.
"""


def codes(report: str, evidence=EVIDENCE) -> list[str]:
    return [p.code for p in check_report(report, evidence, CATALOG)]


def test_a_complete_grounded_report_passes():
    assert codes(GOOD) == []


def test_missing_section_is_reported():
    assert codes(GOOD.replace("## Timeline\n- 05:15 UTC migration applied [deploy:CHG-2026-0907]\n", "")) == \
        ["missing_section"]


def test_every_claim_needs_a_citation_but_next_steps_do_not():
    report = GOOD.replace("- p95 rose fifteenfold [metric:trackline.p95_ms]", "- p95 rose fifteenfold")
    assert codes(report) == ["uncited_claim"]
    assert codes(GOOD.replace("1. Recreate the index concurrently.", "1. Recreate it. 2. Watch it.")) == []


def test_citations_must_point_at_observed_sources():
    report = GOOD.replace("[inc-2026-02-tracking-latency]", "[inc-2025-11-pos-outage]")
    assert codes(report) == ["unknown_citation"]


def test_an_invented_runbook_is_caught():
    report = GOOD.replace("Follow [it-incident-response-runbook].", "Follow [it-db-index-rebuild-runbook].")
    assert codes(report) == ["unknown_citation", "runbook_not_in_catalog", "runbook_missing"]


def test_a_real_runbook_that_was_never_retrieved_is_caught():
    report = GOOD.replace("Follow [it-incident-response-runbook].", "Follow [it-database-failover-runbook].")
    assert codes(report) == ["unknown_citation", "runbook_not_retrieved"]


def test_claims_split_bullets_and_sentences_and_skip_tables():
    body = "First fact [a]. Second fact v3.6.2 [b].\n- bullet one\n| col | col |\n1. numbered"
    assert claims(body) == ["First fact [a].", "Second fact v3.6.2 [b].", "bullet one", "numbered"]
    assert set(parse_sections(GOOD)) == {"summary", "impact", "timeline", "likely cause", "recommended runbook",
                                        "next steps"}


def test_round_score_orders_dod_failures_below_any_pass():
    assert round_score(1, None) < round_score(0, 0.0)
    assert round_score(4, None) < round_score(2, None)
    assert round_score(0, 0.75) < round_score(0, 1.0)
