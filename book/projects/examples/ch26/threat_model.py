# path: book/projects/examples/ch26/threat_model.py
"""A small, explicit threat-modeling data model for AI systems.

The model is deliberately plain: dataclasses, no I/O, one renderer. The point is
to make assets, principals, trust boundaries, entry points, and harmful effects
first-class objects that a team can review in a pull request, and to turn each
threat into a list of controls that Chapter 27 implements.

Categories follow STRIDE, with the AI-specific reading noted on each member.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Trust(str, Enum):
    TRUSTED = "trusted"        # part of the control plane; we wrote or verified it
    SEMI = "semi-trusted"      # authenticated but may be wrong or malicious
    UNTRUSTED = "untrusted"    # arbitrary text or bytes from outside the control plane


class Stride(str, Enum):
    SPOOFING = "Spoofing"                                # fake tool output, impersonated system notices
    TAMPERING = "Tampering"                              # injected instructions, poisoned memory or index
    REPUDIATION = "Repudiation"                          # actions without a trace binding them to a cause
    INFORMATION_DISCLOSURE = "Information disclosure"    # exfiltration, cross-tenant, secrets, prompt leak
    DENIAL_OF_SERVICE = "Denial of service"             # loops, oversized inputs, denial of wallet
    ELEVATION_OF_PRIVILEGE = "Elevation of privilege"   # confused deputy, tool abuse, excessive agency


class Level(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class Asset:
    name: str
    description: str
    classification: str  # e.g. "confidential", "internal", "public", "capability"


@dataclass(frozen=True)
class Principal:
    name: str
    kind: str   # "human", "service", "model", "third-party", "content"
    trust: Trust
    note: str = ""


@dataclass(frozen=True)
class Boundary:
    name: str
    source: str
    target: str
    crosses: str  # what data crosses and in which direction


@dataclass(frozen=True)
class EntryPoint:
    name: str
    boundary: str
    description: str


@dataclass(frozen=True)
class Threat:
    threat_id: str
    title: str
    category: Stride
    entry_point: str
    assets: tuple[str, ...]
    mechanism: str
    harmful_effect: str
    likelihood: Level
    impact: Level
    controls: tuple[str, ...]
    residual_risk: str = ""

    @property
    def risk(self) -> int:
        return int(self.likelihood) * int(self.impact)


@dataclass
class ThreatModel:
    system: str
    assets: list[Asset] = field(default_factory=list)
    principals: list[Principal] = field(default_factory=list)
    boundaries: list[Boundary] = field(default_factory=list)
    entry_points: list[EntryPoint] = field(default_factory=list)
    threats: list[Threat] = field(default_factory=list)

    # ---- integrity checks -------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of problems. Empty list means the model is internally consistent."""
        problems: list[str] = []
        asset_names = {a.name for a in self.assets}
        entry_names = {e.name for e in self.entry_points}
        boundary_names = {b.name for b in self.boundaries}
        ids: set[str] = set()
        for e in self.entry_points:
            if e.boundary not in boundary_names:
                problems.append(f"entry point {e.name!r} references unknown boundary {e.boundary!r}")
        for t in self.threats:
            if t.threat_id in ids:
                problems.append(f"duplicate threat id {t.threat_id!r}")
            ids.add(t.threat_id)
            if t.entry_point not in entry_names:
                problems.append(f"{t.threat_id}: unknown entry point {t.entry_point!r}")
            for a in t.assets:
                if a not in asset_names:
                    problems.append(f"{t.threat_id}: unknown asset {a!r}")
            if not isinstance(t.assets, tuple) or not isinstance(t.controls, tuple):
                problems.append(f"{t.threat_id}: assets and controls must be tuples (missing trailing comma?)")
            elif not t.controls or any(not isinstance(c, str) or not c.strip() for c in t.controls):
                problems.append(f"{t.threat_id}: controls must be a non-empty tuple of non-blank names")
            if not t.harmful_effect.strip():
                problems.append(f"{t.threat_id}: harmful effect is empty")
        return problems

    def coverage_gaps(self) -> list[str]:
        """Entry points with no threat and boundaries with no entry point. Not errors, but each
        one is a question for the next review: is it really safe, or just not thought about yet?"""
        threatened = {t.entry_point for t in self.threats}
        used = {e.boundary for e in self.entry_points}
        gaps = [f"entry point {e.name!r} has no threat" for e in self.entry_points if e.name not in threatened]
        gaps += [f"boundary {b.name!r} has no entry point" for b in self.boundaries if b.name not in used]
        return gaps

    def by_risk(self) -> list[Threat]:
        return sorted(self.threats, key=lambda t: (-t.risk, t.threat_id))

    def controls(self) -> list[str]:
        """Deduplicated control list, in first-seen order. This is the input to Chapter 27."""
        seen: dict[str, None] = {}
        for t in self.by_risk():
            for c in t.controls:
                seen.setdefault(c, None)
        return list(seen)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def render_threat_table(model: ThreatModel) -> str:
    """Threats ordered by risk, STRIDE-style columns."""
    rows = (
        (
            t.threat_id,
            t.category.value,
            t.entry_point,
            ", ".join(t.assets),
            t.mechanism,
            t.harmful_effect,
            f"{t.likelihood.name.title()} / {t.impact.name.title()} = {t.risk}",
            "; ".join(t.controls),
        )
        for t in model.by_risk()
    )
    return _table(
        ["ID", "STRIDE", "Entry point", "Assets", "Mechanism",
         "Harmful effect", "L / I = risk", "Primary controls"],
        rows,
    )


def render_markdown(model: ThreatModel) -> str:
    parts = [f"## Threat model: {model.system}", ""]
    parts += ["### Assets", "",
              _table(["Asset", "Classification", "Description"],
                     ((a.name, a.classification, a.description) for a in model.assets)), ""]
    parts += ["### Principals", "",
              _table(["Principal", "Kind", "Trust", "Note"],
                     ((p.name, p.kind, p.trust.value, p.note) for p in model.principals)), ""]
    parts += ["### Trust boundaries", "",
              _table(["Boundary", "From", "To", "What crosses"],
                     ((b.name, b.source, b.target, b.crosses) for b in model.boundaries)), ""]
    parts += ["### Entry points", "",
              _table(["Entry point", "Boundary", "Description"],
                     ((e.name, e.boundary, e.description) for e in model.entry_points)), ""]
    parts += ["### Threats", "", render_threat_table(model), ""]
    parts += ["### Control requirements", ""] + [f"- {c}" for c in model.controls()]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# Worked models for the Northwind running example
# --------------------------------------------------------------------------- #


def _common_assets() -> list[Asset]:
    return [
        Asset("hr-documents", "HR policies and confidential employee records", "confidential"),
        Asset("it-runbooks", "Operational runbooks including break-glass procedures", "confidential"),
        Asset("ticket-data", "Support tickets with customer names and contact details", "confidential"),
        Asset("system-prompt", "Policy text and tool descriptions given to the model", "internal"),
        Asset("provider-credentials", "API keys for the model provider and internal services", "secret"),
        Asset("tenant-isolation", "The guarantee that retail and logistics never see each other's data", "property"),
        Asset("spend-budget", "Model and tool spend per request and per day", "capability"),
    ]


def _common_principals() -> list[Principal]:
    return [
        Principal("employee", "human", Trust.SEMI, "Authenticated via SSO; may be careless or hostile"),
        Principal("support-staff", "human", Trust.SEMI, "Authenticated; broader ticket access than employees"),
        Principal("document-author", "content", Trust.UNTRUSTED, "Anyone who can get text into the indexed corpus"),
        Principal("ticket-submitter", "content", Trust.UNTRUSTED, "External customers writing ticket text"),
        Principal("llm", "model", Trust.UNTRUSTED, "Reads everything; its proposals carry no authority"),
        Principal("model-provider", "third-party", Trust.SEMI, "Sees prompts and outputs; contractual controls only"),
        Principal("tool-gateway", "service", Trust.TRUSTED, "Validates and executes tool calls; holds credentials"),
        Principal("policy-engine", "service", Trust.TRUSTED, "Allowlists, ACL filters, approval rules, validators"),
    ]


def northwind_rag_model() -> ThreatModel:
    """Project 3: the RAG knowledge assistant. Read-only, no outbound tools, markdown UI."""
    m = ThreatModel(system="Northwind Assist RAG knowledge assistant")
    m.assets = _common_assets()
    m.principals = _common_principals()
    m.boundaries = [
        Boundary("B1 user->app", "employee browser", "API", "question text, session identity"),
        Boundary("B2 corpus->context", "vector/lexical index", "prompt", "retrieved chunks with ACL metadata"),
        Boundary("B3 app->provider", "API", "model provider", "assembled prompt; returns generated text"),
        Boundary("B4 model->UI", "generated text", "employee browser", "markdown rendered as HTML, links, images"),
        Boundary("B5 app->telemetry", "API", "trace store", "prompts, chunks, outputs, latencies"),
        Boundary("B6 ingestion->index", "source systems", "index", "documents, chunks, embeddings, ACL tags"),
    ]
    m.entry_points = [
        EntryPoint("chat-message", "B1 user->app", "Free text from an authenticated employee"),
        EntryPoint("retrieved-chunk", "B2 corpus->context", "Top-k chunks placed into the prompt"),
        EntryPoint("rendered-answer", "B4 model->UI", "Model markdown interpreted by the browser"),
        EntryPoint("ingested-document", "B6 ingestion->index", "Any document accepted by the pipeline"),
        EntryPoint("answer-cache", "B2 corpus->context", "Cached answers keyed by question text"),
        EntryPoint("trace-record", "B5 app->telemetry", "Spans written by the application"),
    ]
    L, M, H = Level.LOW, Level.MEDIUM, Level.HIGH
    m.threats = [
        Threat("R1", "Injected instructions in a retrieved document", Stride.TAMPERING, "retrieved-chunk",
               ("hr-documents", "system-prompt"),
               "A chunk contains text addressed to the assistant",
               "Answer carries attacker-chosen content or discloses the system prompt",
               H, M, ("Label retrieved text as data in the prompt", "Keep the RAG path free of tools",
                      "Citation check: claims must map to chunks", "Output URL allowlist"),
               "Model may still produce misleading prose; it gains no authority"),
        Threat("R2", "Data read-out through a rendered image URL", Stride.INFORMATION_DISCLOSURE, "rendered-answer",
               ("hr-documents", "ticket-data"),
               "Model emits a markdown image whose query string holds record data; the browser fetches it",
               "Confidential text leaves to an outside host with no tool call",
               M, H, ("Strip or sandbox markdown images", "Egress allowlist on rendered hosts",
                      "Content Security Policy on the answer pane"),
               "Reduced to allowlisted hosts; a malicious allowlisted host remains a gap"),
        Threat("R3", "Cross-tenant retrieval leakage", Stride.INFORMATION_DISCLOSURE, "retrieved-chunk",
               ("tenant-isolation", "hr-documents"),
               "Retrieval returns chunks the current user is not entitled to read",
               "A retail user receives logistics or HR-only content",
               M, H, ("Apply ACL and tenant filters during retrieval, not after", "Namespaced indexes per tenant",
                      "Cross-tenant retrieval tests in CI"),
               "Depends on correct ACL tags at ingestion"),
        Threat("R4", "Cached-answer leakage across ACLs", Stride.INFORMATION_DISCLOSURE, "answer-cache",
               ("tenant-isolation", "hr-documents"),
               "A cached answer keyed only by question text is served to a less-privileged user",
               "One user sees an answer built from another user's documents",
               M, H, ("Include tenant and authorization context in the cache key", "Cache answers, not raw documents",
                      "Per-user cache partition tests"),
               "Cache key must track every ACL input"),
        Threat("R5", "Index poisoning through ingestion", Stride.TAMPERING, "ingested-document",
               ("hr-documents", "it-runbooks"),
               "A document with an injection payload is accepted into the corpus",
               "Every future retrieval of that chunk re-delivers the payload",
               M, M, ("Control and authenticate ingestion sources", "Scan and quarantine on ingest",
                      "Record provenance and version per chunk", "Monitor anomalous retrieval patterns"),
               "Scanning is best-effort; provenance limits blast radius"),
        Threat("R6", "Secrets copied into traces", Stride.INFORMATION_DISCLOSURE, "trace-record",
               ("provider-credentials", "hr-documents"),
               "Spans record full prompts, chunks, and outputs including sensitive values",
               "Secrets and personal data land in a broad-retention store",
               M, M, ("Redact before the trace sink", "Store hashes or ids where full text is not needed",
                      "Make debug capture an explicit, scoped capability"),
               "Redaction rules need coverage tests"),
        Threat("R7", "Denial of wallet via oversized inputs", Stride.DENIAL_OF_SERVICE, "chat-message",
               ("spend-budget",),
               "A user pastes a very large question or forces repeated expensive retrievals",
               "Token and retrieval spend spikes; latency budgets break",
               M, L, ("Input size caps", "Per-user and per-day spend limits", "Load shedding on breach"),
               "Caps must match legitimate long questions"),
    ]
    return m


def northwind_agent_model() -> ThreatModel:
    """Project 4: the tool-using support agent with a send_reply side effect."""
    m = ThreatModel(system="Northwind Assist support agent with send_reply")
    m.assets = _common_assets()
    m.principals = _common_principals()
    m.boundaries = [
        Boundary("B1 user->app", "support-staff browser", "API", "instruction text, session identity"),
        Boundary("B2 corpus->context", "ticket and KB index", "prompt", "retrieved tickets and articles"),
        Boundary("B3 model->gateway", "proposed tool call", "tool gateway", "tool name and arguments"),
        Boundary("B4 gateway->external", "tool gateway", "email and ticket systems", "validated side effects"),
        Boundary("B5 tool->context", "tool result", "prompt", "search output fed back to the model"),
        Boundary("B6 app->telemetry", "API", "trace store", "steps, arguments, approvals, results"),
    ]
    m.entry_points = [
        EntryPoint("agent-instruction", "B1 user->app", "Free text telling the agent what to do"),
        EntryPoint("retrieved-ticket", "B2 corpus->context", "Ticket text written by external customers"),
        EntryPoint("tool-result", "B5 tool->context", "Output of search_tickets and lookup_employee"),
        EntryPoint("proposed-call", "B3 model->gateway", "A tool call the model wants to execute"),
        EntryPoint("trace-record", "B6 app->telemetry", "Per-step records written by the harness"),
    ]
    L, M, H = Level.LOW, Level.MEDIUM, Level.HIGH
    m.threats = [
        Threat("A1", "Indirect injection drives an outbound send", Stride.ELEVATION_OF_PRIVILEGE, "retrieved-ticket",
               ("ticket-data", "hr-documents"),
               "Ticket text instructs the agent to send collected data to an outside address",
               "The agent proposes send_reply to an attacker destination",
               H, H, ("send_reply requires human approval bound to concrete arguments",
                      "Recipient allowlist enforced in the gateway", "Least-privilege tool set per task",
                      "Treat tool results as untrusted data"),
               "Approval UI must show real arguments; approver fatigue is a residual risk"),
        Threat("A2", "Confused deputy on lookup_employee", Stride.ELEVATION_OF_PRIVILEGE, "proposed-call",
               ("hr-documents", "tenant-isolation"),
               "Model is talked into querying records outside the requester's scope",
               "Agent reads employee data the requester may not see",
               M, H, ("Authorize tool arguments against the requesting user, not the model",
                      "Scope credentials to the user's tenant and role", "Deny by default on scope mismatch"),
               "Authorization must live in the gateway, not the prompt"),
        Threat("A3", "Fake tool output in a ticket", Stride.SPOOFING, "retrieved-ticket",
               ("ticket-data",),
               "Ticket embeds JSON shaped like a harness tool result suggesting a next action",
               "Model follows the forged result and proposes an unauthorized call",
               M, M, ("Separate real tool results from document text by provenance",
                      "Never parse model-visible text as control messages", "Validate every call in the gateway"),
               "Depends on strict provenance tagging"),
        Threat("A4", "Replayed or duplicated send", Stride.TAMPERING, "proposed-call",
               ("ticket-data",),
               "A retry after a timeout re-executes send_reply",
               "The same external message or action fires twice",
               M, M, ("Idempotency keys on side-effecting tools", "Harness records completed actions",
                      "Distinguish retryable from non-retryable failures"),
               "Idempotency must be enforced at the external API"),
        Threat("A5", "Runaway tool-call loop", Stride.DENIAL_OF_SERVICE, "agent-instruction",
               ("spend-budget",),
               "The agent keeps calling tools without making progress",
               "Cost and latency climb; the task never terminates",
               M, M, ("Step and token budgets", "Repeated-state detection", "Per-step and per-task cost caps"),
               "Budgets must be generous enough for real tasks"),
        Threat("A6", "Prompt and tool-description disclosure", Stride.INFORMATION_DISCLOSURE, "agent-instruction",
               ("system-prompt",),
               "User or document asks the agent to reveal its instructions and tool schemas",
               "Internal policy and tool surface leak, easing further attacks",
               M, L, ("Keep no secrets in the system prompt", "Treat prompt text as semi-public",
                      "Do not rely on secrecy of instructions for safety"),
               "Disclosure is low-impact only if nothing sensitive lives in the prompt"),
        Threat("A7", "Poisoned tool description in the supply chain", Stride.TAMPERING, "proposed-call",
               ("provider-credentials", "tenant-isolation"),
               "A tool or MCP server ships a description that biases the agent before any call",
               "Agent behavior changes though application code did not",
               L, H, ("Review tool and MCP descriptions as dependencies", "Pin and verify server versions",
                      "Give the agent the minimum tool set per task"),
               "Requires change review on non-code artifacts"),
    ]
    return m


if __name__ == "__main__":  # pragma: no cover
    for build in (northwind_rag_model, northwind_agent_model):
        model = build()
        assert not model.validate(), model.validate()
        print(render_markdown(model))
        print()
