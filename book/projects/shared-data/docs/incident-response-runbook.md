---
id: it-incident-response-runbook
title: Production Incident Response Runbook
version: "4.1"
updated_at: 2026-03-12
owner: Platform Engineering
tenant: shared
acl_groups: ["it-oncall"]
tags: [it, incident, runbook, on-call, sev1, postmortem]
---

# Production Incident Response Runbook

This runbook is for on-call engineers and incident commanders. It covers every production service in
both tenants. Paging is done through **Siren**; incidents are tracked in Beacon with the prefix
`INC-YYYY-NNNN`.

## Severity levels

| Severity | Definition                                                                    | Response target | Update cadence |
|----------|-------------------------------------------------------------------------------|-----------------|----------------|
| **SEV1** | Customer-facing outage or data loss affecting a whole tenant or more than 100 stores or more than 20% of shipments | **15 minutes** | every **30 minutes** |
| **SEV2** | Significant degradation; a core workflow is impaired for many users; a workaround exists | 30 minutes | every 60 minutes |
| **SEV3** | Minor degradation or a single site affected; no data at risk                 | 4 business hours | daily |
| **SEV4** | Cosmetic or low-impact; fix in normal sprint work                             | next business day | at closure |

If unsure between two levels, pick the higher one. Severity can be lowered later by the incident
commander.

## Roles

- **Incident Commander (IC)**: owns the incident, decides severity, coordinates responders, and is the
  only person who changes incident status. The primary on-call engineer is IC by default until a
  dedicated IC takes over.
- **Communications Lead**: posts status updates to the `#incidents` channel and the status page,
  and answers stakeholder questions so that responders are not interrupted.
- **Responders**: subject-matter engineers working the problem.
- **Scribe**: keeps the timeline in the Beacon incident record. For SEV1 this role is mandatory.

## Procedure

1. **Acknowledge** the Siren page within the response target. Unacknowledged SEV1 pages escalate to the
   secondary on-call after 10 minutes and to the engineering manager after 20.
2. **Open the incident** in Beacon (`New incident`), set severity, and start the bridge call linked from
   the incident record. Post the first update in `#incidents` within 15 minutes for SEV1/SEV2.
3. **Stabilise first.** Prefer rollback, feature-flag disable, or traffic shedding over root-causing
   under pressure. Any production change during an incident is recorded in the timeline with who, what,
   and when.
4. **Communicate** on the cadence above, even if the update is "no change". Use the template:
   `Impact / Current status / Next update at HH:MM`.
5. **Resolve** when customer impact has ended for at least 30 minutes and monitoring is green. Set the
   status to `Resolved`, post the final update, and schedule the postmortem.
6. **Monitor** for 24 hours. Recurrence within 24 hours reopens the same incident.

## Postmortem

- Required for every SEV1 and SEV2, and for any SEV3 that recurred.
- Draft due within **5 business days** of resolution; review meeting within 10 business days.
- Blameless. Focus on contributing factors, detection gaps, and action items with owners and due
  dates. Action items are tracked as Beacon tasks linked to the incident.
- Published to the `Incident reports` space with ACL `it-oncall` and `managers`.

## Communication templates

**Initial (SEV1)**: "We are investigating an issue affecting <service> in the <tenant> tenant. Impact:
<impact>. Next update at <time>."

**Resolved**: "The issue affecting <service> was resolved at <time>. Duration: <duration>. Root cause and
follow-up actions will be published in the postmortem by <date>."

## Tooling checklist

- Dashboards: `grafana/northwind-overview`, per-service boards linked from each runbook.
- Logs: `logs.northwind.internal` (14-day retention for prod).
- Feature flags: `flags.northwind.internal`; emergency disable requires IC confirmation in the timeline.
- Status page: `status.northwind.example`, updated by the Communications Lead only.

## Related documents

- Database Failover Runbook (`it-database-failover-runbook`)
- Incident report INC-2025-1142 (`inc-2025-11-pos-outage`)
- Incident report INC-2026-0217 (`inc-2026-02-tracking-latency`)
