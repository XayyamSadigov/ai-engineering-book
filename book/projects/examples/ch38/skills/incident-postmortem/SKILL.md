---
name: incident-postmortem
description: Write a blameless incident postmortem with timeline, impact, root cause, and follow-up actions from incident tickets and metrics.
version: 0.4.1
triggers: [postmortem, incident, outage, rca]
owner: sre
---
# Postmortem procedure

1. Collect the incident ticket, alert timeline, and the relevant runbook.
2. Fill `template.md` section by section. Every timeline entry needs a source id in brackets.
3. Root cause: state the triggering change and the missing safeguard separately.
4. Follow-ups: each one gets an owner team and a ticket; do not create tickets without approval.
