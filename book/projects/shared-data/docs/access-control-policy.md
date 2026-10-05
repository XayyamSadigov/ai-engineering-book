---
id: sec-access-control-policy
title: Access Control Policy
version: "3.1"
updated_at: 2026-02-02
owner: Security Office
tenant: shared
acl_groups: ["security", "it-oncall", "managers"]
tags: [security, access, iam, policy, privileged-access, reviews]
---

# Access Control Policy

Classification: Confidential. This policy sets how access to Northwind systems and data is granted,
used, reviewed, and removed. It applies to employees, contractors, service accounts, and automated
agents.

## Principles

1. **Least privilege**: grant the minimum access needed for the role, for the shortest time needed.
2. **Group-based**: access is granted through groups managed in Northwind ID, never to individuals
   directly, except for break-glass accounts.
3. **Tenant isolation**: data belonging to the `retail` or `logistics` tenant is accessible only to
   groups scoped to that tenant or to explicitly cross-tenant `shared` groups.
4. **Auditable**: every grant, use of privileged access, and removal is logged and retained for 5 years.

## Standard groups

| Group          | Grants                                                       | Approver                       |
|----------------|--------------------------------------------------------------|--------------------------------|
| `all`          | Intranet, policies, Beacon, self-service tools               | Automatic on hire              |
| `retail` / `logistics` | Tenant-scoped product documentation and tools        | Automatic from PeopleHub       |
| `managers`     | Team PeopleHub data, approvals, incident reports             | Department Director            |
| `hr`           | PeopleHub admin, employee records (Restricted)               | Head of People Operations      |
| `finance`      | Ledgerly approvals, invoice data                             | Finance Operations lead        |
| `it-oncall`    | Production runbooks, Siren schedules, `NW-Admin` VPN profile  | Platform Engineering lead      |
| `security`     | Security tooling, audit logs, this policy's admin sections   | Head of Security               |
| `contractors`  | Replaces `all` for non-employees; no self-service HR         | Sponsoring manager             |

Requests go through Beacon `IT / Access request` and must name the group, the business reason, and an
end date where applicable. Approvals are recorded in the ticket; verbal approvals are not valid.

## Privileged access

- Privileged access (production databases, cloud administration, identity administration) is granted
  **just-in-time** through the privileged access tool for a maximum of **4 hours** per activation, with
  a ticket reference. Standing privileged access is not permitted except for break-glass accounts.
- Privileged sessions are recorded. Session recordings are retained for 1 year.
- Privileged accounts use a hardware security key and a 16-character minimum password rotated every
  180 days (see the Password Reset Runbook).

## Break-glass accounts

Two break-glass accounts per tenant exist for use when identity systems are down. Credentials are
sealed in the secrets manager with a dual-control policy; opening them pages the Security on-call and
creates a Security incident automatically. Every use is reviewed within 2 business days.

## Access reviews

- **Quarterly** for `managers`, `hr`, `finance`, `it-oncall`, `security`, and all privileged roles.
- **Annually** for tenant groups and application-level roles.
- Reviews are run from Northwind ID; approvers have 10 business days to confirm or revoke. Unreviewed
  access is **revoked automatically** on day 11.

## Joiners, movers, leavers

- **Joiners**: baseline groups are created 3 business days before start (see the Onboarding Accounts
  Runbook).
- **Movers**: when PeopleHub records a department or tenant change, role-specific groups from the
  previous role are removed after a **14-day** grace period unless the new manager re-requests them.
- **Leavers**: all access is disabled **within 24 hours** of the termination time recorded in PeopleHub;
  for involuntary terminations, at the time of notification. Mailboxes are forwarded to the manager for
  30 days and then deleted.

## Service accounts and automated agents

- Each service account has a named human owner and a documented purpose in the service catalogue.
- Secrets are stored in the secrets manager, never in code or documents, and rotated at least every
  180 days or immediately on suspected exposure.
- Automated agents (including Northwind Assist tools) act with a scoped identity and may never hold
  broader access than the human on whose behalf they act. Tool calls that change state (creating tickets,
  sending replies) require an explicit human approval step recorded in the audit log.

## Exceptions

Exceptions to this policy require written approval from the Head of Security, have an end date no more
than 12 months out, and are reviewed quarterly. Exceptions are tracked in the Security exception register.

## Related documents

- Data Classification Policy (`sec-data-classification-policy`)
- Password Reset Runbook (`it-password-reset-runbook`)
- Onboarding Accounts Runbook (`it-onboarding-accounts-runbook`)
