---
id: it-onboarding-accounts-runbook
title: New Hire Account Provisioning Runbook
version: "2.2"
updated_at: 2025-12-01
owner: IT Service Desk
tenant: shared
acl_groups: ["it-oncall", "hr"]
tags: [it, onboarding, accounts, runbook, identity, hr]
---

# New Hire Account Provisioning Runbook

Audience: IT Service Desk agents and People Operations partners. This runbook describes how accounts
are created for new employees and contractors, and what the new hire should have by Day 1.

## Trigger

Account creation is triggered automatically when People Operations sets a new hire's record in
**PeopleHub** to status `Hired - pending start` with a confirmed start date. PeopleHub sends the event
to the identity system (**Northwind ID**) nightly at 22:00. Manual creation is used only for contractors
and for emergency starts (see below).

## Timeline

| When                              | What happens                                                         |
|-----------------------------------|----------------------------------------------------------------------|
| Start date minus 10 business days | PeopleHub record complete; manager completes the `New hire checklist` |
| Start date minus **3 business days** | Northwind ID account, mailbox, and chat created automatically; laptop shipped or prepared |
| Start date minus 1 business day   | Welcome email with first-login instructions sent to the personal address on file |
| Day 1                             | New hire sets password, enrols Northwind Authenticator, receives laptop |
| Day 1 to Day 5                    | Manager requests role-specific access via Beacon                      |

## What is created automatically

- **Northwind ID** account in the format `firstname.lastname` (a numeric suffix is added for duplicates).
- Mailbox `firstname.lastname@northwind.example` and chat account.
- Membership of the baseline groups: `all`, the tenant group (`retail` or `logistics` or `shared`), and
  the department group from PeopleHub.
- Access to the intranet, PeopleHub self-service, Beacon, Ledgerly, and TripDesk.
- A Beacon onboarding ticket `ONB-YYYY-NNNN` assigned to the manager with the Day 1 checklist.

## What requires a request

Role-specific access is **not** granted automatically. The manager requests it in Beacon under
`IT / Access request`, one request per system:

| System or group                 | Approver                     | Typical turnaround   |
|---------------------------------|------------------------------|----------------------|
| `managers` group                | Department Director          | 1 business day       |
| `hr` group (PeopleHub admin)    | Head of People Operations    | 1 business day       |
| `it-oncall` and `NW-Admin` VPN  | Platform Engineering lead    | 2 business days      |
| `finance` (Ledgerly approver)   | Finance Operations lead      | 1 business day       |
| Source code repositories        | Engineering manager          | same day             |
| Production database read access | Security Office              | 3 business days      |

All access follows the Access Control Policy: least privilege, time-limited where possible, and reviewed
quarterly.

## Day 1 steps for the new hire

1. Open the welcome email and follow the link to `id.northwind.example/activate`. The link is valid for
   **72 hours**; if it expired, the manager requests a new one in the onboarding ticket.
2. Set a password that meets the Password Standard (minimum 12 characters; see the Password Reset
   Runbook).
3. Install Northwind Authenticator on a phone and scan the QR code shown after the first login.
4. Sign in on the laptop. Apps and cloud folders sync in 30 to 60 minutes.
5. Complete the mandatory trainings in PeopleHub within 30 days: Security Awareness, Code of Conduct,
   Data Classification.

## Contractors

Contractor accounts are created manually by the Service Desk from a Beacon request by the sponsoring
manager, with a mandatory end date no more than **6 months** ahead (renewable). Contractors receive the
`contractors` group instead of `all`, no PeopleHub self-service, and only the systems named in the
request. Accounts are disabled automatically at 23:59 on the end date.

## Emergency starts

If a start date is moved forward with less than 3 business days' notice, People Operations opens a
Beacon ticket flagged `Expedite onboarding`. The Service Desk creates the account within 4 business
hours. Laptop delivery may still take up to 5 business days; a loaner is provided meanwhile.

## Troubleshooting

- **No welcome email**: check the personal email address in PeopleHub; resend from Northwind ID admin.
- **Activation link expired**: regenerate from the onboarding ticket. Each link is single use.
- **Wrong name or duplicate**: fix in PeopleHub first; Northwind ID syncs the change overnight.
- **Missing baseline group**: usually the department field in PeopleHub was empty at creation time.

## Related documents

- Password Reset Runbook (`it-password-reset-runbook`)
- Access Control Policy (`sec-access-control-policy`)
- Laptop Replacement Runbook (`it-laptop-replacement-runbook`)
