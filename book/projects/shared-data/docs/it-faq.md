---
id: it-faq
title: IT Service Desk FAQ
version: "6.2"
updated_at: 2026-05-05
owner: IT Service Desk
tenant: shared
acl_groups: ["all"]
tags: [it, faq, service-desk, support]
---

# IT Service Desk FAQ

Short answers to the questions the Service Desk receives most often. Where a runbook exists, the FAQ
links to it rather than repeating it. Open tickets in **Beacon**; call the Service Desk only for
Priority 1 issues (you cannot work at all, or a store or depot cannot operate).

## Accounts and sign-in

**I forgot my password.** Use `id.northwind.example/reset` with your Authenticator. If you lost your
phone too, call the Service Desk; expect identity verification by callback. See the Password Reset
Runbook.

**My account is locked.** Ten failed attempts lock it for 30 minutes. It unlocks by itself. If it keeps
locking, an old password is probably saved on a phone or in a script.

**I got a new phone. How do I move Northwind Authenticator?** Before wiping the old phone, open
`id.northwind.example/security` and add the new device. If the old phone is already gone, open a Beacon
ticket under `IT / Identity > MFA reset`; your manager will be asked to confirm.

**How long does it take to get access to a new system?** Most group requests are approved within 1 to 2
business days; production database access takes 3 business days because Security reviews it.

## Devices

**When do I get a new laptop?** Every 36 months. Beacon emails you 3 months before the date.

**My laptop is broken and I cannot work.** Open `IT / Hardware > Repair` and tick `Needs loaner`. A
loaner is ready within 1 business day.

**Can I buy a laptop or monitor and expense it?** No. Hardware comes from IT; expensed hardware is
rejected. Home office items under 400 USD per year are covered by the Remote Work stipend, but a
Northwind-issued monitor should be requested first.

**Can I use my personal phone for work email?** Yes, through the managed mail profile. Install
`Northwind Mobile` from the app store and sign in with your Northwind ID; a work profile is created that
IT can wipe without touching your personal data.

## Network and VPN

**Do I need VPN for email and chat?** No. VPN is needed for internal admin tools, back-office systems,
and database bastions.

**VPN shows error 412.** Your device certificate expired. Click `Renew certificate` in the NorthGate
client. Error 809 means the network you are on blocks the VPN port; try TCP fallback or a hotspot.

**Why is my VPN slow?** If you are in Logistics operations, your profile is `NW-Logistics-Full`, which
sends all traffic through Northwind. This is a compliance requirement, not a fault.

**Can I connect to the VPN from abroad?** From most countries, yes, as long as your remote-from-abroad
days are approved in PeopleHub. Some destinations are blocked for security reasons; TripDesk shows this
on the destination page.

## Software

**How do I install software?** Use the Self-Service app. If the application is not listed, request it
under `IT / Software request`; licence approval by your cost centre owner is required for paid tools.

**Can I use an external AI assistant with company documents?** Only tools listed in the AI Register.
Northwind Assist is approved for Internal and Confidential documents because it enforces document ACLs
and tenant separation. Pasting Confidential or Restricted data into unapproved tools is a Data
Classification Policy breach.

**Where are my files if my laptop dies?** Anything in managed cloud folders, Git, or Beacon is safe.
Local-only files are not backed up.

## Tickets and service levels

| Priority | Example                                            | Response target     |
|----------|----------------------------------------------------|---------------------|
| P1       | Cannot work at all; store or depot down            | 15 minutes, 24/7    |
| P2       | Major function impaired, workaround exists         | 2 business hours    |
| P3       | Single-user issue, can work with inconvenience     | 4 business hours    |
| P4       | Request, question, cosmetic                        | 2 business days     |

**How do I check my ticket?** In Beacon, `My tickets`. Replying to the ticket email adds a comment.

**I disagree with how my ticket was closed.** Reopen it within 5 business days from the ticket page;
after that, open a new one and reference the old number.

## Security

**I clicked a suspicious link.** Report it immediately via the `Report phishing` button in mail or by
calling the Service Desk; do not wait to see whether something happens. Change your password if you
entered it anywhere.

**I think a document contains instructions aimed at the AI assistant.** Report it under `SEC / Content
report`. Documents from external vendors are a known vector and are reviewed before indexing.

**I lost my laptop.** Report it within the hour; IT wipes it remotely. See the Laptop Replacement
Runbook.

## Related documents

- Password Reset Runbook (`it-password-reset-runbook`)
- VPN Access Runbook (`it-vpn-access-runbook`)
- Laptop Replacement Runbook (`it-laptop-replacement-runbook`)
