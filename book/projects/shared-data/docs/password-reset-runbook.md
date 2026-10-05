---
id: it-password-reset-runbook
title: Password Reset and Account Lockout Runbook
version: "3.0"
updated_at: 2026-04-14
owner: IT Service Desk
tenant: shared
acl_groups: ["all"]
tags: [it, password, identity, runbook, lockout, mfa]
---

# Password Reset and Account Lockout Runbook

## Password standard (summary)

| Rule                                   | Value                                                |
|----------------------------------------|------------------------------------------------------|
| Minimum length                         | **12 characters** (16 for privileged accounts)       |
| Composition                            | No composition rules; passphrases encouraged         |
| Rotation                               | **None** for standard accounts; **180 days** for privileged accounts (`it-oncall`, `NW-Admin`, service accounts) |
| Reuse                                  | Last 10 passwords cannot be reused                   |
| Lockout                                | **10 failed attempts** within 15 minutes locks the account for **30 minutes** |
| Breached password check                | Passwords found in known breach lists are rejected   |

MFA via Northwind Authenticator is mandatory for all accounts. A password alone never grants access.

## Self-service reset (preferred)

1. Go to `id.northwind.example/reset` from any device.
2. Enter your Northwind ID and approve the push in Northwind Authenticator (or enter the 6-digit code).
3. Choose a new password. It takes effect immediately for web applications and within 15 minutes for
   the laptop login (connect to any network so the device can sync).
4. If you have lost access to your Authenticator device, self-service is not possible. Proceed to the
   Service Desk reset below.

Self-service is unavailable while an account is locked; wait for the 30-minute lockout to end or call
the Service Desk.

## Service Desk assisted reset

Used when self-service fails or MFA is lost. The agent **must** verify identity before resetting; a
reset performed without verification is a security incident.

### Identity verification

The caller must pass **both** of the following:

1. **Record check**: employee number and the manager's name as recorded in PeopleHub, plus one of: start
   month and year, or last 4 digits of the corporate card, or the office location on record.
2. **Callback or manager confirmation**: the agent calls the mobile number on file in PeopleHub, **or**
   the caller's manager confirms the request in the Beacon ticket. Email or chat confirmation from the
   caller's own account is not accepted, because the account may be compromised.

If verification fails, the agent does not reset the password and opens a Security ticket
(`SEC / Suspicious reset attempt`). Never read a temporary password aloud over a call that the caller
initiated through an unknown number.

### Procedure

1. Open or locate the Beacon ticket (`IT / Identity > Password reset`). Record the verification method.
2. In Northwind ID admin, choose `Reset password` and select `Deliver via SMS to number on file`. The
   temporary password is valid for **1 hour** and must be changed at first login.
3. If MFA must be re-enrolled, use `Reset MFA` only after the manager has confirmed in the ticket. The
   user receives a new enrolment link valid for 24 hours.
4. Close the ticket with the verification method noted. Tickets without a verification note fail the
   monthly audit.

## Account lockouts

- Lockouts clear automatically after 30 minutes. The Service Desk may unlock earlier **after identity
  verification**.
- Repeated lockouts are usually caused by a saved old password on a phone mail client or a background
  script. Ask the user to update saved credentials before unlocking a third time.
- More than 50 failed attempts in an hour triggers a Security alert automatically; the Service Desk
  should not unlock until Security clears the alert.

## Privileged accounts

Privileged accounts (`it-oncall`, `NW-Admin`, database roles) have a 180-day rotation. Reminders go out
14 and 3 days before expiry. Expired privileged passwords can only be reset by the Service Desk with
manager confirmation and are reported to the Security Office weekly.

## Service accounts

Service account secrets live in the secrets manager and are rotated by their owning team; the Service
Desk does not reset them. Requests for service account resets are redirected to the owning team listed
in the service catalogue.

## Related documents

- Onboarding Accounts Runbook (`it-onboarding-accounts-runbook`)
- Access Control Policy (`sec-access-control-policy`)
- IT FAQ (`it-faq`)
