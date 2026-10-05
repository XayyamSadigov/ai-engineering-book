---
name: vpn-triage
description: Triage employee VPN connection problems such as login loops, certificate errors, and split-tunnel issues.
version: 2.0.0
triggers: [vpn, certificate, tunnel]
owner: it-oncall
---
# VPN triage

1. Ask for the client version and the exact error text.
2. Login loop: clear the cached profile, then re-enroll the device certificate (see `[it-vpn-access-runbook]`).
3. Certificate expired: run `renew_cert.sh` only from the IT sandbox host, never on the employee laptop.
4. Escalate to network on-call if more than five users at one site report the same error.
