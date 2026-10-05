---
id: inc-2025-11-pos-outage
title: "Incident Report INC-2025-1142: Card Payment Outage in Retail Stores"
version: "1.2"
updated_at: 2025-11-26
owner: Retail Systems
tenant: retail
acl_groups: ["it-oncall", "managers"]
tags: [incident, postmortem, retail, pos, paybridge, certificate, sev1]
---

# Incident Report INC-2025-1142: Card Payment Outage in Retail Stores

| Field               | Value                                            |
|---------------------|--------------------------------------------------|
| Incident ID         | **INC-2025-1142**                                |
| Severity            | SEV1                                             |
| Tenant              | retail                                           |
| Started             | 2025-11-18 09:04 local                           |
| Detected            | 2025-11-18 09:11 (alert `paybridge-auth-failure-rate`) |
| Resolved            | 2025-11-18 12:16                                 |
| **Duration**        | **3 hours 12 minutes**                           |
| Incident Commander  | Retail Systems on-call (primary)                 |
| Stores affected     | **412 of 1,240**                                 |
| Customer impact     | Card payments declined; cash and gift cards unaffected |

## Summary

Card payments failed in 412 stores for just over three hours because the **PayBridge client
certificate** used by the PayBridge Adapter on the affected Store Servers expired at 09:00 on
2025-11-18. The certificate had been issued on 2024-11-18 with a 12-month validity. Stores that had
been migrated to Lumen POS 4.7 before 2025-11-18 had already received a newer certificate as part of the
upgrade; stores still on 4.6 had not.

## Timeline

| Time (local) | Event                                                                                   |
|--------------|-----------------------------------------------------------------------------------------|
| 09:00        | Certificate `paybridge-store-2024` expires                                              |
| 09:04        | First card authorisation failures; Store Servers log `TLS handshake failed: certificate expired` |
| 09:11        | Siren pages Retail Systems on-call; `paybridge-auth-failure-rate` at 38%                |
| 09:18        | Incident opened as SEV1; bridge started; first `#incidents` update                      |
| 09:40        | Root cause identified from Store Server logs on 3 sample stores                          |
| 09:55        | Decision: push the 4.7 certificate bundle to 4.6 stores via emergency sync instead of accelerating the 4.7 upgrade |
| 10:20        | Emergency sync job prepared and tested on 5 pilot stores; payments restored there        |
| 10:35        | Sync pushed to all 412 stores in 4 waves                                                 |
| 11:50        | 395 stores confirmed healthy; 17 stores required a Store Server restart by store staff   |
| 12:16        | Last store confirmed; incident resolved                                                 |
| 2025-11-24   | Postmortem review meeting                                                               |

## Root cause

The PayBridge certificate was tracked in a spreadsheet owned by a single engineer who had left Retail
Systems in August 2025. No alert existed for certificate expiry. The 4.7 upgrade happened to include a
renewed certificate, which masked the problem for upgraded stores and made the failure pattern (only 4.6
stores) initially confusing.

## Contributing factors

- No central certificate inventory with expiry alerts for store-side certificates.
- The 4.7 rollout schedule (waves of 200 stores per night) meant a third of stores were still on 4.6 on
  the expiry date.
- Store staff had no guidance on restarting the Store Server; 17 stores lost an extra 20 to 40 minutes.

## What went well

- Detection within 7 minutes by the existing failure-rate alert.
- Pilot-first approach for the emergency fix prevented a bad push to all stores.
- Communications Lead kept the status page and store managers updated every 30 minutes.

## Action items

| ID         | Action                                                                     | Owner             | Due        | Status |
|------------|----------------------------------------------------------------------------|-------------------|------------|--------|
| `RS-2201`  | Register all store-side certificates in the central certificate inventory with alerts at 30 and 7 days before expiry | Retail Systems | 2025-12-19 | Done |
| `RS-2202`  | Add Store Server restart instructions to the store IT quick card           | Store Support     | 2025-12-05 | Done   |
| `RS-2203`  | Synthetic card authorisation check every 5 minutes from 20 reference stores | Retail Systems    | 2026-01-30 | Done   |
| `PLAT-2240`| Platform-wide review of certificates not managed by the central inventory  | Platform Engineering | 2026-02-27 | In progress |

## Lessons

Certificates with a fixed lifetime are a known failure class; every certificate must have an owner
that is a team, not a person, and an alert. The renewal process is now documented in the Lumen POS
Platform Overview (Payments section).

## Related documents

- Lumen POS Platform Overview (`prod-retail-pos-overview`)
- Incident Response Runbook (`it-incident-response-runbook`)
