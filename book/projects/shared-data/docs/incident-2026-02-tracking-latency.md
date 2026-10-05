---
id: inc-2026-02-tracking-latency
title: "Incident Report INC-2026-0217: Trackline API Latency Degradation"
version: "1.1"
updated_at: 2026-02-18
owner: Logistics Platform
tenant: logistics
acl_groups: ["it-oncall", "managers"]
tags: [incident, postmortem, logistics, tracking, latency, database, webhooks, sev2]
---

# Incident Report INC-2026-0217: Trackline API Latency Degradation

| Field               | Value                                               |
|---------------------|-----------------------------------------------------|
| Incident ID         | **INC-2026-0217**                                   |
| Severity            | SEV2                                                |
| Tenant              | logistics                                           |
| Started             | 2026-02-09 06:02 UTC                                |
| Detected            | 2026-02-09 06:09 UTC (alert `trackline-p95-latency`) |
| Resolved            | 2026-02-09 07:49 UTC                                |
| **Duration**        | **1 hour 47 minutes**                               |
| Incident Commander  | Logistics Platform on-call                          |
| Customer impact     | Tracking lookups slow (**p95 rose from 240 ms to 4.8 s**); 3.1% of requests timed out; webhook deliveries delayed up to 90 minutes |

## Summary

A schema migration deployed at 05:40 UTC on 2026-02-09 (`CHG-2026-0188`) added a column to the
`shipment_events` table in `pg-logi-prod` and recreated the table's indexes. The migration script
omitted the composite index on `(tracking_id, occurred_at)`. Event-history queries fell back to a
sequential scan as morning scan volume ramped up. Slow responses caused client webhooks to time out,
and the webhook retry logic (then unlimited rate) multiplied load on the same table.

## Timeline (UTC)

| Time   | Event                                                                                   |
|--------|-----------------------------------------------------------------------------------------|
| 05:40  | Migration `CHG-2026-0188` applied; completes in 6 minutes                               |
| 06:02  | Depot scan volume rises; `GET /v3/shipments/{id}` p95 passes 1 s                        |
| 06:09  | Siren pages Logistics Platform on-call; incident opened as SEV2                         |
| 06:20  | Database CPU at 96%; `pg_stat_activity` shows hundreds of sequential scans              |
| 06:31  | Missing index identified by comparing `pg_indexes` with the pre-migration snapshot      |
| 06:38  | Decision: create the index concurrently rather than roll back the migration             |
| 06:41  | `CREATE INDEX CONCURRENTLY` started                                                     |
| 07:05  | Webhook retries identified as 60% of query load; retry workers paused                   |
| 07:22  | Index build completes; p95 drops to 310 ms within 3 minutes                             |
| 07:30  | Retry workers resumed at reduced concurrency; backlog of 48,000 deliveries drains       |
| 07:49  | Backlog cleared; p95 at 250 ms for 30 minutes; incident resolved                        |

## Root cause

The migration tool generated the index list from the development database, where the composite index
had been created manually and never committed to the migration history. The review of `CHG-2026-0188`
did not include a diff of production indexes.

## Contributing factors

- Webhook retries had no per-account rate limit, so each failed delivery produced a new query within a
  minute, creating a retry storm.
- The change window (05:40) was 20 minutes before the daily scan ramp, leaving no time to observe the
  system at low load.
- No synthetic query test ran after the migration.

## What went well

- Alerting fired within 7 minutes of the degradation.
- Choosing a concurrent index build avoided a rollback that would have required a second maintenance
  window.
- No data was lost; all events were eventually delivered.

## Action items

| ID          | Action                                                                | Owner               | Due        | Status |
|-------------|-----------------------------------------------------------------------|---------------------|------------|--------|
| `LP-1877`   | Add production index diff to the migration checklist                  | Logistics Platform  | 2026-02-27 | Done   |
| `LP-1878`   | Rate-limit webhook retries to 50 per minute per account               | Logistics Platform  | 2026-03-13 | Done (v3.4) |
| `LP-1879`   | Move logistics change window to 01:00-03:00 UTC, at least 3 hours before scan ramp | Logistics Platform | 2026-02-20 | Done |
| `PLAT-2251` | Post-migration synthetic query check in the deployment pipeline       | Platform Engineering| 2026-03-31 | In progress |

## Related documents

- Trackline Shipment Tracking API (`prod-logistics-tracking-api`)
- PostgreSQL Failover Runbook (`it-database-failover-runbook`)
- Incident Response Runbook (`it-incident-response-runbook`)
