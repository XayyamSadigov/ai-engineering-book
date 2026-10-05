---
id: it-database-failover-runbook
title: PostgreSQL Failover Runbook
version: "3.3"
updated_at: 2026-02-20
owner: Platform Engineering
tenant: shared
acl_groups: ["it-oncall"]
tags: [it, database, postgresql, failover, runbook, patroni]
---

# PostgreSQL Failover Runbook

Applies to the production PostgreSQL clusters managed by Platform Engineering. All clusters run
**PostgreSQL 15** under **Patroni 3.2** with one primary and two streaming replicas (one synchronous,
one asynchronous) in separate availability zones.

## Clusters

| Cluster name      | Tenant     | Main consumers                                   | Size (Mar 2026) |
|-------------------|------------|--------------------------------------------------|-----------------|
| `pg-retail-prod`  | retail     | Lumen POS back office, Returns API, loyalty       | 1.8 TB          |
| `pg-logi-prod`    | logistics  | Trackline tracking API, RoutePilot, ScanHub sync  | 2.6 TB          |
| `pg-shared-prod`  | shared     | PeopleHub, Beacon, Northwind ID audit log         | 640 GB          |

## Objectives

- **RTO (recovery time objective): 15 minutes.**
- **RPO (recovery point objective): 5 minutes** for asynchronous failover; **zero** when the synchronous
  replica is healthy.
- Automatic failover is enabled. Manual intervention is needed only when Patroni cannot elect a leader or
  when the application layer does not follow the new primary.

## When to use this runbook

- Siren alert `pg-primary-unreachable` or `pg-replication-lag-critical` (lag > 300 s).
- Patroni reports `no leader` for more than 60 seconds.
- Planned maintenance requiring a controlled switchover (use section "Planned switchover").

## Unplanned failover: checklist

1. **Confirm the alert.** From the bastion, run `patronictl -c /etc/patroni.yml list <cluster>`. Expect
   one `Leader`. If a new leader was elected automatically, skip to step 5.
2. **Check the candidates.** The synchronous replica (`role: sync_standby`) is the preferred target. Note
   its `Lag in MB`; above 50 MB means potential data loss and must be stated in the incident record.
3. **Promote manually** only if no leader exists after 60 seconds:
   `patronictl -c /etc/patroni.yml failover <cluster> --candidate <sync-replica>`. Confirm when prompted.
4. **Fence the old primary.** If it is still reachable, stop Patroni on it to avoid split brain. If it is
   not reachable, mark its host as `drained` in the inventory so it is not restarted automatically.
5. **Verify the application layer.** The connection proxy (`pgbouncer-<cluster>`) follows the Patroni
   REST endpoint and should repoint within 10 seconds. Verify with
   `psql -h pgbouncer-<cluster> -c "select pg_is_in_recovery();"` which must return `f`.
6. **Verify replication.** `select client_addr, state, sync_state from pg_stat_replication;` should show
   at least one `streaming` replica within 2 minutes.
7. **Record** the timeline in the incident (see the Incident Response Runbook). A failover is at least a
   SEV2 for the tenant.
8. **Rebuild the old primary** as a replica using `patronictl reinit` after the incident is stable, not
   during it.

## Planned switchover

Used for kernel patches and instance resizes. Schedule in a maintenance window (retail: Tue/Thu
02:00-04:00 local; logistics: Wed 01:00-03:00 local, avoiding quarter-end week).

```
patronictl -c /etc/patroni.yml switchover <cluster> --master <current> --candidate <sync-replica> --scheduled now
```

Expected client-visible impact: connection resets for 5 to 15 seconds. Announce in `#platform-changes`
at least 24 hours ahead with the change ticket `CHG-YYYY-NNNN`.

## Verification queries

```sql
select now() - pg_last_xact_replay_timestamp() as replica_delay;   -- on replicas
select count(*) from pg_stat_activity where state = 'active';      -- on primary, expect < 200
select * from pg_stat_replication;                                  -- on primary
```

## Known pitfalls

- Long-running transactions on the old primary (> 5 minutes) block a clean switchover. Terminate them
  with `pg_terminate_backend` after confirming with the owning team.
- Application pools that pin to a host IP instead of the pgbouncer DNS name will not follow the failover.
  Known offenders are tracked in `PLAT-2211`; as of this version all production services use DNS names.
- Replication slots left behind by a decommissioned replica fill the disk. Check `pg_replication_slots`
  after every failover.

## Post-failover actions

- Run `ANALYZE` on the new primary if the replica had been receiving read traffic for a long time.
- Validate backups: the next nightly base backup must succeed from the new primary; check the
  `backup-pg-<cluster>` job in the morning.
- Postmortem within 5 business days for unplanned failovers.

## Related documents

- Incident Response Runbook (`it-incident-response-runbook`)
- Incident report INC-2026-0217 (`inc-2026-02-tracking-latency`)
