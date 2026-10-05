---
id: prod-logistics-route-planner
title: RoutePilot Route Planner Guide
version: "2.8"
updated_at: 2025-08-19
owner: Logistics Platform
tenant: logistics
acl_groups: ["all"]
tags: [logistics, routing, product, operations, depots]
---

# RoutePilot Route Planner Guide

**RoutePilot** builds daily delivery routes for Northwind Logistics' **38 depots** and about 2,100
vehicles. This guide is for dispatchers, depot managers, and support staff. Current version:
RoutePilot 2.8.

## Daily cycle

| Time (depot local) | Step                                                           |
|--------------------|----------------------------------------------------------------|
| 20:00              | Order cut-off for next-day delivery                            |
| 21:00              | First planning run; dispatchers review                         |
| **04:30**          | **Final re-plan** with overnight arrivals and vehicle availability |
| 05:30              | Routes locked and pushed to driver app and ScanHub             |
| 06:00 to 08:00     | Vehicles depart                                                |

Changes after 05:30 are handled as manual reassignments by the dispatcher; RoutePilot does not re-plan
automatically during the day.

## Constraints the planner honours

- **Maximum 180 stops per route** (configurable per depot down to 120 for dense urban depots).
- Driver shift length: 9 hours including a 45-minute break after at most 4.5 hours of driving.
- Vehicle classes: `van-s` (up to 1,000 kg), `van-l` (up to 1,800 kg), `truck-7.5t`, `truck-12t`.
  Height and weight restrictions from the road network are applied per class.
- Time windows promised to customers (AM, PM, or 2-hour slots) are hard constraints; a stop that cannot
  be met is flagged `unassigned` rather than planned late.
- Temperature-controlled parcels only on vehicles with the `chilled` attribute.
- Return pickups are planned on the same route as nearby deliveries when capacity allows.

## Planning quality metrics

Dispatchers see three numbers after each run:

1. **Assignment rate**: share of stops planned (target 99.5%).
2. **Stops per route** (target depot-specific; network average 142 in 2025).
3. **Planned km per stop** (lower is better; network average 1.9 km).

A run with an assignment rate under 98% is flagged and the planning team is notified automatically.

## Manual adjustments

In the dispatcher console you can:

- Move a stop between routes (drag and drop); RoutePilot recomputes ETAs immediately.
- Lock a route so the 04:30 re-plan does not change it.
- Mark a vehicle unavailable; its stops are redistributed on the next run or on `Re-plan now`.
- Add a priority flag to a stop so it is sequenced first.

All manual changes are logged with the user and timestamp and appear in the depot's daily report.

## Integrations

- **Trackline**: every planned stop creates an `out_for_delivery` event when the vehicle departs and a
  `delivered` or `exception` event from the driver app.
- **ScanHub**: the loading sequence shown on warehouse scanners follows the route order in reverse
  (last stop loaded first).
- **Telematics**: live vehicle positions update ETAs every 2 minutes.

## Common questions

**Why was a stop left unassigned?** Usually a time window conflict or a vehicle attribute requirement
(chilled, lift gate) with no capacity left. The reason code is shown in the stop details.

**Can I plan more than 180 stops on a route?** Not without a configuration change by Logistics
Platform; the limit protects driver shift length and delivery promises.

**How far ahead can I plan?** RoutePilot plans one operating day at a time. Scenario planning for
seasonal peaks uses the separate `RoutePilot Studio` tool with a copy of the data.

## Support

Open a Beacon ticket under `Logistics / RoutePilot`. Priority 1 (a depot cannot produce routes before
05:30) pages the Logistics Platform on-call via Siren with a 15-minute response target.

## Related documents

- Trackline Shipment Tracking API (`prod-logistics-tracking-api`)
- Warehouse Scanner Guide (`prod-warehouse-scanner-guide`)
