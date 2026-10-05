---
id: prod-logistics-tracking-api
title: Trackline Shipment Tracking API v3
version: "3.4"
updated_at: 2026-03-30
owner: Logistics Platform
tenant: logistics
acl_groups: ["all"]
tags: [logistics, api, tracking, product, reference, webhooks]
---

# Trackline Shipment Tracking API v3

**Trackline** is Northwind Logistics' shipment tracking service. It exposes shipment status to business
clients, the customer portal, and internal tools. Base URL: `https://api.logistics.northwind.example`.

## Identifiers

- Tracking IDs have the format **`NW` followed by 12 digits**, for example `NW482910037755`. The last
  digit is a mod-10 check digit.
- Client references (`client_ref`) are free text up to 64 characters, unique per client account.

## Authentication and limits

- API keys issued per client account; sent in the `X-Api-Key` header. Keys can be rotated in the client
  portal; the old key stays valid for 24 hours after rotation.
- Rate limit: **600 requests/minute** per account, burst of 100. HTTP 429 with `Retry-After` on excess.
- Bulk lookups (`POST /v3/shipments/batch`, up to 200 IDs) count as one request.

## Endpoints

### `GET /v3/shipments/{tracking_id}`

Returns the current status and the full event history.

```json
{
  "tracking_id": "NW482910037755",
  "status": "in_transit",
  "estimated_delivery": "2026-03-31",
  "events": [
    { "at": "2026-03-29T08:12:00Z", "code": "PICKED_UP", "location": "Depot North 2" },
    { "at": "2026-03-30T02:40:00Z", "code": "HUB_SCAN", "location": "Central Hub" }
  ]
}
```

### `GET /v3/shipments?client_ref=...`

Lookup by client reference.

### `POST /v3/shipments/batch`

Up to 200 tracking IDs; returns the same payload per ID, with `not_found` entries for unknown IDs.

### `POST /v3/webhooks` and `DELETE /v3/webhooks/{id}`

Register a callback URL for status changes. One webhook per account per event type.

## Status model

| Status          | Meaning                                         | Terminal |
|-----------------|-------------------------------------------------|----------|
| `created`       | Label created, not yet picked up                | no       |
| `picked_up`     | Collected from shipper                          | no       |
| `in_transit`    | Moving between facilities                       | no       |
| `out_for_delivery` | On a vehicle for final delivery              | no       |
| `delivered`     | Delivered with proof of delivery                | yes      |
| `exception`     | Delay, damage, or address problem; see `reason` | no       |
| `returned`      | Returned to shipper                             | yes      |

Event codes are stable; new codes may be added and clients must ignore unknown codes.

## Webhooks

- Delivered as `POST` with JSON body and `X-Trackline-Signature` (HMAC-SHA256 of the body with the
  account's webhook secret).
- Retried on non-2xx responses with exponential backoff: 1 min, 5 min, 30 min, 2 h, 12 h (**5 attempts
  over about 15 hours**). After the last failure the webhook is paused and the account owner is emailed.
- Since v3.4 (2026-03), retries are rate-limited to **50 per minute per account** to prevent retry
  storms like the one that contributed to INC-2026-0217.

## Service levels

- Availability target: 99.95% monthly.
- Latency target: **p95 under 300 ms** for single lookups; under 1.5 s for batch.
- Data freshness: scan events visible within 60 seconds of the physical scan.
- Dashboard: `grafana/logistics-trackline`.

## Data retention

Event history is available via the API for **18 months** after the terminal status. Older data is in
the analytics warehouse and can be requested through Logistics Platform.

## Errors

| Code      | HTTP | Meaning                            |
|-----------|------|------------------------------------|
| `TRK-001` | 404  | Tracking ID not found              |
| `TRK-002` | 400  | Invalid tracking ID (check digit)  |
| `TRK-003` | 429  | Rate limit exceeded                |
| `TRK-004` | 400  | Batch size over 200                |

## Related documents

- RoutePilot Route Planner (`prod-logistics-route-planner`)
- Warehouse Scanner Guide (`prod-warehouse-scanner-guide`)
- Incident report INC-2026-0217 (`inc-2026-02-tracking-latency`)
