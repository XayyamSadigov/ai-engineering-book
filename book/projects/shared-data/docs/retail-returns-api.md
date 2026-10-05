---
id: prod-retail-returns-api
title: Retail Returns API v2 Reference
version: "2.3"
updated_at: 2025-12-15
owner: Retail Systems
tenant: retail
acl_groups: ["all"]
tags: [retail, api, returns, product, reference]
---

# Retail Returns API v2 Reference

The Returns API validates and records product returns for Northwind Retail. It is used by Lumen POS
registers, the online store, and the customer service console. Base URL (production):
`https://api.retail.northwind.example/returns`.

## Authentication

OAuth 2.0 client credentials. Request a token from `https://auth.northwind.example/oauth2/token` with
scope `returns.write` or `returns.read`. Tokens are valid for **1 hour**. Client registrations are
requested in Beacon under `Retail / API access` and approved by Retail Systems.

## Rate limits

| Client type           | Limit                 |
|-----------------------|-----------------------|
| Store Server          | 200 requests/minute   |
| Online store          | 1,000 requests/minute |
| Internal tools        | 100 requests/minute   |

Exceeding the limit returns HTTP 429 with a `Retry-After` header. Clients must back off exponentially
starting at 1 second.

## Return eligibility rules

- Default return window: **30 days** from the purchase date.
- Northwind Rewards members: **90 days**.
- Final-sale items (catalogue flag `final_sale = true`) are not returnable.
- Items returned without a receipt are refunded to a gift card at the lowest price in the last 60 days,
  limited to 3 no-receipt returns per customer per 12 months.
- Refunds go back to the original payment method when the original transaction can be found; otherwise
  to a gift card.

## Endpoints

### `POST /v2/returns/validate`

Checks eligibility without creating a return. Body:

```json
{ "transaction_id": "TX-2026-0192-003-0041", "items": [{ "sku": "RT-55120", "qty": 1 }] }
```

Response includes `eligible` per item, the `window_ends_at` date, and the refund `method`.

### `POST /v2/returns`

Creates a return. Requires `Idempotency-Key` header (UUID). Returns `201 Created` with the return id in
the format `RET-YYYYMMDD-NNNNNN` and the refund breakdown.

### `GET /v2/returns/{return_id}`

Fetches a return. Status values: `created`, `inspected`, `refunded`, `rejected`, `cancelled`.

### `POST /v2/returns/{return_id}/inspect`

Records the inspection outcome at the store or warehouse (`condition`: `resellable`, `damaged`,
`missing_parts`). Damaged items trigger a partial refund rule set maintained by Retail Finance.

### `GET /v2/returns?customer_id=...&from=...&to=...`

Lists returns for a customer; paginated with `cursor`, page size up to 100.

## Error codes

| Code      | HTTP | Meaning                                              |
|-----------|------|------------------------------------------------------|
| `RET-001` | 404  | Transaction not found                                |
| `RET-002` | 422  | Return window expired                                |
| `RET-003` | 422  | Item is final sale                                   |
| `RET-004` | 422  | Quantity exceeds purchased quantity                  |
| `RET-005` | 409  | Return already exists for this item (idempotency)    |
| `RET-006` | 422  | No-receipt return limit reached for this customer    |
| `RET-010` | 503  | Payment provider unavailable; retry later            |

## Webhooks

Clients may subscribe to `return.created`, `return.refunded`, and `return.rejected`. Deliveries are
signed with HMAC-SHA256 using the client's webhook secret and retried 5 times over 24 hours.

## Service levels

- Availability target: 99.9% monthly.
- Latency target: p95 under **400 ms** for `validate` and `GET`, under 800 ms for `POST /v2/returns`.
- Dashboard: `grafana/retail-returns-api`.

## Versioning and deprecation

v1 was retired on **2025-06-30**. Breaking changes are introduced only in a new major version, with a
minimum 6-month overlap. Non-breaking additions are announced in the Retail Systems changelog.

## Related documents

- Lumen POS Platform Overview (`prod-retail-pos-overview`)
