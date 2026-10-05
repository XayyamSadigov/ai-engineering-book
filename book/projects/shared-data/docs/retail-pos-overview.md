---
id: prod-retail-pos-overview
title: Lumen POS Platform Overview
version: "4.7"
updated_at: 2026-01-09
owner: Retail Systems
tenant: retail
acl_groups: ["all"]
tags: [retail, pos, product, architecture, payments]
---

# Lumen POS Platform Overview

**Lumen POS** is the point-of-sale platform used in all **1,240 Northwind Retail stores**. This overview
is for support staff, store IT coordinators, and engineers joining the Retail Systems team. The current
production release is **Lumen POS 4.7** (rolled out between 2025-11-03 and 2025-12-12).

## Components

| Component            | Runs on                    | Purpose                                                     |
|----------------------|----------------------------|-------------------------------------------------------------|
| Lumen Register       | Till terminals (Windows IoT) | Checkout UI, barcode scanning, receipt printing            |
| Lumen Store Server   | One server per store       | Local catalogue, prices, offline transaction queue           |
| Lumen Cloud          | Northwind cloud (retail)   | Master catalogue, pricing, reporting, store server sync      |
| PayBridge Adapter    | Store server               | Card payments through the PayBridge gateway                  |
| Returns API client   | Store server               | Creates and validates returns via the Returns API v2         |

Each store has between 2 and 24 registers. Registers talk only to their Store Server; the Store Server
talks to Lumen Cloud and to PayBridge.

## Offline mode

The Store Server keeps a full copy of the catalogue and prices so that stores keep trading when the
connection to Lumen Cloud is lost.

- Sales continue offline for up to **72 hours**. After 72 hours the registers show a warning and
  managers must call the Store Support line.
- Card payments offline use PayBridge's stand-in authorisation up to **150 USD per transaction**; higher
  amounts require cash or a second card.
- Returns cannot be created offline because they need the Returns API; the register records a
  "pending return" that is completed when connectivity returns.
- Offline transactions are uploaded automatically when the connection is restored, oldest first.

## Nightly synchronisation

Lumen Cloud pushes catalogue and price changes to Store Servers at **02:00 local store time**. Price
changes scheduled in the Pricing tool before 18:00 appear in stores the next morning. Emergency price
changes can be pushed on demand by Retail Systems (`Pricing > Push now`), which takes about 20 minutes
for all stores.

## Payments

Card payments go through the **PayBridge** gateway via the PayBridge Adapter on the Store Server. The
adapter authenticates with a client certificate issued by PayBridge and valid for **12 months**.
Certificates are tracked in the certificate inventory and renewed 30 days before expiry (process
tightened after incident INC-2025-1142). Supported methods: chip and PIN, contactless, mobile wallets,
and Northwind gift cards. Cash and gift-card transactions do not touch PayBridge.

## Monitoring and support

- Store health dashboard: `grafana/retail-store-health` shows per-store connectivity, queue depth of
  offline transactions, and PayBridge success rate.
- Alerts: `pos-store-offline` (store unreachable > 15 minutes), `paybridge-auth-failure-rate` (> 5% over
  5 minutes, pages Retail Systems on-call via Siren).
- Store staff reach **Store Support** by phone or via Beacon category `Retail / POS`. Priority 1 tickets
  (a store cannot take payments) have a 15-minute response target.

## Release process

Lumen POS releases quarterly. Each release goes to **10 pilot stores** for 2 weeks, then to 10% of
stores, then to all stores in waves of 200 stores per night. Store Servers update first; registers
update on the next restart. Rollback is done by pinning the previous version in Lumen Cloud, which
takes effect at the next nightly sync.

## Data

Transaction data is classified **Confidential**; card data never leaves the PayBridge Adapter (which
is PCI DSS scoped) and is never stored on Store Servers. Daily sales summaries are written to
`pg-retail-prod` and to the analytics warehouse.

## Roadmap (indicative)

- 4.8 (Q2 2026): self-checkout register support, improved returns flow.
- 5.0 (Q4 2026): new Store Server on Linux, removal of the legacy receipt printer driver.

## Related documents

- Retail Returns API v2 (`prod-retail-returns-api`)
- Incident report INC-2025-1142 (`inc-2025-11-pos-outage`)
