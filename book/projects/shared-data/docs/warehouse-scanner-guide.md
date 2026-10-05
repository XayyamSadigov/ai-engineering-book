---
id: prod-warehouse-scanner-guide
title: ScanHub Warehouse Scanner Guide
version: "2.3"
updated_at: 2025-07-22
owner: Logistics Platform
tenant: logistics
acl_groups: ["all"]
tags: [logistics, warehouse, scanner, product, operations, hardware]
---

# ScanHub Warehouse Scanner Guide

**ScanHub** is the handheld application used in Northwind Logistics warehouses and depots for receiving,
put-away, picking, packing, and loading. It runs on **Zebra TC-58** rugged handhelds. Current app
version: **ScanHub 2.3.1**.

## Devices and accounts

- Each warehouse has a pool of shared handhelds in charging cradles at the shift desk.
- Sign in with your Northwind ID and a 6-digit **warehouse PIN** (set in PeopleHub). MFA is not required
  on handhelds inside the warehouse network.
- Sessions end automatically after 20 minutes of inactivity or when the device is docked.
- Batteries are hot-swappable: tap `Menu > Swap battery`, swap within 60 seconds, and the session is
  preserved.

## Supported barcodes

| Label type            | Symbology       | Prefix / format              |
|-----------------------|-----------------|------------------------------|
| Shipment label        | Code 128        | `NW` + 12 digits (tracking ID) |
| Location label        | Data Matrix     | `LOC-<zone>-<aisle>-<bay>-<level>`, e.g. `LOC-A-14-03-2` |
| Tote                  | Code 128        | `TOTE-` + 8 digits           |
| Product (inbound)     | EAN-13 / GS1-128| Supplier barcode             |
| Vehicle / trailer     | QR              | `VEH-` + fleet number        |

## Workflows

### Receiving
1. Scan the inbound shipment label.
2. Scan each product barcode; the expected quantity is shown. Deviations over 2% require a supervisor
   PIN.
3. Confirm. Items move to status `received` and appear in put-away.

### Put-away
1. Scan the tote. 2. Walk to the suggested location shown on screen. 3. Scan the location label.
A mismatch shows error `SH-104` (wrong location); scan the correct label or choose `Override` with a
reason.

### Picking
Pick lists are released in waves every 30 minutes. The screen shows location, product, and quantity.
Scan location, then product; short picks are recorded with reason codes (`out of stock`, `damaged`,
`not found`). Three `not found` for the same location in one shift triggers a cycle count task.

### Packing
Scan the tote, then each item, then print the shipment label. The label printer must be paired via
`Menu > Printers`; the default is the printer nearest the packing station.

### Loading
Scan the vehicle QR, then each shipment label in the sequence shown (reverse route order from
RoutePilot). Loading a shipment that belongs to another vehicle shows error `SH-201` and must not be
overridden.

## Error codes

| Code     | Meaning                                  | Action                                     |
|----------|------------------------------------------|--------------------------------------------|
| `SH-101` | Barcode not recognised                   | Clean label or key in the number manually  |
| `SH-104` | Wrong location                           | Scan the correct location or override      |
| `SH-120` | Quantity deviation over tolerance        | Supervisor PIN required                    |
| `SH-201` | Shipment belongs to another vehicle      | Do not load; inform the dispatcher         |
| `SH-305` | Offline: sync pending                    | Keep working; scans sync when back online  |
| `SH-500` | Server error                             | Retry; if persistent, open a Beacon ticket |

## Offline behaviour

ScanHub buffers up to **2,000 scans** per device while offline and syncs them in order when the network
returns. Picking and loading are allowed offline; receiving is not (expected quantities need the server).

## Troubleshooting

- **Device will not sign in**: check the warehouse Wi-Fi indicator; dock the device for 30 seconds to
  force a sync; verify your PIN in PeopleHub.
- **Scanner beam not firing**: hold the trigger for 3 seconds to reset; if still dead, swap devices and
  place the faulty one in the red "repair" tray.
- **Label printer not found**: re-pair under `Menu > Printers`; printers are named by station, for
  example `PRN-PACK-04`.

Open Beacon tickets under `Logistics / ScanHub`. Shift supervisors can escalate a warehouse-wide outage
to the Logistics Platform on-call through Siren.

## Related documents

- RoutePilot Route Planner Guide (`prod-logistics-route-planner`)
- Trackline Shipment Tracking API (`prod-logistics-tracking-api`)
