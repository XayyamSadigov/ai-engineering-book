---
id: it-laptop-replacement-runbook
title: Laptop Replacement and Repair Runbook
version: "1.9"
updated_at: 2025-10-06
owner: IT Service Desk
tenant: shared
acl_groups: ["all"]
tags: [it, hardware, laptop, runbook, lifecycle]
---

# Laptop Replacement and Repair Runbook

## Standard models

| Role profile                          | Standard device                 | Alternative (on request)     |
|---------------------------------------|---------------------------------|------------------------------|
| General office                        | Dell Latitude 7450 (16 GB)      | MacBook Air M3 (16 GB)       |
| Engineering, data, design             | MacBook Pro 14" M3 Pro (36 GB)  | Dell Precision 5490 (32 GB)  |
| Store and warehouse back-office       | Dell Latitude 5450 (16 GB)      | none                         |

Devices are ordered, imaged, and enrolled in device management by the IT Service Desk. Users may not
purchase laptops through expenses; such claims are rejected under the Expense Policy.

## Refresh cycle

- Laptops are replaced every **36 months** from the delivery date recorded in the asset register.
- Three months before the refresh date, the user receives an automated email from Beacon with a
  pre-filled replacement request. Approving it is all that is required; no manager approval is needed
  for a like-for-like refresh.
- A change of model (for example, moving from Latitude to MacBook) requires manager approval in Beacon.

## Requesting a replacement (planned)

1. Open Beacon and choose `IT / Hardware > Laptop replacement`.
2. Select the reason: `Scheduled refresh`, `Role change`, or `Performance insufficient`.
3. Confirm the delivery location (office pickup or home address for remote staff).
4. Service level: the replacement is delivered within **5 business days** of approval for stock models.
   Non-stock models take up to 15 business days.

## Broken, lost, or stolen devices (unplanned)

1. **Stolen or lost**: report immediately via Beacon `IT / Security > Lost or stolen device` or by
   phone to the Service Desk. IT remotely wipes the device and revokes its certificates within 1 hour of
   the report. Also report theft to the local police and keep the report number for the ticket.
2. **Broken**: open `IT / Hardware > Repair`. If the device cannot be used for work, tick `Needs loaner`.
   A loaner is provided within **1 business day** at an office, or shipped the same day to remote staff.
3. Repairs under warranty take 5 to 10 business days. Out-of-warranty damage is repaired or the device
   is replaced at IT's discretion; the user's cost centre is charged for accidental damage above 2
   incidents in 36 months.

## Data migration

- All work data should live in managed cloud storage (Drive, Git, ticketing systems). Local-only data
  is the user's responsibility; IT does not back up local disks.
- The new device is enrolled so that the user's apps, settings, and cloud folders sync on first login.
  Allow 30 to 60 minutes for the first sync.
- Loaners are wiped when returned.

## Returning the old device

- Return the old laptop within **10 business days** of receiving the new one, by office drop-off or the
  prepaid return box shipped with the replacement.
- Devices are kept in quarantine for **14 days** after return, then wiped and either redeployed or
  recycled. If you realise you forgot to move data, open a ticket within those 14 days.
- Unreturned devices after 30 days are reported to the user's manager and remotely wiped.

## Accessories

Headsets, external monitors, docks, keyboards, and mice are requested separately under `IT / Hardware
> Accessories`. Standard accessories are delivered within 5 business days. Monitors for home use fall
under the Remote Work stipend only if a Northwind-issued monitor is not available.

## Service levels summary

| Request                       | Target                      |
|-------------------------------|-----------------------------|
| Scheduled replacement         | 5 business days from approval |
| Loaner for broken device      | 1 business day              |
| Remote wipe after theft report| 1 hour                      |
| Accessory delivery            | 5 business days             |

## Related documents

- Onboarding Accounts Runbook (`it-onboarding-accounts-runbook`)
- Remote Work Policy (`hr-remote-work-policy`)
- Data Classification Policy (`sec-data-classification-policy`)
