---
id: it-vpn-access-runbook
title: NorthGate VPN Access Runbook
version: "2.4"
updated_at: 2026-01-28
owner: IT Service Desk
tenant: shared
acl_groups: ["all"]
tags: [it, vpn, runbook, networking, remote-access]
---

# NorthGate VPN Access Runbook

NorthGate is Northwind's VPN service. It is required for access to internal systems (PeopleHub admin,
Beacon admin, store and warehouse back-office tools, database bastions) from outside a Northwind office.
Email, chat, and the intranet home page do **not** require VPN.

## Client requirements

| Item                   | Requirement                                   |
|------------------------|-----------------------------------------------|
| Client version         | **NorthGate Client 5.2.1** or later           |
| Operating systems      | Windows 11 23H2+, macOS 14+, Ubuntu 22.04+    |
| Authentication         | Northwind ID + **Northwind Authenticator** push (MFA) |
| Device                 | Northwind-managed device with current compliance status |
| Ports                  | UDP 443 preferred, TCP 443 fallback            |

The client is installed by default on managed laptops. If it is missing, install it from the Self-Service
app (`Northwind Apps > NorthGate Client`). Personal devices cannot connect.

## Connection profiles

- **`NW-Standard`** (default for most employees): split tunnelling enabled. Only traffic to internal
  ranges goes through the VPN.
- **`NW-Logistics-Full`**: used by Logistics operations staff and anyone handling carrier integrations.
  **Split tunnelling is disabled** for compliance with carrier contracts; all traffic goes through
  Northwind.
- **`NW-Admin`**: for IT on-call and database administrators. Requires membership of the `it-oncall`
  group and a hardware security key in addition to the Authenticator push.

Profiles are assigned automatically from group membership. If you believe you have the wrong profile,
open a Beacon ticket in category `IT / Network access`.

## Connecting

1. Open NorthGate Client and select your profile.
2. Click **Connect**, enter your Northwind ID password, and approve the push notification.
3. The status turns green and shows the assigned internal IP. First connection on a new device takes up
   to 60 seconds while the device certificate is issued.

## Troubleshooting

| Symptom / error code | Meaning                                | Fix                                                        |
|----------------------|----------------------------------------|------------------------------------------------------------|
| **Error 412**        | Device certificate expired             | Click `Renew certificate` in the client; if that fails, run `Repair` from Self-Service |
| **Error 809**        | UDP/TCP 443 blocked by local network   | Switch to TCP fallback in `Settings > Transport`, or use a phone hotspot |
| **Error 633**        | Another VPN client holds the adapter   | Quit other VPN software and reboot                          |
| Push never arrives   | Authenticator not enrolled or phone offline | Use the 6-digit code in the Authenticator app instead of push |
| Connected, no access | Device out of compliance               | Open Self-Service, run `Compliance check`, install pending updates |
| Connected, slow      | Full-tunnel profile on a poor link     | Expected for `NW-Logistics-Full`; move closer to the router, avoid video in parallel |

If the issue persists after these steps, collect diagnostics with `Help > Export logs` and attach the
file to a Beacon ticket. The Network queue's response target is 4 business hours for access issues.

## Access for contractors

Contractors receive VPN access only via a sponsoring manager's request in Beacon (`IT / Contractor
access`). Access is time-limited to the contract end date, maximum 6 months per request, and uses the
`NW-Standard` profile only.

## Known limitations

- NorthGate does not support connections from countries on the Security restricted list; travellers
  should check TripDesk's destination page before departure.
- Only one active session per user. Connecting from a second device disconnects the first.
- The idle timeout is 12 hours; the hard session limit is 24 hours.

## Escalation

- Service Desk (Beacon, `IT / Network access`): first line, 4 business hours response.
- Network on-call via Siren (service `network-vpn`): for outages affecting multiple users, 24/7.

## Related documents

- Remote Work Policy (`hr-remote-work-policy`)
- Password Reset Runbook (`it-password-reset-runbook`)
- Access Control Policy (`sec-access-control-policy`)
