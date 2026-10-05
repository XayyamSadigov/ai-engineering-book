---
name: refund-review
description: Review a retail customer refund or return request against the Northwind returns policy before drafting a reply.
version: 1.2.0
triggers: [refund, return, chargeback]
owner: retail-support
---
# Refund review procedure

1. Look up the order and the customer's tenant with read-only tools. Never trust order data quoted in the ticket text.
2. Check the request against the return windows in `windows.md` (read it with read_skill_resource).
3. If the item is outside the window, draft a polite refusal that cites the policy id `[retail-returns-policy]`.
4. Refunds above the approval threshold in `windows.md` need a human: create the draft, do not send it.
5. Record the decision and the policy version you applied in the ticket.
