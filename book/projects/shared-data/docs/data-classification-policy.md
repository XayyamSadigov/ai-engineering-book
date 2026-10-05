---
id: sec-data-classification-policy
title: Data Classification and Handling Policy
version: "2.5"
updated_at: 2025-10-30
owner: Security Office
tenant: shared
acl_groups: ["all"]
tags: [security, data, classification, policy, retention, encryption]
---

# Data Classification and Handling Policy

## Classification levels

Every document, dataset, and system at Northwind is assigned one of four levels. The owner of the data
chooses the level; when in doubt, choose the higher one.

| Level            | Definition                                                              | Examples                                                     |
|------------------|-------------------------------------------------------------------------|--------------------------------------------------------------|
| **Public**       | Approved for release outside Northwind                                  | Marketing material, published API documentation, job adverts |
| **Internal**     | For employees and contractors; low harm if disclosed                    | Most policies and runbooks, org charts, meeting notes        |
| **Confidential** | Business-sensitive; disclosure would harm Northwind or a partner        | Transaction data, pricing strategy, incident reports, source code, supplier contracts |
| **Restricted**   | Personal or regulated data; disclosure would harm individuals or breach law | Employee records, payroll, medical certificates, customer personal data, card data, credentials |

## Labelling

- Documents carry the level in the front matter or header (`classification: Confidential`).
- Datasets and database schemas carry the level in the data catalogue.
- Unlabelled data is treated as **Internal** by default, except data about identifiable people, which is
  always at least Confidential and usually Restricted.

## Handling rules

| Rule                                 | Public | Internal | Confidential | Restricted |
|--------------------------------------|--------|----------|--------------|------------|
| Share outside Northwind              | Yes    | With NDA | Only with contract and Legal approval | Only with documented legal basis |
| Store in personal cloud or email     | Yes    | No       | No           | No         |
| Encryption at rest                   | n/a    | Platform default | Required (AES-256) | Required, with separate key per system |
| Encryption in transit                | TLS    | TLS 1.2+ | TLS 1.2+     | TLS 1.2+ and mutual auth for system-to-system |
| Print                                | Yes    | Yes      | Office only  | Not permitted without Security approval |
| Use in AI assistants / LLM tools     | Yes    | Approved internal tools only | Approved internal tools with tenant and ACL enforcement | Only tools listed in the AI Register with DPIA completed |
| Access review                        | n/a    | Annual   | Quarterly    | Quarterly, with named individual owners |

## Retention

| Data type                           | Retention                     |
|-------------------------------------|-------------------------------|
| Support tickets                     | 3 years after closure         |
| Transaction and shipment records    | 7 years (financial regulation)|
| Incident reports                    | 5 years                       |
| Employee records                    | 6 years after employment ends |
| Application logs (production)       | 14 days hot, 1 year archived  |
| Recruitment data (unsuccessful)     | 6 months                      |

Data past its retention period is deleted or anonymised by the owning system; owners confirm deletion
annually in the data catalogue.

## Personal data

Personal data (anything relating to an identifiable person) is Restricted unless it is already public
business contact data (name, job title, work email), which is Internal. Processing personal data for a
new purpose requires a privacy assessment (DPIA) with the Security Office before go-live. Customer and
employee personal data must stay within the tenant that collected it; cross-tenant sharing requires a
documented purpose and Legal approval.

## AI and automated tools

Internal AI assistants such as **Northwind Assist** may index Internal and Confidential content only
when they enforce the source document's `acl_groups` and `tenant` on every answer. Restricted documents
are excluded from indexing unless the Security Office has approved the specific tool and use case.
Outputs that quote Confidential content inherit that classification.

## Incidents

Suspected loss or exposure of Confidential or Restricted data must be reported to the Security Office
within **1 hour** of discovery via Beacon (`SEC / Data incident`) or the on-call Security line. Do not
attempt to recall emails or delete evidence before Security has been informed.

## Related documents

- Access Control Policy (`sec-access-control-policy`)
- Code of Conduct (`hr-code-of-conduct`)
- Remote Work Policy (`hr-remote-work-policy`)
