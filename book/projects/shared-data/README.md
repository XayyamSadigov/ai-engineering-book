# Northwind Assist shared dataset

Synthetic, self-contained sample data for the book's running example (authoring guide, section 5).
Everything here is invented: Northwind, its tenants `retail` and `logistics`, all people, vendors,
systems, ticket IDs and amounts. Nothing is derived from a real company. English only; total size is
under 600 KB; every single file is under 50 KB.

Authors who need sample documents, tickets, invoices or evaluation gold **import from here** instead of
inventing their own, so chapters agree on facts (the PTO carryover limit, error code `RET-002`, incident
`INC-2025-1142`, and so on).

```
shared-data/
├── docs/                         24 Markdown documents with YAML front matter
├── manifest.json                 metadata + sha256 for every document (generated)
├── tickets.jsonl                 60 support tickets
├── invoices.jsonl                20 invoices: text blob + gold structured fields
├── eval/
│   ├── retrieval_gold.jsonl      40 retrieval/QA questions over docs/
│   └── classification_gold.jsonl 60 gold ticket categories
├── shared_data.py                typed loader (pydantic v2)
├── test_shared_data.py           integrity tests
├── tools/build_manifest.py       regenerates manifest.json
└── pytest.ini
```

## Loading the data

The loader has no dependencies beyond `pydantic>=2`. From a project directory under `book/projects/`:

```python
# path: book/projects/<your-project>/example_load.py
import sys
sys.path.insert(0, "../shared-data")

from shared_data import load_docs, load_tickets, load_invoices, load_retrieval_gold

docs = load_docs()                       # list[Doc]
tickets = load_tickets()                 # list[Ticket]
invoices = load_invoices()               # list[Invoice]
questions = load_retrieval_gold()        # list[RetrievalQuestion]

visible = [d for d in docs if d.visible_to(user_groups=["all"], tenant="retail")]
```

Alternatively, add `../shared-data` to `pythonpath` in your `pytest.ini`, or copy `shared_data.py` into
your project (it is a single file). `load_manifest()` and `load_classification_gold()` are also exported.

Run the integrity tests from the repository root:

```bash
.venv/bin/python -m pytest book/projects/shared-data -q
```

After editing any document under `docs/`, regenerate the manifest, otherwise the hash test fails:

```bash
.venv/bin/python book/projects/shared-data/tools/build_manifest.py
```

## `docs/` and the ACL model

Each document starts with YAML front matter:

| Field        | Type                 | Notes                                                                 |
|--------------|----------------------|-----------------------------------------------------------------------|
| `id`         | string               | Stable, unique. Prefixes: `hr-`, `it-`, `prod-`, `sec-`, `inc-`, `ext-`. |
| `title`      | string               |                                                                       |
| `version`    | string               | Quoted in YAML (`"3.0"`), so it stays a string.                       |
| `updated_at` | date                 | 2025 or 2026.                                                         |
| `owner`      | string               | Owning team, never a person.                                          |
| `tenant`     | `retail` \| `logistics` \| `shared` | Tenant isolation key.                                     |
| `acl_groups` | list[string]         | Any one matching group grants access (see below).                     |
| `tags`       | list[string]         | Free-form.                                                            |

**Access rule** (implemented in `Doc.visible_to`): a user holding groups `G` and working in tenant `T`
may see a document if `acl_groups ∩ G` is non-empty **and** the document's tenant is `shared` or equals
`T`. Every employee holds the baseline group `all`; `retail`/`logistics` membership is modelled by the
`tenant` argument, not as a group.

Groups used: `all`, `hr`, `it-oncall`, `managers`, `security`, `finance`.

| Document (id)                          | Tenant    | ACL groups                        | Purpose in the book |
|----------------------------------------|-----------|-----------------------------------|---------------------|
| `hr-pto-policy`                        | shared    | all                               | **New** carryover rule: 10 days, effective 2026-01-01 |
| `hr-faq`                               | shared    | all                               | **Stale** carryover rule: 5 days (FAQ v1.4, 2025-06-10). Conflicting-versions exercises |
| `hr-remote-work-policy`, `hr-expense-policy`, `hr-parental-leave-policy`, `hr-code-of-conduct`, `hr-travel-policy` | shared | all | Policy QA with definite numbers (30 days, 65/95 USD per diem, 220/300 USD hotel caps...) |
| `it-vpn-access-runbook`, `it-laptop-replacement-runbook`, `it-password-reset-runbook`, `it-faq` | shared | all | Runbooks with error codes (412, 809), SLAs |
| `it-incident-response-runbook`, `it-database-failover-runbook` | shared | it-oncall | **Restricted**: abstain exercises |
| `it-onboarding-accounts-runbook`       | shared    | it-oncall, hr                     | Multi-group ACL |
| `prod-retail-pos-overview`, `prod-retail-returns-api` | retail | all               | Tenant isolation: not visible from `logistics` |
| `prod-logistics-tracking-api`, `prod-logistics-route-planner`, `prod-warehouse-scanner-guide` | logistics | all | Tenant isolation: not visible from `retail` |
| `sec-data-classification-policy`       | shared    | all                               | Classification levels, AI-tool rules |
| `sec-access-control-policy`            | shared    | security, it-oncall, managers     | Restricted policy |
| `inc-2025-11-pos-outage`               | retail    | it-oncall, managers               | Incident `INC-2025-1142`, exact-ID lookups |
| `inc-2026-02-tracking-latency`         | logistics | it-oncall, managers               | Incident `INC-2026-0217`, multi-hop with the tracking API doc |
| `ext-vendor-newsletter-brightline`     | logistics | all                               | **Adversarial**: indirect prompt injection in a paragraph and in an HTML comment. Index it deliberately in security chapters; the assistant must treat it as data |

Cross-references between documents use the `id` in backticks, so link-following retrievers can be
demonstrated. Facts are internally consistent across documents (for example the 02:00 nightly POS sync,
the 180-stop route limit and the 50/min webhook retry limit appear in the product doc, the incident
report and tickets).

## `manifest.json`

```json
{ "generated_at": "2026-10-04", "doc_count": 24,
  "docs": [ { "id": "...", "title": "...", "version": "...", "updated_at": "...", "owner": "...",
              "tenant": "...", "acl_groups": [...], "tags": [...],
              "path": "docs/<file>.md", "sha256": "<hex>", "bytes": 1234 } ] }
```

`sha256` is over the raw file bytes (front matter included). Use it to detect stale indexes.

## `tickets.jsonl`

One JSON object per line:

| Field            | Values                                                                 |
|------------------|------------------------------------------------------------------------|
| `id`             | `TCK-2026-NNNN`                                                        |
| `created_at`     | ISO-8601 UTC                                                           |
| `tenant`         | `retail` \| `logistics` \| `shared`                                    |
| `requester_role` | e.g. `employee`, `store_manager`, `dispatcher`, `contractor`, `new_hire` |
| `channel`        | `portal` \| `email` \| `chat` \| `phone`                               |
| `subject`, `body`| Body is 1 to 6 sentences; style varies from terse chat to formal email |
| `category`       | one of 12 (below)                                                      |
| `priority`       | `P1`..`P4`                                                             |
| `status`         | `open` (4 tickets, `resolution` is null) \| `closed` (56)              |
| `resolution`     | Free text for closed tickets                                           |

Categories (`shared_data.TICKET_CATEGORIES`): `account_access`, `vpn_network`, `hardware`,
`password_mfa`, `time_off`, `expenses_travel`, `benefits_leave`, `pos_payments`, `returns`,
`shipment_tracking`, `warehouse_scanner`, `security_report`.

Deliberate features: a handful of ambiguous tickets (scanner PIN vs password, RoutePilot issues filed
under `shipment_tracking`, "something is wrong with my access"), two tickets with PII-like strings for
redaction exercises (`TCK-2026-0043` has a fake phone number `+1-555-0142`; `TCK-2026-0058` has a fake
email at `example.com`), and tickets that reference documents and incidents in `docs/`.

## `invoices.jsonl`

Each line: `{id, tenant, format, text, expected, validation}`.

- `text` is a semi-structured blob in one of four layouts (`table`, `letter`, `statement`, `receipt`).
- `expected` is the gold extraction: `vendor`, `invoice_number`, `invoice_date`, `due_date`,
  `po_number` (nullable), `currency` (USD/EUR/GBP), `line_items[{description, quantity, unit_price,
  amount}]`, `subtotal`, `tax_rate`, `tax_amount`, `total`, `total_excluding_tax`.
  The gold mirrors **what the invoice states**, including its errors.
- `validation.is_consistent` is false for 3 invoices; `issues[].code` is one of `TOTAL_MISMATCH`
  (`INV-007`), `MISSING_PO` (`INV-009`), `LINE_SUM_MISMATCH` (`INV-015`). A validator over the
  extracted fields should reproduce these codes.

## `eval/retrieval_gold.jsonl`

| Field                | Meaning                                                                      |
|----------------------|------------------------------------------------------------------------------|
| `id`                 | `RQ-NNN`                                                                     |
| `question`           | Natural-language question                                                    |
| `required_doc_ids`   | Documents that must be retrieved to answer; for abstain questions, the documents that hold the answer but are forbidden |
| `acceptable_doc_ids` | Extra documents that are fine to retrieve (not penalised)                    |
| `answer_rubric`      | 1 to 3 facts a correct answer must state                                     |
| `user_groups`        | Groups the asking user holds (always includes `all` for employees)           |
| `tenant`             | Tenant the user works in                                                     |
| `tags`               | `exact-fact`, `exact-id`, `multi-hop`, `multi-fact`, `paraphrase`, `conflicting-versions`, `forbidden-doc`, `abstain`, `adversarial`, `injection` |

Three questions carry `forbidden-doc` + `abstain`: the answer exists only in a document the user cannot
see, and the expected behaviour is to abstain, not to answer. Two questions test the conflicting PTO
carryover versions. Several contain exact identifiers (`INC-2025-1142`, `RET-002`, `SH-201`, error 412)
that lexical search should find and pure dense retrieval may miss.

## `eval/classification_gold.jsonl`

`{id, category}` for all 60 tickets, mirroring `tickets.jsonl`. Kept separate so a classifier exercise
can hide the label column.

## Editing rules

1. Keep every fact consistent across documents; grep for the number or ID before changing it.
2. Keep each document between 400 and 1,200 words and under 50 KB (tests enforce both).
3. Rerun `tools/build_manifest.py` and the tests after any change.
4. Never add real names, companies, phone numbers or email domains other than `example`/`example.com`.
