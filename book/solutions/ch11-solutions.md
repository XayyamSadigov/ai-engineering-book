# Chapter 11 — Solutions: Ingestion and Chunking

Identifiers match the exercises in `book/chapters/11-ingestion-and-chunking.md`. Code references are to `book/projects/ragkit/`.

## Knowledge questions

**K1.** The three identity fields are `id`, `version`, and `content_hash`. `id` names the logical document. It changes when the document is re-keyed, for example a wiki migration that assigns new ids, while the version and the text stay the same. `version` is the owner's declared edition. It changes in a metadata-only release that bumps "2.3" to "2.4" without touching the text, so the hash stays the same. `content_hash` is the hash of the normalized text. It changes when an editor fixes a typo without bumping the version, and also when the normalization configuration or the parser version changes, with id and version unchanged. That last case is why parser and normalization versions are recorded: a hash change with no edit points at the pipeline, not the author.

**K2.** Position is excluded because inserting a paragraph near the top would otherwise change the id of every later chunk. The bug it prevents is the re-embedding storm with orphaned feedback and cache entries. Version is excluded because a version bump that leaves a section untouched should not re-embed that section. The bug it prevents is a full re-index on every release. The chunker fingerprint is included so two configurations, say budgets of 300 and 400, can coexist in one index for an A/B test. Without it, the second run would overwrite the first run's chunks under the same ids, or worse, mix chunks from both configurations under one id space. The section path is included so the same sentence under two headings ("Applies to contractors." under both "Leave" and "Expenses") gets two ids with different section metadata. Without it, the second occurrence would collide with the first, and the occurrence counter would make its id depend on document order across sections.

**K3.** Causes: (1) the page is a scan or image with no text layer; (2) the text is drawn as vector outlines or uses a font without a usable character map, so glyphs cannot be mapped to characters; (3) the page really is blank or a full-page figure. Other causes include encryption that the parser handled incorrectly, or text in an annotation layer the extractor ignores. The parser flags the page instead of raising because the document is valid and other pages may be fine. Raising would lose them. Indexing the empty page silently would make the document invisible to retrieval. A flag lets the pipeline decide per policy: queue the page for OCR, index the readable pages, or hold the document. It also makes the scanned slice measurable.

**K4.** The threshold is roughly (1/b)^(1/r) = (1/32)^(1/4) ≈ 0.42. With more, shorter bands, a pair needs to agree on only 4 consecutive values in any of 32 bands to collide. Many more dissimilar pairs become candidates: at s = 0.5 the collision probability is 1 - (1 - 0.0625)^32 ≈ 0.87, compared with about 0.06 for 16 bands of 8 rows. The final threshold check on the full signature still rejects them, so correctness holds, but candidate verification work rises sharply. Recall for pairs near 0.9 is essentially 1 in both settings. The 32 by 4 setting suits a low duplicate threshold such as 0.5. It wastes work at 0.9.

**K5.** Overlap exists to keep a fact near an arbitrary boundary whole in at least one chunk. Fixed windows put boundaries at arbitrary places, so a sentence or a header-row pair is regularly cut, and overlap repairs part of that damage. The evaluation measured fixed-128 integrity rising from 0.75 to 1.00 with 16 tokens of overlap. Section-aware boundaries fall at headings and block edges, where authors already separated ideas, so there is little to repair. Parent-child chunking hands the generator the whole parent, so a child boundary does not limit the evidence. In both cases overlap mostly adds near-duplicate hits and index size.

**K6.** Span integrity asks whether all the evidence spans a question needs sit together inside one chunk, independent of any retriever. It is a property of the chunking alone. Recall@k asks whether the spans are all present somewhere in the top k results. Recall@3 can exceed span integrity because retrieval can return two adjacent chunks that each hold half of a split piece of evidence. The union then contains everything, while no single chunk does. That rescue depends on k and luck, and it costs extra context. It also disappears at k=1: fixed-64 dropped from 0.81 at k=3 to 0.38 at k=1.

**K7.** (1) Deterministic breadcrumb (title and section path). Enough for well-structured documentation with descriptive titles and headings, such as Northwind's runbooks and policies. Sign you need more: misses concentrate on documents with uninformative titles or generic headings ("Notes", "Details"), or on files whose title is a filename. (2) Document summary prepended to every chunk. Enough when the problem is that the document as a whole is unnamed (exported PDFs, scanned contracts) but sections are self-contained. Sign you need more: on a slice of terse, self-referential chunks ("this limit", "the steps above"), recall is still low after the summary was added, because the missing context is specific to the chunk, not the document. (3) Model-written per-chunk context (contextual retrieval, Chapter 12). Justified only when an evaluation slice shows a recall gap the first two levels did not close, and worth its cost (one model call per chunk, repeated when neighborhoods change, plus model-written text from untrusted documents in the index). In all three cases the header stays out of `Chunk.text`, and changing its format means a new index version.

## Engineering questions

**E1.** Use `JsonlParser(id_field="id", text_fields=["subject", "body", "resolution"], title_field="subject", tenant_field="tenant", acl_field=None, updated_at_field="created_at", metadata_fields=["category", "priority", "status", "channel", "requester_role", "created_at"])`. The ACL comes from an authoritative mapping rather than from the export. Pass `DocDefaults(acl_groups=["support"])`, or better, compute groups per ticket from the ticket system's permission model, for example adding `security` for `security_report` tickets. A ticket with an unknown tenant is rejected. Questions by mechanism:
- *Metadata filter plus retrieval:* "how did we fix register declines last time" (filter category `pos_payments`, status `closed`, semantic search on body and resolution).
- *Retrieval alone:* "has anyone seen a scanner pairing loop like this" (free-text similarity).
- *SQL against the ticket system:* counts, rates, and trends such as "how many P1 tickets this week" or "average time to close by category". Retrieval returns k examples. It cannot count.

The test of the configuration is a gold set of support-agent questions, each tagged with the mechanism that should answer it.

**E2.** The plan:
- **Slices.** Structured docs (policies, runbooks), unstructured text (transcripts, long emails, newsletters), and reference tables, at least 15 to 20 evidence-span questions per slice with tags.
- **Baselines.** The fixed-size baseline, and the current default (section-aware, plus recursive for text without headings).
- **Candidates.** Semantic at two or three percentiles and window sizes, with the production embedding model.
- **Metrics per slice.** Span integrity, recall@1 and recall@5, MRR, and context tokens at k, with bootstrap confidence intervals.
- **Cost figures.** Ingestion embedding tokens per corpus token, measured, not estimated. Ingestion wall time under the provider's rate limit. Re-chunk cost on an embedding model change.
- **Operational check.** Id stability under a typical edit, because semantic boundaries shift.
- **Decision rule.** Adopt semantic only for slices where it beats the default beyond the confidence interval, and only if the added ingestion cost fits the budget. Otherwise keep the cheaper strategy. Expect it to help at most on the unstructured slice.

**E3.** The path:
- **Separate OCR stage.** Parse each PDF; pages with `needs_ocr` go to an OCR queue served by dedicated workers, and the document is held with status `ocr_pending`.
- **Partial indexing.** Pages with a text layer are indexed only if the document's policy allows partial indexing. For contracts, prefer holding the whole document, so that retrieval never answers from page 1 while page 7 holds an amendment. Expose the pending status to users ("2,000 contracts are being processed").
- **Confidence handling.** Calibrate a per-engine floor on about 100 labeled pages. Pages below the floor go to human review or are indexed with a `low_ocr_confidence` flag that the generator's evidence label shows. Chunks carry `ocr` and `ocr_confidence_min` so retrieval metrics can be sliced.
- **Tables.** Detect payment-schedule tables with a layout-aware engine or a dedicated extraction step (Chapter 6 style schema: date, amount, currency, milestone). Store the extracted rows in a table queried by tool or SQL, and also index a rendered pipe table with repeated headers for retrieval and citation. Validate totals against the contract value.
- **Slice evaluation.** Track recall for the OCR slice separately and spot-check extraction accuracy.

**E4.** If ids change, every document id changes. Because chunk ids are prefixed with the document id, every chunk id changes too. The naive pipeline would embed everything again, orphan cached answers keyed on old chunk ids, and invalidate evaluation labels that reference old document ids.

The migration:
1. Build an old-to-new id mapping from the migration tool, or by matching content hashes, which survive the move because the text is unchanged.
2. Re-chunk with the same fingerprint.
3. For each new chunk, look up the old chunk with the same content hash, section path, occurrence, and mapped document, and reuse its embedding through `CachedEmbeddings`. Its key is model plus text, so identical text hits the cache even under a new id.
4. Write new chunk records and delete the old ones in one versioned index swap.
5. Rewrite gold labels and feedback through the mapping, and keep an alias table so old citations in stored conversations still resolve.
6. Invalidate cached answers whose evidence ids are in the mapping, or rewrite their keys.

Acceptance: zero embedding calls for unchanged text, unchanged retrieval metrics on the gold set, and no dangling ids in caches or labels.

## Practical exercises

**P1.** Add `content_selector: str | None` to `HtmlParser` (format `#id` or `.class`). In `_Extractor`, keep a `capture_depth` counter. Content is emitted only while `capture_depth > 0`, entered when a start tag's `id` or `class` list matches and tracked by tag nesting. The existing drop and hidden rules still apply inside the captured region. Title extraction from `<title>` stays global. Acceptance tests:
- With `<div id="content">` holding the article and an `<aside>`-free sidebar `<div class="sidebar">` holding decoy text, the selector `#content` yields only article blocks.
- The selector `.sidebar` yields only the decoy.
- An unmatched selector yields an empty document that the loader rejects with "no extractable text".
- Offsets still satisfy the block invariant.

**P2.** `RecursiveChunker.for_python(chunk_size)` returns a chunker with separators `("\nclass ", "\ndef ", "\n    def ", "\n\n", "\n", " ", "")`. Note that the separator stays with the left part, so a split before `def` keeps the keyword with the next piece only if the separator string is chosen as the boundary *before* the keyword. Either adjust `_split_span` to attach these separators to the right part, or use separators ending at the newline (`"\n"` lookahead) and test the boundary placement. `CodeContextChunker(inner)` delegates `split` and overrides `finalize` to add `metadata["file_path"]` and `metadata["enclosing_class"]` (the last `class X` line at or before the chunk start), and its `embedding_text` header becomes `path > Class`. Acceptance:
- On a 200-line module with functions under budget, every `def` block is wholly inside one chunk (parse each chunk with `ast` after dedenting and expect success, or check that each function's line range is contained).
- Methods carry the class name.
- The fingerprint differs from the plain recursive chunker.

**P3.** Subclass `MarkdownSectionChunker` and override `_pack` so that table blocks are replaced by one piece per row, with text `"; ".join(f"{h}: {v}" for h, v in zip(header, row)) + "."`, kind `table`, and `meta={"table_row": i}`. Span offsets cover the row's line. Non-table blocks are packed as before. Gold evidence spans like "Internal tools | 100 requests/minute" no longer match the serialized form, so evaluate with row-aware evidence. Either add alternative spans per question (`"Client type: Internal tools; Limit: 100 requests/minute"`), or compare after canonicalizing both forms to cell values. Report span integrity and recall on CE-02, CE-04, CE-05, CE-08, and CE-11. Expected outcome: recall@1 on lookup questions at least as good as section-300, with smaller context tokens. Questions needing two rows or a comparison get worse.

**P4.** Add `"tags": [...]` to `EvidenceQuestion` (default empty) and a `--slice` flag. `evaluate_chunker` returns per-tag aggregates alongside the overall `ConfigResult`, for example by running the loop per tag subset, and `format_markdown` prints one table per tag. Tag CE-02, 04, 05, 08, and 11 as `table`; CE-15 and 16 as `code`; the rest as `prose`. New questions over the POS outage incident should cover the timeline (a table), the root cause, the remediation action items, the detection gap, and the customer impact, each with verbatim evidence spans checked by a test that every span occurs in the normalized document. Acceptance:
- The script runs offline and the JSON output includes per-tag results.
- A unit test asserts that a table-only slice exists.
- The write-up names the winning configuration per slice, with the caveat that each question is worth 1/n of the slice.

## Debugging exercises

**D1.** All three causes change every chunk id or hash while text and retrieval stay the same. In order of probability:
1. **A default changed in the upgraded library.** A tokenizer name, a default budget, or `algorithm_version` changed, so the chunker fingerprint changed and every id changed. Confirm by comparing the `chunker` field on stored chunks before and after: two fingerprints.
2. **A normalization or parser version change** altered every content hash, for example a new boilerplate pattern or a different Unicode form. Confirm with `parser` and `metadata["normalized"]` on stored chunks, and by diffing the `content_hash` of a document whose source did not change.
3. **The tokenizer** switched (for example `RAGKIT_TOKENIZER` now resolves to tiktoken), which shifts every token-based boundary. Confirm with the tokenizer name inside the fingerprint config.

Retrieval quality is unchanged because the text barely changed. Fixes: pin chunker configuration explicitly in the ingestion config. Treat parser, normalization, and fingerprint changes as planned re-index events with a cost estimate. Let `CachedEmbeddings`, keyed on model plus text, absorb re-embedding when only ids changed. Alert when an ingestion run's added fraction exceeds a threshold relative to source changes.

**D2.** The table was split without its header row. The chunk holds two data rows and no `| Client type | Limit |` line, so the model sees "Internal tools" next to "100 requests/minute" but has no column label telling it that the number is a limit. The likely causes are a fixed-size or recursive chunker that cut at a line boundary inside the table, or the section chunker with `repeat_table_header=False` at a small budget. The section path shows "Rate limits", but section paths live in `embedding_text()`, and if the evidence packer passed only `text`, the model never saw that context either. The model's hedge is reasonable: values without labels are ambiguous, and a grounded model should not assume what the second column means. Fix: keep tables whole, or split them with headers repeated, and have the evidence packer include the breadcrumb (Chapter 13). Verify with the table questions' span integrity.

**D3.** Chain of events: HR published two documents with identical text. `hr-comp-bands` is restricted to `hr`. `hr-comp-bands-public` was intended for everyone, or the reverse; either way the two had different ACLs. A deduplication step that keyed on content hash alone, without the permission scope, treated them as duplicates and kept one of them. If it kept the restricted id but carried over the public copy's ACL, or merged metadata, the indexed chunks of `hr-comp-bands` ended up with a broader ACL than the source. The retriever correctly filtered by the chunk's ACL, which was wrong. The same chain applies if a stale `acl_groups` came from the dropped copy's record. The defect: deduplication crossed a permission scope. The fix is to key duplicates on tenant plus sorted ACL groups plus hash, as `dedupe_documents` does, and never let a duplicate merge copy permission fields. Confirm by comparing the indexed chunk's `acl_groups` with the source system's ACL for `hr-comp-bands`, and add a test that two copies with different ACLs are both kept.
