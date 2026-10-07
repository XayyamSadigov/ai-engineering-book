# Exercises — Chapter 11 — Ingestion and Chunking

Solutions: `../solutions/ch11-solutions.md`


**Start here:** K2, K6, E2, P3, D1 (about 4 hours). The rest go deeper.

### Knowledge questions

**K1.** Name the three identity fields of a `Document` and explain, for each, a change that alters it while leaving the other two unchanged.

**K2.** Why are position and document version excluded from `ragkit`'s chunk id, while the chunker fingerprint and section path are included? Describe one bug each inclusion or exclusion prevents.

**K3.** A PDF page yields an empty string from text extraction. List three different causes, and explain why the parser flags the page rather than raising an error.

**K4.** With 128 MinHash values split into 32 bands of 4 rows, roughly where is the similarity threshold of the LSH S-curve, and what happens to the false-candidate rate compared with 16 bands of 8 rows?

**K5.** Explain why overlap is usually unnecessary with section-aware and parent-child chunking but useful with fixed-size windows.

**K6.** What does span integrity measure that recall@k does not, and why can recall@3 exceed span integrity?

**K7.** Name the three levels of contextual chunk header in increasing cost. For each, give a corpus where it is enough and one sign that you need the next level.

### Engineering questions

**E1.** Northwind wants to index 30,000 support tickets alongside its policies. Design the JSONL parser configuration (text fields, metadata fields, title, ACL source) and state which ticket questions should be answered by metadata filters, by retrieval, or by SQL against the ticket system.

**E2.** A team proposes semantic chunking for all documents "because it understands meaning." Write the evaluation plan you would require before approving it: corpus slices, gold questions, metrics, the baseline, and the cost figures you would report.

**E3.** The procurement team uploads 2,000 scanned contracts. Design the ingestion path: OCR placement, confidence handling, quality slicing, what is indexed while OCR is pending, and how the contracts' tables (payment schedules) are represented.

**E4.** Northwind's wiki is being migrated, and every page will get a new URL and a new internal id. Explain what happens to chunk ids, embeddings, cached answers, and evaluation labels, and design a migration that avoids re-embedding and keeps labels valid.

### Practical exercises

**P1.** (about 60 min) Add an `HtmlParser` option `content_selector` that, when set to an element id or class, keeps only content inside the matching element. Write tests with a page whose article sits inside `<div id="content">` and whose sidebar contains decoy text.

**P2.** (about 2 hours) Add `RecursiveChunker.for_python()`, a preset whose separators split Python source on class and function boundaries first, and a `CodeContextChunker` wrapper that prefixes each chunk's `embedding_text()` with the file path and enclosing class name. Test it on a 200-line module and assert no function is split when it fits the budget.

**P3.** (about 2 hours) Implement a `RowSentenceTableChunker` that serializes each table row as "Column: value." sentences, keeps non-table blocks as the section chunker does, and add it to the evaluation grid. Report its span integrity and recall on the table questions only.

**P4.** (about 90 min) Extend `chunk_size.py` with a `--slice` option that reports metrics per question tag (table, code, prose), add tags to the gold file, and write five new questions with evidence spans over `incident-2025-11-pos-outage.md`. Report which configuration wins each slice.

### Debugging exercises

**D1.** After a library upgrade, the nightly ingestion log shows that 94 percent of chunks were "added" and 94 percent "removed" across a corpus where only a dozen documents changed. Embedding spend for the night was 15 times normal. Retrieval quality is unchanged. Diagnose the likely causes in order of probability and name the field on the stored chunks that confirms each.

**D2.** Users report that the assistant answers "the table does not list a limit for internal tools" for the Returns API, although the table clearly lists it. The trace shows the retrieved chunk is `| Online store | 1,000 requests/minute |\n| Internal tools | 100 requests/minute |` with section path `["Retail Returns API v2 Reference", "Rate limits"]`. Explain what went wrong at ingestion and why the model's answer is reasonable given its input.

**D3.** A logistics employee receives an answer that quotes a paragraph from an HR document restricted to the `hr` group. The retriever's ACL filter is correct and tested. The load report from the last run lists one exact duplicate: `dropped_id: hr-comp-bands-public`, `kept_id: hr-comp-bands`. Reconstruct the chain of events and name the defect.
