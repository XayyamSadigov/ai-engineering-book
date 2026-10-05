# ragkit

Ingestion and chunking for the RAG part of the book. Chapter 11 builds and explains it; Chapters 12-15
and Project 3 import it. It turns sources (Markdown, HTML, PDF, plain text and transcripts, JSONL
records) into `Document`s with identity, permissions, and structure, cleans and deduplicates them, and
cuts them into `Chunk`s with six strategies that share one identity and metadata scheme.

## Layout

```
ragkit/
  pyproject.toml          .env.example          README.md
  eval_data/chunk_eval_gold.jsonl     # evidence-span questions over shared-data/docs
  ragkit/
    __init__.py           # re-exports the public API below
    documents.py          # Document, Chunk, Block, PageInfo, SourceType, hashing helpers
    tokenizers.py         # Tokenizer protocol, RegexTokenizer (default), TiktokenTokenizer
    settings.py           # RagkitSettings (RAGKIT_* env vars)
    normalize.py          # unicode/whitespace/boilerplate cleanup, MinHash + LSH dedup
    pipeline.py           # load_documents, chunk_documents, diff_chunks
    parsers/              # base (Parser, DocDefaults, DocumentBuilder), markdown, html, pdf, text, jsonl
    chunking/             # base, fixed, recursive, sentence(s), section, semantic, parent_child
    eval/chunk_size.py    # chunk-size and strategy comparison (CLI: ragkit-chunk-eval)
  tests/                  # offline; PDFs are generated in memory by tests/pdf_fixtures.py
```

## Install and test

```bash
uv pip install -e ../aie_core -e .        # or: pip install -e ../aie_core && pip install -e .
pytest -q
python -m ragkit.eval.chunk_size --docs ../shared-data/docs --gold eval_data/chunk_eval_gold.jsonl --k 3
```

No network or API keys are needed. `--embeddings fake` adds the semantic chunker with the hashing fake;
`--embeddings settings` uses the embedding client configured through `aie_core` settings.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RAGKIT_TOKENIZER` | `regex` | `regex` (deterministic, offline) or `tiktoken` (needs its encoding file) |
| `RAGKIT_TIKTOKEN_ENCODING` | `cl100k_base` | encoding for `TiktokenTokenizer` |
| `RAGKIT_PDF_MIN_CHARS_PER_PAGE` | `20` | a page with a thinner text layer is flagged `needs_ocr` |
| `RAGKIT_REQUIRE_ACL` | `true` | `load_documents` rejects documents without tenant and ACL groups |
| `RAGKIT_NEAR_DUPLICATE_THRESHOLD` | `0.9` | estimated Jaccard similarity above which a document is a near duplicate |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `fake` | read by `aie_core`; used by `SemanticChunker` via `--embeddings settings` |

## Public API

```python
from ragkit import (
    # model
    Document, Chunk, Block, PageInfo, SourceType,
    # parsing
    MarkdownParser, HtmlParser, PdfParser, TextParser, JsonlParser, DocDefaults, ParseError,
    OcrEngine, OcrResult, parse_file,
    # cleaning and dedup
    normalize_document, dedupe_documents, NearDuplicateIndex,
    # chunking
    Chunker, BaseChunker, FixedTokenChunker, RecursiveChunker, SentenceChunker,
    MarkdownSectionChunker, SemanticChunker, ParentChildChunker, split_roles, expand_to_parents,
    # pipeline
    load_documents, chunk_documents, diff_chunks, LoadReport, ChunkDiff,
    # tokens
    Tokenizer, RegexTokenizer,
)
```

Key contracts:

- `Chunk.id` is `"{doc_id}:{hash}"` over chunker fingerprint, section path, content hash, occurrence,
  and parent id. It is deterministic, does not change when unrelated text moves, and differs between
  chunker configurations.
- `Chunk.text` is exact source text (or a table part with its header repeated); `Chunk.embedding_text()`
  prefixes the title and section breadcrumb for embedding and lexical indexing.
- `Chunk.tenant` and `Chunk.acl_groups` are copied from the document. Filter on them at retrieval time.
- Parent-child output contains both roles; index `role != "parent"` and resolve hits with
  `expand_to_parents`.

## Retrieval (Chapter 12)

`ragkit.retrieval` holds the retrieval funnel. The shared contract is `ragkit/retrieval/types.py`
(Principal, RetrievalQuery, ScoredChunk, RetrievalResult, Retriever, Reranker, visible).

```
ragkit/retrieval/
  common.py      # metadata filters (fail loudly on unknown keys), indexed_text, Stopwatch
  bm25.py        # BM25Tokenizer, BM25Index: from scratch, ACL-before-scoring, JSON persistence
  dense.py       # DenseRetriever: aie_core embeddings in a semsearch VectorStore namespace
  hybrid.py      # reciprocal_rank_fusion, weighted_score_fusion (min-max), HybridRetriever
  rerank.py      # LexicalOverlapReranker, CrossEncoderReranker (lazy, with fallback), LLMReranker
  query.py       # QueryPlan, QueryRewriter, MultiQueryExpander, QueryDecomposer, HyDEGenerator, ChainedTransformer
  parent.py      # ParentDocumentRetriever, parent_child_index
  contextual.py  # ContextualEnricher (LLM context prefix per chunk, cached by content hash), JsonFileCache
  pipeline.py    # RetrievalPipeline (transform -> retrievers in parallel -> fusion -> rerank -> ACL check), stage_candidates
  settings.py    # RetrievalSettings (RAGKIT_RETRIEVAL_* variables)
  sql/lexical_tsvector.sql   # the PostgreSQL full-text variant (not run offline)
ragkit/eval/compare_retrievers.py   # bm25 / dense / hybrid / hybrid+rerank on shared-data gold
```

```bash
pip install -e ../p2-semantic-search          # DenseRetriever stores vectors in semsearch
python -m ragkit.eval.compare_retrievers --by-tag
pytest -q tests/test_retrieval_*.py
```

| Variable | Default | Meaning |
|---|---|---|
| `RAGKIT_RETRIEVAL_CANDIDATE_K` | `50` | hits requested from each first-stage retriever per search text |
| `RAGKIT_RETRIEVAL_RERANK_K` | `20` | fused candidates handed to the reranker |
| `RAGKIT_RETRIEVAL_FINAL_K` | `8` | hits returned to the generator |
| `RAGKIT_RETRIEVAL_FUSION` | `rrf` | `rrf` or `weighted` |
| `RAGKIT_RETRIEVAL_RRF_K` | `60` | RRF damping constant |
| `RAGKIT_RETRIEVAL_BM25_K1` / `_BM25_B` | `1.2` / `0.75` | BM25 term saturation and length normalization |
| `RAGKIT_RETRIEVAL_CROSS_ENCODER_MODEL` | unset | local cross-encoder name or path; unset means the lexical fallback |
| `RAGKIT_RETRIEVAL_RERANK_BATCH_SIZE` | `8` | candidates per LLM reranking call |
| `RAGKIT_RETRIEVAL_PARALLEL` | `true` | run first-stage retrievers concurrently |

`sentence-transformers` is optional (`pip install -e '.[rerank]'`); without it `CrossEncoderReranker`
logs one warning and delegates to `LexicalOverlapReranker`.

## RAG evaluation (Chapter 14)

```
ragkit/eval/
  rag_dataset.py      # RagExpectation, RagInput, RagOutput, from_grounded_qa, load_gold_dataset, synthesize_questions
  rag_metrics.py      # hit/recall/precision@k, MRR, nDCG (required=2, acceptable=1), leak_report, citations, abstention
  rag_judges.py       # FaithfulnessJudge (claims + support), RubricCoverageJudge, ContextRelevanceJudge, evalkit rubric judges
  stage_isolation.py  # FailureStage, diagnose, diagnose_run, stage_counts, stage_shift
  rag_report.py       # render_rag_report (leaks first, paired deltas, stage table, slices)
  run_rag_eval.py     # offline demo system, PRESETS, DEFAULT_GATE, compare_configs, CLI
```

Depends on `evalkit` (path dependency). Everything runs offline; `--judges llm` uses
`aie_core.make_llm_client()` and the usual `LLM_PROVIDER` / `LLM_MODEL` / API-key variables.

```bash
python -m ragkit.eval.run_rag_eval --out out/rag_eval                       # pack1 vs rerank+pack4, gate passes
python -m ragkit.eval.run_rag_eval --candidate lexical-k5-pack4-noacl       # leaks, gate fails, exit 1
pytest -q tests/test_rag_eval_*.py
```

Any RAG system is evaluated by passing a `(question, principal) -> RagOutput` callable to
`make_target` / `evaluate_system`. Stage isolation reads `RetrievalResult.trace["stages"]` in the
layout `RetrievalPipeline` writes (`kind` = retrieve/fusion/rerank, `candidate_ids`).
