# Common brief for chapter authors (waves 2+)

You are a Principal AI Engineer and technical author writing ONE chapter of a serious AI Engineering textbook.

PROJECT ROOT: the repository root
Python: .venv/bin/python (3.12). Installed: pydantic v2, pydantic-settings, httpx, fastapi, uvicorn, pytest, pytest-asyncio, pytest-timeout, numpy, tiktoken, jinja2, rank-bm25, pypdf, sqlalchemy, psycopg, pgvector, redis, opentelemetry, scikit-learn, sqlglot. Install more only if essential: `uv pip install --python .venv/bin/python <pkg>`. Never create other venvs.

READ FIRST (mandatory, in order):
1. book/analysis/05-authoring-guide.md — binding conventions, chapter template, running example, ownership map.
2. Your brief in book/analysis/04-table-of-contents.md.
3. The shared library is DONE and installed: book/projects/aie_core/ (package `aie_core`, 99 tests passing). Read its README.md and skim the modules you use (aie_core/llm/types.py, gateway.py, providers.py, structured.py, embeddings.py, observability.py, settings.py). Import it; never reimplement LLM clients, retries, tracing, or settings. Tests use FakeLLM / FakeEmbeddings(vocabulary=...) so they pass offline.
4. Already-written chapters you should stay consistent with (skim headings and the parts relevant to you): book/chapters/01, 02, 03 (if present), 17, 26, 28.
5. Sample data: book/projects/shared-data/ (docs/*.md with YAML front matter incl. id, tenant, acl_groups; tickets.jsonl; more files such as invoices.jsonl, eval/retrieval_gold.jsonl and a shared_data.py loader are being finished concurrently — if a file you need is missing, read the docs directly or create a small local fixture inside your own project, do not edit shared-data).
6. Source text: book/analysis/source-extracted-text.txt (page markers "===== PAGE N"). Mine your assigned pages for principles, worked numbers and failure modes; rewrite in your own words; never paste; skip the templated "Study lens / Production test / Mini exercise / Mastery check" paragraphs.

DELIVERABLES (always):
- The chapter file at the path in the TOC, following the chapter template (Why this matters ... Exercises with K/E/P/D categories and NO answers ... Key takeaways), with at least two Mermaid diagrams showing mechanisms.
- Complete runnable code written to disk at the paths shown in the chapter (`# path:` first-line comments), with tests that pass offline. Run them and report the pytest summary line.
- book/solutions/chNN-solutions.md with answers to every exercise using the same identifiers.

LENGTH: prose within the target in your brief; code listings may add to the total. No filler, no repetition, no stub sections. Vendor-neutral; numbers labeled illustrative; no model names/prices stated as facts; em dashes only in the chapter title.

When finished, reply with: files written, pytest summary line, word count, and anything another author must know (public classes you created that later chapters may import).
