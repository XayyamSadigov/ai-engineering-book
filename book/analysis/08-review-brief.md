# Review brief (four-lens review, then fix)

Project root: the repository root. Python: .venv/bin/python.
Read first: book/analysis/05-authoring-guide.md (conventions), 04-table-of-contents.md (briefs and
ownership), 02-competency-map.md (target competencies and mental models), 07-integration-notes.md
(package APIs and known issues; lines starting "REVIEW TODO" may be in your scope).

You review a GROUP of chapters. For each chapter read the whole chapter, its solutions file, and skim
its code and tests. Review it in four roles, one after another:

1. **Principal AI Engineer**: Would someone who understands this chapter be able to build serious
   production AI applications with it? Look for: missing must-cover items from the TOC brief;
   concepts that are named but not taught (what/why/how/when/when-not/alternatives/trade-offs/
   implementation/failure/testing/production); incorrect or outdated technical claims; hand-waving
   where a concrete number, mechanism, or code is needed; vendor facts stated as truth; claims
   dated after mid-2026.
2. **Senior Software Architect**: weak architecture, layering violations, scalability issues,
   reliability gaps (timeouts, retries, idempotency), security issues (authz in the model, secrets,
   injection paths, tenant leakage), maintainability problems in the code (god modules, hidden globals,
   untestable code), code in the chapter that differs from code on disk.
3. **AI Educator**: bad learning order inside the chapter, prerequisites used before they are explained
   (check against earlier chapters), abrupt jumps, shallow sections, repetition of material owned by
   another chapter (replace with a cross-reference), missing or weak exercises (need K/E/P/D categories,
   debugging exercises that present a broken system/trace), answers leaking into the Exercises section,
   solutions that do not match exercise ids, broken cross-references (chapter numbers/titles must match
   the TOC), mental-model callouts that are decorative.
4. **Production/SRE Engineer**: missing observability (what to trace/measure/alert), deployment gaps,
   cost control, failure recovery and degraded modes, operational runbook gaps, capacity reasoning.

Then FIX what you found, directly in the files:
- Edit chapters, solutions, and code. Keep code listings in chapters identical to code on disk (if you
  change code, re-paste the listing). Keep public APIs stable unless broken; if you must change a
  shared package API, do it backward compatibly and run every dependent test suite.
- Prefer substantive improvements (add a missing mechanism, a worked number, a failure-mode table,
  an observability subsection, a debugging exercise) over cosmetic edits. Do not inflate length with
  filler; cut repetition.
- Fix cross-references, terminology consistency (use names from the TOC and the packages), style rules
  (no em dashes outside chapter titles; numbers labeled illustrative; vendor-neutral).
- After edits, run the tests of every code directory you touched (from repo root:
  `.venv/bin/python -m pytest <dir> -q -p no:cacheprovider`) and `.venv/bin/python book/tools/build_book.py --check`.

Then REVIEW AGAIN (second pass, all four roles, quicker) and fix anything remaining.

Write your gap list and what you changed to book/analysis/reviews/<group-id>.md with sections:
"Gaps found (by reviewer role)", "Changes applied", "Second-pass findings", "Remaining open items"
(only items you could not fix, with reason). Reply with a short summary and test results.
