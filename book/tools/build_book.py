#!/usr/bin/env python3
"""Assemble the single-file book and the exercises directory from chapter files.

Usage:
    python book/tools/build_book.py            # writes book/AI_ENGINEERING_BOOK.md and book/exercises/*.md
    python book/tools/build_book.py --check    # only validate structure, exit 1 on problems
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]
CHAPTERS = BOOK / "chapters"
EXERCISES = BOOK / "exercises"
OUT = BOOK / "AI_ENGINEERING_BOOK.md"

PARTS: list[tuple[str, list[int]]] = [
    ("PART I — AI Engineering Foundations", [1, 2, 3]),
    ("PART II — LLM Application Development", [4, 5, 6, 7]),
    ("PART III — Embeddings and Retrieval", [8, 9]),
    ("PART IV — Production RAG", [10, 11, 12, 13, 14, 15]),
    ("PART V — Tools and Workflows", [16, 17, 18]),
    ("PART VI — Agents", [19, 20, 21, 22, 23]),
    ("PART VII — Evaluation", [24, 25]),
    ("PART VIII — Security and Guardrails", [26, 27]),
    ("PART IX — Production AI Engineering", [28, 29, 30, 31, 32, 33, 34]),
    ("PART X — AI System Design", [35, 36]),
    ("PART XI — Advanced Patterns", [37, 38]),
    ("PART XII — Capstone", [39]),
]

FRONT = [
    ("README.md", "About this book"),
    ("00-learning-roadmap.md", "Learning roadmap"),
]
BACK = [
    ("appendix-c-interview-preparation.md", "Appendix C — Interview preparation"),
    ("glossary.md", "Glossary"),
    ("references.md", "References"),
    ("coverage-matrix.md", "Coverage matrix"),
]

REQUIRED_SECTIONS = ["## Why this matters", "## Exercises", "## Key takeaways"]
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def chapter_file(n: int) -> Path:
    matches = sorted(CHAPTERS.glob(f"{n:02d}-*.md"))
    if not matches:
        raise FileNotFoundError(f"chapter {n:02d} missing in {CHAPTERS}")
    return matches[0]


def slugify(text: str) -> str:
    text = re.sub(r"[`*_]", "", text)
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", text).strip("-")


def shift_headings(md: str, by: int) -> str:
    """Demote headings so chapter H1 becomes H2 under part H1 in the combined file.

    Fenced code blocks are left untouched.
    """
    out: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m = HEADING_RE.match(line)
            if m:
                level = min(6, len(m.group(1)) + by)
                line = "#" * level + " " + m.group(2)
        out.append(line)
    return "\n".join(out) + "\n"


def check_chapter(path: Path, md: str) -> list[str]:
    problems: list[str] = []
    first = md.lstrip().splitlines()[0] if md.strip() else ""
    if not first.startswith("# Chapter"):
        problems.append(f"{path.name}: first line should be '# Chapter N — Title', got {first[:60]!r}")
    for sec in REQUIRED_SECTIONS:
        if sec not in md:
            problems.append(f"{path.name}: missing section {sec!r}")
    if sum(1 for ln in md.splitlines() if ln.lstrip().startswith("```")) % 2:
        problems.append(f"{path.name}: unbalanced code fences")
    words = len(re.findall(r"\w+", md))
    if words < 3500:
        problems.append(f"{path.name}: only {words} words")
    # answers must not be inline
    ex = md.split("## Exercises", 1)[-1].split("## Key takeaways", 1)[0]
    if re.search(r"(?im)^\s*(answer|solution)\s*:", ex):
        problems.append(f"{path.name}: exercises section appears to contain answers")
    return problems


def extract_exercises(n: int, md: str) -> str | None:
    if "## Exercises" not in md:
        return None
    title = md.lstrip().splitlines()[0].lstrip("# ").strip()
    body = md.split("## Exercises", 1)[1].split("## Key takeaways", 1)[0]
    return f"# Exercises — {title}\n\nSolutions: `../solutions/ch{n:02d}-solutions.md`\n{body.rstrip()}\n"


def main(check_only: bool) -> int:
    problems: list[str] = []
    toc: list[str] = ["## Table of contents", ""]
    parts_md: list[str] = []
    EXERCISES.mkdir(exist_ok=True)

    for part_title, numbers in PARTS:
        toc.append(f"- **{part_title}**")
        parts_md.append(f"\n\n# {part_title}\n")
        for n in numbers:
            try:
                path = chapter_file(n)
            except FileNotFoundError as e:
                problems.append(str(e))
                continue
            md = path.read_text(encoding="utf-8")
            problems += check_chapter(path, md)
            title = md.lstrip().splitlines()[0].lstrip("# ").strip()
            toc.append(f"  - [{title}](#{slugify(title)})")
            parts_md.append(shift_headings(md, 1))
            if not check_only:
                ex = extract_exercises(n, md)
                if ex:
                    (EXERCISES / f"ch{n:02d}-exercises.md").write_text(ex, encoding="utf-8")

    if problems:
        print("\n".join(problems))
    if check_only:
        return 1 if problems else 0

    front_md: list[str] = []
    for fname, label in FRONT:
        p = BOOK / fname
        if p.exists():
            toc.insert(2, f"- [{label}](#{slugify(label)})")
            front_md.append(f"\n\n# {label}\n" + shift_headings(p.read_text(encoding='utf-8'), 1))
    back_md: list[str] = []
    for fname, label in BACK:
        p = BOOK / fname
        if p.exists():
            toc.append(f"- [{label}](#{slugify(label)})")
            back_md.append(f"\n\n# {label}\n" + shift_headings(p.read_text(encoding='utf-8'), 1))

    sol_dir = BOOK / "solutions"
    sol_md: list[str] = []
    sol_files = sorted(sol_dir.glob("ch*-solutions.md"))
    if sol_files:
        toc.append(f"- [Solutions to exercises](#{slugify('Solutions to exercises')})")
        sol_md.append("\n\n# Solutions to exercises\n\nAnswers are collected here, separate from the "
                      "exercises, so you can attempt each exercise before reading its solution.\n")
        for f in sol_files:
            sol_md.append("\n" + shift_headings(f.read_text(encoding="utf-8"), 1))

    header = (
        "# AI Engineering: From Software Engineer to Production AI Engineer\n\n"
        "The complete book in one file. Chapters, projects, exercises and appendices are also available "
        "as separate files under `book/`.\n\n"
    )
    OUT.write_text(header + "\n".join(toc) + "\n" + "".join(front_md) + "".join(parts_md) + "".join(back_md) + "".join(sol_md),
                   encoding="utf-8")
    words = len(re.findall(r"\w+", OUT.read_text(encoding="utf-8")))
    print(f"wrote {OUT} ({words:,} words); problems: {len(problems)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check" in sys.argv))
