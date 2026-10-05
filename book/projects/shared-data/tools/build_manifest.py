# path: book/projects/shared-data/tools/build_manifest.py
"""Rebuild manifest.json from docs/*.md. Run after editing any document:

    python book/projects/shared-data/tools/build_manifest.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from shared_data import DATA_DIR, DOCS_DIR, parse_front_matter, sha256_of, DocMeta  # noqa: E402


def build() -> dict:
    entries = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        validated = DocMeta(**meta)
        entry = json.loads(validated.model_dump_json())
        entry["path"] = f"docs/{path.name}"
        entry["sha256"] = sha256_of(path)
        entry["bytes"] = path.stat().st_size
        entries.append(entry)
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate doc ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return {"generated_at": date.today().isoformat(), "doc_count": len(entries), "docs": entries}


if __name__ == "__main__":
    manifest = build()
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} with {manifest['doc_count']} docs")
