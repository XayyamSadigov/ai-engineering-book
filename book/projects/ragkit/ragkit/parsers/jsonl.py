# path: book/projects/ragkit/ragkit/parsers/jsonl.py
"""JSONL records (tickets, FAQ entries, CRM notes) to one Document per record.

Structured sources are different from prose: some fields are content worth embedding (subject,
body, resolution) and some are facts worth filtering on (category, priority, status, dates).
The parser renders the text fields as labeled paragraphs and keeps the rest as metadata, so
"P1 tickets about payments" becomes a filter plus a semantic query instead of a hope that the
embedding captured the priority.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from ..documents import Document, SourceType, sha256_text
from .base import DocDefaults, DocumentBuilder, ParseError, decode


class JsonlParser:
    source_type = SourceType.JSONL
    version = "jsonl/1"

    def __init__(
        self,
        *,
        id_field: str = "id",
        text_fields: Sequence[str] = ("text",),
        title_field: str | None = None,
        tenant_field: str | None = "tenant",
        acl_field: str | None = "acl_groups",
        version_field: str | None = None,
        updated_at_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,  # None: every scalar field not rendered as text
        label_fields: bool = True,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        self.id_field = id_field
        self.text_fields = list(text_fields)
        self.title_field = title_field
        self.tenant_field = tenant_field
        self.acl_field = acl_field
        self.version_field = version_field
        self.updated_at_field = updated_at_field
        self.metadata_fields = list(metadata_fields) if metadata_fields is not None else None
        self.label_fields = label_fields
        self.on_error = on_error
        self.errors: list[str] = []

    def parse(self, raw: bytes | str, *, source_uri: str, defaults: DocDefaults | None = None) -> list[Document]:
        self.errors = []
        docs: list[Document] = []
        seen: set[str] = set()
        for n, line in enumerate(decode(raw).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
                doc = self._record_to_document(record, source_uri, defaults)
                if doc.id in seen:
                    raise ValueError(f"duplicate id {doc.id!r}")
                seen.add(doc.id)
                docs.append(doc)
            except (ValueError, ParseError) as exc:
                message = f"line {n}: {exc}"
                if self.on_error == "raise":
                    raise ParseError(message, source_uri=source_uri) from exc
                self.errors.append(message)
        return docs

    def _record_to_document(self, record: dict[str, Any], source_uri: str, defaults: DocDefaults | None) -> Document:
        rid = record.get(self.id_field)
        if rid in (None, ""):
            raise ValueError(f"missing id field {self.id_field!r}")
        builder = DocumentBuilder()
        rendered = 0
        for field in self.text_fields:
            value = record.get(field)
            if value in (None, ""):
                continue
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            label = field.replace("_", " ").capitalize()
            builder.paragraph(f"{label}: {text}" if self.label_fields else text, field=field)
            rendered += 1
        if rendered == 0:
            raise ValueError(f"none of the text fields {self.text_fields} has content")

        rendered_set = set(self.text_fields)
        if self.metadata_fields is None:
            meta = {k: v for k, v in record.items() if k not in rendered_set and isinstance(v, (str, int, float, bool))}
        else:
            meta = {k: record[k] for k in self.metadata_fields if k in record}
        declared: dict[str, Any] = {"id": str(rid)}
        if self.title_field and record.get(self.title_field):
            declared["title"] = str(record[self.title_field])
        if self.tenant_field and record.get(self.tenant_field):
            declared["tenant"] = record[self.tenant_field]
        if self.acl_field and record.get(self.acl_field):
            declared["acl_groups"] = record[self.acl_field]
        if self.updated_at_field and record.get(self.updated_at_field):
            declared["updated_at"] = str(record[self.updated_at_field])
        if self.version_field and record.get(self.version_field) is not None:
            declared["version"] = str(record[self.version_field])
        else:  # content-derived version: changes exactly when the record changes
            declared["version"] = sha256_text(json.dumps(record, sort_keys=True, ensure_ascii=False))[:12]
        for key in ("id", self.tenant_field, self.acl_field):
            meta.pop(key or "", None)
        return builder.build(
            source_uri=f"{source_uri}#{rid}",
            source_type=self.source_type,
            parser=self.version,
            defaults=defaults,
            declared=declared,
            fallback_title=str(rid),
            metadata=meta,
        )


__all__ = ["JsonlParser"]
