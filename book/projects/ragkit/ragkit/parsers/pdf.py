# path: book/projects/ragkit/ragkit/parsers/pdf.py
"""PDF parser over pypdf's text layer, with an OCR seam and per-page quality flags.

A PDF page has a text layer only if the producer wrote one. Scans and image-only exports have
none, and pypdf returns an empty string for them. This parser never guesses: each page gets a
`PageInfo` with the extracted character count and `needs_ocr=True` when the layer is thinner
than `min_chars_per_page`. If an `OcrEngine` is supplied, those pages are sent to it and the
result is marked with its confidence; otherwise the document reports `needs_ocr` and the
ingestion pipeline decides (queue for OCR, index partially, or reject).

What pypdf does not give you: reading order for multi-column layouts, table cell boundaries,
or heading levels. Layout-aware extractors exist for that; plug one in behind the same
`Parser` protocol when your corpus needs it.
"""
from __future__ import annotations

import io
import re
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..documents import Document, PageInfo, SourceType
from ..normalize import dehyphenate, strip_repeated_lines
from .base import DocDefaults, DocumentBuilder, ParseError


class OcrResult(BaseModel):
    text: str
    confidence: float  # 0..1 as reported by the engine; engines differ in calibration


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def ocr_page(self, pdf_bytes: bytes, page_number: int) -> OcrResult: ...


_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)


class PdfParser:
    source_type = SourceType.PDF
    version = "pdf/1"

    def __init__(
        self,
        *,
        min_chars_per_page: int | None = None,
        ocr: OcrEngine | None = None,
        remove_repeated_lines: bool = True,
    ) -> None:
        if min_chars_per_page is None:
            from ..settings import RagkitSettings

            min_chars_per_page = RagkitSettings().pdf_min_chars_per_page
        self.min_chars_per_page = min_chars_per_page
        self.ocr = ocr
        self.remove_repeated_lines = remove_repeated_lines

    def parse(self, raw: bytes | str, *, source_uri: str, defaults: DocDefaults | None = None) -> list[Document]:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        if isinstance(raw, str):
            raise ParseError("PDF input must be bytes", source_uri=source_uri)
        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ParseError("encrypted PDF needs a password", source_uri=source_uri)
            page_texts = [(p.extract_text() or "") for p in reader.pages]
        except PdfReadError as exc:
            raise ParseError(f"unreadable PDF: {exc}", source_uri=source_uri) from exc

        pages: list[PageInfo] = []
        page_lines: list[list[str]] = []
        for number, text in enumerate(page_texts, start=1):
            chars = len(text.strip())
            info = PageInfo(number=number, char_count=chars, needs_ocr=chars < self.min_chars_per_page)
            if info.needs_ocr and self.ocr is not None:
                result = self.ocr.ocr_page(raw, number)
                text = result.text
                info.ocr_applied = True
                info.ocr_confidence = result.confidence
            elif info.needs_ocr:
                text = ""
            pages.append(info)
            page_lines.append(text.replace("\r\n", "\n").split("\n"))

        if self.remove_repeated_lines and len(page_lines) >= 3:
            page_lines = strip_repeated_lines(page_lines)

        builder = DocumentBuilder()
        for info, lines in zip(pages, page_lines):
            builder.page = info.number
            lines = [ln for ln in lines if not _PAGE_NUMBER_LINE.match(ln)]
            for para in _paragraphs("\n".join(lines)):
                meta = {"ocr": True, "ocr_confidence": info.ocr_confidence} if info.ocr_applied else {}
                builder.paragraph(para, **meta)

        info_meta = reader.metadata or {}
        declared_title = str(info_meta.get("/Title") or "").strip()
        meta: dict[str, object] = {"page_count": len(pages)}
        missing = [p.number for p in pages if p.needs_ocr and not p.ocr_applied]
        if missing:
            meta["pages_needing_ocr"] = missing
        if any(p.ocr_applied for p in pages):
            meta["ocr_engine"] = getattr(self.ocr, "name", "unknown")
        doc = builder.build(
            source_uri=source_uri,
            source_type=self.source_type,
            parser=self.version,
            defaults=defaults,
            declared={"title": declared_title} if declared_title else {},
            pages=pages,
            fallback_title=PurePosixPath(source_uri).stem,
            metadata=meta,
        )
        return [doc]


def _paragraphs(text: str) -> list[str]:
    """Blank lines separate paragraphs; single newlines inside a paragraph are line wraps."""
    text = dehyphenate(text)
    out = []
    for block in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        if joined:
            out.append(joined)
    return out


__all__ = ["OcrEngine", "OcrResult", "PdfParser"]
