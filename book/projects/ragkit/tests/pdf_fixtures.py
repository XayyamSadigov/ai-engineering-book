# path: book/projects/ragkit/tests/pdf_fixtures.py
"""Build tiny, valid PDFs in memory so PDF tests need no binary fixtures and no network."""
from __future__ import annotations


def _esc(text: str) -> bytes:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")


def make_pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    """One list of lines per page. An empty list makes an image-only-like page with no text layer."""
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    next_id = 4
    page_ids: list[int] = []
    for lines in pages:
        if lines:
            ops = b" ".join(b"(" + _esc(line) + b") Tj T*" for line in lines)
            stream = b"BT /F1 11 Tf 14 TL 72 740 Td " + ops + b" ET"
        else:
            stream = b""
        cid, pid = next_id, next_id + 1
        next_id += 2
        objs[cid] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objs[pid] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % cid
        )
        page_ids.append(pid)
    objs[2] = b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_ids) + b"] /Count %d >>" % len(page_ids)
    info_ref = b""
    if title:
        objs[next_id] = b"<< /Title (" + _esc(title) + b") >>"
        info_ref = b"/Info %d 0 R " % next_id
    out = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for i in sorted(objs):
        offsets[i] = len(out)
        out += b"%d 0 obj\n" % i + objs[i] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objs) + 1
    out += b"xref\n0 %d\n" % size + b"0000000000 65535 f \n"
    out += b"".join(b"%010d 00000 n \n" % offsets[i] for i in range(1, size))
    out += b"trailer\n<< /Size %d /Root 1 0 R " % size + info_ref + b">>\nstartxref\n%d\n" % xref_at + b"%%EOF\n"
    return out
