# path: book/projects/examples/ch02/tokenizer_experiment.py
"""Tokenizer cost experiment: how many tokens does the same amount of "meaning" cost?

Chapter 2 claims that tokens are not words, that the tokenizer penalizes some inputs
(non-Latin scripts, numbers, identifiers, pretty-printed JSON), and that you must count
with the real tokenizer rather than with ``len(text) / 4``. This script makes those claims
measurable. It uses ``tiktoken`` when an encoding can be loaded (the BPE merge table is
downloaded once and cached) and falls back to a labeled heuristic otherwise, so it never
fails in an offline CI job; it only becomes less precise and says so.

Run:
    python tokenizer_experiment.py              # table over the built-in samples
    python tokenizer_experiment.py --pieces     # also print BPE pieces for short strings
    python tokenizer_experiment.py --encoding cl100k_base
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Callable

DEFAULT_ENCODING = "o200k_base"

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

_Encoder = Callable[[str], list[int]]
_ENCODER_CACHE: dict[str, _Encoder | None] = {}


def _load_tiktoken(encoding_name: str) -> _Encoder | None:
    """Return a ``str -> token ids`` callable, or None if tiktoken is unavailable.

    Any failure (package missing, no network for the merge table, bad name) means
    fallback. The caller reports which path was used.
    """
    if encoding_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoding_name]
    encoder: _Encoder | None
    try:
        import tiktoken  # type: ignore[import-not-found]

        encoder = tiktoken.get_encoding(encoding_name).encode
    except Exception:  # noqa: BLE001 - deliberately broad
        encoder = None
    _ENCODER_CACHE[encoding_name] = encoder
    return encoder


def heuristic_token_count(text: str) -> int:
    """Rough estimate: ~4 ASCII chars, ~4 non-ASCII UTF-8 bytes, or ~2 digits per token.

    Loosely calibrated against public BPE vocabularies. For budgeting, not billing.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128 and not ch.isdigit())
    digit_chars = sum(1 for ch in text if ch.isdigit())
    other_bytes = sum(len(ch.encode("utf-8")) for ch in text if ord(ch) >= 128)
    estimate = ascii_chars / 4.0 + digit_chars / 2.0 + other_bytes / 4.0
    return max(1, math.ceil(estimate))


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    method: str  # "tiktoken:<encoding>" or "heuristic"

    @property
    def exact(self) -> bool:
        return self.method.startswith("tiktoken:")


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> TokenCount:
    """Count tokens with the real tokenizer if possible, else with the heuristic."""
    encoder = _load_tiktoken(encoding_name)
    if encoder is None:
        return TokenCount(heuristic_token_count(text), "heuristic")
    return TokenCount(len(encoder(text)), f"tiktoken:{encoding_name}")


def token_pieces(text: str, encoding_name: str = DEFAULT_ENCODING) -> list[str] | None:
    """Return the decoded BPE pieces for ``text``, or None without a real tokenizer.

    Useful for seeing *where* a tokenizer cuts: ``"unbelievable"`` may be one piece,
    ``"  indented"`` may put the leading spaces in their own piece, and a 10-digit
    number is typically several pieces.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding(encoding_name)
    except Exception:  # noqa: BLE001
        return None
    ids = enc.encode(text)
    return [enc.decode([i]) for i in ids]


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    label: str
    text: str


def _ticket_json(indent: int | None) -> str:
    ticket = {
        "ticket_id": "NW-2026-004417",
        "tenant": "retail",
        "priority": "P2",
        "created_at": "2026-03-14T09:12:44Z",
        "requester": {"employee_id": "E-104233", "department": "store-ops"},
        "subject": "POS terminal 7 freezes after refund",
        "tags": ["pos", "refund", "hardware"],
    }
    return json.dumps(ticket, indent=indent, separators=None if indent else (",", ":"))


ENGLISH = (
    "Employees may carry over up to five unused vacation days into the next calendar year. "
    "Days beyond that limit are forfeited unless a manager approves an exception in writing."
)

# The same policy sentence, translated. Same information, different script.
RUSSIAN = (
    "Сотрудники могут перенести на следующий календарный год до пяти неиспользованных дней отпуска. "
    "Дни сверх этого лимита сгорают, если руководитель письменно не одобрит исключение."
)

AZERBAIJANI = (
    "İşçilər istifadə olunmamış beş günə qədər məzuniyyəti növbəti təqvim ilinə keçirə bilərlər. "
    "Bu limitdən artıq günlər menecer yazılı istisna təsdiqləməsə, itirilir."
)

JAPANESE = (
    "従業員は未使用の休暇を最大5日まで翌暦年に繰り越すことができます。"
    "上限を超える日数は、管理者が書面で例外を承認しない限り失効します。"
)

PYTHON_CODE = '''def carry_over_days(unused: int, approved_exception: bool) -> int:
    """Return how many unused vacation days survive the year boundary."""
    if approved_exception:
        return unused
    return min(unused, 5)
'''

NUMBERS = "Invoice totals: 1284.50, 99731.07, 4017.00, 23.99, 158200.45, 7.25, 61044.10, 3.00"

IDENTIFIERS = (
    "req 3f9a1c2e-7b4d-4e0a-9c1f-5d2b8a6e4f10 trace 7c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f "
    "sha 9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
)

SAMPLES: list[Sample] = [
    Sample("English prose", ENGLISH),
    Sample("Russian prose (same text)", RUSSIAN),
    Sample("Azerbaijani prose (same text)", AZERBAIJANI),
    Sample("Japanese prose (same text)", JAPANESE),
    Sample("Python function", PYTHON_CODE),
    Sample("JSON, compact", _ticket_json(indent=None)),
    Sample("JSON, indent=2", _ticket_json(indent=2)),
    Sample("Decimal numbers", NUMBERS),
    Sample("UUIDs and hashes", IDENTIFIERS),
]

PIECE_DEMOS: list[str] = [
    "unbelievable",
    "Northwind Assist",
    "отпуск",
    "1234567890",
    "2026-03-14T09:12:44Z",
    "        return min(unused, 5)",
    "{\"ticket_id\": \"NW-2026-004417\"}",
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    label: str
    chars: int
    utf8_bytes: int
    words: int
    tokens: int
    method: str

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens if self.tokens else float("nan")

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words if self.words else float("nan")


_WORD_RE = re.compile(r"\S+")


def measure(samples: list[Sample], encoding_name: str = DEFAULT_ENCODING) -> list[Row]:
    rows: list[Row] = []
    for s in samples:
        tc = count_tokens(s.text, encoding_name)
        rows.append(
            Row(
                label=s.label,
                chars=len(s.text),
                utf8_bytes=len(s.text.encode("utf-8")),
                words=len(_WORD_RE.findall(s.text)),
                tokens=tc.tokens,
                method=tc.method,
            )
        )
    return rows


def format_table(rows: list[Row]) -> str:
    header = f"{'sample':<30} {'chars':>6} {'bytes':>6} {'words':>6} {'tokens':>7} {'chars/tok':>9} {'tok/word':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.label:<30} {r.chars:>6} {r.utf8_bytes:>6} {r.words:>6} {r.tokens:>7} "
            f"{r.chars_per_token:>9.2f} {r.tokens_per_word:>8.2f}"
        )
    methods = sorted({r.method for r in rows})
    lines.append("")
    lines.append("counting method: " + ", ".join(methods))
    if any(not m.startswith("tiktoken:") for m in methods):
        lines.append("note: heuristic counts are budgeting estimates, not billing numbers")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="tiktoken encoding name")
    parser.add_argument("--pieces", action="store_true", help="also show BPE pieces for short strings")
    args = parser.parse_args(argv)

    print(format_table(measure(SAMPLES, args.encoding)))

    if args.pieces:
        print()
        for demo in PIECE_DEMOS:
            pieces = token_pieces(demo, args.encoding)
            if pieces is None:
                print(f"{demo!r}: pieces unavailable without tiktoken (heuristic={heuristic_token_count(demo)})")
            else:
                shown = " | ".join(p.replace(" ", "␣") for p in pieces)
                print(f"{demo!r} -> {len(pieces)} tokens: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
