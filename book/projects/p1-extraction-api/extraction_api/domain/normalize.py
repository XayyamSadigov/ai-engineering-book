# path: book/projects/p1-extraction-api/extraction_api/domain/normalize.py
"""Deterministic post-processing: turn what the model wrote into canonical values.

Everything here is pure and boring on purpose. The model is good at *finding* a value in
messy text; code is better at *converting* it, because code can refuse an ambiguous input
instead of guessing. Each parser raises NormalizationError with a stable code that the
rules layer turns into a RuleViolation.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

CENT = Decimal("0.01")


class NormalizationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- text
def clean_text(value: str | None) -> str | None:
    """Collapse internal whitespace; empty strings become None."""
    if value is None:
        return None
    collapsed = " ".join(str(value).split())
    return collapsed or None


_NULL_TOKENS = {"", "n/a", "na", "none", "null", "not provided", "(not provided)", "-", "unknown"}


def clean_identifier(value: str | None) -> str | None:
    """Identifiers keep their case but lose surrounding noise and placeholder values."""
    text = clean_text(value)
    if text is None or text.casefold() in _NULL_TOKENS:
        return None
    return text.strip(" .,;:#")


# -------------------------------------------------------------------------- numbers
_CURRENCY_NOISE = re.compile(r"[\s$€£¥]|USD|EUR|GBP|CAD|US\$", re.IGNORECASE)


def _to_decimal(raw: str | float | int | Decimal) -> Decimal:
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, bool):
        raise NormalizationError("INVALID_NUMBER", f"boolean is not a number: {raw!r}")
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))
    text = _CURRENCY_NOISE.sub("", raw.strip())
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if "," in text and "." in text:
        # the right-most separator is the decimal separator: 1,234.50 or 1.234,50
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        head, _, tail = text.rpartition(",")
        # "1234,50" is a decimal comma; "1,234" or "12,400" is a thousands separator
        text = f"{head.replace(',', '')}.{tail}" if len(tail) == 2 else text.replace(",", "")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise NormalizationError("INVALID_NUMBER", f"cannot parse number from {raw!r}") from exc
    if not value.is_finite():
        raise NormalizationError("INVALID_NUMBER", f"non-finite number {raw!r}")
    return -value if negative else value


def parse_money(raw: str | float | int | Decimal | None) -> Decimal | None:
    if raw is None or (isinstance(raw, str) and clean_identifier(raw) is None):
        return None
    return _to_decimal(raw).quantize(CENT)


def parse_unit_price(raw: str | float | int | Decimal | None) -> Decimal | None:
    """Unit prices keep their printed precision (0.0045 must not become 0.00); the number
    of decimals also tells the rules layer how much rounding to tolerate."""
    if raw is None or (isinstance(raw, str) and clean_identifier(raw) is None):
        return None
    return _to_decimal(raw)


def parse_quantity(raw: str | float | int | Decimal | None) -> Decimal | None:
    if raw is None or (isinstance(raw, str) and clean_identifier(raw) is None):
        return None
    value = _to_decimal(raw)
    # 4.82e+07 becomes 48200000; 12.40 becomes 12.4
    return value.quantize(Decimal(1)) if value == value.to_integral_value() else value.normalize()


def parse_rate(raw: str | float | int | Decimal | None) -> Decimal | None:
    """'8%', 8, and 0.08 all mean eight percent. Rates above 1 are read as percentages."""
    if raw is None or (isinstance(raw, str) and clean_identifier(raw) is None):
        return None
    text = raw.strip().rstrip("%") if isinstance(raw, str) else raw
    value = _to_decimal(text)
    if (isinstance(raw, str) and raw.strip().endswith("%")) or value > 1:
        value = value / Decimal(100)
    if value < 0 or value > 1:
        raise NormalizationError("INVALID_RATE", f"tax rate out of range: {raw!r}")
    return value.quantize(Decimal("0.0001")).normalize()


# ---------------------------------------------------------------------------- dates
_UNAMBIGUOUS_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y", "%d-%b-%Y", "%Y%m%d",
)
_NUMERIC_DMY = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$")


def parse_date(raw: str | None) -> date | None:
    """Parse a date as written. Numeric day/month orders that could be read both ways
    (03/04/2026) raise AMBIGUOUS_DATE instead of guessing; a person resolves those."""
    text = clean_identifier(raw)
    if text is None:
        return None
    for fmt in _UNAMBIGUOUS_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    m = _NUMERIC_DMY.match(text)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 and b <= 12:
            return date(year, b, a)  # day first
        if b > 12 and a <= 12:
            return date(year, a, b)  # month first
        if a == b:
            return date(year, a, b)
        raise NormalizationError("AMBIGUOUS_DATE", f"cannot tell day from month in {raw!r}")
    raise NormalizationError("INVALID_DATE", f"unrecognized date {raw!r}")


# ------------------------------------------------------------------------- currency
_CURRENCY_ALIASES: dict[str, str] = {
    "usd": "USD", "us$": "USD", "us dollar": "USD", "us dollars": "USD", "dollars": "USD",
    "eur": "EUR", "€": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "£": "GBP", "pound": "GBP", "pounds": "GBP", "pounds sterling": "GBP",
    "cad": "CAD", "c$": "CAD", "canadian dollars": "CAD",
}


def parse_currency(raw: str | None, *, dollar_means: str = "USD") -> str | None:
    """Map symbols and names to ISO 4217 codes. A bare '$' is ambiguous worldwide, so it
    maps to a configured default rather than a hard-coded assumption."""
    text = clean_text(raw)
    if text is None:
        return None
    key = text.casefold()
    if key == "$":
        return dollar_means
    if key in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[key]
    if re.fullmatch(r"[A-Za-z]{3}", text):
        return text.upper()
    raise NormalizationError("UNKNOWN_CURRENCY", f"unrecognized currency {raw!r}")


# ------------------------------------------------------------------ evidence spans
def locate(quote: str | None, text: str) -> tuple[int, int] | None:
    """Find `quote` in `text`, tolerating whitespace and case differences. Returns offsets
    into the original text, or None. Models often re-flow whitespace when quoting tables;
    they should never change the characters themselves."""
    if not quote or not quote.strip():
        return None
    idx = text.find(quote)
    if idx != -1:
        return idx, idx + len(quote)
    # build a whitespace-collapsed, casefolded copy with a map back to original offsets
    norm_chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            norm_chars.append(" ")
            prev_space = True
        else:
            norm_chars.append(ch.casefold())
            prev_space = False
        index_map.append(i)
    haystack = "".join(norm_chars)
    needle = " ".join(quote.split()).casefold()
    pos = haystack.find(needle)
    if pos == -1:
        return None
    start = index_map[pos]
    end = index_map[pos + len(needle) - 1] + 1
    return start, end


__all__ = [
    "NormalizationError", "clean_text", "clean_identifier", "parse_money", "parse_unit_price", "parse_quantity",
    "parse_rate", "parse_date", "parse_currency", "locate", "CENT",
]
