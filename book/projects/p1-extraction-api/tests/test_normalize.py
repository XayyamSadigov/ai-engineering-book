# path: book/projects/p1-extraction-api/tests/test_normalize.py
from datetime import date
from decimal import Decimal

import pytest

from extraction_api.domain.normalize import (
    NormalizationError, clean_identifier, locate, parse_currency, parse_date, parse_money,
    parse_quantity, parse_rate, parse_unit_price,
)


@pytest.mark.parametrize("raw, expected", [
    ("$1,488.00", Decimal("1488.00")),
    ("1.488,00 EUR", Decimal("1488.00")),
    ("1234,5", Decimal("12345.00")),     # one decimal after a comma reads as thousands
    ("1234,50", Decimal("1234.50")),
    ("(12.00)", Decimal("-12.00")),
    (3327.48, Decimal("3327.48")),
    ("£6,200.00", Decimal("6200.00")),
    (None, None),
    ("n/a", None),
])
def test_parse_money(raw, expected):
    assert parse_money(raw) == expected


def test_parse_money_rejects_garbage():
    with pytest.raises(NormalizationError) as exc:
        parse_money("about three hundred")
    assert exc.value.code == "INVALID_NUMBER"


def test_quantity_and_unit_price_keep_precision():
    assert parse_quantity("4.82e+07") == Decimal("48200000")
    assert str(parse_quantity("4.82e+07")) == "48200000"
    assert parse_quantity("12.40") == Decimal("12.4")
    assert parse_unit_price("0.0045") == Decimal("0.0045")  # parse_money would round this to 0.00


@pytest.mark.parametrize("raw, expected", [("8%", Decimal("0.08")), (8, Decimal("0.08")),
                                           ("0.08", Decimal("0.08")), ("20 %", Decimal("0.2"))])
def test_parse_rate(raw, expected):
    assert parse_rate(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("2026-01-14", date(2026, 1, 14)),
    ("14 Jan 2026", date(2026, 1, 14)),
    ("January 14, 2026", date(2026, 1, 14)),
    ("14/01/2026", date(2026, 1, 14)),   # 14 cannot be a month
    ("01/14/2026", date(2026, 1, 14)),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_ambiguous_date_is_refused_not_guessed():
    with pytest.raises(NormalizationError) as exc:
        parse_date("03/04/2026")
    assert exc.value.code == "AMBIGUOUS_DATE"


def test_currency_aliases_and_dollar_policy():
    assert parse_currency("€") == "EUR"
    assert parse_currency("usd") == "USD"
    assert parse_currency("$", dollar_means="CAD") == "CAD"
    with pytest.raises(NormalizationError):
        parse_currency("dollaroos")


def test_placeholder_identifiers_become_none():
    assert clean_identifier("(not provided)") is None
    assert clean_identifier("  PO-NW-2026-10412. ") == "PO-NW-2026-10412"


def test_locate_tolerates_whitespace_and_case():
    text = "Subtotal     $3,081.00\nTOTAL DUE     $3,327.48"
    start, end = locate("total due $3,327.48", text)
    assert text[start:end] == "TOTAL DUE     $3,327.48"
    assert locate("TOTAL DUE $3,999.00", text) is None
    assert locate("", text) is None
