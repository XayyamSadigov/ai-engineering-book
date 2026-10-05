# path: book/projects/p1-extraction-api/extraction_api/eval/__init__.py
"""Offline evaluation: gold data, field-level metrics, threshold calibration."""
from .calibration import choose_threshold, expected_calibration_error, reliability
from .dataset import GoldInvoice, load_invoice_gold
from .metrics import FieldScore, InvoiceEvalReport, score_invoices

__all__ = [
    "choose_threshold", "expected_calibration_error", "reliability", "GoldInvoice", "load_invoice_gold",
    "FieldScore", "InvoiceEvalReport", "score_invoices",
]
